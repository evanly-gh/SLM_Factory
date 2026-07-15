# agent/nodes/curate.py
import json
import os
from collections import Counter
from agent.state import AgentState
from data.curriculum import (
    build_initial_curriculum,
    synthesize_hard_negatives,
    apply_quality_controls,
    get_teacher_client,
    annotate_cot,
)

ARTIFACTS_DIR = "artifacts"


def _log(model_id: str, msg: str):
    print(f"[curate][{model_id}] {msg}")


def _log_dataset_report(model_id: str, dataset: list[dict], n_gold: int, n_hard: int,
                        task_type: str = "classification",
                        replay_count: int = 0, surgical_added: int = 0):
    """Print a structured dataset composition report."""
    total = len(dataset)
    gold_ratio = n_gold / total * 100 if total else 0
    hard_ratio = n_hard / total * 100 if total else 0

    if task_type == "NER":
        # NER examples carry entities, not a single label; report entity-type counts.
        label_dist: Counter = Counter()
        for ex in dataset:
            ents = ex.get("entities", [])
            if not ents:
                label_dist["no_entity"] += 1
            for e in ents:
                label_dist[e.get("type", "?")] += 1
    else:
        label_dist = Counter(ex.get("label", ex.get("type", "?")) for ex in dataset)
    lengths = [len(ex.get("text", ex.get("prompt", ""))) for ex in dataset]
    mean_len = sum(lengths) / len(lengths) if lengths else 0
    sorted_lens = sorted(lengths)
    median_len = sorted_lens[len(sorted_lens) // 2] if sorted_lens else 0

    _log(model_id, f"  ┌─ Dataset report ─────────────────────────")
    _log(model_id, f"  │ Total examples : {total}")
    _log(model_id, f"  │ Gold           : {n_gold} ({gold_ratio:.1f}%)")
    _log(model_id, f"  │ Hard negatives : {n_hard} ({hard_ratio:.1f}%)")
    if replay_count:
        _log(model_id, f"  │ Replay buffer  : {replay_count}")
    if surgical_added:
        _log(model_id, f"  │ Surgical added : {surgical_added}")
    _log(model_id, f"  │ Label distribution:")
    for label, count in sorted(label_dist.items(), key=lambda x: -x[1]):
        _log(model_id, f"  │   {label}: {count}")
    _log(model_id, f"  │ Text length    : mean={mean_len:.0f}  median={median_len}")
    _log(model_id, f"  └────────────────────────────────────────────")


def curate_node(state: AgentState) -> AgentState:
    """
    Node 3: build or augment Dcold = Dgold ∪ Dhard based on current intervention type.
    Writes updated dataset to disk. Increments dataset_version.
    """
    import anthropic
    from config.config import ANTHROPIC_API_KEY

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    task_type = state["task_type"]
    selected = state["selected_model"]
    if selected is None:
        raise RuntimeError("curate_node called before task_analysis selected a model")
    model_id = selected.model_id
    intervention = state.get("last_intervention", "data_rebuild")
    eval_set = state["eval_set"]
    train_examples = state["train_examples"]
    failures = state["last_eval"].failures if state.get("last_eval") else []

    llm_decision = state.get("llm_iterate_decision") or {}
    targeted_pattern = llm_decision.get("targeted_patterns") or ""

    if intervention == "data_rebuild" or state["current_dataset_path"] is None:
        if eval_set is None:
            raise RuntimeError(
                "curate_node: eval_set is None in production mode — "
                "production data_rebuild is not supported; curate_node must not be called "
                "with intervention=data_rebuild before eval_set is populated."
            )
        N_TOTAL_BY_TYPE = {
            "classification":             150,
            "multi_label_classification": 300,
            "NER":                        300,
            "structured_extraction":      400,
            "math_reasoning":            1000,
            "code_generation":           1000,
            "multilingual":               400,
            "generation":                1000,
        }
        N_TOTAL = N_TOTAL_BY_TYPE.get(task_type, 150)
        n_gold_target = int(N_TOTAL * 0.65)
        n_hard_target = N_TOTAL - n_gold_target

        # Vary the sampling seed each rebuild so successive data_rebuild rounds do not
        # regenerate a byte-identical gold slice (which would make re-training pointless
        # and feed the rollback→re-train loop). When the acquired corpus is larger than
        # the gold target, this reshuffles which examples are drawn; when the corpus is
        # smaller than the target (all examples used regardless), the gold set is
        # necessarily the same and only the synthetic hard negatives vary.
        rebuild_seed = 42 + int(state.get("dataset_version", 0))
        n_available = len(train_examples)

        _log(model_id, f"DATA REBUILD: building fresh curriculum (seed={rebuild_seed})")
        _log(model_id, f"  task_type={task_type}  N_TOTAL={N_TOTAL}  "
             f"gold_target={n_gold_target}  hard_target={n_hard_target}")
        # Provenance: where the underlying examples came from + how each slice is produced.
        _log(model_id, f"  Data source (gold): {state.get('data_source', 'unknown')}")
        _log(model_id, f"  Acquired train examples available for gold: {n_available}")
        _log(model_id, f"  Hard negatives: SYNTHETIC — generated by the orchestrator LLM "
                       f"(contrastive 2-for-1 rule); not scraped/downloaded")
        if failures:
            _log(model_id, f"  Using {min(len(failures), n_hard_target)} failure examples as hard-negative seeds")

        gold = build_initial_curriculum(
            train_examples, eval_set, n_total=n_gold_target, gold_fraction=1.0,
            seed=rebuild_seed,
        )
        _log(model_id, f"  Gold examples built: {len(gold)} (drawn from {n_available} acquired examples)")
        if len(gold) < n_gold_target:
            _log(model_id,
                 f"  ⚠ gold capped at {len(gold)} < target {n_gold_target}: only {n_available} real "
                 f"examples were acquired for this task (see [acquire] log for the source). "
                 f"Raise the acquisition cap (web_acquire.max_train / n_per_label) to grow the pool.")

        # Preserve the intended gold:hard ratio (~65:35) even when gold is scarce.
        # Generating the full hard target against a tiny gold set produces a dataset that
        # is mostly synthetic negatives (the NER run was 100% hard after quality controls),
        # which is both unbalanced and misreported. Scale the hard target to the ACTUAL
        # gold count so the proportion stays right; only hit the full target once gold does.
        hard_ratio = n_hard_target / max(n_gold_target, 1)          # ~0.538 for 65:35
        n_hard_effective = min(n_hard_target, max(1, round(len(gold) * hard_ratio)))
        if n_hard_effective != n_hard_target:
            _log(model_id, f"  Hard target scaled {n_hard_target} → {n_hard_effective} to preserve "
                           f"~{n_gold_target}:{n_hard_target} gold:hard ratio against {len(gold)} gold")

        # Draw hard-negative seed examples with the same rotating seed so the synthetic
        # negatives also differ across rebuilds when the failure/train pool has surplus.
        import random as _random
        _pool = list(failures) + list(train_examples)
        _random.Random(rebuild_seed).shuffle(_pool)
        source_examples = _pool[:n_hard_effective] if len(_pool) >= n_hard_effective else _pool
        hard = synthesize_hard_negatives(
            source_examples,
            n=n_hard_effective,
            anthropic_client=client,
            task_type=task_type,
        )
        _log(model_id, f"  Hard negatives synthesized: {len(hard)} (source: LLM {'; math/code return gold unchanged' if task_type in ('math_reasoning', 'code_generation') else 'contrastive generation'})")

        if task_type in ("math_reasoning", "code_generation", "generation"):
            plan = state.get("task_plan") or {}
            benchmark = plan.get("benchmark", plan.get("task_name", ""))
            teacher_client, teacher_model, client_type = get_teacher_client(task_type, benchmark)
            _log(model_id, f"  CoT annotation: teacher={teacher_model} ({client_type})")
            gold = annotate_cot(gold, teacher_client, teacher_model, client_type, task_type=task_type)

        # Tag provenance BEFORE quality controls so the composition report reflects what
        # actually survived filtering, instead of inferring gold = total − hard (which
        # underflowed to 0 gold whenever synth count exceeded the filtered total — B131).
        for ex in gold:
            if isinstance(ex, dict):
                ex["_slice"] = "gold"
        for ex in hard:
            if isinstance(ex, dict):
                ex.setdefault("_slice", "hard")

        dataset = apply_quality_controls(gold + hard, task_type=task_type)
        n_hard_added = len(hard)

    elif intervention == "surgical":
        if not failures:
            _log(model_id, "SURGICAL: no failures to target — holding dataset")
            return state

        with open(state["current_dataset_path"]) as f:
            dataset = [json.loads(line) for line in f if line.strip()]
        n_surgical = min(max(len(failures) * 2, 10), 20)

        _log(model_id, f"SURGICAL: augmenting existing dataset (v{state['dataset_version']})")
        _log(model_id, f"  Failure count: {len(failures)}")
        _log(model_id, f"  Targeting {n_surgical} new examples")
        if targeted_pattern:
            _log(model_id, f"  LLM targeted pattern: {targeted_pattern}")

        targeted = synthesize_hard_negatives(
            failures[:n_surgical],
            n=n_surgical,
            anthropic_client=client,
            task_type=task_type,
            targeted_pattern=targeted_pattern,
        )
        dataset = apply_quality_controls(dataset + targeted, task_type=task_type)
        n_hard_added = len(targeted)
        _log(model_id, f"  Synthesized {len(targeted)} targeted examples")

    else:
        _log(model_id, f"SKIP: intervention={intervention} — dataset held fixed")
        return state

    # Mix replay buffer for production mode
    replay = state.get("replay_buffer") or []
    replay_count = 0
    if state.get("mode") == "production" and replay:
        dataset = dataset + replay
        replay_count = len(replay)
        _log(model_id, f"  Mixed {replay_count} replay examples (production mode)")

    # Composition accounting: prefer the provenance tags set on the data_rebuild path
    # (accurate after quality-control filtering); fall back to the count-based estimate
    # for the surgical path (which appends to a dataset loaded from disk without tags).
    tagged_gold = sum(1 for ex in dataset if isinstance(ex, dict) and ex.get("_slice") == "gold")
    tagged_hard = sum(1 for ex in dataset if isinstance(ex, dict) and ex.get("_slice") == "hard")
    if tagged_gold or tagged_hard:
        n_gold, n_hard = tagged_gold, tagged_hard
    else:
        n_hard = min(n_hard_added, len(dataset))
        n_gold = len(dataset) - n_hard

    # Strip the transient provenance tag so it never lands in the training JSONL.
    for ex in dataset:
        if isinstance(ex, dict):
            ex.pop("_slice", None)

    # Save dataset
    state["dataset_version"] += 1
    path = os.path.join(ARTIFACTS_DIR, f"dataset_v{state['dataset_version']}.jsonl")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(path, "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")
    state["current_dataset_path"] = path

    def _label_key(ex):
        if task_type == "NER":
            types = [e.get("type", "?") for e in ex.get("entities", [])]
            return types[0] if types else "no_entity"
        return ex.get("label", ex.get("type", "?"))
    state["last_curation"] = {
        "n_gold": n_gold,
        "n_hard": n_hard,
        "label_dist": dict(Counter(_label_key(ex) for ex in dataset)),
    }

    _log_dataset_report(
        model_id, dataset, n_gold, n_hard, task_type=task_type,
        replay_count=replay_count,
        surgical_added=n_hard_added if intervention == "surgical" else 0,
    )
    _log(model_id, f"  Saved: {path}")

    return state
