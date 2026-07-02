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
                        replay_count: int = 0, surgical_added: int = 0):
    """Print a structured dataset composition report."""
    total = len(dataset)
    gold_ratio = n_gold / total * 100 if total else 0
    hard_ratio = n_hard / total * 100 if total else 0

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

        _log(model_id, f"DATA REBUILD: building fresh curriculum")
        _log(model_id, f"  task_type={task_type}  N_TOTAL={N_TOTAL}  "
             f"gold_target={n_gold_target}  hard_target={n_hard_target}")
        if failures:
            _log(model_id, f"  Using {min(len(failures), n_hard_target)} failure examples as hard-negative seeds")

        gold = build_initial_curriculum(
            train_examples, eval_set, n_total=n_gold_target, gold_fraction=1.0,
        )
        _log(model_id, f"  Gold examples built: {len(gold)}")

        source_examples = (failures[:n_hard_target] if len(failures) >= n_hard_target
                           else failures + train_examples[:n_hard_target - len(failures)])
        hard = synthesize_hard_negatives(
            source_examples,
            n=n_hard_target,
            anthropic_client=client,
            task_type=task_type,
        )
        _log(model_id, f"  Hard negatives synthesized: {len(hard)}")

        if task_type in ("math_reasoning", "code_generation", "generation"):
            plan = state.get("task_plan") or {}
            benchmark = plan.get("benchmark", plan.get("task_name", ""))
            teacher_client, teacher_model, client_type = get_teacher_client(task_type, benchmark)
            _log(model_id, f"  CoT annotation: teacher={teacher_model} ({client_type})")
            gold = annotate_cot(gold, teacher_client, teacher_model, client_type)

        dataset = apply_quality_controls(gold + hard, task_type=task_type)
        n_hard_added = len(hard)

    elif intervention == "surgical":
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

    # Save dataset
    state["dataset_version"] += 1
    path = os.path.join(ARTIFACTS_DIR, f"dataset_v{state['dataset_version']}.jsonl")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(path, "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")
    state["current_dataset_path"] = path

    n_hard = min(n_hard_added, len(dataset))
    n_gold = len(dataset) - n_hard
    state["last_curation"] = {
        "n_gold": n_gold,
        "n_hard": n_hard,
        "label_dist": dict(Counter(ex.get("label", "?") for ex in dataset)),
    }

    _log_dataset_report(
        model_id, dataset, n_gold, n_hard,
        replay_count=replay_count,
        surgical_added=n_hard_added if intervention == "surgical" else 0,
    )
    _log(model_id, f"  Saved: {path}")

    return state
