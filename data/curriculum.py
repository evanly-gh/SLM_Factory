# data/curriculum.py
import logging
import random
from collections import Counter
from agent.cost import (
    tracked_anthropic_messages_create,
    tracked_openai_chat_create,
)
from config.config import TEACHER_MODEL_CLAUDE  # legacy hard-negative fallback only; never CoT
from data.eval_set import EvalSet, _infer_pos_label, _infer_neg_label
from data.loaders.dataset_integrity import normalize_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CoT fallback routing (paper §2.3 quality control #5, §2.5)
# Primary = local Qwen3.6 (wired in curate). Fallbacks only:
# DeepSeek V4 Flash thinking mode for math/science first; GPT-4.1 for code/QA/general first.
# ---------------------------------------------------------------------------

_MATH_SCIENCE_BENCHMARKS = {
    "gsm8k", "arcchallenge", "arc", "ai2arc", "arcc", "math", "scienceqa", "science",
}
_CODE_QA_BENCHMARKS = {"humaneval", "mbpp", "code", "triviaqa", "qa"}


def get_cot_fallbacks(
    task_type: str, benchmark: str | None = None,
) -> list[tuple[object, str]]:
    """Return ordered OpenAI-compatible fallback clients for CoT annotation.

    Paper §2.5's math/science-first specialist route now uses DeepSeek V4 Flash
    in thinking mode; GPT-4.1 remains preferred for code and general-knowledge
    tasks (e.g., HumanEval, TriviaQA).

    Qwen3.6 is always the primary backend and is supplied separately as ``generate_fn``.
    This function contains ONLY cloud fallbacks:
      - math/science: DeepSeek → OpenAI
      - code/QA/general generation: OpenAI → DeepSeek
    Missing keys omit that backend. Claude/the orchestrator is never constructed here.
    """
    from config.config import (
        DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, TEACHER_MODEL_DEEPSEEK,
        OPENAI_API_KEY, TEACHER_MODEL_GPT,
    )
    from openai import OpenAI

    bm = "".join(ch for ch in (benchmark or "").lower() if ch.isalnum())
    deepseek = (
        (OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL),
         TEACHER_MODEL_DEEPSEEK)
        if DEEPSEEK_API_KEY else None
    )
    openai = (
        (OpenAI(api_key=OPENAI_API_KEY), TEACHER_MODEL_GPT)
        if OPENAI_API_KEY else None
    )
    math_like = task_type == "math_reasoning" or (
        task_type == "generation" and bm in _MATH_SCIENCE_BENCHMARKS
    )
    ordered = [deepseek, openai] if math_like else [openai, deepseek]
    return [backend for backend in ordered if backend is not None]


def annotate_cot(
    examples: list[dict],
    teacher_client=None,
    teacher_model: str = "",
    client_type: str = "openai",
    task_type: str = "generation",
    generate_fn=None,
    fallback_teachers: list[tuple[object, str]] | None = None,
    log=print,
) -> list[dict]:
    """Add chain-of-thought reasoning to generation examples.

    Paper §2.3 quality control #5: 'A teacher model generates step-by-step reasoning
    chains for training examples, teaching the model WHY an answer is correct rather
    than only WHAT the answer is.'

    Backend order is per example: local Qwen3.6 ``generate_fn`` first, then each
    OpenAI-compatible ``fallback_teachers`` entry in order. Empty output and exceptions
    advance to the next backend. Claude/the orchestrator is never called. Generation is
    NON-FATAL: if every backend fails, the original example remains CoT-less.

    ``teacher_client``/``teacher_model`` remain as a compatibility bridge for an explicitly
    supplied OpenAI-compatible specialist; non-OpenAI clients are ignored.

    Returns examples with an added 'cot_reasoning' field. The CoT is prepended to the
    response during training formatting. The prompt is task-aware: code-generation gets
    an implementation-plan style explanation rather than a prose math-style derivation.

    Efficiency (B141): examples that ALREADY carry a non-empty 'cot_reasoning' are left
    untouched — e.g. the GSM8K loader ships real gold chains-of-thought, so re-generating
    them via the teacher was both wasteful (300 sequential calls per curate, per tier) and
    quality-reducing (gold CoT replaced by a weaker teacher CoT). The examples that DO need
    annotation are processed concurrently with a bounded thread pool.
    """
    def _build_prompt(prompt_text: str, gold_answer: str) -> str:
        if task_type == "code_generation":
            return (
                f"Explain the reasoning behind this code solution as a concise implementation "
                f"plan a developer would follow: the approach, key steps, and any edge cases "
                f"handled. Do NOT restate the full code.\n\n"
                f"Problem:\n{prompt_text}\n\n"
                f"Correct solution:\n{gold_answer}\n\n"
                f"Reply with only the step-by-step implementation reasoning, not the code and "
                f"not the final answer."
            )
        return (
            f"Solve this problem step by step, showing your reasoning clearly.\n\n"
            f"Problem: {prompt_text}\n\n"
            f"The correct answer is: {gold_answer}\n\n"
            f"Provide a clear step-by-step explanation of how to arrive at this answer. "
            f"Reply with only the reasoning steps, not the final answer."
        )

    fallbacks = list(fallback_teachers or [])
    if teacher_client is not None and teacher_model and client_type == "openai":
        fallbacks.insert(0, (teacher_client, teacher_model))
    from threading import BoundedSemaphore
    cloud_slots = BoundedSemaphore(16)

    def _present(value) -> bool:
        return value is not None and bool(str(value).strip())

    def _first_present(ex: dict, *keys: str):
        for key in keys:
            if key in ex and _present(ex[key]):
                return ex[key]
        return ""

    def _annotate_one(ex: dict) -> tuple[dict, str | None]:
        # Preserve any existing gold CoT (e.g. GSM8K) — do not regenerate.
        if _present(ex.get("cot_reasoning")):
            return ex, None
        prompt_text = _first_present(ex, "prompt", "text")
        gold_answer = _first_present(ex, "answer", "response", "label")
        if not _present(prompt_text) or not _present(gold_answer):
            return ex, None
        cot_prompt = _build_prompt(prompt_text, gold_answer)
        if generate_fn is not None:
            try:
                # LOCAL synth model (Qwen3.6-35B via vLLM) authors the CoT. Low temperature
                # for focused reasoning; 512 tokens covers verbose math/code chains.
                cot = (generate_fn(cot_prompt, 0.3, 512) or "").strip()
                if cot:
                    return {**ex, "cot_reasoning": cot}, "Qwen3.6"
            except Exception:
                pass

        for client, model in fallbacks:
            try:
                # Keep local Qwen at high concurrency while limiting paid cloud fallback
                # traffic to the conservative 16-call cap.
                with cloud_slots:
                    if model.lower().startswith("deepseek-v4"):
                        resp = tracked_openai_chat_create(
                            client,
                            stage="cot_fallback",
                            model=model,
                            messages=[{"role": "user", "content": cot_prompt}],
                            max_tokens=500,
                            reasoning_effort="high",
                            extra_body={"thinking": {"type": "enabled"}},
                        )
                    else:
                        resp = tracked_openai_chat_create(
                            client,
                            stage="cot_fallback",
                            model=model,
                            messages=[{"role": "user", "content": cot_prompt}],
                            max_tokens=500,
                        )
                cot = (resp.choices[0].message.content or "").strip()
                if cot:
                    return {**ex, "cot_reasoning": cot}, model
            except Exception:
                continue
        return ex, None

    # Only spend API calls on examples that actually need a CoT.
    need_idx = [i for i, ex in enumerate(examples)
                if not _present(ex.get("cot_reasoning"))
                and _present(_first_present(ex, "prompt", "text"))
                and _present(_first_present(ex, "answer", "response", "label"))]
    if not need_idx:
        return list(examples)

    from concurrent.futures import ThreadPoolExecutor
    annotated = list(examples)
    # The local synth server continuous-batches, so match the synth concurrency when using
    # generate_fn; cloud teachers keep the conservative 16-way cap.
    max_workers = _synth_concurrency(len(need_idx)) if generate_fn is not None else min(16, len(need_idx))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        outcomes = list(pool.map(lambda i: _annotate_one(examples[i]), need_idx))

    backend_counts: Counter = Counter()
    for i, (result, backend) in zip(need_idx, outcomes):
        annotated[i] = result
        if backend:
            backend_counts[backend] += 1
    succeeded = sum(backend_counts.values())
    log(f"      [cot] annotated={succeeded}/{len(need_idx)} "
        f"by_backend={dict(backend_counts)} failed={len(need_idx) - succeeded}")
    return annotated


def build_initial_curriculum(
    train_examples: list[dict],
    eval_set: EvalSet,
    n_total: int = 150,
    gold_fraction: float = 0.65,
    seed: int = 42,
) -> list[dict]:
    """
    Build Dcold = Dgold ∪ Dhard at 65:35 from train_examples.
    Excludes any example that appears in eval_set.
    Applies label balancing (no label > 3x any other).
    Reads task_type from eval_set.task_type internally.
    """
    task_type = eval_set.task_type
    eval_texts = {
        normalize_text(e.get("text", e.get("prompt", "")))
        for e in eval_set.all
    }
    eval_texts.discard("")
    available = [
        e
        for e in train_examples
        if normalize_text(e.get("text", e.get("prompt", ""))) not in eval_texts
    ]

    # n_gold is the gold-data portion in a mixed dataset (gold + hard negatives).
    # For the initial curriculum (training data only, no synthetic hard negatives yet),
    # we target n_total examples from the gold pool.
    n_gold = int(n_total * gold_fraction)
    # Use n_total as the selection budget so the initial dataset is large enough.
    selection_budget = n_total

    rng = random.Random(seed)

    if task_type == "classification":
        # Balance by label — same for binary and multi-class.
        # multi_label tasks also land here; individual label balance is enforced
        # in apply_quality_controls via per-label counting.
        by_label: dict[str, list[dict]] = {}
        for ex in available:
            lbl = ex.get("label", "unknown")
            by_label.setdefault(lbl, []).append(ex)

        for lbl in by_label:
            rng.shuffle(by_label[lbl])

        labels = list(by_label.keys())
        per_label = selection_budget // max(len(labels), 1)
        gold = []
        for lbl in labels:
            gold.extend(by_label[lbl][:per_label])

        selected_texts = {e["text"] for e in gold}
        remainder = [e for e in available if e["text"] not in selected_texts]
        rng.shuffle(remainder)
        gold.extend(remainder[: selection_budget - len(gold)])

    else:
        # NER, math_reasoning, code_generation, generation:
        # shuffle and take — quality controls and CoT annotation handle diversity downstream.
        shuffled = list(available)
        rng.shuffle(shuffled)
        gold = shuffled[:selection_budget]

    return apply_quality_controls(gold, task_type=task_type)


def apply_quality_controls(
    dataset: list[dict],
    task_type: str = "classification",
) -> list[dict]:
    """
    Enforce quality controls from the paper (§2.3):
    1. Label balancing: no label exceeds 3x count of any other (classification family)
    2. Context-length matching: remove outliers >3x median length (all types)
    3. Entity diversification: cap any entity value at 3 occurrences (NER)
    4. Surface-form dedup: remove near-duplicate texts (all types, Jaccard >0.9)

    New types route to their natural family:
      classification          → label balancing + dedup
      NER                     → entity diversification + length filter
      math_reasoning          → length filter on prompt + dedup
      code_generation         → length filter on prompt + dedup
      generation              → length filter on prompt
    """
    if not dataset:
        return dataset

    if task_type == "classification":
        clean = [e for e in dataset if "text" in e and "label" in e]

        # 1. Label balancing (single-label argmax — for multi_label the flag is on
        # the EvalSet, not here; we balance by the primary label field).
        counts = Counter(e["label"] for e in clean)
        if counts:
            min_count = max(min(counts.values()), 1)
            max_allowed = 3 * min_count
            balanced = []
            seen: Counter = Counter()
            for ex in clean:
                if seen[ex["label"]] < max_allowed:
                    balanced.append(ex)
                    seen[ex["label"]] += 1
            clean = balanced

        clean = _filter_length_outliers(clean)
        clean = _dedup_surface_forms(clean)
        return clean

    elif task_type == "NER":
        clean = [e for e in dataset if "text" in e and "entities" in e]

        # 3. Entity diversification: no entity surface value appears >3 times.
        # Also applies to structured_extraction (schema field values).
        entity_counts: Counter = Counter()
        for ex in clean:
            for ent in ex.get("entities", []):
                entity_counts[ent.get("text", "").lower()] += 1
        over_represented = {k for k, v in entity_counts.items() if v > 3}
        if over_represented:
            result = []
            running: Counter = Counter()
            for ex in clean:
                ent_values = [e.get("text", "").lower() for e in ex.get("entities", [])]
                skip = False
                for v in ent_values:
                    if v in over_represented and running[v] >= 3:
                        skip = True
                        break
                if not skip:
                    result.append(ex)
                    for v in ent_values:
                        running[v] += 1
            clean = result

        clean = _filter_length_outliers(clean)
        return clean

    elif task_type in ("math_reasoning", "code_generation", "generation"):
        clean = [
            e for e in dataset
            if ("prompt" in e and "response" in e) or ("text" in e and "label" in e)
        ]
        key = "text" if task_type == "code_generation" else "prompt"
        clean = _filter_length_outliers(clean, key=key)
        # Dedup on math/code — repeated problem templates inflate the dataset
        # without adding coverage. Generation is diverse enough to skip dedup.
        if task_type in ("math_reasoning", "code_generation"):
            clean = _dedup_surface_forms(clean, key=key)
        return clean

    else:
        return dataset


def _filter_length_outliers(
    examples: list[dict], key: str = "text", max_ratio: float = 3.0,
) -> list[dict]:
    """Remove examples whose text length exceeds max_ratio × median. Paper §2.3 item 3."""
    lengths = [len(e.get(key, "")) for e in examples]
    if not lengths:
        return examples
    lengths.sort()
    median = lengths[len(lengths) // 2] or 1
    cutoff = median * max_ratio
    return [e for e in examples if len(e.get(key, "")) <= cutoff]


def _dedup_surface_forms(
    examples: list[dict], threshold: float = 0.9, key: str = "text"
) -> list[dict]:
    """Remove near-duplicate texts using word-set Jaccard similarity."""
    result = []
    seen_word_sets: list[set] = []
    for ex in examples:
        words = set(ex.get(key, "").lower().split())
        if not words:
            result.append(ex)
            continue
        is_dup = False
        for seen in seen_word_sets[-50:]:
            intersection = len(words & seen)
            union = len(words | seen)
            if union > 0 and intersection / union >= threshold:
                is_dup = True
                break
        if not is_dup:
            result.append(ex)
            seen_word_sets.append(words)
    return result


def _synth_concurrency(n_tasks: int) -> int:
    """How many hard-negative generations to run CONCURRENTLY against the synth endpoint.

    Synthesis is I/O-bound on the vLLM server, which continuous-batches concurrent requests
    (its --max-num-seqs). Firing them one-at-a-time was the single biggest wall-clock cost
    (~15 min/rebuild in run 37372065); at concurrency W it becomes ~n/W round-trips. Tunable
    via SLM_SYNTH_CONCURRENCY (default 16); set higher when the server has more GPUs / a
    larger max-num-seqs. Capped at the number of tasks; 1 restores fully-sequential behavior.
    """
    import os
    try:
        c = int(os.environ.get("SLM_SYNTH_CONCURRENCY", "16"))
    except (TypeError, ValueError):
        c = 16
    return max(1, min(c, max(1, n_tasks)))


def synthesize_hard_negatives(
    examples: list[dict],
    n: int,
    anthropic_client=None,
    task_type: str = "classification",
    pattern_hint: str = "",
    temperature: float = 1.0,
    generate_fn=None,
    source_label: str = "synth",
) -> list[dict]:
    """
    Generate hard negatives using the 2-for-1 rule (paper §2.3).

    For each challenging case, returns BOTH the original gold example AND one
    synthetic hard negative — a contrastive pair that teaches the model what
    TO predict and what NOT to predict for similar surface forms.

    Returns up to 2*n examples (n originals + n synthetics). Each synthetic example is
    tagged with `_source=source_label` for data-lineage logging (B161).

    Generation backend (B161): if `generate_fn` is provided (a
    `generate(prompt, temperature, max_tokens) -> str` from the LOCAL vLLM synthesis
    endpoint), it is used. Otherwise falls back to `anthropic_client` (legacy path / tests).

    `temperature` controls generation diversity. curate_node rotates it across
    successive data_rebuild rounds so a rebuild produces DIFFERENT negatives each time.
    """
    # Synthesis must be NON-FATAL (B161): a per-call failure (endpoint down, proxy 5xx,
    # timeout) must SKIP that example, not crash curate. _gen returns None on failure; the
    # loops skip synthetics when None, and bail early after too many consecutive failures
    # (endpoint effectively dead) so we degrade to gold-only instead of hammering a dead server.
    _fail_state = {"consecutive": 0, "aborted": False}
    _MAX_CONSEC_FAILS = 3

    def _gen(prompt: str, max_tokens: int):
        if _fail_state["aborted"]:
            return None
        try:
            if generate_fn is not None:
                out = generate_fn(prompt, temperature, max_tokens)
            else:
                response = tracked_anthropic_messages_create(
                    anthropic_client.messages,
                    stage="hard_negative_synthesis",
                    model=TEACHER_MODEL_CLAUDE,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                out = response.content[0].text.strip()
            _fail_state["consecutive"] = 0
            return out
        except Exception as e:  # noqa: BLE001
            _fail_state["consecutive"] += 1
            if _fail_state["consecutive"] >= _MAX_CONSEC_FAILS:
                _fail_state["aborted"] = True
                logger.warning("synthesize_hard_negatives: %d consecutive generation failures "
                               "(%s) — aborting synthesis, returning gold-only",
                               _fail_state["consecutive"], str(e)[:80])
            return None

    results = []

    if task_type == "classification":
        # Use the full set of examples (all labels) as source material, not just
        # the minority class. Generate a boundary-crossing counterexample for each:
        # given an example with label X, synthesize one that superficially resembles
        # it but belongs to a different label. This covers all failure modes, not just
        # minority-class confusion.
        all_labels = list({e.get("label") for e in examples if e.get("label")})
        candidates = examples[:n]
        pattern_hint = (
            f"\nThe example should specifically exercise this failure mode: "
            f"{pattern_hint}. Construct text that a model failing in that way "
            f"would misclassify.\n"
            if pattern_hint else ""
        )

        def _synth_one(ex):
            src_label = ex.get("label", "unknown")
            # Pick a target label different from the source
            target_labels = [l for l in all_labels if l != src_label]
            target_label = target_labels[0] if target_labels else src_label
            prompt = (
                f"You are generating a HARD NEGATIVE for a text classifier: a realistic "
                f"example that superficially resembles the '{src_label}' class but genuinely "
                f"belongs to the '{target_label}' class. The surface features should mislead "
                f"toward '{src_label}' while the true meaning is unambiguously '{target_label}'."
                f"{pattern_hint}\n\n"
                f"Reference '{src_label}' example:\n{ex['text']}\n\n"
                f"Output ONLY the new example text for the '{target_label}' class — no preamble, "
                f"no explanation, no quotation marks, no label prefix."
            )
            generated_text = _gen(prompt, 200)
            if generated_text:  # skip the synthetic on generation failure (non-fatal)
                return ex, {"text": generated_text, "label": target_label, "_source": source_label}
            return ex, None

        # Run the 2-for-1 generations CONCURRENTLY (vLLM continuous-batches them). Order is
        # preserved so each gold example stays adjacent to its synthetic counterpart.
        workers = _synth_concurrency(len(candidates))
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pairs = list(pool.map(_synth_one, candidates))
        else:
            pairs = [_synth_one(ex) for ex in candidates]
        for ex, synth in pairs:
            results.append(ex)
            if synth is not None:
                results.append(synth)

    elif task_type == "NER":
        # Hard negatives for NER are HARDER-TO-TAG examples with CORRECT labels — a
        # passage where the same entity mentions sit in a more ambiguous context, with
        # the gold entity types the model SHOULD predict. (An earlier version asked for
        # WRONG types and stored them as targets, which trains the model to mis-tag —
        # negative transfer. We keep labels correct so the SFT signal is positive.)
        candidates = examples[:n]
        for ex in candidates:
            original_entities = ex.get("entities", [])
            valid_types = {
                str(entity.get("type")).strip()
                for entity in original_entities
                if isinstance(entity, dict) and str(entity.get("type", "")).strip()
            }
            entity_desc = ", ".join(
                f'"{e.get("text", "")}" ({e.get("type", "")})' for e in original_entities[:5]
            ) if original_entities else "unknown entities"
            prompt = (
                f"You are generating a HARD training example for a named-entity recognizer. "
                f"Rewrite the passage so the SAME entities appear in a more ambiguous or "
                f"confusable context (e.g. a word that could read as either a company or a "
                f"common noun), so their correct type is harder to infer from surface form "
                f"alone — but keep each entity's CORRECT type unchanged.\n\n"
                f"Original entities (text → correct type): {entity_desc}\n"
                f"Original passage: {ex.get('text', '')}\n\n"
                f"Aggregate pattern to emphasize: {pattern_hint or 'general ambiguity'}\n"
                f"Reply with JSON only: {{\"text\": \"<rewritten passage>\", "
                f"\"entities\": [{{\"text\": \"<span>\", \"type\": \"<CORRECT_TYPE>\"}}]}}\n"
                f"The 'entities' list must contain the spans with their TRUE types (what the "
                f"model SHOULD predict for the rewritten passage). JSON only, no prose."
            )
            raw = _gen(prompt, 400)
            import json as _json, re as _re
            results.append(ex)
            if not raw:  # generation failed — keep gold, skip synthetic (non-fatal)
                continue
            try:
                match = _re.search(r'\{.*\}', raw, _re.DOTALL)
                parsed = _json.loads(match.group()) if match else None
            except (TypeError, ValueError):
                continue
            if not isinstance(parsed, dict):
                continue
            rewritten = parsed.get("text")
            entities = parsed.get("entities")
            if (
                not isinstance(rewritten, str)
                or not rewritten.strip()
                or not isinstance(entities, list)
                or not entities
                or not valid_types
            ):
                continue
            valid_entities = all(
                isinstance(entity, dict)
                and isinstance(entity.get("text"), str)
                and bool(entity["text"].strip())
                and entity["text"] in rewritten
                and isinstance(entity.get("type"), str)
                and entity["type"].strip() in valid_types
                for entity in entities
            )
            if not valid_entities:
                continue
            results.append({
                "text": rewritten,
                "entities": entities,
                "_source": source_label,
            })

    elif task_type == "math_reasoning":
        # SFT on wrong answers actively harms math models — skip wrong-answer negatives.
        # Instead, return the gold examples as-is (CoT annotation in curate_node provides
        # the real augmentation value for math tasks).
        if pattern_hint:
            logger.warning(
                "[curriculum] Pattern-guided positive synthesis is not supported for "
                "math_reasoning; "
                "returning gold examples unchanged."
            )
        return list(examples[:n]) if n < len(examples) else list(examples)

    elif task_type == "code_generation":
        # Wrong-code SFT examples teach the model to produce bugs — skip.
        if pattern_hint:
            logger.warning(
                "[curriculum] Pattern-guided positive synthesis is not supported for "
                "code_generation; "
                "returning gold examples unchanged."
            )
        return list(examples[:n]) if n < len(examples) else list(examples)

    elif task_type == "generation":
        # Rejected answers require a preference objective. Until one exists, open
        # generation remains gold/CoT-only and this compatibility API returns anchors
        # unchanged rather than turning plausible wrong answers into positive SFT targets.
        if pattern_hint:
            logger.warning(
                "[curriculum] Pattern-guided positive synthesis is not supported for "
                "generation without "
                "verified-positive synthesis or preference training; returning gold "
                "examples unchanged."
            )
        return list(examples[:n]) if n < len(examples) else list(examples)

    return results
