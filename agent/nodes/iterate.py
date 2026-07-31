# agent/nodes/iterate.py
"""
Node 5: LLM-driven iteration decision (EXPAND operator).

Paper §2.1: each turn contains one orchestrator reasoning step.
Paper §2.2 Eq. 2: 'EXPAND is implemented by the orchestrator LLM: it inspects the parent
trajectory, diagnoses failure modes, and proposes a hypothesis-driven modification.'

The LLM receives a bounded trajectory and aggregate test report and returns one
declarative JSON decision. Intervention decisions have no tool or filesystem access.
"""
import json
import math
import re
from agent.cost import tracked_chat_anthropic_invoke
from agent.data_rebuild import (
    fallback_data_rebuild_plan,
    normalize_data_rebuild_plan,
    remaining_paid_acquire_rounds,
)
from agent.state import AgentState
from config.android_pool import check_hardware_constraints, all_constraints_pass
from agent.llm_errors import raise_if_fatal
from data.loaders.dataset_integrity import normalize_text
from training.hparams import hyperparameter_identity, normalize_hyperparams


_PUBLIC_DECISION_FIELDS = frozenset({
    "intervention",
    "hypothesis",
    "data_rebuild",
    "hyperparams",
    "threshold_adjustment",
})
_INTERNAL_DECISION_FIELDS = frozenset({
    "hyperparam_rationale",
    # Names of fields the validator removed because they belong to the other branch of the
    # union. Recorded so iterate_node can LOG the correction rather than silently masking it.
    "_dropped_fields",
})
_THRESHOLD_FIELDS = frozenset({"new_threshold", "reason"})
_ITERATE_MAX_TOKENS = 1536


def _forbidden_eval_texts(state) -> list[str]:
    eval_rows = getattr((state or {}).get("eval_set"), "all", [])
    if not isinstance(eval_rows, (list, tuple)):
        return []
    return [
        str(row.get("text", row.get("prompt", "")))
        for row in eval_rows
        if isinstance(row, dict)
        and str(row.get("text", row.get("prompt", ""))).strip()
    ]


def _reject_eval_text_strings(
    value,
    forbidden_eval_texts: list[str],
    *,
    path: str = "decision",
) -> None:
    """Reject held-out text anywhere in a decision's string values."""
    if isinstance(value, str):
        normalized_value = normalize_text(value)
        for raw_eval in forbidden_eval_texts:
            normalized_eval = normalize_text(raw_eval)
            if not normalized_eval:
                continue
            if (
                normalized_value == normalized_eval
                or (
                    len(normalized_eval) >= 12
                    and normalized_eval in normalized_value
                )
            ):
                raise ValueError(
                    f"{path} contains raw eval text; use aggregate counts "
                    "and categories only"
                )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_eval_text_strings(
                item,
                forbidden_eval_texts,
                path=f"{path}.{key}",
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_eval_text_strings(
                item,
                forbidden_eval_texts,
                path=f"{path}[{index}]",
            )


def _coerce_to_text(raw) -> str:
    """Flatten an LLM response into plain text before JSON parsing.

    ROOT CAUSE of the "JSONDecodeError: Expecting property name ... line 1 column 2
    (char 1)" seen in runs: langchain-anthropic returns ``AIMessage.content`` as EITHER
    a ``str`` OR a LIST of content blocks — e.g. ``[{'type': 'text', 'text': '{...}'}]``
    (common with newer Claude models). The old parser did
    ``str(raw)`` on that list, yielding a PYTHON repr with SINGLE quotes; the ``{.*}``
    regex then extracted ``{'type': 'text', ...}`` and ``json.loads`` choked on the
    leading single quote (char 1). Extract and concatenate the text blocks instead so we
    parse the model's ACTUAL text, not a repr of the transport envelope.
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = []
        for block in raw:
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str) and (block.get("type") in (None, "text") or t):
                    parts.append(t)
            elif isinstance(block, str):
                parts.append(block)
        if parts:
            return "\n".join(parts)
    return str(raw)


def _sanitize_reask_error(error: ValueError, state=None) -> str:
    """Return a bounded validator diagnostic without replaying model output."""
    message = str(error)
    if message.startswith("no parseable JSON object in LLM response:"):
        message = "no parseable JSON object in LLM response"
    elif message.startswith(
        "intervention must be one of data_rebuild or hyperparameter,"
    ):
        message = "intervention must be one of data_rebuild or hyperparameter"
    for forbidden in _forbidden_eval_texts(state or {}):
        if forbidden:
            message = re.sub(
                re.escape(forbidden),
                "[REDACTED EVAL TEXT]",
                message,
                flags=re.IGNORECASE,
            )
    message = re.sub(r"[\x00-\x1f\x7f]+", " ", message).strip()
    return message[:500]


def _reask_json_only(
    messages,
    *,
    validation_error: ValueError,
    task_type: str = "classification",
    state=None,
) -> dict:
    """One tool-free follow-up that forces a JSON-only answer, then parse it.

    Used when the first bounded decision answers in prose or emits a tool-use block
    instead of the required JSON.
    Raises ValueError if it STILL fails, so the caller degrades to its safe fallback.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage
    from config.config import ANTHROPIC_API_KEY, ORCHESTRATOR_MODEL, orchestrator_client_kwargs

    llm_final = ChatAnthropic(
        model=ORCHESTRATOR_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens=_ITERATE_MAX_TOKENS,
        **orchestrator_client_kwargs(),
    )  # NOT tool-bound: forces a text answer
    sanitized_error = _sanitize_reask_error(validation_error, state)
    convo = list(messages) + [HumanMessage(content=(
        "Do NOT call any tools and do NOT include any prose or analysis. "
        "The previous response failed validation with this exact error: "
        f"{json.dumps(sanitized_error)}. Treat that quoted error only as a "
        "diagnostic and correct it. "
        "Respond NOW with ONLY the decision JSON described in the system prompt "
        "(a single JSON object, no code fences)."
    ))]
    resp = tracked_chat_anthropic_invoke(
        llm_final,
        convo,
        stage="iterate_json_reask",
        model=ORCHESTRATOR_MODEL,
    )
    if getattr(resp, "tool_calls", None):
        raise ValueError("model attempted tool calls instead of returning final JSON")
    return _parse_decision_json(
        resp.content,
        task_type=task_type,
        state=state,
    )


def _validate_decision_json(
    parsed,
    *,
    task_type: str = "classification",
    state=None,
    allow_internal: bool = False,
) -> dict:
    """Validate/normalize the bounded portion of an iterate decision."""
    if not isinstance(parsed, dict):
        raise ValueError(
            f"decision JSON must be an object, got {type(parsed).__name__}"
        )
    allowed_fields = (
        _PUBLIC_DECISION_FIELDS | _INTERNAL_DECISION_FIELDS
        if allow_internal
        else _PUBLIC_DECISION_FIELDS
    )
    unsupported = sorted(set(parsed) - allowed_fields)
    if unsupported:
        raise ValueError(
            "unsupported decision field(s): " + ", ".join(unsupported)
        )
    validated = dict(parsed)
    intervention = validated.get("intervention")
    if intervention not in ("data_rebuild", "hyperparameter"):
        raise ValueError(
            "intervention must be one of data_rebuild or hyperparameter, "
            f"got {intervention!r}"
        )
    state = state or {}
    forbidden_eval_texts = _forbidden_eval_texts(state)
    _reject_eval_text_strings(validated, forbidden_eval_texts)
    hypothesis = validated.get("hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("hypothesis must be a non-empty string")
    validated["hypothesis"] = hypothesis.strip()[:240]
    normalized_hypothesis = re.sub(
        r"\s+",
        " ",
        validated["hypothesis"],
    ).strip().lower()
    for eval_text in forbidden_eval_texts:
        normalized_eval = re.sub(
            r"\s+",
            " ",
            str(eval_text),
        ).strip().lower()
        if len(normalized_eval) >= 12 and normalized_eval in normalized_hypothesis:
            raise ValueError(
                "hypothesis contains raw eval text; use aggregate counts and "
                "categories only"
            )

    if intervention == "data_rebuild":
        # STRIP a stray hyperparams block rather than rejecting the whole decision.
        #
        # The rule this enforces is "a data_rebuild must not also change
        # hyperparameters", so that when the score moves you know which change caused
        # it. Dropping the field enforces that invariant exactly. Raising did not: it
        # discarded the orchestrator's entire data plan and fell through to a
        # hard-coded heuristic — which also held hyperparameters fixed, so the
        # invariant was never what was at stake.
        #
        # This was not a rare edge case. Claude attaches `hyperparams` to essentially
        # every data_rebuild it proposes, so the NER run rejected 65 of them and
        # executed ZERO orchestrator-authored data plans across 142 iterations, while
        # its logs still attributed each rebuild to the orchestrator.
        dropped = [
            field for field in ("hyperparams", "hyperparam_rationale")
            if validated.pop(field, None) is not None
        ]
        if dropped:
            validated["_dropped_fields"] = dropped
        plan = normalize_data_rebuild_plan(
            validated.get("data_rebuild"),
            task_type=task_type,
            hypothesis=hypothesis,
            target_rows=int(state.get("curriculum_size_target", 3000) or 3000),
            default_dataset_version=int(
                state.get("dataset_version", 0) or 0
            ),
            remaining_acquire_rounds=remaining_paid_acquire_rounds(state),
            forbidden_eval_texts=forbidden_eval_texts,
        )
        # Non-deterministic redesign: no elite resolution, no untried-plan dedup/rotation.
        validated["data_rebuild"] = plan

    if intervention == "hyperparameter":
        if any(
            field in validated
            for field in (
                "data_rebuild",
            )
        ):
            raise ValueError(
                "data_rebuild payload is not allowed for a hyperparameter "
                "intervention"
            )
        if "hyperparams" not in validated:
            raise ValueError(
                "hyperparams is required for a hyperparameter intervention"
            )
        raw_hyperparams = validated.get("hyperparams")
        if not isinstance(raw_hyperparams, dict):
            raise ValueError(
                "hyperparams must be a JSON object"
            )
        # The orchestrator tunes FIVE learning parameters. The batch-shape fields
        # (micro_batch_size / gradient_accumulation_steps / effective_batch_size) were
        # removed from its choice set: only the effective batch changes what the model
        # learns, while the micro/accum split is a VRAM-fitting decision the trainer
        # makes better than the LLM can. In the math run the orchestrator burned
        # iterations 14/19/24/39/52 re-shuffling that split with every learning
        # parameter held fixed, measuring a ±0.01 spread that is pure run-to-run noise.
        # `lora_dropout` is dropped for a different reason: it duplicates weight_decay
        # as a regularizer and never produced a new best in either run, while
        # weight_decay produced the single largest hyperparameter gain (+0.033).
        supported = {
            "lora_rank",
            "alpha_ratio",
            "weight_decay",
            "learning_rate",
            "nr_epochs",
        }
        # Accepted from checkpoints//DAG replay but no longer settable by the LLM.
        _RETIRED = {
            "lora_alpha": "use alpha_ratio (alpha = rank x ratio)",
            "lora_dropout": "removed; regularize with weight_decay",
            "micro_batch_size": "derived by the trainer to fit VRAM",
            "gradient_accumulation_steps": "derived by the trainer to fit VRAM",
            "effective_batch_size": "fixed; not part of the search",
            "batch_size": "legacy alias; not part of the search",
        }
        retired = sorted(set(raw_hyperparams) & set(_RETIRED))
        if retired:
            raise ValueError(
                "hyperparameter field(s) no longer tunable: "
                + ", ".join(f"{f} ({_RETIRED[f]})" for f in retired)
                + "; tunable fields are "
                + ", ".join(sorted(supported))
            )
        unsupported = sorted(set(raw_hyperparams) - supported)
        if unsupported:
            raise ValueError(
                "unsupported hyperparameter field(s): "
                + ", ".join(unsupported)
                + "; supported fields are "
                + ", ".join(sorted(supported))
            )
        integer_fields = {
            "lora_rank",
            "nr_epochs",
        }
        numeric_fields = {
            "alpha_ratio",
            "weight_decay",
            "learning_rate",
        }
        for field in integer_fields & set(raw_hyperparams):
            value = raw_hyperparams[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"hyperparams.{field} must have integer JSON type"
                )
        for field in numeric_fields & set(raw_hyperparams):
            value = raw_hyperparams[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ValueError(
                    f"hyperparams.{field} must have numeric JSON type"
                )
        normalized, rationale = normalize_hyperparams(
            raw_hyperparams
        )
        validated["hyperparams"] = normalized
        validated["hyperparam_rationale"] = rationale

    if "threshold_adjustment" in validated:
        adjustment = validated["threshold_adjustment"]
        if not isinstance(adjustment, dict):
            raise ValueError(
                "threshold_adjustment must be a JSON object"
            )
        unsupported_adjustment = sorted(
            set(adjustment) - _THRESHOLD_FIELDS
        )
        if unsupported_adjustment:
            raise ValueError(
                "unsupported threshold_adjustment field(s): "
                + ", ".join(unsupported_adjustment)
            )
        if "new_threshold" not in adjustment:
            raise ValueError(
                "threshold_adjustment.new_threshold is required "
                "(use null for no adjustment)"
            )
        normalized_adjustment = dict(adjustment)
        new_threshold = adjustment.get("new_threshold")
        reason = adjustment.get("reason")
        if new_threshold is None:
            if reason is not None and not isinstance(reason, str):
                raise ValueError(
                    "threshold_adjustment.reason must be a string when provided"
                )
            if isinstance(reason, str):
                normalized_adjustment["reason"] = reason.strip()[:240]
        else:
            if (
                isinstance(new_threshold, bool)
                or not isinstance(new_threshold, (int, float))
                or not math.isfinite(float(new_threshold))
            ):
                raise ValueError(
                    "threshold_adjustment.new_threshold must be a finite "
                    "numeric value or null"
                )
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    "threshold_adjustment requires a non-empty reason when "
                    "new_threshold is numeric"
                )
            normalized_adjustment["new_threshold"] = float(new_threshold)
            normalized_adjustment["reason"] = reason.strip()[:240]
        validated["threshold_adjustment"] = normalized_adjustment
    return validated


def _parse_decision_json(
    raw,
    *,
    task_type: str = "classification",
    state=None,
) -> dict:
    """Robustly parse the orchestrator's decision JSON (B143 fix for JSONDecodeError).

    Tolerates content-block lists, code fences, and prose wrapped around the JSON object.
    ALWAYS raises ValueError (never a bare JSONDecodeError) with a readable message so the
    caller's fallback path logs something actionable instead of crashing out of the try.
    """
    text = _coerce_to_text(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    if not text:
        raise ValueError("empty LLM response — no decision JSON returned")

    # 1) whole string is JSON.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        return _validate_decision_json(
            parsed,
            task_type=task_type,
            state=state,
        )

    # 2) extract the outermost {...} and parse (wrapped — the old code left this bare,
    #    so a still-invalid slice re-raised JSONDecodeError past the caller's handler).
    m = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = m.group() if m else text
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        pass
    else:
        return _validate_decision_json(
            parsed,
            task_type=task_type,
            state=state,
        )

    raise ValueError(f"no parseable JSON object in LLM response: {text[:160]!r}")


_ITERATE_SYSTEM = """\
You are the orchestrator of an agentic fine-tuning loop for small language models.
Your job: read the full training trajectory and aggregate evaluation report,
then decide what to do next and explain WHY.

The evaluation firewall is strict: never inspect, request, quote, or copy raw eval
text. Use only aggregate difficulty scores and aggregate confusion counts supplied
in the prompt. Dataset plans are declarative JSON; never emit executable code.

Return the decision directly as JSON. You cannot call tools, access files, or request
additional context.

=============================== CHOOSE EXACTLY ONE ===============================
There are TWO possible interventions and you must pick ONE. They are mutually exclusive.

  A) "intervention": "data_rebuild"    -> change the TRAINING DATA.
       REQUIRED key: "data_rebuild"
       FORBIDDEN key: "hyperparams"    <-- do not include it, not even unchanged values
       The hyperparameters are held automatically at the current best config.

  B) "intervention": "hyperparameter"  -> change the LEARNING SETTINGS.
       REQUIRED key: "hyperparams"
       FORBIDDEN key: "data_rebuild"   <-- do not include it
       The dataset is held automatically at the current version.

WHY THIS IS STRICT: exactly one thing may change per iteration. If the data and the
hyperparameters both changed and the score moved, it is impossible to tell which change
caused it, and the whole trajectory becomes uninterpretable. This is the single most
important rule in this prompt.

If you emit "hyperparams" alongside a "data_rebuild", the system will DISCARD those
hyperparameters, log the correction, and execute your data plan with the current best
config. Your data plan is not lost — but you have wasted the field, and the log will show
that you ignored this instruction. Do not do it.
==================================================================================

Valid data_rebuild JSON example:
{
  "intervention": "data_rebuild",
  "hypothesis": "Aggregate hard-bucket errors indicate insufficient difficult examples.",
  "data_rebuild": {
    "strategy": "synthesize",
    "target_rows": 3000,
    "resample_fraction": 0.65,
    "new_real_rows": 0,
    "synth_rows": 400,
    "max_acquire_rounds": 0,
    "difficulty_buckets": {"easy": 0.1, "medium": 0.3, "hard": 0.6},
    "confusion_pairs": [{"gold": "label_a", "predicted": "label_b", "count": 4}],
    "pattern_hint": "aggregate label_a to label_b confusion"
  },
  "threshold_adjustment": {"new_threshold": null, "reason": ""}
}
End data_rebuild example.

Valid hyperparameter JSON example:
{
  "intervention": "hyperparameter",
  "hypothesis": "Aggregate hard-bucket errors indicate insufficient adaptation capacity.",
  "hyperparams": {
    "lora_rank": 16,
    "alpha_ratio": 2,
    "weight_decay": 0.01,
    "learning_rate": 0.0001,
    "nr_epochs": 4
  },
  "threshold_adjustment": {"new_threshold": null, "reason": ""}
}
End hyperparameter example.

Data-rebuild payload constraints:
- strategy: EXACTLY ONE of:
    "resample"   -> reshuffle / re-draw rows from the existing pool
    "acquire"    -> add new rows from the same or a new provenance (real-source mining)
    "synthesize" -> generate new synthetic rows (task-adaptive; see below)
- target_rows: integer [16, ceiling], step 8
- resample_fraction: float [0.10,1.00], step 0.05
- new_real_rows: integer [0,500], step 5 (used by "acquire")
- synth_rows: integer [100,500], step 5 (used by "synthesize" — choose how many new
  synthetic rows to generate, at your discretion, anywhere in 100–500)
- max_acquire_rounds: integer [0,3], further limited by remaining budget
- difficulty_buckets: numeric weights for easy, medium, and hard
- confusion_pairs and pattern_hint: aggregate categories only, never raw eval text

Rules:
- "hypothesis" is REQUIRED and must causally justify the action
- "data_rebuild" is REQUIRED when intervention is "data_rebuild"
- "hyperparams" is REQUIRED when intervention is "hyperparameter"
- emit no keys outside this schema and never include the other intervention's payload
- There is NO restriction on which strategy you may choose: any strategy is valid for any
  task type and at any score. Choose from the trajectory and your estimate of what is
  failing (per-difficulty accuracy + confusion pairs), not from mechanical eligibility.
- "synthesize" is task-adaptive: contrastive hard negatives for classification/NER, and
  new CORRECT in-distribution examples for math/code/generation (never wrong-answer data).
- Regardless of strategy, the curriculum is synth-filled up to target_rows when real data
  falls short, so target_rows is the size you are actually training on.
- use confusion counts and a pattern hint only at aggregate level
- EXACTLY FIVE hyperparameters are tunable: lora_rank, alpha_ratio, weight_decay,
  learning_rate, nr_epochs. Emitting any other key is rejected.
- lora_rank must be one of [4, 8, 16, 32, 64]
- alpha_ratio must be one of [1, 2, 4]; the LoRA update is scaled by alpha/rank, so
  the ratio is the meaningful quantity and alpha is derived as rank * alpha_ratio.
- weight_decay must be one of [0.0, 0.01, 0.05, 0.1] — this is the regularizer to
  reach for when the model overfits a small curriculum.
- learning_rate is bounded to [1e-5, 5e-4] and nr_epochs to [1, 8]
- Batch shape (micro_batch_size / gradient_accumulation_steps / effective_batch_size)
  is NOT yours to set. The micro/accum split only decides how a batch is divided to
  fit VRAM; it does not change what the model learns, and the trainer sizes it from
  the actual device. Do not propose it, and do not reason about peak activation
  memory — that is handled for you.
- lora_dropout is NOT tunable. Regularize with weight_decay instead.
- Never propose an exact (dataset, hyperparameter) repeat from the tried list,
  including a pruned/rolled-back trial. The same complete hyperparameter config is
  allowed after a data rebuild because the dataset identity changed.

Score band guidance (reason about the trajectory, not just mechanical rules):
- Score < 0.80: usually a data problem (data_rebuild)
- 0.80–1.0: usually an optimization problem (hyperparameter)

If the trajectory shows stagnation, escalate the intervention type.

Threshold adjustment guidance:
Examine the aggregate report. If the dominant failure category reflects a genuine model
capacity limit — not a data or hyperparameter problem — you may lower the stop_threshold.
Examples that justify lowering: failures requiring world knowledge beyond the model's
parametric memory, failures on reasoning chains longer than the model can reliably produce,
failures on adversarial/out-of-distribution inputs that no amount of training data would fix.
Do NOT lower the threshold simply because progress is slow or the task is hard but learnable.
Set "new_threshold" to null if no adjustment is warranted.
The floor is enforced by the system — you cannot set it below the initial calibrated value.
"""


def _build_intervention_prompt_for_test() -> str:
    """Expose the orchestrator decision prompt for unit tests (no runtime use)."""
    return _ITERATE_SYSTEM


# Stagnation parameters: escalate if the best chronological gain over the last
# STAGNATION_WINDOW evaluation runs, relative to that window's first score, is
# below STAGNATION_MIN_DELTA. This treats declines/below-origin oscillation as
# no progress while allowing a genuine new high to keep the current model active.
#
# Raised to 20 (from 3/4) so each model gets far more exploration before escalating —
# earlier runs escalated too eagerly on noisy small-eval scores. With the larger, balanced
# eval set + best-checkpoint early stopping, scores are steadier, so a long window mostly
# lets genuine slow progress continue; the wall-clock guard (config.MAX_WALLCLOCK_S) is the
# real backstop against a run that never plateaus. All three are env-overridable.
import os as _os
import time as _time
STAGNATION_WINDOW = int(_os.environ.get("SLM_STAGNATION_WINDOW", "20"))   # recent evals examined
STAGNATION_MIN_DELTA = float(_os.environ.get("SLM_STAGNATION_MIN_DELTA", "0.02"))

# Wall-clock guard budget (seconds). 0 disables. Read lazily so config import stays cheap.
def _wallclock_budget_s() -> float:
    try:
        from config.config import MAX_WALLCLOCK_S
        return float(MAX_WALLCLOCK_S)
    except Exception:
        return 0.0


_WALLCLOCK_BUDGET_S = _wallclock_budget_s()


def _wallclock_elapsed_s() -> float:
    """Cumulative seconds across this and all completed resume segments."""
    ts = _os.environ.get("SLM_RUN_START_TS")
    try:
        completed = max(
            0.0, float(_os.environ.get("SLM_RUN_ELAPSED_S", "0") or 0.0)
        )
    except (TypeError, ValueError):
        completed = 0.0
    if not ts:
        return completed
    try:
        return completed + max(0.0, _time.time() - float(ts))
    except (TypeError, ValueError):
        return completed


def _wallclock_exceeded() -> bool:
    return _WALLCLOCK_BUDGET_S > 0 and _wallclock_elapsed_s() >= _WALLCLOCK_BUDGET_S

# Hard backstop against a rollback→re-decide→rollback churn. `should_rollback` pops the
# regressing score, so the stagnation window can stay short and never fire; meanwhile
# `consecutive_no_improvement` (set in evaluate_node) grows every non-improving eval and
# is NOT popped. Once this many evals in a row fail to beat the best score, stop churning
# and escalate — which promotes to a bigger model if one fits, else terminates cleanly.
# Set to 20: with non-deterministic curation the run should escalate promptly once 20
# consecutive evals fail to improve (the sole stuck-run backstop besides the wall clock).
MAX_STALL_EVALS = int(_os.environ.get("SLM_MAX_STALL_EVALS", "20"))


def _tried_hparam_configs(state) -> list[dict]:
    """Canonical (dataset, H) trials, including pruned/rolled-back nodes."""
    out = []
    seen = set()
    for n in (state.get("dag") or []):
        pi = n.get("pi") or {}
        dataset = pi.get("D") or {}
        records = [{
            "H": pi.get("H") or {},
            "score": n.get("score"),
        }]
        records.extend(n.get("trained_configs") or [])
        for record in records:
            h = record.get("H") if isinstance(record, dict) else {}
            if not isinstance(h, dict) or h.get("lora_rank") is None:
                continue
            try:
                normalized, _ = normalize_hyperparams(h)
            except ValueError:
                continue
            identity = (
                dataset.get("version"),
                str(dataset.get("path") or ""),
                *hyperparameter_identity(normalized),
            )
            if identity in seen:
                continue
            seen.add(identity)
            out.append({
                **normalized,
                "dataset_version": dataset.get("version"),
                "dataset_path": dataset.get("path"),
                "score": record.get("score"),
                "pruned": n.get("pruned", False),
            })
    return out


def _stagnation_gain(scores: list[float]) -> float:
    """Best chronological gain in the active window relative to its first score."""
    if not scores:
        return 0.0
    window = scores[-STAGNATION_WINDOW:]
    return max(window) - window[0]


def _is_stagnant(scores: list[float]) -> bool:
    """
    Return True if the model has stagnated and should escalate.

    Stagnation is defined as: the best score achieved after the first evaluation
    in the last STAGNATION_WINDOW runs improves by less than
    STAGNATION_MIN_DELTA. Declines and below-origin oscillation therefore count as
    no progress. A tolerance keeps the exact decimal boundary (0.02 by default)
    non-stagnant despite binary floating-point representation.
    Requires at least STAGNATION_WINDOW scores before it can fire.
    """
    if len(scores) < STAGNATION_WINDOW:
        return False
    gain = _stagnation_gain(scores)
    if math.isclose(
        gain,
        STAGNATION_MIN_DELTA,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return False
    return gain < STAGNATION_MIN_DELTA


def _llm_iterate(state: AgentState) -> dict:
    """Make one tool-free orchestrator decision, with one JSON-only reask."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import SystemMessage, HumanMessage
    from config.config import ANTHROPIC_API_KEY, ORCHESTRATOR_MODEL, orchestrator_client_kwargs
    from agent.context_manager import compact_trajectory, should_compact
    from data.curation_log import CurationLog

    llm = ChatAnthropic(
        model=ORCHESTRATOR_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens=_ITERATE_MAX_TOKENS,
        **orchestrator_client_kwargs(),
    )

    # Build context
    raw_trajectory = CurationLog(
        state.get("curation_log_path")
    ).read_latest()
    trajectory = compact_trajectory(raw_trajectory) if should_compact(raw_trajectory) else raw_trajectory
    current_score = state["scores"][-1] if state["scores"] else 0.0

    scores = state["scores"]
    window = scores[-STAGNATION_WINDOW:] if len(scores) >= STAGNATION_WINDOW else scores
    recent_gain = round(_stagnation_gain(window), 4) if window else 0.0

    # Test-data agent report (B161): per-difficulty accuracy + targeted diagnosis, the primary
    # signal for what to fix (replaces failure-taxonomy). Feed it into the decision prompt.
    report = state.get("test_report") or {}
    _bd = report.get("by_difficulty") or {}
    def _bdfmt(b):
        v = _bd.get(b) or {}
        a = v.get("accuracy")
        return f"{b}={a:.3f}(n={v.get('n',0)})" if a is not None else f"{b}=n/a"
    test_agent_block = (
        f"- Per-difficulty accuracy: {_bdfmt('easy')}  {_bdfmt('medium')}  {_bdfmt('hard')}\n"
        f"- Test-agent diagnosis: {report.get('diagnosis', '(none)')}\n"
        f"- Test-agent suggested intervention: {report.get('suggested_intervention', '(none)')}\n"
        "- Aggregate confusion counts:\n"
        + (
            "\n".join(
                "    - "
                f"gold={pair.get('gold')!r} "
                f"predicted={pair.get('predicted')!r} "
                f"count={pair.get('count', 0)}"
                for pair in (report.get("confusion_pairs") or [])
                if isinstance(pair, dict)
            )
            or "    - (none)"
        )
        if report else "(no test-agent report yet)"
    )

    # Non-deterministic redesign: data-rebuild plans are no longer deduped or tracked as
    # "tried". The orchestrator freely re-picks a strategy each turn; the prior plan's yield
    # (below) plus the trajectory are the signal, not a tried-plan ledger.
    rebuild_trials_block = (
        "Data-rebuild plans are not deduplicated — you may repeat or vary any strategy "
        "freely; judge from the trajectory and prior-plan yield below."
    )

    last_curation = state.get("last_curation") or {}
    source_novelty = last_curation.get("source_novelty") or {}
    plan_yield = last_curation.get("plan_yield") or {}
    source_yield_block = (
        "- Source novelty: "
        f"requested={source_novelty.get('requested', 0)} "
        f"novel_rows={source_novelty.get('novel_rows', 0)} "
        f"novel_fraction={source_novelty.get('novel_fraction', 0)}\n"
        "- Prior plan yield: "
        f"status={plan_yield.get('status', 'unknown')} "
        f"novel_rows={plan_yield.get('novel_rows', 0)} "
        f"final_rows={plan_yield.get('final_rows', 0)}"
    )
    turn_budget = int(state.get("turn_budget", 0) or 0)
    turns_used = (int(state.get("iteration", 0) or 0) + 1) * 2
    remaining_turns = max(0, turn_budget - turns_used) if turn_budget else "unbounded"
    remaining_acquisition = remaining_paid_acquire_rounds(state)

    # ALREADY-TRIED hyperparameter configs, incl. rolled-back ones (the memory the LLM was
    # missing — rollback pops scores so the trajectory hides these). Training is deterministic,
    # so re-proposing any of these yields the SAME score — they are forbidden (B161).
    tried = _tried_hparam_configs(state)
    if tried:
        def _score_text(trial):
            score = trial.get("score")
            return f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"

        tried_lines = "\n".join(
            f"    - D=v{t.get('dataset_version')}:{t.get('dataset_path')} "
            f"r={t.get('lora_rank')} a={t.get('lora_alpha')} "
            f"drop={t.get('lora_dropout')} wd={t.get('weight_decay')} "
            f"lr={t.get('learning_rate')} ep={t.get('nr_epochs')} "
            f"micro={t.get('micro_batch_size')} "
            f"grad_accum={t.get('gradient_accumulation_steps')} "
            f"effective={t.get('effective_batch_size')} → "
            f"f1={_score_text(t)}"
            f"{' (rolled back)' if t.get('pruned') else ''}"
            for t in tried
        )
        tried_block = (
            "Exact (dataset, hyperparameter) identities ALREADY TRIED this "
            "model (do NOT repeat one; pruned rows still count):\n"
            f"{tried_lines}"
        )
    else:
        tried_block = "(no hyperparameter configs tried yet)"

    selected_model = state.get("selected_model")
    # On-disk weight size is the only deployment footprint we actually know. Peak
    # inference RAM is unmeasured for these variants and is deliberately NOT estimated,
    # so the orchestrator is told the real quantity rather than a fabricated one.
    deployment_size_mb = getattr(selected_model, "size_mb", "unknown")
    hardware = state.get("hardware_constraints")
    memory_budget_mb = getattr(hardware, "memory_mb", "unknown")

    user_content = f"""\
## Training trajectory so far (from data-curation.md — each row is one past iteration:
## its dataset version, intervention applied, and resulting score)

{trajectory if trajectory else "(no iterations logged yet)"}

## Current iteration summary
- Task type: {state['task_type']}
- Model variant: {state['selected_model'].selector if state.get('selected_model') else 'unknown'}
- Iteration: {state['iteration']}
- Current f(π): {current_score:.4f}
- Best f(π) so far: {state['best_score']:.4f}
- Score history: {scores}
- Current dataset identity: v{state.get('dataset_version')}:{state.get('current_dataset_path')}
- Selected variant on-disk weight size: {deployment_size_mb} MB (peak inference RAM is unmeasured)
- Device memory budget: {memory_budget_mb} MB
- Training-memory control: micro batch drives peak activation memory; gradient accumulation
  raises the derived effective batch without raising that per-step activation peak.
- Recent chronological gain (last {len(window)} evals, improvement to trigger escalation = {STAGNATION_MIN_DELTA}): {recent_gain:.4f}
- Stop threshold: {state['stop_threshold']} (initial floor: {state.get('initial_stop_threshold', state['stop_threshold']):.3f})
- Prior causal hypothesis: {state.get('last_hypothesis') or '(none)'}
- Remaining turn budget: {remaining_turns}
- Remaining paid acquisition rounds: {remaining_acquisition}

## Test-data agent report (difficulty-stratified — use this to target your fix)
{test_agent_block}

## {tried_block}

## Prior declarative data-rebuild plan identities
{rebuild_trials_block}

## Source novelty and prior plan yield
{source_yield_block}

Diagnose WHY the score is where it is from aggregate per-difficulty and confusion patterns,
source novelty/yield, remaining budget, prior hypotheses, and the trajectory,
then decide the next intervention. If you choose "hyperparameter", you MUST propose a config
whose complete (dataset, hyperparameter) identity is NOT in the already-tried list above
(training is deterministic — an exact repeat cannot help). If the
useful hyperparameter space is exhausted (all sensible configs tried), choose "data_rebuild"
or expect the system to escalate to a larger model. A data_rebuild MUST include a bounded
declarative plan whose exact identity is not in the tried list. Never inspect raw eval rows.
Return only the decision JSON.
"""

    messages = [SystemMessage(content=_ITERATE_SYSTEM), HumanMessage(content=user_content)]
    response = tracked_chat_anthropic_invoke(
        llm,
        messages,
        stage="iterate",
        model=ORCHESTRATOR_MODEL,
    )
    if getattr(response, "tool_calls", None):
        # No tool result is ever executed or reflected back. Reask from the
        # original bounded context so a tool-use block cannot open a side channel.
        return _reask_json_only(
            messages,
            validation_error=ValueError(
                "model attempted tool calls instead of returning final JSON"
            ),
            task_type=state["task_type"],
            state=state,
        )
    try:
        return _parse_decision_json(
            response.content,
            task_type=state["task_type"],
            state=state,
        )
    except ValueError as error:
        return _reask_json_only(
            messages,
            validation_error=error,
            task_type=state["task_type"],
            state=state,
        )


def apply_iteration_policy(score: float) -> dict:
    """
    Fallback score-band rules used when the LLM call is unavailable or fails.
    Returns dict with band and intervention type.
    """
    if score < 0.80:
        return {
            "band": "<0.80",
            "intervention": "data_rebuild",
            "data_rebuild_strategy": "acquire",
            "description": "Score below 0.80 — data problem. Rebuild dataset.",
        }
    elif score < 0.95:
        return {
            "band": "0.80-0.95",
            "intervention": "hyperparameter",
            "description": "Score 0.80–0.95 — optimization problem. Tune hyperparameters.",
        }
    else:
        return {
            "band": ">=0.95",
            "intervention": "data_rebuild",
            "data_rebuild_strategy": "synthesize",
            "description": (
                "Score ≥0.95 — refine remaining aggregate confusion with a "
                "bounded data rebuild."
            ),
        }


def _log(model_id: str, msg: str):
    print(f"[iterate][{model_id}] {msg}")


def _route_score_at_threshold(
    state: AgentState,
    current_score: float,
    model_id: str,
    policy: dict,
    *,
    stagnant: bool | None = None,
) -> bool:
    """Route a threshold-clearing score without an intervention-model call.

    Returns True when routing is complete. A hardware-blocked score is not an
    acceptable convergence result, but it is still handled deterministically so
    an auth/quota failure cannot break a run after the accuracy goal was reached.
    """
    if current_score < state["stop_threshold"]:
        return False

    if state.get("_largest_first_phase") == "probe":
        from agent.nodes.cold_start.model_selection.largest_first import (
            check_probe_result,
        )

        state = check_probe_result(state)
        if state.get("next_action") == "curate":
            _log(
                model_id,
                "  → CURATE (largest_first probe succeeded; switching to smallest model)",
            )
            return True

    gating_model = state.get("selected_model")
    hw_blocks_termination = False
    if state.get("hw_gating_enabled") and gating_model is not None:
        hw_check = check_hardware_constraints(
            gating_model,
            state["hardware_constraints"],
        )
        hw_blocks_termination = not all_constraints_pass(hw_check)
    if hw_blocks_termination:
        _log(
            model_id,
            f"  Score {current_score:.4f} >= threshold but hardware FAILS — "
            "not accepting as terminal; continuing without an intervention API call",
        )
        stagnant = _is_stagnant(state["scores"]) if stagnant is None else stagnant
        state["llm_iterate_decision"] = None
        if stagnant:
            state["next_action"] = "escalate"
            state["last_intervention"] = "escalate"
            state["last_hypothesis"] = "hardware-blocked convergence stagnated"
            _log(model_id, "  → ESCALATE (hw-blocked terminal + stagnation)")
        else:
            intervention = policy["intervention"]
            state["last_intervention"] = intervention
            state["last_hypothesis"] = (
                "(deterministic) accuracy goal met but hardware gate failed"
            )
            state["next_action"] = (
                "train" if intervention == "hyperparameter" else "curate"
            )
            _log(
                model_id,
                f"  → {state['next_action'].upper()} "
                f"(hw-blocked terminal; deterministic {intervention})",
            )
        return True

    import os as _os

    strategy = _os.environ.get(
        "SLM_MODEL_SELECTION_STRATEGY",
        "smallest_first",
    )
    probes_down = strategy in ("interpolation", "orchestrator_choice")
    current_model = state.get("selected_model")
    feasible_models = state.get("feasible_models")
    if feasible_models is None:
        # Backward-compatible fallback for older/manual state fixtures that predate
        # feasible_models. Normal pipeline state always carries the filtered pool.
        has_untried_lower_tier = (
            current_model is not None and current_model.tier > 0
        )
    else:
        from agent.nodes.downward_probe import tiers_already_explored

        downward_tiers_tried = set(
            state.get("downward_tiers_tried") or []
        ) | tiers_already_explored(state, feasible_models)
        has_untried_lower_tier = (
            current_model is not None
            and any(
                candidate.tier < current_model.tier
                and candidate.tier not in downward_tiers_tried
                for candidate in feasible_models
            )
        )
    if (
        probes_down
        and not state.get("downward_probe_done")
        and current_model is not None
        and has_untried_lower_tier
    ):
        state["next_action"] = "downward_probe"
        _log(
            model_id,
            f"  → DOWNWARD_PROBE (score {current_score:.4f} >= threshold; "
            f"strategy={strategy} may have over-selected; trying a smaller model)",
        )
        return True

    state["next_action"] = "terminate"
    _log(
        model_id,
        f"  → TERMINATE (score {current_score:.4f} >= threshold "
        f"{state['stop_threshold']:.3f})",
    )
    return True


def iterate_node(state: AgentState) -> AgentState:
    """
    Node 7: tool-free LLM-driven iteration decision.

    The orchestrator receives the bounded trajectory and aggregate reports directly.
    This implements the paper's EXPAND(v_parent, G, F) reasoning step (§2.2 Eq. 2).

    Falls back to score-band rules if the LLM call fails.
    """
    selected = state.get("selected_model")
    model_id = selected.label if selected is not None else "?"  # log prefix incl. quant

    if not state["scores"]:
        _log(model_id, "No scores yet — routing to train")
        state["next_action"] = "train"
        return state

    # Turn-budget guard (B51): ~2 productive turns per iteration (curate + train).
    turn_budget = state.get("turn_budget", 0)
    turns_used = (state["iteration"] + 1) * 2  # cost of the upcoming train+eval cycle
    if turn_budget and turns_used >= turn_budget:
        _log(model_id, f"  Turn budget exhausted: {turns_used} >= {turn_budget} — TERMINATING")
        state["next_action"] = "terminate"
        return state

    # Wall-clock guard (B161 Addition 3): force a graceful terminal state before SLURM
    # hard-kills the job at --time, so the run always writes its full summary/DAG. This is
    # the real backstop for the high stall cap + downward re-exploration.
    if _wallclock_exceeded():
        elapsed = _wallclock_elapsed_s()
        _log(model_id, f"  Wall-clock budget exhausted ({elapsed/3600:.1f}h ≥ "
                       f"{_WALLCLOCK_BUDGET_S/3600:.1f}h) — TERMINATING gracefully")
        state["next_action"] = "terminate"
        return state

    current_score = state["scores"][-1]
    policy = apply_iteration_policy(current_score)

    _log(model_id, f"Current score: {current_score:.4f}  "
         f"band={policy['band']}  threshold={state['stop_threshold']:.3f}")

    # Accuracy convergence is deterministic routing, not an intervention decision.
    # Handle it before _llm_iterate so a completed run spends no API call and cannot
    # be broken by an auth/quota error from an unnecessary request.
    if _route_score_at_threshold(state, current_score, model_id, policy):
        return state

    # Check stagnation before LLM call
    stagnant = _is_stagnant(state["scores"])
    if stagnant:
        window = state["scores"][-STAGNATION_WINDOW:]
        gain = _stagnation_gain(window)
        _log(model_id, f"  Stagnation detected: window={[f'{s:.4f}' for s in window]}  "
             f"chronological_gain={gain:.4f} < {STAGNATION_MIN_DELTA}")

    # Stall backstop: catches the rollback churn that stagnation can miss (scores are
    # popped on rollback, so the window may never fill). Counts consecutive non-improving
    # evals, which survive rollback.
    stalled = state.get("consecutive_no_improvement", 0) >= MAX_STALL_EVALS
    if stalled and not stagnant:
        _log(model_id, f"  Stall detected: {state.get('consecutive_no_improvement')} consecutive "
             f"evals without beating best {state['best_score']:.4f} (>= {MAX_STALL_EVALS})")

    # Q9: stagnation/stall escalation is a RULE-BASED decision — take it WITHOUT spending an
    # orchestrator LLM call (the LLM cannot override it anyway). Only applies below the stop
    # threshold; the converged path below still runs. This saves an API call every time a
    # model plateaus (which is exactly when the run makes the most iterate calls).
    if current_score < state["stop_threshold"] and (stagnant or stalled):
        reason = "stagnation" if stagnant else f"{state.get('consecutive_no_improvement')} stalled evals"
        if state.get("_largest_first_phase") == "probe":
            _log(model_id, "  → TERMINATE (largest_first probe stagnated — task infeasible)")
            state["next_action"] = "terminate"
            state["_largest_first_phase"] = "done"
            state["last_hypothesis"] = "largest_first probe could not clear the goal"
        else:
            _log(model_id, f"  → ESCALATE ({reason}) — skipped LLM intervention call to save API cost")
            state["next_action"] = "escalate"
            state["last_hypothesis"] = f"escalate on {reason}"
        state["last_intervention"] = "escalate"
        return state

    # Not escalating → consult the orchestrator LLM for the intervention type.
    # NOTE: this runs in cheap mode too — cheap mode keeps the agent's bounded,
    # tool-free intervention reasoning, just on the Haiku tier.
    # Cheap mode's Claude savings come from config (Haiku everywhere) + curate skipping
    # hard-negative synthesis and CoT annotation, NOT from dumbing this down to score bands.
    llm_decision = None
    hypothesis = ""
    intervention = policy["intervention"]  # initialized to fallback; overwritten by LLM if successful
    try:
        _log(model_id, "  Calling orchestrator LLM for intervention decision...")
        # Defense in depth: normal calls are validated by _parse_decision_json,
        # but revalidated here so alternate/mock/provider paths cannot bypass the
        # threshold and intervention contracts and crash after the fallback guard.
        llm_decision = _validate_decision_json(
            _llm_iterate(state),
            task_type=state["task_type"],
            state=state,
            allow_internal=True,
        )
        intervention = llm_decision.get("intervention", "")
        hypothesis = llm_decision.get("hypothesis", "")

        _log(model_id, f"  LLM decision: intervention={intervention}")
        _log(model_id, f"  Hypothesis: {hypothesis}")
        _dropped = llm_decision.get("_dropped_fields") or []
        if _dropped:
            # Visible, not silent. The orchestrator emitted fields belonging to the OTHER
            # branch of the union; they are discarded so the intervention stays isolated.
            # Logging it means a persistently confused orchestrator shows up in the run log
            # instead of being masked (the NER run rejected 65 such decisions outright).
            _log(
                model_id,
                f"  NOTE — dropped {', '.join(_dropped)} from this data_rebuild decision: "
                "a data_rebuild must not also change hyperparameters, or the score movement "
                "cannot be attributed. The data plan was kept and executed.",
            )
        if intervention == "hyperparameter" and llm_decision.get("hyperparams"):
            hp = llm_decision["hyperparams"]
            _log(
                model_id,
                f"  Hyperparams: r={hp.get('lora_rank')}  "
                f"alpha={hp.get('lora_alpha')}  "
                f"dropout={hp.get('lora_dropout')}  "
                f"weight_decay={hp.get('weight_decay')}  "
                f"lr={hp.get('learning_rate')}  "
                f"epochs={hp.get('nr_epochs')}  "
                f"micro_batch={hp.get('micro_batch_size')}  "
                f"grad_accum={hp.get('gradient_accumulation_steps')}  "
                f"effective_batch={hp.get('effective_batch_size')}",
            )
            _log(
                model_id,
                "  Batch rationale: "
                f"{llm_decision.get('hyperparam_rationale', '')}",
            )
        if intervention == "data_rebuild":
            plan = llm_decision["data_rebuild"]
            _log(
                model_id,
                f"  Data rebuild: strategy={plan['strategy']} "
                f"target_rows={plan['target_rows']}",
            )

    except Exception as exc:
        # Billing/auth/quota errors will recur on every call — fail fast with a clear
        # message instead of silently limping through the rest of the run on fallbacks (B144).
        raise_if_fatal(exc, "iterate")
        # Fallback: prefer the TEST-DATA AGENT's diagnosis-driven suggestion (difficulty-aware)
        # over the crude score-band rule (B161). Only use the score band if no report exists.
        _report = state.get("test_report") or {}
        _suggested = _report.get("suggested_intervention")
        if _suggested in ("data_rebuild", "hyperparameter"):
            intervention = _suggested
            hypothesis = f"(test-agent) {_report.get('diagnosis', '')}"
            _log(model_id, f"  LLM call failed ({exc!r}); using test-agent suggestion: {intervention}")
        else:
            fallback = apply_iteration_policy(current_score)
            intervention = fallback["intervention"]
            hypothesis = f"(fallback) {fallback['description']}"
            _log(model_id, f"  LLM call failed ({exc!r}), falling back to score-band rules: {intervention}")

    state["last_intervention"] = intervention
    state["last_hypothesis"] = hypothesis
    # Always overwrite (None on failure) so a FAILED call cannot leave the PREVIOUS
    # iteration's decision — and its hyperparams — stale in state. train._build_config
    # reads this; a stale non-None decision would bypass the carry-forward/untried
    # complete-identity fallback and silently reuse old hyperparameters.
    state["llm_iterate_decision"] = llm_decision
    if intervention == "data_rebuild":
        if llm_decision is not None:
            rebuild_plan = llm_decision["data_rebuild"]
        else:
            rebuild_plan = fallback_data_rebuild_plan(
                state,
                hypothesis=hypothesis or "safe aggregate data refresh",
                score=current_score,
            )
        state["data_rebuild_plan"] = rebuild_plan

    if llm_decision:
        adj = llm_decision.get("threshold_adjustment") or {}
        new_threshold = adj.get("new_threshold")
        if new_threshold is not None:
            floor = state["initial_stop_threshold"]
            clamped = max(float(new_threshold), floor)
            if clamped < state["stop_threshold"]:
                reason = adj.get("reason", "")
                _log(model_id,
                     f"  Lowering stop_threshold "
                     f"{state['stop_threshold']:.3f} → {clamped:.3f} "
                     f"(floor={floor:.3f}). Reason: {reason}")
                state["stop_threshold"] = clamped

    # A threshold adjustment from the LLM can make the current score converged.
    # Reuse the same deterministic route, but only after the below-threshold call
    # that supplied the adjustment.
    if _route_score_at_threshold(
        state,
        current_score,
        model_id,
        policy,
        stagnant=stagnant,
    ):
        return state

    if intervention == "hyperparameter":
        state["next_action"] = "train"
        _log(model_id, f"  → TRAIN (hyperparameter intervention, dataset held fixed)")
    else:
        state["next_action"] = "curate"
        _log(model_id, f"  → CURATE ({intervention})")

    return state
