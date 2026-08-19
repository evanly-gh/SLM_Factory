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
    mining_available_for_state,
    unexhausted_sources,
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
# Output budget for one decision. Must comfortably fit a full data_rebuild plan PLUS a complete
# hypothesis; a response that hits this ceiling is cut mid-JSON and the whole decision is lost.
# 1536 was survivable only while the hypothesis was silently truncated to 240 chars. Once that
# cap was lifted, the very first iterate call of slm-clinc150-cse-38179864 hit 1536 exactly, the
# reask hit it again, and the orchestrator's chosen `data_rebuild` was discarded in favour of the
# score-band fallback's `hyperparameter` (B240).
# 4096 was too small for the WRONG reason: the decision JSON is ~520 tokens, but max_tokens
# bounds THINKING PLUS answer, and a measured iterate call spent 1923 tokens thinking before
# writing 520 of JSON. The prompt's length instruction cannot help — the model neither sees nor
# budgets its own thinking — so 38 of 69 calls in run 38303490 were truncated mid-object and
# reasked. Output is billed on tokens actually generated, never on the cap, so a high ceiling
# costs nothing and simply stops the truncation. Anthropic requires this parameter (it cannot be
# omitted); claude-sonnet-5 permits up to 128000.
_ITERATE_MAX_TOKENS = int(__import__("os").environ.get("SLM_ITERATE_MAX_TOKENS", "20000"))

# Runaway guard ONLY — deliberately far above any legitimate hypothesis. Length is managed by
# telling the orchestrator its budget in the prompt (see HYPOTHESIS_TARGET_WORDS), not by cutting
# text after the fact: the hypothesis is the densest signal in the run and is replayed into later
# contexts, so a longer, better-reasoned one is genuinely more useful than a short one.
#
# This exists so a pathological response cannot blow up the context, not to shape normal output.
# If the truncation warning ever fires, raise this rather than accept the loss (B238/B240).
HYPOTHESIS_MAX_CHARS = int(__import__("os").environ.get("SLM_HYPOTHESIS_MAX_CHARS", "4000"))
# Stated to the model in the prompt. A soft target it can exceed when it has more to say — the
# point is to stop it writing 1500 tokens of prose and running out of output budget mid-JSON.
HYPOTHESIS_TARGET_WORDS = int(__import__("os").environ.get("SLM_HYPOTHESIS_TARGET_WORDS", "150"))


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


# The FIVE hyperparameters the orchestrator may set. Module-level so the validator and the
# reask salvage path cannot drift apart on what "tunable" means.
_TUNABLE_HYPERPARAMS = frozenset({
    "lora_rank",
    "alpha_ratio",
    "weight_decay",
    "learning_rate",
    "nr_epochs",
})


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
    task: str = "classification",
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
    # A length failure needs the OPPOSITE instruction from a format failure. Without this the
    # reask just repeats "return valid JSON", the model writes another over-long answer, and it
    # fails identically — both attempts burned that way in slm-clinc150-cse-38179864 (B240).
    too_long = "stop_reason=max_tokens" in str(validation_error)
    remedy = (
        "Your previous answer RAN OUT OF OUTPUT SPACE and was cut off mid-object. Send the "
        "same decision again but MUCH SHORTER: keep every required field, and compress "
        f"'hypothesis' to at most {max(60, HYPOTHESIS_TARGET_WORDS // 2)} words. "
        if too_long
        else "Treat that quoted error only as a diagnostic and correct it. "
    )
    convo = list(messages) + [HumanMessage(content=(
        "Do NOT call any tools and do NOT include any prose or analysis. "
        "The previous response failed validation with this exact error: "
        f"{json.dumps(sanitized_error)}. " + remedy +
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
    try:
        return _parse_decision_json(
            resp.content,
            task=task,
            state=state,
        )
    except ValueError as error:
        # Last-resort salvage for the ONE failure mode that is safely droppable: the model
        # re-sent retired hyperparameter knobs. Those knobs are not applied by the trainer under
        # any circumstance, so discarding them yields exactly the decision the orchestrator could
        # legally have written, and keeping the remaining tunable fields preserves its intent.
        # This mirrors the existing `_dropped_fields` precedent for a stray `hyperparams` block on
        # a data_rebuild, whose comment records that rejecting whole decisions cost the NER run 65
        # orchestrator-authored plans. Anything else still raises.
        salvaged = _strip_retired_hyperparams(resp.content)
        if salvaged is None:
            raise
        decision = _validate_decision_json(
            salvaged, task=task, state=state
        )
        decision["_dropped_fields"] = sorted(
            set(decision.get("_dropped_fields") or []) | {"retired_hyperparams"}
        )
        return decision


def _strip_retired_hyperparams(raw):
    """Return the decision with retired hyperparameter keys removed, or None if not applicable.

    Only rewrites a `hyperparameter` decision whose sole defect is retired keys AND which still
    has at least one tunable field left to act on — otherwise there is no decision to salvage.
    """
    text = _coerce_to_text(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        parsed = json.loads(match.group() if match else text)
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(parsed, dict) or parsed.get("intervention") != "hyperparameter":
        return None
    hyperparams = parsed.get("hyperparams")
    if not isinstance(hyperparams, dict):
        return None
    kept = {k: v for k, v in hyperparams.items() if k in _TUNABLE_HYPERPARAMS}
    if not kept or set(hyperparams) <= set(kept):
        return None
    parsed["hyperparams"] = kept
    return parsed


def _validate_decision_json(
    parsed,
    *,
    task: str = "classification",
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
    hypothesis = hypothesis.strip()
    if len(hypothesis) > HYPOTHESIS_MAX_CHARS:
        print(
            f"      [iterate] ⚠ hypothesis truncated {len(hypothesis)} → "
            f"{HYPOTHESIS_MAX_CHARS} chars — raise SLM_HYPOTHESIS_MAX_CHARS; the tail of a "
            f"hypothesis carries the confusion pairs, so this loses actionable signal"
        )
        hypothesis = hypothesis[:HYPOTHESIS_MAX_CHARS]
    validated["hypothesis"] = hypothesis
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
        # A `mine_new_real` plan is rewritten to `surgical_synthesis` once every source is
        # exhausted and web research has spent its allowance, because it provably cannot add a row.
        plan = normalize_data_rebuild_plan(
            validated.get("data_rebuild"),
            task=task,
            hypothesis=hypothesis,
            mining_available=mining_available_for_state(state),
        )
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
        supported = set(_TUNABLE_HYPERPARAMS)
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
        # Store ONLY the orchestrator-facing tunable knobs. normalize_hyperparams also DERIVES
        # the trainer's shape — lora_alpha, lora_dropout, micro_batch_size,
        # gradient_accumulation_steps, effective_batch_size, batch_size — and writing those back
        # into the decision made this validator reject its own output on the second pass
        # (`iterate_node` re-validates as defense in depth), killing every hyperparameter
        # decision with an error naming six fields the orchestrator never proposed (B244).
        #
        # Nothing needs them here: `train._build_config` calls normalize_hyperparams itself at
        # the point of use, so the derived shape is rebuilt where it is actually consumed. Keeping
        # the decision to what was decided also restores `alpha_ratio`, which normalization drops
        # and which the decision log had therefore been printing as None.
        validated["hyperparams"] = _decided_hyperparams(normalized)
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


def _decided_hyperparams(normalized: dict) -> dict:
    """Project a normalized trainer config back to the five knobs the orchestrator chooses.

    `alpha_ratio` is reconstructed from the snapped alpha and rank because normalization emits
    absolute `lora_alpha` and drops the ratio. Reconstructing rather than echoing the model's
    raw value means the decision records what was actually APPLIED after snapping.
    """
    decided = {
        field: normalized[field]
        for field in ("lora_rank", "weight_decay", "learning_rate", "nr_epochs")
        if field in normalized
    }
    rank = normalized.get("lora_rank")
    alpha = normalized.get("lora_alpha")
    if rank and alpha:
        ratio = alpha / rank
        decided["alpha_ratio"] = int(ratio) if float(ratio).is_integer() else ratio
    return decided


def _hit_output_cap(response) -> bool:
    """True when the provider stopped generation because max_tokens was reached."""
    metadata = getattr(response, "response_metadata", None) or {}
    if not isinstance(metadata, dict):
        return False
    return metadata.get("stop_reason") == "max_tokens"


def _parse_decision_json(
    raw,
    *,
    task: str = "classification",
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
            task=task,
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
            task=task,
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
    "strategy": "surgical_synthesis",
    "rows": 400,
    "target_categories": [{"category": "wrong_arguments", "count": 61}],
    "pattern_hint": "arguments extracted from the wrong slot of the request"
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

DRIVE EVERY DATA-REBUILD FIELD FROM THE FAILURE ANALYSIS. The report gives you
per-difficulty accuracy (easy/medium/hard), aggregate confusion counts, a diagnosis, a
suggested intervention, and prior source novelty/yield. Every field below MUST be a
reasoned function of THAT evidence — not a fixed default and not a guess. In your
"hypothesis", name the specific failure evidence (which buckets, which confusion pairs)
that each non-trivial field value is responding to.

Data-rebuild payload constraints (how to set each from the failure analysis):
- strategy: EXACTLY ONE of two. These are the ONLY two ways the curriculum can grow.
    "mine_new_real"      -> add REAL rows. Taken first from datasets this run has already
                            sourced but not exhausted, and only if those are used up, from a
                            new dataset found by web research. Real data carries no teacher
                            error, so prefer this whenever rows are available: the best result
                            this project has measured came from a gold-only curriculum.
                            Choose it when the failure looks like THINNESS — broad errors, the
                            easy bucket failing, or a curriculum much smaller than the task
                            deserves.
    "surgical_synthesis" -> add TEACHER-GENERATED rows aimed at named failure categories.
                            Choose it when the failure is CONCENTRATED: a few categories
                            dominate and you want many more examples of exactly those.
  You will be told when mine_new_real is unavailable (every source exhausted and web research
  already spent). A mine_new_real plan sent after that is rewritten to surgical_synthesis,
  because it provably cannot add a row.
  There is no "resample" and no untargeted "synthesize". Re-drawing from the pool the
  curriculum was built from cannot add information, and generating rows for class balance
  rather than for observed failures wastes the same teacher calls.
- The curriculum is CUMULATIVE and has no target size. It starts at whatever the loader
  supplied and every rebuild ADDS to it; rows leave only via quality control or the eval
  firewall. So "rows" is how many NEW rows to add, not a size to reach.
- rows: integer [50,2000]. How many new rows this rebuild should add. Scale to the size of the
  failing region: toward 2000 when many categories are failing badly or the curriculum is
  thin, toward 50 when one narrow category is left.
- target_categories: the failure categories to aim at, echoed from the report's confusion
  counts, most-costly first, at most 8. Each is {"category": <name>, "count": <failures>}.
  These names come from the task's own scorer, so use the names you were given — do not
  invent one. A category you already targeted whose count did not fall will be skipped as
  exhausted, so prefer ones you have not spent on yet.
- pattern_hint: describe the dominant failure mode in your own words (aggregate categories
  only, never raw eval text). It is inserted into the generation prompt.

Rules:
- "hypothesis" is REQUIRED and must causally justify the action AND tie each non-trivial
  field to the failure evidence it responds to. Aim for about {hypothesis_target_words} words:
  long enough to name the specific buckets and confusion pairs driving the decision, because
  YOU WILL BE SHOWN THIS TEXT AGAIN on later iterations as the record of your own reasoning —
  a vague hypothesis is worthless to your future self. Do not pad it with restatement of the
  numbers above. Your ENTIRE response, including this field, must fit in
  {max_output_tokens} output tokens; if you exceed that, the JSON is cut off mid-object and
  your decision is DISCARDED, so keep the reasoning dense rather than lengthy.
- "data_rebuild" is REQUIRED when intervention is "data_rebuild"
- "hyperparams" is REQUIRED when intervention is "hyperparameter"
- emit no keys outside this schema and never include the other intervention's payload
- There is NO restriction on which strategy you may choose: any strategy is valid for any
  task type and at any score. Choose ONLY from the failure analysis (per-difficulty
  accuracy + confusion pairs + diagnosis), not from mechanical eligibility.
- "surgical_synthesis" always produces CORRECT training targets, never wrong-answer data. For
  a task with a fixed label set the generated row inherits a real row's label, so the target
  cannot be wrong — only the phrasing can. For an open-ended task the teacher writes both the
  input and the answer, and the row is checked first by an exact programmatic verifier where
  one exists (the format-bound tasks) and then by the teacher itself.
- The curriculum is NEVER padded to the system-computed target. That target is an upper
  aspiration; whatever real data and your chosen strategy supply, after quality control, is the
  size you are actually training on, and it is routinely well below target. If you want a bigger
  curriculum you have to choose a strategy that produces rows — nothing tops it up for you.
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

# Substituted rather than .format()ed: the prompt embeds literal JSON examples, so brace
# formatting over the whole string would fail on them.
_ITERATE_SYSTEM = _ITERATE_SYSTEM.replace(
    "{hypothesis_target_words}", str(HYPOTHESIS_TARGET_WORDS)
).replace("{max_output_tokens}", str(_ITERATE_MAX_TOKENS))


def _build_intervention_prompt_for_test() -> str:
    """Expose the orchestrator decision prompt for unit tests (no runtime use)."""
    return _ITERATE_SYSTEM


# Stagnation parameters: escalate if the best chronological gain over the last
# STAGNATION_WINDOW evaluation runs, relative to that window's first score, is
# below STAGNATION_MIN_DELTA. This treats declines/below-origin oscillation as
# no progress while allowing a genuine new high to keep the current model active.
#
# Policy (2026-08-05): ONE stagnation mechanism. Escalate once STAGNATION_WINDOW evals have gone
# by without the score improving by more than STAGNATION_MIN_DELTA.
#
# Measured over EVALS PERFORMED, not over surviving score entries. The old design read
# `state["scores"]`, which `rollback` pops on every regression — so in a run where most
# iterations regress the list never grew, the window never filled, and the check silently never
# fired (slm-clinc150-cse-38155022 sat at 2 entries for all 20 iterations). `eval_history` is
# append-only and is never rewound, so the window means what it says.
#
# Worked example of the intended semantics: an improvement, then 9 regressions, then another
# improvement, then 4 more regressions = 15 evals. If the best score across that whole window
# beat the score at its start by <= 2%, escalate — the two improvements do NOT reset anything,
# because what matters is total progress over the window, not recency of the last gain.
import os as _os
import time as _time
STAGNATION_WINDOW = int(_os.environ.get("SLM_STAGNATION_WINDOW", "15"))   # recent evals examined
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
# Unconditional ceiling. A model that has run this many evals WITHOUT reaching the goal must
# escalate regardless of what the score history looks like. Counts evals actually performed on
# the current model, so rollback cannot hide from it.
MAX_EVALS_BEFORE_ESCALATION = int(
    _os.environ.get("SLM_MAX_EVALS_BEFORE_ESCALATION", "30")
)


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


def _eval_history(state) -> list[float]:
    """Every eval score for the CURRENT model, in order, including rolled-back ones.

    `state["scores"]` is not usable for stagnation: rollback pops the regressing entry, so a run
    that mostly regresses keeps a permanently short list and the window never fills. `eval_history`
    is append-only (written by evaluate_node) and reset only on a tier change, so a window over it
    genuinely means "the last N evals of this model".

    Falls back to `scores` for states created before the field existed (older checkpoints).
    """
    history = state.get("eval_history")
    if isinstance(history, list) and history:
        return [float(s) for s in history]
    return list(state.get("scores") or [])


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
    from agent.run_memory import build_run_memory
    from data.curation_log import CurationLog

    llm = ChatAnthropic(
        model=ORCHESTRATOR_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens=_ITERATE_MAX_TOKENS,
        **orchestrator_client_kwargs(),
    )

    # Build context.
    #
    # Preferred form is the structured run memory: outcomes marked KEPT/ROLLED BACK with deltas,
    # failures since the last improvement aggregated by intervention, and full (never truncated)
    # reasoning. The raw data-curation.md dump is the fallback for the first iteration, before
    # the DAG has any nodes — and for any caller that has no DAG in state.
    trajectory = build_run_memory(state)
    if not trajectory:
        raw_trajectory = CurationLog(
            state.get("curation_log_path")
        ).read_latest()
        trajectory = (
            compact_trajectory(raw_trajectory)
            if should_compact(raw_trajectory)
            else raw_trajectory
        )
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
    # Memo about the attempt that was just rolled back. The per-difficulty block above describes
    # the RESTORED best checkpoint (the live weights); this describes the thing that failed, so
    # the orchestrator knows what not to repeat. Without it a rollback is invisible in the prompt
    # and the same intervention gets proposed again (B231).
    _failed = state.get("last_failed_attempt") or {}
    if _failed:
        _fbd = _failed.get("by_difficulty") or {}

        def _fbdfmt(b):
            v = _fbd.get(b) or {}
            a = v.get("accuracy")
            return f"{b}={a:.3f}" if a is not None else f"{b}=n/a"

        failed_block = (
            "\n## LAST ATTEMPT WAS ROLLED BACK — do not repeat it\n"
            f"- Tried: intervention={_failed.get('intervention')!r}"
            + (
                f" (sub-strategy={_failed.get('sub_strategy')!r})"
                if _failed.get("sub_strategy") else ""
            )
            + f" on iteration {_failed.get('iteration')}\n"
            f"- Result: scored {_failed.get('score')} vs best {_failed.get('best_score')} "
            f"(Δ={_failed.get('delta'):+}) — REGRESSION, so it was discarded and the previous "
            "best checkpoint restored.\n"
            f"- That attempt's difficulty profile: {_fbdfmt('easy')}  {_fbdfmt('medium')}  "
            f"{_fbdfmt('hard')}\n"
            f"- Its stated hypothesis was: {_failed.get('hypothesis', '')}\n"
            "- The scores in the report BELOW describe the restored best checkpoint (the current "
            "live weights), NOT the failed attempt. Choose something materially different from "
            "the failed attempt above.\n"
        )
    else:
        failed_block = ""

    test_agent_block = (
        failed_block
        + f"- Per-difficulty accuracy: {_bdfmt('easy')}  {_bdfmt('medium')}  {_bdfmt('hard')}\n"
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
    if not mining_available_for_state(state):
        rebuild_trials_block += (
            "\nRESAMPLE IS UNAVAILABLE THIS TURN: the entire training pool is already in the "
            "mine_new_real is UNAVAILABLE: every dataset this run has sourced is exhausted and "
            "web research has already spent its allowance without finding another. "
            "surgical_synthesis is the only data intervention that can still add rows."
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
    # State the mining position concretely rather than as a budget count. "3 paid rounds left" told
    # the orchestrator about an API allowance; what it actually needs to know is whether any dataset
    # still has rows it has not seen.
    _unexhausted = unexhausted_sources(state)
    if _unexhausted:
        mining_status = ", ".join(
            f"{s['source']} (taken {s.get('consumed', 0)} so far)" for s in _unexhausted
        )
    else:
        _failed = int(state.get("failed_discovery_rounds", 0) or 0)
        mining_status = (
            f"NONE — all sourced datasets exhausted; web research has failed {_failed} time(s). "
            + ("mine_new_real is RETIRED for this run."
               if not mining_available_for_state(state)
               else "One more web-research round is available.")
        )

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
# YOUR MEMORY OF THIS MODEL SO FAR
# Every attempt on the CURRENT model variant, including ones that were rolled back. Attempts are
# marked KEPT (they set a new best) or ROLLED BACK (they did not and were discarded). Treat
# "FAILED SINCE THE LAST IMPROVEMENT" as the record of what is NOT working: if one intervention
# type dominates that list, choosing it again is very unlikely to help.

{trajectory if trajectory else "(no iterations logged yet)"}

## Current iteration summary
- Task type: {state['task']}
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
- Datasets with rows left to mine: {mining_status}

## Test-data agent report (difficulty-stratified — use this to target your fix)
{test_agent_block}

## {tried_block}

## Data-rebuild plan notes
{rebuild_trials_block}

## Source novelty and prior plan yield
{source_yield_block}

Diagnose WHY the score is where it is from aggregate per-difficulty and confusion patterns,
source novelty/yield, remaining budget, prior hypotheses, and the trajectory,
then decide the next intervention. If you choose "data_rebuild", set EVERY field of the plan
as a reasoned function of that failure analysis (see the per-field guidance above), and name
the driving evidence in your hypothesis. If you choose "hyperparameter", you MUST propose a
config whose complete (dataset, hyperparameter) identity is NOT in the already-tried list
above (training is deterministic — an exact repeat cannot help). If the useful hyperparameter
space is exhausted (all sensible configs tried), choose "data_rebuild" or expect the system to
escalate to a larger model. Never inspect raw eval rows. Return only the decision JSON.
"""

    messages = [SystemMessage(content=_ITERATE_SYSTEM), HumanMessage(content=user_content)]

    # Iteration 1 logs the COMPLETE decision input — system prompt AND user content — so the run
    # log stays self-contained and any orchestrator call can be replayed from it. Later turns log
    # only the user content, because the system prompt is a fixed constant that does not change
    # within a run: repeating it every turn added thousands of identical lines between the parts
    # that actually differ, which is what made the CLINC150 logs hard to read.
    #
    # Set SLM_LOG_FULL_ITERATE_PROMPT=1 to restore the verbatim system prompt on every turn.
    _orch_id = state["selected_model"].label if state.get("selected_model") else "?"
    _full_every_turn = _os.environ.get("SLM_LOG_FULL_ITERATE_PROMPT", "0") == "1"
    _first_turn = not state.get("_iterate_prompt_logged")
    if _first_turn or _full_every_turn:
        _log(_orch_id, "  ===== ORCHESTRATOR CONTEXT (full prompt — system prompt logged once) =====")
        _log(_orch_id, f"  --- system prompt ---\n{_ITERATE_SYSTEM}")
        _log(_orch_id, f"  --- user content ---\n{user_content}")
        _log(_orch_id, "  ===== END ORCHESTRATOR CONTEXT =====")
        state["_iterate_prompt_logged"] = True
    else:
        _log(
            _orch_id,
            "  ===== ORCHESTRATOR CONTEXT (delta only; system prompt + fixed instructions "
            "unchanged since iteration 1) =====",
        )
        _log(_orch_id, f"  --- user content (this turn) ---\n{user_content}")
        _log(_orch_id, "  ===== END ORCHESTRATOR CONTEXT =====")

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
            task=state["task"],
            state=state,
        )
    try:
        if _hit_output_cap(response):
            # Distinguishing "malformed JSON" from "ran out of room" matters: they need
            # OPPOSITE corrections. Reported as a parse error, the reask says "return valid
            # JSON", the model writes another over-long answer, and it fails identically —
            # which is exactly what happened twice in slm-clinc150-cse-38179864 (B240).
            raise ValueError(
                f"response was cut off after {_ITERATE_MAX_TOKENS} output tokens "
                "(stop_reason=max_tokens), so the JSON is incomplete. The decision was too "
                "long, NOT malformed."
            )
        return _parse_decision_json(
            response.content,
            task=state["task"],
            state=state,
        )
    except Exception as error:  # noqa: BLE001 — see below
        # Catch Exception, not just ValueError. Self-correction should be attempted for ANY
        # validation/parse failure; a narrower clause silently skips the reask for anything the
        # validator (or a future validator) raises that is not a ValueError, and the run then
        # degrades straight to the score-band fallback. The CLINC150 run
        # (slm-clinc150-cse-38155022) recorded 19 `iterate` calls, 6 validation failures, and
        # ZERO `iterate_json_reask` events, so the reask demonstrably did not run there even
        # though the failures were ValueErrors — hence also the explicit log line below, so the
        # next run states plainly whether self-correction was attempted (B223).
        _log(
            _orch_id,
            f"  Decision failed validation ({type(error).__name__}: {str(error)[:160]}) "
            "— asking the orchestrator to correct itself (1 reask)",
        )
        try:
            corrected = _reask_json_only(
                messages,
                validation_error=(
                    error if isinstance(error, ValueError) else ValueError(str(error))
                ),
                task=state["task"],
                state=state,
            )
        except Exception as reask_error:  # noqa: BLE001
            _log(
                _orch_id,
                f"  Reask also failed ({type(reask_error).__name__}: "
                f"{str(reask_error)[:160]}) — falling back",
            )
            raise
        _log(_orch_id, "  Reask succeeded — using the corrected decision")
        return corrected


def apply_iteration_policy(score: float) -> dict:
    """
    Fallback score-band rules used when the LLM call is unavailable or fails.
    Returns dict with band and intervention type.
    """
    if score < 0.80:
        return {
            "band": "<0.80",
            "intervention": "data_rebuild",
            "data_rebuild_strategy": "mine_new_real",
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
            "data_rebuild_strategy": "surgical_synthesis",
            "description": (
                "Score ≥0.95 — refine remaining aggregate confusion with a "
                "bounded data rebuild."
            ),
        }


def _log(model_id: str, msg: str):
    print(f"[iterate][{model_id}] {msg}")


# ---------------------------------------------------------------------------
# Stretch goals: raising the accuracy target when a model converges quickly
# ---------------------------------------------------------------------------
# A goal derived from the teacher's zero-shot score (or from the 0.80 floor when the teacher scored
# below it) is a floor on ambition, not a ceiling. BC5CDR cleared 0.8000 in five iterations and 81
# minutes and stopped; the floor had set the bar, not the task's difficulty, and the run had hours
# of budget left. When a score clears the goal the orchestrator is now asked whether the goal should
# be raised, and it is told how quickly the goal was reached so "converged on iteration 2" and
# "converged on iteration 40" can be treated differently.
#
# TERMINATION. Raises are a RATCHET: each one must exceed the run's high-water mark
# (`max_stop_threshold`) by at least _THRESHOLD_RAISE_MIN_STEP, and THRESHOLD_CEILING caps the top.
# So the number of raises in a run is bounded by (ceiling - initial) / min_step, and each one
# additionally requires the model to actually clear the previous goal first. Every pre-existing
# budget (turn budget, wall clock, graph steps, stagnation, eval cap) still applies unchanged, so
# repeated raising cannot produce a run that fails to terminate.
_THRESHOLD_RAISE_MIN_STEP = 0.005

_THRESHOLD_RAISE_SYSTEM = """You set the accuracy goal for an agentic fine-tuning run.

The model just MET its accuracy goal. Your only decision: should the goal be RAISED so the run
keeps pushing, or is this run finished?

Raise the goal when the model reached it EASILY — few iterations, a large margin above the goal, a
still-rising trajectory, or plenty of remaining budget. The point is to find the model's real
ceiling rather than stopping at a bar that turned out to be soft. This matters most when the goal
came from a FLOOR rather than from the teacher's own measured score: a floored goal reflects what we
refused to go below, not what the task actually permits.

Do NOT raise when the model barely scraped over the line, when the trajectory is noisy or falling,
when it took many iterations to get here, or when little budget remains. A raise you cannot justify
just burns compute and ends the run on a failure note.

Reply with STRICT JSON, no prose:
{"raise_goal": true, "new_threshold": <float>, "reason": "<one sentence>"}
or
{"raise_goal": false, "reason": "<one sentence>"}

Constraints on new_threshold: strictly greater than the current goal by at least %(min_step)s, at
most %(ceiling)s, and it must be plausibly reachable — aim near what the trajectory suggests is
achievable, not at the ceiling by reflex."""


def _threshold_raise_enabled() -> bool:
    import os as _os

    if _os.environ.get("SLM_THRESHOLD_RAISE", "1") == "0":
        return False
    # Cheap mode exists to keep API spend near zero; a stretch goal is an optimisation, not a
    # correctness requirement, so it is the right thing to drop there.
    return _os.environ.get("SLM_CHEAP") != "1"


def _llm_threshold_raise(
    state: AgentState,
    current_score: float,
    model_id: str,
) -> dict | None:
    """Ask the orchestrator whether the met accuracy goal should be raised.

    A small, focused call rather than a field on the main intervention decision: the intervention
    prompt is only built for BELOW-threshold scores, and at convergence the only open question is
    the goal itself. Returns the validated decision dict, or None when the call fails or is
    unavailable — a failure always means "do not raise", never a broken run.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import SystemMessage, HumanMessage
    from config.config import (
        ANTHROPIC_API_KEY,
        ORCHESTRATOR_MODEL,
        orchestrator_client_kwargs,
    )
    from agent.llm_text import MIN_THINKING_SAFE_MAX_TOKENS
    from agent.threshold import THRESHOLD_CEILING, describe_threshold_provenance

    threshold = float(state["stop_threshold"])
    calibration = state.get("threshold_calibration") or {}
    history = _eval_history(state)
    iteration = int(state.get("iteration", 0) or 0)
    turn_budget = int(state.get("turn_budget", 0) or 0)
    turns_used = (iteration + 1) * 2
    raises = list(state.get("threshold_raises") or [])

    ceiling_headroom = THRESHOLD_CEILING - threshold
    if ceiling_headroom < _THRESHOLD_RAISE_MIN_STEP:
        _log(
            model_id,
            f"  Stretch goal: already within {_THRESHOLD_RAISE_MIN_STEP} of the "
            f"{THRESHOLD_CEILING} ceiling — not asking to raise",
        )
        return None

    user_content = "\n".join([
        f"Task type            : {state.get('task')}",
        f"Model                : {model_id}",
        f"Current accuracy goal: {threshold:.4f}",
        f"Goal provenance      : {describe_threshold_provenance(calibration)}",
        f"Score just achieved  : {current_score:.4f} "
        f"(margin +{current_score - threshold:.4f} above the goal)",
        f"Iterations used      : {iteration}",
        f"Score trajectory     : {[f'{s:.4f}' for s in history[-12:]]}",
        f"Budget               : {turns_used}/{turn_budget or 'unbounded'} turns used"
        + (
            f"; {turn_budget - turns_used} turns remain"
            if turn_budget
            else ""
        ),
        f"Ceiling              : {THRESHOLD_CEILING} (hard cap)",
        f"Minimum raise step   : {_THRESHOLD_RAISE_MIN_STEP}",
        f"Previous raises       : {len(raises)}"
        + (
            " — " + "; ".join(
                f"{r.get('from')}→{r.get('to')} at iteration {r.get('iteration')}"
                for r in raises[-4:]
            )
            if raises
            else " (none yet)"
        ),
        "",
        "Should the accuracy goal be raised? JSON only.",
    ])

    system = _THRESHOLD_RAISE_SYSTEM % {
        "min_step": _THRESHOLD_RAISE_MIN_STEP,
        "ceiling": THRESHOLD_CEILING,
    }
    try:
        llm = ChatAnthropic(
            model=ORCHESTRATOR_MODEL,
            anthropic_api_key=ANTHROPIC_API_KEY,
            max_tokens=MIN_THINKING_SAFE_MAX_TOKENS,
            **orchestrator_client_kwargs(),
        )
        response = tracked_chat_anthropic_invoke(
            llm,
            [SystemMessage(content=system), HumanMessage(content=user_content)],
            stage="threshold_raise",
            model=ORCHESTRATOR_MODEL,
        )
        return _validate_threshold_raise(
            _loads_json_object(_coerce_to_text(response.content))
        )
    except ValueError:
        # A malformed or unusable stretch-goal reply is not worth a reask: declining to raise is
        # always a safe outcome, and the goal the run was calibrated against is already banked.
        raise
    except Exception as exc:  # noqa: BLE001 - a failed stretch-goal call must never break a run
        raise_if_fatal(exc, "threshold_raise")
        _log(
            model_id,
            f"  Stretch-goal call failed ({exc!r}); keeping goal at {threshold:.4f}",
        )
        return None


def _loads_json_object(text: str) -> dict:
    """Parse a single JSON object out of an LLM reply, tolerating fences and surrounding prose."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    if not text:
        raise ValueError("empty response — no JSON object returned")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object found in response: {text[:160]!r}") from None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise ValueError(f"unparseable JSON in response: {exc}") from exc


def _validate_threshold_raise(decision: object) -> dict | None:
    """Validate the stretch-goal JSON. Returns None for a well-formed decline."""
    if not isinstance(decision, dict):
        raise ValueError("threshold-raise decision must be a JSON object")
    if not decision.get("raise_goal"):
        return None
    new_threshold = decision.get("new_threshold")
    if (
        isinstance(new_threshold, bool)
        or not isinstance(new_threshold, (int, float))
        or not math.isfinite(float(new_threshold))
    ):
        raise ValueError(
            "threshold-raise new_threshold must be a finite numeric value"
        )
    reason = decision.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("threshold-raise requires a non-empty reason")
    return {
        "raise_goal": True,
        "new_threshold": float(new_threshold),
        "reason": reason.strip()[:240],
    }


def _bank_convergence(
    state: AgentState,
    current_score: float,
    model_id: str,
) -> None:
    """Record that this run cleared an accuracy goal, before any raise can move the bar.

    Without this, raising the goal could convert a genuine success into a reported failure: a run
    that converged at 0.81 against 0.80, then raised to 0.87 and finished at 0.83, would print
    "budget exhausted" despite having met the goal it was actually calibrated against.
    """
    threshold = float(state["stop_threshold"])
    banked = state.get("convergence_banked")
    if isinstance(banked, dict) and float(banked.get("threshold", 0.0)) >= threshold:
        return
    state["convergence_banked"] = {
        "threshold": round(threshold, 4),
        "score": round(float(current_score), 4),
        "iteration": int(state.get("iteration", 0) or 0),
        "selector": getattr(state.get("selected_model"), "selector", None),
    }
    _log(
        model_id,
        f"  ✓ GOAL MET and banked: {current_score:.4f} >= {threshold:.4f} at iteration "
        f"{state.get('iteration', 0)} — this result stands even if a stretch goal is missed",
    )


def _maybe_raise_threshold(
    state: AgentState,
    current_score: float,
    model_id: str,
) -> bool:
    """Bank the convergence, then ask the orchestrator whether to raise the goal.

    Returns True when the goal was raised, in which case the caller must NOT treat the score as
    terminal — the run continues against the new goal.
    """
    from agent.threshold import THRESHOLD_CEILING

    _bank_convergence(state, current_score, model_id)
    if not _threshold_raise_enabled():
        return False

    # At most one stretch-goal call per iteration. iterate_node routes through here twice — once
    # before the intervention call and once after, because a threshold ADJUSTMENT can make the same
    # score converge. Without this guard, an orchestrator that lowered the goal would immediately
    # be asked to raise it again on the same score, in the same turn.
    iteration = int(state.get("iteration", 0) or 0)
    if state.get("_threshold_raise_asked_iteration") == iteration:
        return False
    state["_threshold_raise_asked_iteration"] = iteration

    threshold = float(state["stop_threshold"])
    high_water = max(
        float(state.get("max_stop_threshold", 0.0) or 0.0),
        threshold,
    )
    state["max_stop_threshold"] = high_water

    _log(
        model_id,
        f"  Goal {threshold:.4f} met with {current_score:.4f} at iteration "
        f"{state.get('iteration', 0)} — asking the orchestrator whether to raise it",
    )
    try:
        decision = _llm_threshold_raise(state, current_score, model_id)
    except ValueError as exc:
        _log(model_id, f"  Stretch-goal decision invalid ({exc}); keeping the goal")
        return False
    if decision is None:
        return False

    proposed = float(decision["new_threshold"])
    # The ratchet. Clamping to the high-water mark rather than the current goal means a goal that
    # was LOWERED earlier in the run cannot be used to re-raise into the same band repeatedly.
    minimum = high_water + _THRESHOLD_RAISE_MIN_STEP
    if proposed < minimum:
        _log(
            model_id,
            f"  Stretch goal DECLINED: proposed {proposed:.4f} does not exceed the run's "
            f"high-water goal {high_water:.4f} by the minimum step "
            f"{_THRESHOLD_RAISE_MIN_STEP} — keeping {threshold:.4f}",
        )
        return False
    raised = round(min(proposed, THRESHOLD_CEILING), 4)
    if raised <= threshold:
        return False

    reason = decision["reason"]
    _log(
        model_id,
        f"  ▲ RAISING accuracy goal {threshold:.4f} → {raised:.4f} "
        f"(ceiling {THRESHOLD_CEILING}, score was {current_score:.4f}, iteration "
        f"{state.get('iteration', 0)}). Reason: {reason}",
    )
    state["stop_threshold"] = raised
    state["max_stop_threshold"] = raised
    state["threshold_raises"] = list(state.get("threshold_raises") or []) + [{
        "from": round(threshold, 4),
        "to": raised,
        "score_at_raise": round(float(current_score), 4),
        "iteration": int(state.get("iteration", 0) or 0),
        "reason": reason,
    }]
    return True


def _ladder_enabled(strategy: str | None = None) -> bool:
    """False under a single-model strategy, which pins the run to one model for its whole life.

    Consulted at all three ladder gates — stagnation escalation, eval-cap escalation, and the
    post-convergence downward probe. Reading the strategy at each site independently is how one of
    them ends up missed, so they all go through here.
    """
    from config.config import model_ladder_enabled

    return model_ladder_enabled(strategy)


def _route_score_at_threshold(
    state: AgentState,
    current_score: float,
    model_id: str,
    policy: dict,
    *,
    stagnant: bool | None = None,
) -> bool:
    """Route a threshold-clearing score.

    Returns True when routing is complete. A hardware-blocked score is not an
    acceptable convergence result, but it is still handled deterministically so
    an auth/quota failure cannot break a run after the accuracy goal was reached.

    Meeting the goal is no longer automatically terminal: the goal is banked and the orchestrator
    is asked whether to RAISE it (see _maybe_raise_threshold). A raise returns False so the caller
    treats the score as below-threshold again and the run keeps training against the new goal.
    """
    if current_score < state["stop_threshold"]:
        return False

    # Ask about a stretch goal BEFORE the downward-probe/terminate decision. Both of those treat
    # the goal as settled, and if the bar is about to move the whole convergence question reopens:
    # probing for a smaller model that clears a goal we are about to abandon wastes a tier.
    if _maybe_raise_threshold(state, current_score, model_id):
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
        stagnant = _is_stagnant(_eval_history(state)) if stagnant is None else stagnant
        state["llm_iterate_decision"] = None
        if stagnant and not _ladder_enabled():
            state["next_action"] = "terminate"
            state["last_intervention"] = "terminate"
            state["last_hypothesis"] = (
                "hardware-blocked convergence stagnated; single_model pins the run to one model"
            )
            _log(model_id, "  → TERMINATE (hw-blocked + stagnation; single_model — no escalation)")
        elif stagnant:
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
    # Policy 2026-08-05: on meeting the goal ALWAYS try the next tier down, for every selection
    # strategy — the objective is the smallest model that clears the bar, so a converged larger
    # model is only a provisional answer. The two exit conditions are structural rather than
    # strategy-based: no lower tier exists, or every lower tier has already been tried
    # (`has_untried_lower_tier` below covers both). Under `smallest_first` the run already starts
    # at the lowest tier, so in practice it never regresses — which is the expected behaviour,
    # not a special case. The previous strategy allowlist meant a `smallest_first` run that had
    # ESCALATED could never come back down again even when the smaller model might now succeed
    # on the improved dataset.
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
        not state.get("downward_probe_done")
        and current_model is not None
        and has_untried_lower_tier
        # `single_model` answers "can THIS model do it", not "what is the smallest model that can".
        # Probing downward would replace the chosen model and destroy the comparison.
        and _ladder_enabled(strategy)
    ):
        state["next_action"] = "downward_probe"
        _log(
            model_id,
            f"  → DOWNWARD_PROBE (score {current_score:.4f} >= threshold on tier "
            f"{current_model.tier}; trying the next tier down to find the SMALLEST model that "
            f"still clears the goal — strategy={strategy})",
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
    stagnant = _is_stagnant(_eval_history(state))
    if stagnant:
        window = state["scores"][-STAGNATION_WINDOW:]
        gain = _stagnation_gain(window)
        _log(model_id, f"  Stagnation detected: window={[f'{s:.4f}' for s in window]}  "
             f"chronological_gain={gain:.4f} < {STAGNATION_MIN_DELTA}")

    # Stall backstop: catches the rollback churn that stagnation can miss (scores are
    # popped on rollback, so the window may never fill). Counts consecutive non-improving
    # evals, which survive rollback.
    # MAX_STALL_EVALS was removed (2026-08-05): "consecutive evals without a new best" and
    # "no meaningful gain over a window" were two knobs answering the same question, and the
    # consecutive counter reset on every improvement — so an improvement every 14 evals could
    # defer escalation forever. The window over eval_history subsumes it.

    # Unconditional ceiling on evals spent per model. `iteration` counts evals actually run on
    # the CURRENT model and is never rewound by rollback, so unlike the stagnation window this
    # cannot be hidden by popped scores. A model that has burned this many evals without hitting
    # the goal has had its chance.
    evals_run = int(state.get("iteration", 0) or 0)
    eval_cap_hit = evals_run >= MAX_EVALS_BEFORE_ESCALATION
    if eval_cap_hit and not stagnant:
        _log(model_id, f"  Eval cap reached: {evals_run} evals on this model without meeting the "
             f"goal (>= {MAX_EVALS_BEFORE_ESCALATION})")

    # Q9: stagnation/stall escalation is a RULE-BASED decision — take it WITHOUT spending an
    # orchestrator LLM call (the LLM cannot override it anyway). Only applies below the stop
    # threshold; the converged path below still runs. This saves an API call every time a
    # model plateaus (which is exactly when the run makes the most iterate calls).
    if current_score < state["stop_threshold"] and (stagnant or eval_cap_hit):
        if stagnant:
            reason = (
                f"no gain > {STAGNATION_MIN_DELTA:.0%} across the last "
                f"{STAGNATION_WINDOW} evals"
            )
        else:
            reason = f"{evals_run} evals without meeting the goal"
        if state.get("_largest_first_phase") == "probe":
            _log(model_id, "  → TERMINATE (largest_first probe stagnated — task infeasible)")
            state["next_action"] = "terminate"
            state["_largest_first_phase"] = "done"
            state["last_hypothesis"] = "largest_first probe could not clear the goal"
            state["last_intervention"] = "escalate"
        elif not _ladder_enabled():
            # The naive baseline: this model had its chance and did not clear the goal. Report that
            # honestly rather than reaching for a bigger model, because the whole point of the
            # strategy is to measure one model's ceiling under the loop.
            _log(
                model_id,
                f"  → TERMINATE ({reason}) — single_model pins the run to one model, so there is "
                f"no escalation. Best score on this model stands as the result.",
            )
            state["next_action"] = "terminate"
            state["last_intervention"] = "terminate"
            state["last_hypothesis"] = f"single_model: {reason}, no escalation available"
        else:
            _log(model_id, f"  → ESCALATE ({reason}) — skipped LLM intervention call to save API cost")
            state["next_action"] = "escalate"
            state["last_hypothesis"] = f"escalate on {reason}"
            state["last_intervention"] = "escalate"
        return state

    # Resample availability for THIS turn: if the whole train pool is already in the
    # Mining availability is recomputed each turn: a source with rows left keeps it on the menu,
    # and it comes off only once every source is exhausted AND web research has spent its allowance.
    # Advisory here; curate re-derives it precisely from the decontaminated pool at execution.


    # Not escalating → consult the orchestrator LLM for the intervention type.
    # NOTE: this runs in cheap mode too — cheap mode keeps the agent's bounded,
    # tool-free intervention reasoning, just on the Haiku tier.
    # Cheap mode's Claude savings come from config (Haiku everywhere) + curate skipping
    # curriculum synthesis and CoT annotation, NOT from dumbing this down to score bands.
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
            task=state["task"],
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
            # Only the FIVE tunable fields. The retired knobs (lora_alpha, lora_dropout,
            # micro_batch_size, gradient_accumulation_steps, effective_batch_size) were removed
            # from the orchestrator's choice set, so printing them only ever emitted `None`.
            _log(
                model_id,
                f"  Hyperparams: rank={hp.get('lora_rank')}  "
                f"alpha_ratio={hp.get('alpha_ratio')}  "
                f"weight_decay={hp.get('weight_decay')}  "
                f"lr={hp.get('learning_rate')}  "
                f"epochs={hp.get('nr_epochs')}",
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
                # Audit the lower the same way raises are audited. Without this the only record of
                # a lowered goal was one log line, so a converged run could not be checked against
                # the bar it was actually held to.
                state["threshold_lowers"] = list(
                    state.get("threshold_lowers") or []
                ) + [{
                    "from": round(float(state["stop_threshold"]), 4),
                    "to": round(clamped, 4),
                    "score_at_lower": round(float(current_score), 4),
                    "iteration": int(state.get("iteration", 0) or 0),
                    "reason": reason,
                }]
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
