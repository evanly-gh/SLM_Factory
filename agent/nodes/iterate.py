# agent/nodes/iterate.py
"""
Node 5: LLM-driven iteration decision (EXPAND operator).

Paper §2.1: 'each turn is one LLM reasoning step together with its associated tool calls.'
Paper §2.2 Eq. 2: 'EXPAND is implemented by the orchestrator LLM: it inspects the parent
trajectory, diagnoses failure modes, and proposes a hypothesis-driven modification.'

The LLM has access to 4 tools (bash, read_file, edit_file, web_search) via
ChatAnthropic.bind_tools. It can use them to inspect data, read eval results,
or check the curation log before making its decision.
"""
import json
import re
from agent.state import AgentState
from config.android_pool import check_hardware_constraints, all_constraints_pass
from agent.llm_errors import raise_if_fatal


def _parse_decision_json(raw) -> dict:
    """Robustly parse the orchestrator's decision JSON (B143 fix for JSONDecodeError).

    Tolerates code fences and prose wrapped around the JSON object. Raises ValueError
    (not JSONDecodeError) with a readable message on an empty/JSON-less response so the
    caller's fallback path logs something actionable.
    """
    text = raw if isinstance(raw, str) else str(raw)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    if not text:
        raise ValueError("empty LLM response — no decision JSON returned")
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError(f"no JSON object found in LLM response: {text[:160]!r}")


_ITERATE_SYSTEM = """\
You are the orchestrator of an agentic fine-tuning loop for small language models.
Your job: read the full training trajectory and the current evaluation failures,
then decide what to do next and explain WHY.

You have access to tools:
- bash: run shell commands (e.g. to inspect files, check dataset stats)
- read_file: read a file from disk (e.g. data-curation.md, eval results)
- edit_file: write content to a file
- web_search: search the web via Exa API

Use tools if you need more information before deciding. When you are ready,
output your final decision as JSON (no tool call). The JSON format:

{
  "intervention": "<data_rebuild | hyperparameter | surgical>",
  "hypothesis": "<one concise sentence: causal reason the score is where it is>",
  "hyperparams": {
    "lora_rank": <integer from [4,8,16,32,64]>,
    "learning_rate": <float, e.g. 2e-4>,
    "nr_epochs": <integer>,
    "batch_size": <integer, e.g. 4 or 8>
  },
  "targeted_patterns": "<only for surgical: describe the failure pattern to target>",
  "threshold_adjustment": {
    "new_threshold": <float or null>,
    "reason": "<required if new_threshold is not null: explain which failure category is out of distribution>"
  }
}

Rules:
- "hyperparams" is REQUIRED when intervention is "hyperparameter"
- "targeted_patterns" is REQUIRED when intervention is "surgical"
- lora_rank must be one of [4, 8, 16, 32, 64]
- Do not repeat a hyperparameter config identical to the previous iteration's best

Score band guidance (reason about the trajectory, not just mechanical rules):
- Score < 0.80: usually a data problem (data_rebuild)
- 0.80–0.95: usually an optimization problem (hyperparameter)
- >= 0.95: usually surgical hard-negative addition (surgical)

If the trajectory shows stagnation, escalate the intervention type.

Threshold adjustment guidance:
Examine the sample failures. If the dominant failure cluster reflects a genuine model
capacity limit — not a data or hyperparameter problem — you may lower the stop_threshold.
Examples that justify lowering: failures requiring world knowledge beyond the model's
parametric memory, failures on reasoning chains longer than the model can reliably produce,
failures on adversarial/out-of-distribution inputs that no amount of training data would fix.
Do NOT lower the threshold simply because progress is slow or the task is hard but learnable.
Set "new_threshold" to null if no adjustment is warranted.
The floor is enforced by the system — you cannot set it below the initial calibrated value.
"""

MAX_TOOL_ROUNDS = 5

# Stagnation parameters: escalate if the total improvement over the last
# STAGNATION_WINDOW evaluation runs is below STAGNATION_MIN_DELTA.
# This is more robust than a consecutive-zero-improvement count because it
# accounts for very slow but real progress (which should not trigger escalation)
# versus genuine plateaus (which should).
STAGNATION_WINDOW = 3       # number of recent evaluations to look at
STAGNATION_MIN_DELTA = 0.02  # minimum cumulative improvement over that window

# Hard backstop against a rollback→re-decide→rollback churn. `should_rollback` pops the
# regressing score, so the stagnation window can stay short and never fire; meanwhile
# `consecutive_no_improvement` (set in evaluate_node) grows every non-improving eval and
# is NOT popped. Once this many evals in a row fail to beat the best score, stop churning
# and escalate — which promotes to a bigger model if one fits, else terminates cleanly.
MAX_STALL_EVALS = 4


def _is_stagnant(scores: list[float]) -> bool:
    """
    Return True if the model has stagnated and should escalate.

    Stagnation is defined as: the total improvement across the last
    STAGNATION_WINDOW evaluation runs is less than STAGNATION_MIN_DELTA.
    Requires at least STAGNATION_WINDOW scores before it can fire.
    """
    if len(scores) < STAGNATION_WINDOW:
        return False
    window = scores[-STAGNATION_WINDOW:]
    delta = max(window) - min(window)
    return delta < STAGNATION_MIN_DELTA


def _format_failures(failures: list[dict], task_type: str) -> str:
    """Render sample failures in a task-appropriate, readable form.

    The generic `predicted=X gold=Y text=Z` layout is unreadable for NER (gold is a
    list of entity dicts) and generation (gold is a long answer). Formatting per task
    type gives the orchestrator LLM a signal it can actually diagnose.
    """
    lines = []
    for f in failures:
        if task_type == "classification":
            text = str(f.get("text", ""))[:120]
            lines.append(f"  - text={text!r}\n    predicted={f.get('predicted','?')!r}  gold={f.get('label','?')!r}")
        elif task_type == "NER":
            text = str(f.get("text", ""))[:120]
            gold = f.get("entities", f.get("label", "?"))
            pred = f.get("predicted", "?")
            lines.append(f"  - text={text!r}\n    predicted_spans={pred!r}\n    gold_spans={gold!r}")
        else:  # math_reasoning, code_generation, generation
            prompt = str(f.get("prompt", f.get("text", "")))[:120]
            gold = str(f.get("answer", f.get("response", f.get("label", "?"))))[:150]
            pred = str(f.get("predicted", "?"))[:150]
            judge = f.get("judge_score")
            judge_str = f"  (judge={judge})" if judge is not None else ""
            lines.append(f"  - prompt={prompt!r}{judge_str}\n    predicted={pred!r}\n    gold={gold!r}")
    return "\n".join(lines)


def _llm_iterate(state: AgentState) -> dict:
    """Call the orchestrator LLM with tool access to reason about the trajectory.

    Uses ChatAnthropic.bind_tools so the LLM can call bash/read_file/edit_file/web_search
    to inspect data before making its decision. Multi-round tool loop: call LLM → execute
    tool calls → feed results back → repeat until the LLM produces a final text response.

    Uses the Context Manager to compact older iterations before sending to the LLM.
    Paper §2.1, §2.2.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
    from config.config import ANTHROPIC_API_KEY, ORCHESTRATOR_MODEL
    from agent.context_manager import compact_trajectory, should_compact
    from agent.tools import COLD_START_TOOLS
    from data.curation_log import CurationLog

    # Build tool-bound LLM
    llm = ChatAnthropic(
        model=ORCHESTRATOR_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens=1024,
    ).bind_tools(COLD_START_TOOLS)

    tools_by_name = {t.name: t for t in COLD_START_TOOLS}

    # Build context
    raw_trajectory = CurationLog().read_latest()
    trajectory = compact_trajectory(raw_trajectory) if should_compact(raw_trajectory) else raw_trajectory
    current_score = state["scores"][-1] if state["scores"] else 0.0
    last_eval = state.get("last_eval")
    failures_summary = ""
    if last_eval and last_eval.failures:
        failures_summary = _format_failures(last_eval.failures[:10], state["task_type"])

    scores = state["scores"]
    window = scores[-STAGNATION_WINDOW:] if len(scores) >= STAGNATION_WINDOW else scores
    recent_delta = round(max(window) - min(window), 4) if window else 0.0

    user_content = f"""\
## Training trajectory so far (from data-curation.md — each row is one past iteration:
## its dataset version, intervention applied, and resulting score)

{trajectory if trajectory else "(no iterations logged yet)"}

## Current iteration summary
- Task type: {state['task_type']}
- Model: {state['selected_model'].model_id if state.get('selected_model') else 'unknown'}
- Iteration: {state['iteration']}
- Current f(π): {current_score:.4f}
- Best f(π) so far: {state['best_score']:.4f}
- Score history: {scores}
- Recent delta (last {len(window)} evals, improvement to trigger escalation = {STAGNATION_MIN_DELTA}): {recent_delta:.4f}
- Stop threshold: {state['stop_threshold']} (initial floor: {state.get('initial_stop_threshold', state['stop_threshold']):.3f})

## Sample failures from the latest eval (up to 10, formatted for a {state['task_type']} task)
{failures_summary if failures_summary else "(none yet)"}

Diagnose WHY the score is where it is from the failures and the trajectory above,
then decide the next intervention. Do not repeat an intervention that the trajectory
shows already failed to move the score. Use tools if you need to inspect the data,
then output the decision JSON.
"""

    messages = [SystemMessage(content=_ITERATE_SYSTEM), HumanMessage(content=user_content)]

    response = None
    for _ in range(MAX_TOOL_ROUNDS):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return _parse_decision_json(response.content)

        # Execute tool calls, feed results back
        for tc in response.tool_calls:
            tool = tools_by_name.get(tc["name"])
            if tool:
                try:
                    result = tool.invoke(tc["args"])
                except Exception as e:
                    result = f"[tool error] {e}"
            else:
                result = f"[unknown tool] {tc['name']}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # Exhausted the tool-use budget. Rather than giving up straight to the score-band
    # fallback (the frequent "LLM exhausted tool rounds" warning in the logs), make ONE
    # final call with tools removed and an explicit instruction to answer now. This
    # salvages a real, reasoned decision in the common case where the model was still
    # exploring with tools when it ran out of rounds.
    if response is None:
        raise RuntimeError(
            "LLM exhausted tool rounds without producing a final JSON decision"
        )
    if response.tool_calls:
        llm_final = ChatAnthropic(
            model=ORCHESTRATOR_MODEL,
            anthropic_api_key=ANTHROPIC_API_KEY,
            max_tokens=1024,
        )  # NOT tool-bound: forces a text answer
        messages.append(HumanMessage(content=(
            "You have used all available tool rounds. Do NOT call any tools. "
            "Respond NOW with ONLY the decision JSON described in the system prompt "
            "(no prose, no code fences)."
        )))
        response = llm_final.invoke(messages)
        if getattr(response, "tool_calls", None):
            raise RuntimeError(
                "LLM exhausted tool rounds without producing a final JSON decision"
            )
    return _parse_decision_json(response.content)


def apply_iteration_policy(score: float) -> dict:
    """
    Fallback score-band rules used when the LLM call is unavailable or fails.
    Returns dict with band and intervention type.
    """
    if score < 0.80:
        return {
            "band": "<0.80",
            "intervention": "data_rebuild",
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
            "intervention": "surgical",
            "description": "Score ≥0.95 — add 2–3 targeted examples per remaining failure pattern.",
        }


def _log(model_id: str, msg: str):
    print(f"[iterate][{model_id}] {msg}")


def iterate_node(state: AgentState) -> AgentState:
    """
    Node 7: LLM-driven iteration decision with tool access.

    The orchestrator LLM can use bash/read_file/edit_file/web_search to inspect
    the training data, eval results, or curation log before making its decision.
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

    current_score = state["scores"][-1]
    policy = apply_iteration_policy(current_score)

    _log(model_id, f"Current score: {current_score:.4f}  "
         f"band={policy['band']}  threshold={state['stop_threshold']:.3f}")

    # Check stagnation before LLM call
    stagnant = _is_stagnant(state["scores"])
    if stagnant:
        window = state["scores"][-STAGNATION_WINDOW:]
        delta = max(window) - min(window)
        _log(model_id, f"  Stagnation detected: window={[f'{s:.4f}' for s in window]}  "
             f"delta={delta:.4f} < {STAGNATION_MIN_DELTA}")

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
    llm_decision = None
    hypothesis = ""
    intervention = policy["intervention"]  # initialized to fallback; overwritten by LLM if successful
    try:
        _log(model_id, "  Calling orchestrator LLM for intervention decision...")
        llm_decision = _llm_iterate(state)
        intervention = llm_decision.get("intervention", "")
        hypothesis = llm_decision.get("hypothesis", "")
        if intervention not in ("data_rebuild", "hyperparameter", "surgical"):
            raise ValueError(f"Unknown intervention: {intervention!r}")

        _log(model_id, f"  LLM decision: intervention={intervention}")
        _log(model_id, f"  Hypothesis: {hypothesis}")
        if intervention == "hyperparameter" and llm_decision.get("hyperparams"):
            hp = llm_decision["hyperparams"]
            _log(model_id, f"  Hyperparams: lora_rank={hp.get('lora_rank')}  "
                 f"lr={hp.get('learning_rate')}  epochs={hp.get('nr_epochs')}  "
                 f"batch={hp.get('batch_size')}")
        if intervention == "surgical" and llm_decision.get("targeted_patterns"):
            _log(model_id, f"  Targeted patterns: {llm_decision['targeted_patterns']}")

    except Exception as exc:
        # Billing/auth/quota errors will recur on every call — fail fast with a clear
        # message instead of silently limping through the rest of the run on fallbacks (B144).
        raise_if_fatal(exc, "iterate")
        _log(model_id, f"  LLM call failed ({exc!r}), falling back to score-band rules")
        fallback = apply_iteration_policy(current_score)
        intervention = fallback["intervention"]
        hypothesis = f"(fallback) {fallback['description']}"
        _log(model_id, f"  Fallback: intervention={intervention}")

    state["last_intervention"] = intervention
    state["last_hypothesis"] = hypothesis

    if llm_decision:
        state["llm_iterate_decision"] = llm_decision

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

    # Largest-first strategy: when the probe model hits the threshold,
    # switch to the smallest model and restart the training loop.
    if state.get("_largest_first_phase") == "probe" and current_score >= state["stop_threshold"]:
        from agent.nodes.cold_start.model_selection.largest_first import check_probe_result
        state = check_probe_result(state)
        if state.get("next_action") == "curate":
            _log(model_id, f"  → CURATE (largest_first probe succeeded; switching to smallest model)")
            return state

    if current_score >= state["stop_threshold"]:
        hw_blocks_termination = False
        _gating_model = state.get("selected_model")
        if state.get("hw_gating_enabled") and _gating_model is not None:
            hw_check = check_hardware_constraints(
                _gating_model, state["hardware_constraints"]
            )
            if not all_constraints_pass(hw_check):
                hw_blocks_termination = True
        if hw_blocks_termination:
            _log(model_id,
                 f"  Score {current_score:.4f} >= threshold but hardware FAILS — "
                 f"not accepting as terminal; continuing")
            if stagnant:
                state["next_action"] = "escalate"
                _log(model_id, "  → ESCALATE (hw-blocked terminal + stagnation)")
            elif intervention == "hyperparameter":
                state["next_action"] = "train"
            else:
                state["next_action"] = "curate"
            return state
        # Active downward probe: route to the downward_probe node ONCE to try a smaller
        # model. Q3: only meaningful for strategies that do NOT already start at the bottom
        # of the feasible set — interpolation and orchestrator_choice pick a mid/large model
        # by prediction, so a smaller one might also clear the goal. smallest_first and
        # largest_first already end up at the smallest model, so a downward probe is redundant.
        # Read the strategy from the env directly (same source config.py uses) so this
        # routing decision needs no API-key-bearing config import.
        import os as _os
        _strategy = _os.environ.get("SLM_MODEL_SELECTION_STRATEGY", "smallest_first")
        _probes_down = _strategy in ("interpolation", "orchestrator_choice")
        _current_model = state.get("selected_model")
        if (_probes_down
                and not state.get("downward_probe_done")
                and _current_model is not None
                and _current_model.tier > 0):
            state["next_action"] = "downward_probe"
            _log(model_id,
                 f"  → DOWNWARD_PROBE (score {current_score:.4f} >= threshold; strategy="
                 f"{_strategy} may have over-selected; trying a smaller model)")
            return state
        state["next_action"] = "terminate"
        _log(model_id, f"  → TERMINATE (score {current_score:.4f} >= threshold {state['stop_threshold']:.3f})")
    elif intervention == "hyperparameter":
        state["next_action"] = "train"
        _log(model_id, f"  → TRAIN (hyperparameter intervention, dataset held fixed)")
    else:
        state["next_action"] = "curate"
        _log(model_id, f"  → CURATE ({intervention})")

    return state
