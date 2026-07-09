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
from agent.state import AgentState


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
    import agent.nodes.evaluate as _eval_mod
    CurationLog = _eval_mod.CurationLog

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
        sample = last_eval.failures[:10]
        failures_summary = "\n".join(
            f"  - predicted={f.get('predicted','?')} gold={f.get('label', f.get('entities', f.get('response', '?')))!r} text={str(f.get('text', f.get('prompt', '')))[:80]}"
            for f in sample
        )

    scores = state["scores"]
    window = scores[-STAGNATION_WINDOW:] if len(scores) >= STAGNATION_WINDOW else scores
    recent_delta = round(max(window) - min(window), 4) if window else 0.0

    user_content = f"""\
## Current training trajectory

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

## Sample failures (up to 10)
{failures_summary if failures_summary else "(none yet)"}

Decide the next intervention. Use tools if needed, then output JSON.
"""

    messages = [SystemMessage(content=_ITERATE_SYSTEM), HumanMessage(content=user_content)]

    for _ in range(MAX_TOOL_ROUNDS):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # Final response — parse JSON from content
            raw = response.content if isinstance(response.content, str) else str(response.content)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())

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

    # Exhausted tool rounds — try to parse the last response
    raw = response.content if isinstance(response.content, str) else str(response.content)
    return json.loads(raw.strip())


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
    model_id = selected.model_id if selected is not None else "?"

    if not state["scores"]:
        _log(model_id, "No scores yet — routing to train")
        state["next_action"] = "train"
        return state

    # Turn-budget guard (B51): ~2 productive turns per iteration (curate + train).
    turn_budget = state.get("turn_budget", 0)
    turns_used = state["iteration"] * 2
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

    # Try LLM-driven decision with tool access
    llm_decision = None
    hypothesis = ""
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
            floor = state.get("initial_stop_threshold") or state["stop_threshold"]
            clamped = max(float(new_threshold), floor)
            if clamped < state["stop_threshold"]:
                reason = adj.get("reason", "")
                _log(model_id,
                     f"  Lowering stop_threshold "
                     f"{state['stop_threshold']:.3f} → {clamped:.3f} "
                     f"(floor={floor:.3f}). Reason: {reason}")
                state["stop_threshold"] = clamped

    if current_score >= state["stop_threshold"]:
        state["next_action"] = "terminate"
        _log(model_id, f"  → TERMINATE (score {current_score:.4f} >= threshold {state['stop_threshold']:.3f})")
    elif stagnant:
        state["next_action"] = "escalate"
        _log(model_id, f"  → ESCALATE (stagnation overrides LLM decision)")
    elif intervention == "hyperparameter":
        state["next_action"] = "train"
        _log(model_id, f"  → TRAIN (hyperparameter intervention, dataset held fixed)")
    else:
        state["next_action"] = "curate"
        _log(model_id, f"  → CURATE ({intervention})")

    return state
