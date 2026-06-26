# agent/nodes/iterate.py
import json
from agent.state import AgentState


_ITERATE_SYSTEM = """\
You are the orchestrator of an agentic fine-tuning loop for small language models.
Your job: read the full training trajectory in data-curation.md and the current evaluation
failures, then decide what to do next and explain WHY.

Output ONLY valid JSON — no markdown, no commentary, no trailing text:
{
  "intervention": "<data_rebuild | hyperparameter | surgical>",
  "hypothesis": "<one concise sentence: causal reason the score is where it is and what you expect to change>",
  "hyperparams": {
    "lora_rank": <integer from [4,8,16,32,64] or null for full fine-tune>,
    "learning_rate": <float, e.g. 2e-4>,
    "nr_epochs": <integer>,
    "batch_size": <integer, e.g. 4 or 8>
  },
  "targeted_patterns": "<only for surgical intervention: describe the specific failure pattern to synthesize examples for>"
}

Rules:
- "hyperparams" is REQUIRED when intervention is "hyperparameter". Omit or set null for other interventions.
- "targeted_patterns" is REQUIRED when intervention is "surgical". Omit or set null for other interventions.
- lora_rank must be one of [4, 8, 16, 32, 64] or null (null = full fine-tune, slower but higher capacity).
- Do not produce a hyperparameter config identical to the previous iteration's best — the whole
  point is to explore a different region of the search space.

Score band guidance (but reason about the trajectory, do not just apply mechanical rules):
- Score < 0.80: usually a data problem (data_rebuild)
- 0.80–0.95: usually an optimization problem (hyperparameter)
- >= 0.95: usually surgical hard-negative addition (surgical)

If the trajectory shows stagnation (same intervention repeated with no improvement), escalate
the intervention type (e.g. try hyperparameter if data_rebuild stagnated, try surgical if
hyperparameter stagnated).
"""


def _llm_iterate(state: AgentState) -> dict:
    """Call the orchestrator LLM with the full trajectory and return a parsed decision dict."""
    import anthropic
    from config import ANTHROPIC_API_KEY, ORCHESTRATOR_MODEL
    from data.curation_log import CurationLog

    trajectory = CurationLog().read_latest()
    current_score = state["scores"][-1] if state["scores"] else 0.0
    last_eval = state.get("last_eval")
    failures_summary = ""
    if last_eval and last_eval.failures:
        sample = last_eval.failures[:10]
        failures_summary = "\n".join(
            f"  - predicted={f.get('predicted','?')} gold={f.get('label', f.get('entities', f.get('response', '?')))!r} text={str(f.get('text', f.get('prompt', '')))[:80]}"
            for f in sample
        )

    user_content = f"""\
## Current training trajectory

{trajectory if trajectory else "(no iterations logged yet)"}

## Current iteration summary
- Iteration: {state['iteration']}
- Current f(π): {current_score:.4f}
- Best f(π) so far: {state['best_score']:.4f}
- Consecutive rounds without improvement: {state['consecutive_no_improvement']}
- Score history: {state['scores']}
- Stop threshold: {state['stop_threshold']}

## Sample failures (up to 10)
{failures_summary if failures_summary else "(none yet)"}

Decide the next intervention. Output JSON only.
"""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=ORCHESTRATOR_MODEL,
        max_tokens=512,
        system=_ITERATE_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
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


def iterate_node(state: AgentState) -> AgentState:
    """
    Node 5: LLM-driven iteration decision.

    Reads the full data-curation.md trajectory and current failures, calls the orchestrator
    LLM (Claude Sonnet 4.6) to reason about WHY the score is where it is and propose the
    next (D, H, S) modification with a causal hypothesis. This implements the paper's
    EXPAND(v_parent, G, F) reasoning step.

    Falls back to score-band rules if the LLM call fails.
    """
    if not state["scores"]:
        state["next_action"] = "train"
        return state

    current_score = state["scores"][-1]

    # Try LLM-driven decision
    llm_decision = None
    hypothesis = ""
    try:
        llm_decision = _llm_iterate(state)
        intervention = llm_decision.get("intervention", "")
        hypothesis = llm_decision.get("hypothesis", "")
        if intervention not in ("data_rebuild", "hyperparameter", "surgical"):
            raise ValueError(f"Unknown intervention: {intervention!r}")
    except Exception as exc:
        # Fallback: score-band rules
        print(f"[iterate_node] LLM call failed ({exc!r}), falling back to score-band rules")
        fallback = apply_iteration_policy(current_score)
        intervention = fallback["intervention"]
        hypothesis = f"(fallback) {fallback['description']}"

    state["last_intervention"] = intervention
    state["last_hypothesis"] = hypothesis

    # Store additional LLM guidance so curate/train nodes can use it
    if llm_decision:
        state["llm_iterate_decision"] = llm_decision

    if current_score >= state["stop_threshold"]:
        state["next_action"] = "terminate"
    elif state["consecutive_no_improvement"] >= 2:
        state["next_action"] = "escalate"
    elif intervention == "hyperparameter":
        # Hyperparameter tuning: hold dataset fixed, skip curation, go straight to retrain
        state["next_action"] = "train"
    else:
        # data_rebuild and surgical interventions need curation first
        state["next_action"] = "curate"

    return state
