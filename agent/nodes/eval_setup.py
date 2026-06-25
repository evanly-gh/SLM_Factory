# agent/nodes/eval_setup.py
import os
import json
from agent.state import AgentState
from data.eval_set import build_eval_set

ARTIFACTS_DIR = "artifacts"


def eval_setup_node(state: AgentState) -> AgentState:
    """
    Node 2: download data and build E = Epos ∪ Eneg ∪ Eboundary.
    Eval set is built BEFORE any training. Fixed throughout all iterations.
    task_type flows from state — no hardcoding.
    """
    task_type = state["task_type"]
    plan = state.get("task_plan")

    if plan is not None:
        # Autonomous, general path: acquire the dataset from the web per the
        # orchestrator's plan (works for any task / task_type).
        from data.loaders.web_acquire import acquire_dataset
        train_examples, test_examples = acquire_dataset(
            plan, description=state.get("description", "")
        )
    elif task_type == "classification":
        from data.loaders.sms_spam import download_sms_spam
        train_examples, test_examples = download_sms_spam()
    else:
        raise NotImplementedError(
            f"eval_setup_node does not yet have a data loader for task_type={task_type!r}. "
            "Add a loader branch here when NER or generation data sources are available."
        )

    state["train_examples"] = train_examples
    eval_set = build_eval_set(test_examples, task_type=task_type)
    state["eval_set"] = eval_set

    # Persist the held-out eval set as a durable artifact (it is otherwise only in state).
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACTS_DIR, "eval_set.json"), "w") as f:
        json.dump({
            "task_type": eval_set.task_type,
            "counts": {"pos": len(eval_set.pos), "neg": len(eval_set.neg),
                       "boundary": len(eval_set.boundary), "total": len(eval_set.all)},
            "pos": eval_set.pos,
            "neg": eval_set.neg,
            "boundary": eval_set.boundary,
        }, f, indent=2)
    return state
