# agent/nodes/eval_setup.py
from agent.state import AgentState
from data.eval_set import build_eval_set


def eval_setup_node(state: AgentState) -> AgentState:
    """
    Node 2: download data and build E = Epos ∪ Eneg ∪ Eboundary.
    Eval set is built BEFORE any training. Fixed throughout all iterations.
    task_type flows from state — no hardcoding.
    """
    task_type = state["task_type"]

    if task_type == "classification":
        from data.loaders.sms_spam import download_sms_spam
        train_examples, test_examples = download_sms_spam()
    else:
        raise NotImplementedError(
            f"eval_setup_node does not yet have a data loader for task_type={task_type!r}. "
            "Add a loader branch here when NER or generation data sources are available."
        )

    state["train_examples"] = train_examples
    state["eval_set"] = build_eval_set(test_examples, task_type=task_type)
    return state
