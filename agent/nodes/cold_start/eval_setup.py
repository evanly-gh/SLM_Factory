# agent/nodes/cold_start/eval_setup.py
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

    acquire_meta: dict = {}
    if plan is not None:
        # Autonomous, general path: acquire the dataset from the web per the orchestrator's
        # plan. Size the acquisition to the task's GOLD TARGET (curate uses N_TOTAL*0.65),
        # so a real benchmark loads enough train examples for curate to actually reach the
        # target (previously hard-capped at 300 → gold stuck at 300 for math's 650 target).
        # A little headroom (×1.15) covers eval-set overlap removal + quality-control drops.
        # `target_examples` is the web/synthesis-fallback ceiling (real benchmarks ignore it).
        from config.config import DATASET_SIZE_BY_TYPE
        _N_TOTAL = DATASET_SIZE_BY_TYPE.get(task_type, 150)
        _gold_target = int(_N_TOTAL * 0.65)
        _bench_train = min(int(_gold_target * 1.15) + 40, 1200)   # enough to reach gold target
        _bench_test = 80                                          # eval cost is ~N×tokens; keep modest
        from data.loaders.web_acquire import acquire_dataset
        train_examples, test_examples = acquire_dataset(
            plan, description=state.get("description", ""),
            target_examples=max(_gold_target, 120),
            benchmark_max_train=_bench_train, benchmark_max_test=_bench_test,
            meta=acquire_meta,
        )
    elif task_type == "classification":
        from data.loaders.sms_spam import download_sms_spam
        train_examples, test_examples = download_sms_spam()
        acquire_meta["source"] = "bundled SMS Spam dataset (UCI)"
    else:
        raise NotImplementedError(
            f"eval_setup_node does not yet have a data loader for task_type={task_type!r}. "
            "Add a loader branch here when NER or generation data sources are available."
        )

    state["train_examples"] = train_examples
    state["data_source"] = acquire_meta.get("source", "unknown")

    # Forward planner flags so the eval set carries multi_label/schema/multilingual
    # context for downstream scorer dispatch.
    plan = state.get("task_plan") or {}
    eval_set = build_eval_set(
        test_examples,
        task_type=task_type,
        multi_label=plan.get("multi_label", False),
        schema=plan.get("schema", None),
        multilingual=plan.get("multilingual", False),
    )
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
