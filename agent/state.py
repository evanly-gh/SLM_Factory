# agent/state.py
from typing import TypedDict, Optional
from data.eval_set import EvalSet
from eval.harness import EvalResult
from config.android_pool import ModelSpec, HardwareConstraints

class AgentState(TypedDict):
    # Task specification
    description: str
    target_metric: str
    hardware_constraints: HardwareConstraints

    # Task analysis outputs
    task: str                         # registry name: "xlam_bfcl", "clinc150", ... (see tasks/)
    selected_model: Optional[ModelSpec]
    feasible_models: list[ModelSpec]       # models that passed hardware_filter (Stages 1+2), largest→smallest
    stop_threshold: float             # calibrated target; iterate_node may lower or RAISE mid-run
    initial_stop_threshold: float     # immutable floor; stop_threshold can never go below this
    # Provenance of the accuracy target (B32): which source set it, the anchor and headroom
    # used, and whether calibration is still pending the first measurement. Audit trail for
    # "where did this goal come from" — see agent/threshold.py.
    threshold_calibration: Optional[dict]
    # Stretch-goal machinery. When a score clears the goal the orchestrator is asked whether the
    # goal should be RAISED, so a model that converged quickly is pushed further instead of
    # stopping at a bar it cleared on iteration 2.
    #   convergence_banked — the first goal this run actually cleared, and the score/iteration
    #     that cleared it. Banked so raising can never turn an already-successful run into a
    #     reported failure if the stretch goal is then missed.
    #   threshold_raises   — append-only audit of every raise (from, to, score, iteration, reason).
    #   threshold_lowers   — the same for every LOWER. The orchestrator may lower the goal down to
    #     `initial_stop_threshold` when the failures look like a capacity limit; that used to leave
    #     no durable trace, so the goal a run was actually held to was unrecoverable afterwards.
    #   max_stop_threshold — high-water mark. Raises must strictly exceed it, which makes the goal
    #     a ratchet toward THRESHOLD_CEILING and bounds how many raises a run can perform.
    # The per-iteration value is additionally stamped on each DAG node as `stop_threshold`, which
    # is what the post-run accuracy chart plots as a step line.
    convergence_banked: Optional[dict]
    threshold_raises: list[dict]
    threshold_lowers: list[dict]
    max_stop_threshold: float
    # The task's CLOSED label vocabulary, pinned once from the frozen eval set by eval_setup:
    # {"labels": [...], "definitions": {label: what it means}, "benchmark": str|None, "source": str}.
    # Downstream stages may only DROP rows outside it, never extend it, and no LLM may introduce a
    # class — see data/label_space.py (B259). None for task types whose `label` is a constant tag.
    task_label_space: Optional[dict]
    task_plan: Optional[dict]         # orchestrator's autonomous plan (labels, exa_queries, ...)
    autonomous: bool                  # if True, task_analysis derives task/plan via LLM

    # Data
    train_examples: list[dict]
    eval_set: Optional[EvalSet]
    data_source: Optional[str]        # provenance of train/eval examples (real benchmark vs web/Exa)
    current_dataset_path: Optional[str]  # path to Dcold JSONL on disk
    dataset_version: int              # incremented each curate call
    data_rebuild_plan: Optional[dict]  # validated declarative rebuild plan
    data_rebuild_plan_identity: Optional[str]  # canonical plan hash
    # --- Curriculum growth bookkeeping (2026-08-19) ---
    # The curriculum is CUMULATIVE: cold start loads gold rows, every data_rebuild ADDS to them, and
    # rows leave only via quality control or the eval firewall. Nothing re-draws from a pool it has
    # already drawn from, which is what the removed `resample`/gold-fill did.
    #   source_progress          — {source_id: {consumed, asked_for, url, exhausted}}. What we have
    #                              taken from each dataset, so `mine_new_real` can tell an exhausted
    #                              corpus from one we only read the first few thousand rows of.
    #   failed_discovery_rounds  — consecutive web-research rounds that contributed zero novel rows.
    #                              At MAX_FAILED_DISCOVERY_ROUNDS, mine_new_real is retired and the
    #                              orchestrator is told only surgical_synthesis remains.
    #   task_brief               — the orchestrator's own description of this benchmark, authored
    #                              once at cold start from real rows. Every synthesis and
    #                              verification prompt is built from it.
    source_progress: dict
    failed_discovery_rounds: int
    task_brief: Optional[dict]
    #   teacher_fitness — the five-shot score the teacher achieved on THIS task's eval set, and the
    #                     resulting go/no-go on synthetic data. Measured once at cold start. Absent
    #                     or unmeasured means synthesis is REFUSED: "we could not check" is not
    #                     evidence of a fit teacher. See agent/teacher_fitness.py.
    teacher_fitness: Optional[dict]
    #   run_health — cross-iteration counters (consecutive empty rebuilds, verification wipeouts,
    #                mining shutouts, load failures) plus a bounded per-iteration curriculum ledger.
    #                A single bad iteration is survivable; a pattern is a run that cannot learn, and
    #                nothing used to be watching across iterations. See agent/run_health.py.
    run_health: dict
    _last_synth_attempted: int
    _last_synth_kept: int
    curation_log_path: str            # run-local durable trajectory path

    # Search state
    best_weights_ref: Optional[str]
    best_score: float
    lifetime_best_score: float          # max best_score ever seen across all tiers/models in this run
    iteration: int
    scores: list[float]               # f(π) per iteration
    dag: list[dict]                   # lineage DAG nodes
    consecutive_no_improvement: int
    downward_probe_done: bool         # True once post-convergence lower-tier probing is complete
    retained_gguf_paths: list[str]    # GGUFs of new-best iterations; all others are reaped

    # Last iteration results
    last_eval: Optional[EvalResult]
    last_curation: Optional[dict]     # composition, provenance, plan config, source novelty/yield
    last_intervention: str            # "data_rebuild" | "hyperparameter" | "rollback"
    last_hypothesis: str              # LLM-generated causal reasoning for the next intervention
    llm_iterate_decision: Optional[dict]  # full LLM decision JSON from iterate_node
    next_action: str                  # "train" | "curate" | "rollback" | "escalate" | "terminate"

    # Baseline tracking: one entry per model tried, records zero-shot vs best fine-tuned
    model_baselines: list[dict]       # [{selector,model_id,quant,baseline_f1,best_finetuned_f1}, ...]

    # Phase 2 flags
    quantize_enabled: bool                # True runs INT4 quantization after eval
    hw_gating_enabled: bool               # True makes latency/power hard gates

    # Turn budget: charged at ~2 productive turns per iteration (curate + train).
    turn_budget: int
    _graph_steps: int                     # durable cumulative completed-node count
    _wallclock_terminated_before: Optional[str]  # long node skipped at wall guard
    # "the full orchestrator system prompt has already been logged this run". Undeclared keys
    # are dropped when LangGraph merges a node's returned state against this schema, so an
    # undeclared flag reads back False on the next call: iterate re-logged the entire system
    # prompt on all 69 turns instead of once (B256).
    _iterate_prompt_logged: bool
    # Iteration at which the stretch-goal question was last put to the orchestrator. iterate_node
    # routes through the threshold check twice per turn, so without this the same score would be
    # asked about twice — and, per the note above, an undeclared key reads back as absent, which
    # would make the guard silently do nothing.
    _threshold_raise_asked_iteration: Optional[int]

    # Model selection strategy state
    _largest_first_phase: Optional[str]   # "probe" | "escalate" | "done" (largest_first strategy only)
    escalation_history: list[dict]        # per-variant {selector, model_id, quant, tier, best_score, iterations, scores}
                                          # recorded by escalate_node before it resets, for the run summary

    # Internal: carry training results between train_node and evaluate_node
    _pending_weights_refs: Optional[dict]  # label -> weights_ref
    _pending_training_outputs: Optional[dict]  # label -> TrainingOutput
    _pending_configs: Optional[dict]       # label -> config dict

    # --- Data-size targets chosen by the orchestrator (task_planner), clamped to config
    # floors/ceiling. curate/eval_setup read these instead of a fixed per-type constant.
    # Curriculum and eval sizes are per-task caps on the spec (`initial_train_cap`, `eval_cap`), not
    # run state: the loader returns as many rows as it has up to those, and the curriculum then grows
    # by rebuild. The old `curriculum_size_target`/`eval_size_target` pair was recomputed per model
    # tier by a novelty x capacity formula whose result nothing read.

    # --- Data provenance + contamination control ---
    # Explicit source/split restrictions associated with held-out eval data. This is
    # decision metadata for acquisition code; eval_setup separately enforces normalized
    # text disjointness for the train/test rows it receives.
    eval_source_ban: list[dict]            # [{"kind":"hf|url|split","id":...}, ...]
    data_sources: list[dict]               # running lineage: every source used (unique records, with url)
    data_source_usage: list[dict]          # per-build provenance: [{iteration,dataset_version,strategy,sources:[...]}]

    # --- Difficulty-stratified eval + test-data agent ---
    # eval example ids/texts bucketed by difficulty (base-model zero-shot gradient), and the
    # test agent's per-difficulty scores + diagnosis from the latest eval.
    eval_difficulty: Optional[dict]        # {"easy":[...],"medium":[...],"hard":[...]}
    test_report: Optional[dict]            # {"overall":f,"by_difficulty":{...},"diagnosis":...}
    # Memo describing the intervention that was just rolled back: what was tried, its score and
    # delta, and its difficulty profile. `test_report` describes the CURRENT (restored) model, so
    # this is the only record of the failed attempt and exists so the orchestrator can avoid
    # repeating it (B227/B231). Cleared once a new attempt is evaluated.
    last_failed_attempt: Optional[dict]
    # Append-only per-model eval scores, including rolled-back ones. Stagnation is measured over
    # this because rollback pops state["scores"]. Reset on tier change.
    eval_history: Optional[list]

    # --- Post-convergence downward re-exploration ---
    downward_tiers_tried: list[int]        # tiers already re-probed downward (probe each once)
    converged_model_ref: Optional[dict]    # {selector,model_id,quant,tier,score} of the variant we converged on
    downward_probe_history: dict           # {origin, fixed_H, attempts: exact-selector/H probe records}
    downward_probe_pending: Optional[dict] # durable exact-selector + fixed H plan awaiting train/eval
