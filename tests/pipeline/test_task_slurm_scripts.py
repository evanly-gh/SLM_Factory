import os
import re
import subprocess
from pathlib import Path


PIPELINE_DIR = Path(__file__).parent

# The curated-benchmark pipeline scripts: 2x L40S on gpu-l40s-intelligentsystems, each pinning one
# task via SLM_BENCHMARK_TASK. The value IS the registry key: the filename suffix, the exported
# variable and `tasks.TASKS` all use the same name, so there is no second table to keep in step —
# which is what the two hand-synchronised dicts here used to be
# (`NAMED_BENCHMARK_TASK_TYPES` plus a parallel loader table).
#
# `coedit` and `medqa` were removed 2026-08-15 by decision; neither had ever produced a run. The
# two legacy autonomous scripts (`run_math_l40s.slurm`, `run_ner_l40s.slurm`) went with the
# channels on 2026-08-18: math and NER are now curated tasks with their own launchers.
L40S_BENCHMARK_SCRIPTS = {
    "run_clinc150_l40s.slurm": "clinc150",
    "run_routerbench_l40s.slurm": "routerbench",
    "run_dialogsum_l40s.slurm": "dialogsum",
    "run_gsm8k_l40s.slurm": "gsm8k",
    "run_xlam_bfcl_l40s.slurm": "xlam_bfcl",
    # Added 2026-08-15: LlamaPIE when-to-respond (arXiv:2505.04066), vendored bundle, no live fetch.
    "run_proactive_listening_l40s.slurm": "proactive_listening",
    # Added 2026-08-13 alongside the registry entries. These exist on BOTH accounts because the
    # shared CSE queue was priority-starved (28th of 53 pending, projected start 35h out) while
    # the dedicated quota had 5 free GPUs; running on whichever frees first is the point.
    "run_calendar_json_l40s.slurm": "calendar_json",
    "run_ner_bc5cdr_l40s.slurm": "ner_bc5cdr",
    # Re-added 2026-08-23. An sms_spam loader existed before the 2026-08-18 registry rebuild and
    # was deleted with the `task_type` channel; the new one is a rewrite that deduplicates before
    # splitting and stratifies the holdout (the old one did neither — B44).
    "run_sms_spam_l40s.slurm": "sms_spam",
    # Added 2026-08-24: ToolBench / ToolEval pass rate, reimplementing arXiv:2512.15943.
    "run_toolbench_l40s.slurm": "toolbench",
}

L40S_TASK_SCRIPTS = tuple(L40S_BENCHMARK_SCRIPTS)

# The PREEMPTIBLE family (2026-08-27). Same pipeline body and same benchmark keys as their `_l40s`
# twins, run on `ckpt-g2` under QOS `ckpt-gpu` because the dedicated queue had become unusable:
# priority here is almost entirely fair-share (`PriorityWeightFairShare=5000` against
# `PriorityWeightAge=0`, so a job never improves by waiting), the account was at ~2x its
# entitlement, and a fresh submission projected 30+ hours out. `PriorityWeightQOS` is 100000 and
# `ckpt-gpu` holds the cluster's highest QOS priority, so the same job starts in minutes.
#
# DELIBERATELY LIMITED TO CHEAP TASKS. `PreemptMode=REQUEUE` returns a preempted job to the queue
# rather than killing it, but `GraceTime=0` and `KillWait=10s` leave ~10 seconds — not enough to
# checkpoint a training step — so a preempted run resumes from its last COMMITTED checkpoint and
# loses what was in flight. These three train in minutes, so that costs minutes. `toolbench` trains
# for ~93 minutes a step and is NOT in this family for that reason.
CKPT_BENCHMARK_SCRIPTS = {
    "run_sms_spam_ckpt.slurm": "sms_spam",
    "run_clinc150_ckpt.slurm": "clinc150",
    "run_ner_bc5cdr_ckpt.slurm": "ner_bc5cdr",
}

# The overnight-campaign scripts on the shared CSE account (2026-08-13). Same pipeline body and
# same benchmark keys, but gpu-l40s-cse caps wall-clock at 24h instead of 7 days, so these rely
# on the USR1 checkpoint-and-requeue contract to cross the boundary rather than treating it as a
# failure. They are a SEPARATE family from the `_l40s` scripts on purpose: a 24h preemptible-ish
# regime produces numbers that are not directly comparable with the weeklong dedicated runs, and
# keeping the filenames distinct keeps that visible.
CSE_BENCHMARK_SCRIPTS = {
    "run_xlam_bfcl_cse.slurm": "xlam_bfcl",
    "run_calendar_json_cse.slurm": "calendar_json",
    "run_ner_bc5cdr_cse.slurm": "ner_bc5cdr",
    "run_routerbench_cse.slurm": "routerbench",
    "run_proactive_listening_cse.slurm": "proactive_listening",
    # Added 2026-08-23. clinc150 was the only classification task that could run on one quota, so
    # it queued behind the intelligentsystems account while its peers could take whichever freed
    # first — which is the entire reason this family exists.
    "run_clinc150_cse.slurm": "clinc150",
    "run_sms_spam_cse.slurm": "sms_spam",
    # Added 2026-08-24. This task's eval is the suite's most expensive (765 long generations plus
    # ~2,295 judge calls per pass), so the 24h box matters more here than elsewhere — the judge
    # cache surviving a requeue is what makes it workable.
    "run_toolbench_cse.slurm": "toolbench",
}

# Measurement scripts that are NOT pipeline runs. They stand up a model and record numbers, without
# training anything or touching the agent loop, so they are deliberately excluded from the
# per-loader coverage invariant above — that invariant exists to guarantee every curated loader has
# exactly one launcher, and a probe is not a launcher. They are still held to the same allocation
# rules by `test_every_slurm_script_runs_on_a_sanctioned_l40s_allocation`, because an unaccounted
# script on an unsanctioned queue is exactly what that test is for.
PROBE_SCRIPTS = {
    # Teacher zero-/few-shot measurement incl. the Min et al. corrupt-label ablation (2026-08-17).
    "run_teacher_fewshot_probe.slurm",
    # Sub-billion candidates (gemma-3-270m, SmolLM2-360M/135M) measured zero-shot, fine-tuned and
    # Q4_K_M on ner_bc5cdr and xlam_bfcl (2026-08-23). Two accounts for the same reason the
    # benchmark launchers have two: whichever quota frees first. These are the only scripts in the
    # tree that ask for ONE GPU — they run no teacher, so the second would sit idle.
    "run_small_model_probe.slurm",
    "run_small_model_probe_cse.slurm",
    # Quantization-only pass over checkpoints an earlier probe already trained. Separate from the
    # probe itself because it is resumable by design: retraining six models to recover one column
    # would cost hours, and the checkpoints are on disk.
    "run_small_model_quant_pass.slurm",
    # Does the vLLM eval backend score the same as the in-process path (2026-08-27)? A probe rather
    # than a verification run: it runs no agent loop and no teacher, just the two inference backends
    # over identical rows, so it asks for ONE GPU — a like-for-like comparison of two engines wants
    # them on the same card anyway. See eval/student_server.py.
    "verify_eval_backend.slurm",
}

# Bounded verification runs. These DO run the full agent loop, so they are not probes, but they are
# not launchers either: their job is to prove an intervention adds rows before a full-length results
# run pays to find out it does not. Excluded from the per-loader coverage invariant because a task may
# have several of them or none, and pinned to a NON-requeued time box on purpose — a verification run
# that dies should stay dead so the failure gets examined rather than silently restarted.
VERIFICATION_SCRIPTS = {
    "run_xlam_bfcl_single_verify.slurm",
    "run_xlam_bfcl_single_verify_cse.slurm",
    # Added 2026-08-24 for the `toolbench` bring-up. A PAIR on purpose: same task, same bounds, same
    # overrides, one model changed, so anything that differs between them is model capability rather
    # than harness behaviour. SmolLM2-360M is the pool's best sub-billion structured-output model and
    # is the run expected to score; gemma-3-270m-it is measured near the floor on function calling
    # and is the negative control that says whether a floor score looks like a floor score or like a
    # broken harness.
    "run_toolbench_smollm2_verify.slurm",
    "run_toolbench_gemma_verify.slurm",
    # CSE twins, for the same reason the benchmark launchers have two accounts: whichever quota
    # frees first. The dedicated account is capped at 10 GPUs shared across the whole group and was
    # fully consumed by other users when these runs were launched, so a run that can only ever sit on
    # one account is a run that does not happen.
    "run_toolbench_smollm2_verify_cse.slurm",
    "run_toolbench_gemma_verify_cse.slurm",
}


def _sbatch_directives(source: str) -> dict[str, str]:
    directives = {}
    for line in source.splitlines():
        if not line.startswith("#SBATCH --"):
            continue
        directive = line[len("#SBATCH --"):]
        if "=" not in directive:
            directives[directive] = ""
            continue
        key, value = directive.split("=", 1)
        directives[key] = value
    return directives


def _shell_function(source: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^{re.escape(name)}\(\) \{{\n.*?^\}}\n",
        source,
    )
    assert match is not None
    return match.group(0)


def _run_gpu_profile(**overrides: str) -> subprocess.CompletedProcess[str]:
    source = (PIPELINE_DIR / "_l40s_task_body.sh").read_text(encoding="utf-8")
    functions = "\n".join(
        _shell_function(source, name)
        for name in (
            "_detect_allocated_gpu_count",
            "_validate_gpu_id",
            "_configure_gpu_profile",
        )
    )
    harness = (
        f"{functions}\n"
        "_configure_gpu_profile || exit $?\n"
        "printf '%s\\n' "
        '"$SLM_GPU_PROFILE" "$SLM_GPU_COUNT" "$SLM_SYNTH_GPU_IDS" '
        '"$SLM_SYNTH_TP" "$SLM_SYNTH_GPU_UTILIZATION" '
        '"$SLM_SYNTH_MAX_NUM_SEQS" "$SLM_SYNTH_CONCURRENCY" '
        '"$SLM_PIPELINE_GPU_ID"\n'
    )
    env = os.environ.copy()
    for name in (
        "SLURM_GPUS_ON_NODE",
        "CUDA_VISIBLE_DEVICES",
        "SLM_GPU_PROFILE",
        "SLM_GPU_COUNT",
        "SLM_SYNTH_GPU_IDS",
        "SLM_SYNTH_TP",
        "SLM_SYNTH_GPU_UTILIZATION",
        "SLM_SYNTH_MAX_NUM_SEQS",
        "SLM_SYNTH_CONCURRENCY",
        "SLM_PIPELINE_GPU_ID",
    ):
        env.pop(name, None)
    env.update(overrides)
    return subprocess.run(
        ["bash", "-c", harness],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def _assert_bounded_term_contract(source: str) -> None:
    assert "TERM_REQUESTED=0" in source
    assert 'SLM_TERM_GRACE_S="${SLM_TERM_GRACE_S:-120}"' in source
    assert "_wait_for_pipeline_with_grace" in source
    assert 'sleep "$grace_s"' in source
    assert 'kill -KILL "$PIPELINE_PID"' in source
    assert "TERM grace expired; force-killing pipeline" in source
    assert "trap '' TERM USR1" in source
    term_exit = (
        'if [ "$TERM_REQUESTED" -eq 1 ]; then\n'
        '    exit "$PIPELINE_STATUS"\n'
        "fi"
    )
    assert term_exit in source
    assert source.index(term_exit) < source.index(
        'if [ "$REQUEUE_REQUESTED" -eq 1 ]; then\n'
        "    if _durable_resume_ready"
    )


# --- Curated-benchmark l40s scripts (every registry task on 2x L40S) ------------

def test_l40s_benchmark_scripts_cover_exactly_the_registry():
    """Every task has exactly one weeklong launcher, and every launcher names a real task.

    Both halves matter: a task with no script cannot be run at all, and a typo'd
    SLM_BENCHMARK_TASK is rejected by `get_task` hours after submission, inside a job that has
    already allocated GPUs.
    """
    import tasks

    on_disk = {p.name for p in PIPELINE_DIR.glob("run_*_l40s.slurm")}
    assert on_disk == set(L40S_BENCHMARK_SCRIPTS)
    assert set(L40S_BENCHMARK_SCRIPTS.values()) == set(tasks.TASKS)


def test_l40s_benchmark_scripts_use_2xl40s_on_int_sys_with_requeue_contract():
    for filename, key in L40S_BENCHMARK_SCRIPTS.items():
        source = (PIPELINE_DIR / filename).read_text(encoding="utf-8")
        directives = _sbatch_directives(source)
        assert directives["partition"] == "gpu-l40s", filename
        assert directives["account"] == "gpu-l40s-intelligentsystems", filename
        assert directives["qos"] == "normal", filename
        assert directives["gres"] == "gpu:l40s:2", filename
        assert directives["mem"] == "160G", filename
        assert directives["time"] == "7-00:00:00", filename
        assert "#SBATCH --requeue" in source, filename
        assert "#SBATCH --signal=B:USR1@7200" in source, filename
        assert f"export SLM_BENCHMARK_TASK={key}" in source, filename
        assert "export SLM_CUDA_ISOLATION=1" in source, filename
        assert "source " in source and "_l40s_task_body.sh" in source, filename
        assert "gpu:a40" not in source.lower(), filename
        assert "gpu:a100" not in source.lower(), filename
        assert "partition=ckpt" not in source, filename


def test_a_task_is_configured_identically_on_all_accounts():
    """The `_l40s` and `_cse` launchers for one task must export the same task-level settings.

    They exist as a pair only because whichever quota frees first is the one that runs. That makes any
    difference between them a lottery on the science: the same task would be measured under different
    settings depending on which account had a free GPU, and nothing in the log would say so.

    It happened. `run_toolbench_l40s.slurm` carried `SLM_VERIFY_SYNTH=0` (the fix for the teacher
    rejecting rows its own exact verifier had just accepted) and `SLM_EVAL_SIZE_CAP=500` (which halves
    the suite's most expensive eval); its CSE twin carried neither, because the fixes were applied to
    the launcher that had most recently been run. A run submitted to CSE would have silently re-hit
    both problems.

    ACCOUNT-level differences are expected and excluded: the account name, the job name, the output
    path, and the wall clock, since gpu-l40s-cse enforces MaxWall=1-00:00:00.
    """
    account_specific = {"SLM_RUN_DIR"}
    offenders: list[str] = []
    variants = {**CSE_BENCHMARK_SCRIPTS, **CKPT_BENCHMARK_SCRIPTS}
    for cse_name, key in variants.items():
        l40s_name = next(
            (name for name, other in L40S_BENCHMARK_SCRIPTS.items() if other == key), None
        )
        if l40s_name is None:
            continue
        exports = {}
        for name in (l40s_name, cse_name):
            source = (PIPELINE_DIR / name).read_text(encoding="utf-8")
            exports[name] = {
                match.group(1): match.group(2).strip()
                for match in re.finditer(r"^export (SLM_\w+)=(.*)$", source, re.MULTILINE)
                if match.group(1) not in account_specific
            }
        for variable in sorted(set(exports[l40s_name]) | set(exports[cse_name])):
            l40s_value = exports[l40s_name].get(variable)
            cse_value = exports[cse_name].get(variable)
            if l40s_value != cse_value:
                offenders.append(
                    f"{key}: {variable} is {l40s_value!r} in {l40s_name} but {cse_value!r} in "
                    f"{cse_name}"
                )
    assert not offenders, (
        "the two launchers for a task must configure it identically, or the result depends on "
        "which account had a free GPU:\n  " + "\n  ".join(offenders)
    )


def test_cse_scripts_match_the_registry_and_use_the_24h_requeue_contract():
    import tasks

    # Verification launchers and probes are excluded by name, not by pattern: this invariant is
    # about the per-task CSE launchers being one-to-one with their registry key, and neither a
    # bounded verification run nor a measurement probe that happens to sit on the CSE account is
    # one of those.
    on_disk = (
        {p.name for p in PIPELINE_DIR.glob("run_*_cse.slurm")}
        - VERIFICATION_SCRIPTS
        - PROBE_SCRIPTS
    )
    assert on_disk == set(CSE_BENCHMARK_SCRIPTS)

    for filename, key in CSE_BENCHMARK_SCRIPTS.items():
        source = (PIPELINE_DIR / filename).read_text(encoding="utf-8")
        directives = _sbatch_directives(source)
        assert directives["account"] == "gpu-l40s-cse", filename
        assert directives["partition"] == "gpu-l40s", filename
        assert directives["gres"] == "gpu:l40s:2", filename
        # gpu-l40s-cse enforces MaxWall=1-00:00:00; anything longer is rejected at submit.
        assert directives["time"] == "24:00:00", filename
        # The checkpoint signal must fire with enough margin to publish the atomic
        # JSON + SQLite checkpoint before the scheduler kills the job.
        assert "#SBATCH --signal=B:USR1@7200" in source, filename
        assert "#SBATCH --requeue" in source, filename
        assert f"export SLM_BENCHMARK_TASK={key}" in source, filename
        assert "export SLM_CUDA_ISOLATION=1" in source, filename
        assert "_l40s_task_body.sh" in source, filename
        # Registry lockstep: a typo'd key here is rejected by run.py at startup, hours after
        # submission and after the vLLM server has already been paid for.
        assert key in tasks.TASKS, filename


def test_every_registry_task_can_actually_be_launched():
    """Each task carries its own loader and human label on its spec.

    These used to be two dicts — `NAMED_BENCHMARK_TASK_TYPES` and a parallel loader table — that
    had to be kept in step by hand, and a key present in one and absent from the other failed only
    at run time, inside a job that had already allocated GPUs. `TaskSpec` has no field with a
    default, so a task that reaches the registry has both.
    """
    import tasks

    for name, spec in tasks.TASKS.items():
        assert callable(spec.load), name
        assert spec.title, name


def test_l40s_benchmark_scripts_log_to_logs_slurm():
    for filename in L40S_BENCHMARK_SCRIPTS:
        directives = _sbatch_directives((PIPELINE_DIR / filename).read_text(encoding="utf-8"))
        assert "/logs/slurm/" in directives["output"], filename


# --- Shared 2-GPU task body: GPU-profile math + durability contract -------------

def test_shared_l40s_body_exercises_agent_discovery_before_local_fallback():
    source = (PIPELINE_DIR / "_l40s_task_body.sh").read_text(encoding="utf-8")

    assert "SLM_AGENT_FIRST_DATASET_DISCOVERY=1" in source
    assert "cost-observability" in source
    assert "local fallback" in source


def test_shared_l40s_body_sets_a_context_ceiling_and_leaves_the_reserve_to_the_task():
    """The output-token reserve is declared per TASK, so the job script must not pin one.

    The body used to export five per-channel `SLM_EVAL_MAX_NEW_TOKENS_*` variables. Nothing reads
    them any more — `eval.harness.eval_output_token_reserve` reads `TaskSpec.max_new_tokens` and
    validates it against that task's own context window — and a stale export that nothing reads is
    worse than none, because it reads as configuration.
    """
    source = (PIPELINE_DIR / "_l40s_task_body.sh").read_text(encoding="utf-8")

    assert 'SLM_MAX_SEQ_LENGTH="${SLM_MAX_SEQ_LENGTH:-4096}"' in source
    for stale in (
        "SLM_EVAL_MAX_NEW_TOKENS_CLASSIFICATION",
        "SLM_EVAL_MAX_NEW_TOKENS_NER",
        "SLM_EVAL_MAX_NEW_TOKENS_MATH",
        "SLM_EVAL_MAX_NEW_TOKENS_GENERATION",
        "SLM_EVAL_MAX_NEW_TOKENS_APPS",
    ):
        assert f'export {stale}=' not in source, stale
    # The one-off override survives as an override, and every task's own reserve fits its context.
    from eval.harness import eval_output_token_reserve
    from tasks import TASKS

    for name, spec in TASKS.items():
        assert 0 < eval_output_token_reserve(name) < spec.max_seq_length, name


def test_two_gpu_profile_isolates_synth_and_pipeline_devices():
    completed = _run_gpu_profile(SLURM_GPUS_ON_NODE="2")

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "auto-2gpu",
        "2",
        "0",
        "1",
        "0.90",
        "64",
        "48",
        "1",
    ]
    source = (PIPELINE_DIR / "_l40s_task_body.sh").read_text(encoding="utf-8")
    assert 'CUDA_VISIBLE_DEVICES="$SLM_SYNTH_GPU_IDS"' in source
    assert '--tensor-parallel-size "$SLM_SYNTH_TP"' in source
    assert 'export CUDA_VISIBLE_DEVICES="$SLM_PIPELINE_GPU_ID"' in source


def test_synth_server_runs_with_cuda_graphs_unless_explicitly_disabled():
    """--enforce-eager disables torch.compile AND CUDA graphs, which is the wrong trade for a
    ~3B-active MoE whose decode is kernel-launch bound. It stays reachable as a rollback."""
    source = (PIPELINE_DIR / "_l40s_task_body.sh").read_text(encoding="utf-8")

    assert 'if [ "${SLM_SYNTH_ENFORCE_EAGER:-0}" = "1" ]; then' in source
    assert 'SYNTH_EAGER_FLAG="--enforce-eager"' in source
    assert "$SYNTH_EAGER_FLAG \\" in source
    # The bare flag must never be hard-coded back into the serve line.
    serve_line = next(
        line for line in source.splitlines() if "--max-num-seqs" in line and "vllm" not in line
    )
    assert "--enforce-eager" not in serve_line


def test_gpu_profile_detects_visible_devices_and_allows_valid_overrides():
    detected = _run_gpu_profile(CUDA_VISIBLE_DEVICES="0,1,2")
    assert detected.returncode == 0, detected.stderr
    assert detected.stdout.splitlines()[:4] == [
        "auto-3gpu",
        "3",
        "0,1",
        "2",
    ]
    assert detected.stdout.splitlines()[-1] == "2"

    overridden = _run_gpu_profile(
        SLURM_GPUS_ON_NODE="4",
        SLM_GPU_COUNT="5",
        SLM_SYNTH_GPU_IDS="1,2",
        SLM_SYNTH_TP="2",
        SLM_SYNTH_GPU_UTILIZATION="0.70",
        SLM_SYNTH_MAX_NUM_SEQS="20",
        SLM_SYNTH_CONCURRENCY="10",
        SLM_PIPELINE_GPU_ID="4",
    )
    assert overridden.returncode == 0, overridden.stderr
    assert overridden.stdout.splitlines() == [
        "auto-5plus-gpu",
        "5",
        "1,2",
        "2",
        "0.70",
        "20",
        "10",
        "4",
    ]


def test_four_and_five_gpu_profiles_keep_compatibility_and_exclusive_layouts():
    four_gpu = _run_gpu_profile(SLURM_GPUS_ON_NODE="4")
    assert four_gpu.returncode == 0, four_gpu.stderr
    assert four_gpu.stdout.splitlines() == [
        "compat-4gpu-shared",
        "4",
        "0,1,2,3",
        "4",
        "0.45",
        "96",
        "64",
        "0",
    ]

    five_gpu = _run_gpu_profile(SLURM_GPUS_ON_NODE="5")
    assert five_gpu.returncode == 0, five_gpu.stderr
    assert five_gpu.stdout.splitlines() == [
        "auto-5plus-gpu",
        "5",
        "0,1,2,3",
        "4",
        "0.82",
        "96",
        "64",
        "4",
    ]


def test_gpu_profile_rejects_invalid_tp_ids_and_exclusive_overlap():
    invalid_tp = _run_gpu_profile(
        SLM_GPU_COUNT="2",
        SLM_SYNTH_TP="3",
    )
    assert invalid_tp.returncode == 2
    assert "SLM_SYNTH_TP must be one of 1, 2, or 4" in invalid_tp.stderr

    invalid_id = _run_gpu_profile(
        SLM_GPU_COUNT="2",
        SLM_PIPELINE_GPU_ID="2",
    )
    assert invalid_id.returncode == 2
    assert "outside the 2-GPU allocation" in invalid_id.stderr

    overlap = _run_gpu_profile(
        SLM_GPU_COUNT="3",
        SLM_PIPELINE_GPU_ID="1",
    )
    assert overlap.returncode == 2
    assert "must not overlap" in overlap.stderr


def test_shared_l40s_body_restarts_synth_and_resumes_stable_run():
    source = (PIPELINE_DIR / "_l40s_task_body.sh").read_text(encoding="utf-8")

    assert 'SLM_RUN_DIR="${SLM_RUN_DIR:-' in source
    assert "SLM_RESUME" in source
    assert 'SLM_MAX_WALLCLOCK_S="${SLM_MAX_WALLCLOCK_S:-0}"' in source
    assert "aggregate wall-clock auto-termination is disabled" in source
    assert "python tests/pipeline/run.py" in source
    assert "--resume" in source
    assert "vllm serve" in source
    assert "scontrol requeue" in source
    assert '[ -s "$SLM_RUN_DIR/run-manifest.json" ]' in source
    assert '[ -s "$SLM_RUN_DIR/checkpoint.json" ]' in source
    assert "durable_resume_available" in source
    _assert_bounded_term_contract(source)


def test_term_grace_waits_for_finalization_then_kills_hung_child(tmp_path):
    source = (PIPELINE_DIR / "_l40s_task_body.sh").read_text(encoding="utf-8")
    wait_function = _shell_function(
        source,
        "_wait_for_pipeline_with_grace",
    )
    graceful_child = tmp_path / "graceful-child.sh"
    graceful_ready = tmp_path / "graceful.ready"
    graceful_child.write_text(
        "#!/bin/bash\n"
        "trap 'sleep 0.2; exit 7' TERM\n"
        f'touch "{graceful_ready}"\n'
        "while true; do sleep 0.05; done\n",
        encoding="utf-8",
    )
    hung_child = tmp_path / "hung-child.sh"
    hung_ready = tmp_path / "hung.ready"
    hung_child.write_text(
        "#!/bin/bash\n"
        "trap '' TERM\n"
        f'touch "{hung_ready}"\n'
        "while true; do sleep 0.05; done\n",
        encoding="utf-8",
    )
    harness = tmp_path / "term-harness.sh"
    harness.write_text(
        "#!/bin/bash\n"
        "set -o pipefail\n"
        f"{wait_function}\n"
        f'bash "{graceful_child}" &\n'
        "PIPELINE_PID=$!\n"
        f'while [ ! -e "{graceful_ready}" ]; do sleep 0.01; done\n'
        'kill -TERM "$PIPELINE_PID"\n'
        "started=$(date +%s%N)\n"
        "set +e\n"
        "_wait_for_pipeline_with_grace 2\n"
        "status=$?\n"
        "set -e\n"
        "elapsed=$(( $(date +%s%N) - started ))\n"
        '[ "$status" -eq 7 ]\n'
        '[ "$elapsed" -ge 150000000 ]\n'
        f'bash "{hung_child}" &\n'
        "PIPELINE_PID=$!\n"
        f'while [ ! -e "{hung_ready}" ]; do sleep 0.01; done\n'
        'kill -TERM "$PIPELINE_PID"\n'
        "started=$(date +%s%N)\n"
        "set +e\n"
        "_wait_for_pipeline_with_grace 1\n"
        "status=$?\n"
        "set -e\n"
        "elapsed=$(( $(date +%s%N) - started ))\n"
        '[ "$status" -eq 137 ]\n'
        '[ "$elapsed" -ge 900000000 ]\n'
        '[ "$elapsed" -lt 3000000000 ]\n'
        'if kill -0 "$PIPELINE_PID" 2>/dev/null; then exit 9; fi\n',
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["bash", str(harness)],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, (
        completed.stdout + completed.stderr
    )


def test_all_task_scripts_and_shared_body_have_valid_bash_syntax():
    for filename in (*L40S_TASK_SCRIPTS, "_l40s_task_body.sh"):
        subprocess.run(["bash", "-n", str(PIPELINE_DIR / filename)], check=True)


def test_pipeline_docs_explain_continuous_requeue_guard_semantics():
    source = (PIPELINE_DIR.parents[1] / "docs" / "PIPELINE.md").read_text(
        encoding="utf-8"
    )

    assert "checkpoint/requeue-enabled weeklong jobs" in source
    assert "SLM_MAX_WALLCLOCK_S=0" in source
    assert "USR1" in source
    assert "aggregate" in source
    assert "continuous progress" in source
    assert "structural durability" in source
    assert "SLM_TERM_GRACE_S" in source
    assert "`TERM` takes precedence over requeue" in source


# --- Log directory split: pipeline runs vs GPU infrastructure -------------------
# logs/slurm/ holds pipeline run logs; logs/gpu_setup/ holds the vLLM synth server and
# GPU toolchain builds. Slurm does not create an --output directory and fails the job
# when it is missing, so the paths and the mkdir guards are both load-bearing.

REPO_ROOT = PIPELINE_DIR.parent.parent


def test_pipeline_task_scripts_log_to_logs_slurm():
    for name in L40S_TASK_SCRIPTS:
        directives = _sbatch_directives((PIPELINE_DIR / name).read_text())
        output = directives["output"]
        assert "/logs/slurm/" in output, f"{name} must log to logs/slurm/, got {output}"


def test_every_slurm_script_runs_on_a_sanctioned_l40s_allocation():
    """Two allocations, both 2x L40S on the same partition, and nothing else.

    This test used to require `gpu-l40s-intelligentsystems` everywhere. The reason was sound —
    the old gpu-l40s-cse and ckpt-g2 variants ran the same pipeline under a different wall clock
    and a preemptible regime, which made their numbers incomparable with the dedicated runs — but
    it is a reason to keep the two families SEPARATE and clearly labelled, not to forbid the
    second one. The 2026-08-13 campaign runs on CSE because the dedicated quota was saturated.

    A THIRD family was sanctioned on 2026-08-27: `ckpt-g2` under QOS `ckpt-gpu`, for the three cheap
    tasks in CKPT_BENCHMARK_SCRIPTS. This docstring used to name "a preemptible ckpt account" as the
    thing that must never come back, so the reversal is worth stating plainly rather than quietly
    editing. The original objection was correct but narrower than the wording: a preemptible run under
    a different wall clock produces numbers that must not be POOLED with dedicated ones. The remedy is
    labelling, not prohibition — the job name carries `-ckpt`, which is asserted below, so the two
    families stay distinguishable in `logs/slurm/`. And it is limited to tasks that train in minutes,
    so a preemption costs minutes; `toolbench` at ~93 minutes a training step stays off it.

    What still must not reappear is a configuration nobody has accounted for: a different GPU type, an
    unpinned `--gres`, or a script missing from the registries above.
    """
    # Scoped to the source tree. `REPO_ROOT.glob("**/*.slurm")` walked the whole repository, which
    # includes a 152 GB `logs/` directory on a network filesystem — 19 seconds that grows with every
    # run. The invariant is about SOURCE scripts, and every one lives under tests/pipeline or scripts/,
    # so searching those two directories tests the same thing and cannot be outgrown by run output.
    scripts = sorted(
        path
        for directory in ("tests/pipeline", "scripts")
        for path in (REPO_ROOT / directory).glob("**/*.slurm")
    )
    accounted = (set(L40S_TASK_SCRIPTS) | set(CSE_BENCHMARK_SCRIPTS)
                 | set(CKPT_BENCHMARK_SCRIPTS) | PROBE_SCRIPTS | VERIFICATION_SCRIPTS)
    assert {p.name for p in scripts} == accounted
    for path in scripts:
        source = path.read_text(encoding="utf-8")
        directives = _sbatch_directives(source)
        if path.name in CKPT_BENCHMARK_SCRIPTS:
            # The preemptible family, sanctioned 2026-08-27 for the three cheap tasks named in
            # CKPT_BENCHMARK_SCRIPTS. Its rules are checked here and nowhere else, so the ONE thing
            # that made ckpt unacceptable before cannot come back: an unlabelled ckpt run whose
            # numbers get pooled with dedicated ones. The `-ckpt` job name is what keeps them
            # distinguishable in `logs/slurm/`, and it is asserted.
            assert directives["account"] == "ckpt-intelligentsystems", path.name
            assert directives["partition"] == "ckpt-g2", path.name
            assert directives["gres"] == "gpu:l40s:2", (
                f"{path.name}: ckpt-g2 also offers l40 and h200, and the device database is "
                "calibrated on L40S — an unpinned request silently changes the hardware"
            )
            assert "-ckpt" in directives["job-name"], (
                f"{path.name}: the job name must carry -ckpt so a preemptible run is never mistaken "
                "for a dedicated one in the log directory"
            )
            assert "--requeue" in source, (
                f"{path.name}: PreemptMode=REQUEUE only returns the job to the queue if the job "
                "allows requeue; without this a preemption is just a death"
            )
            assert directives["time"] < "24:00:00", (
                f"{path.name} is preemptible and must be time-boxed"
            )
            continue
        assert directives["account"] in {
            "gpu-l40s-intelligentsystems", "gpu-l40s-cse"}, path.name
        assert directives["partition"] == "gpu-l40s", path.name
        assert "gpu:a40" not in source.lower(), path.name
        assert "gpu:a100" not in source.lower(), path.name
        assert "partition=ckpt" not in source, path.name
        if path.name in PROBE_SCRIPTS:
            # A probe measures and exits, so it must NOT hold a pipeline-sized reservation — asking
            # for days, or for GPUs it will not use, is how a shared queue gets starved. A pipeline
            # run needs two L40S because the vLLM teacher occupies one; a probe that runs no teacher
            # needs one, and pinning it to two would reserve an idle GPU for hours. Both bounds are
            # asserted rather than left open so "probe" cannot become a label a results run wears to
            # escape the allocation rules.
            assert directives["gres"] in {"gpu:l40s:1", "gpu:l40s:2"}, path.name
            assert directives["time"] < "24:00:00", (
                f"{path.name} is a probe and must request less than a full pipeline day"
            )
            continue
        assert directives["gres"] == "gpu:l40s:2", path.name
        if path.name in VERIFICATION_SCRIPTS:
            # A verification run is time-boxed and NOT requeued, on purpose: it exists to answer one
            # question cheaply, and a requeue would silently restart the very failure it was launched
            # to surface. Anything asking for a pipeline-sized reservation is a results run wearing a
            # verification label.
            assert directives["time"] < "24:00:00", (
                f"{path.name} is a verification run and must be time-boxed"
            )
            assert "--requeue" not in source, (
                f"{path.name} must not requeue: a verification run that dies should stay dead so "
                "the failure gets examined"
            )
            continue
        # The CSE account is the only place a 24h cap is legitimate; a weeklong request there is
        # rejected at submit time, and a 24h request on the dedicated account is a mistake.
        expected_wall = "24:00:00" if directives["account"].endswith("-cse") else "7-00:00:00"
        assert directives["time"] == expected_wall, path.name


def _body_default_max_seq_length() -> int:
    """The context the shared body falls back to when a launcher does not set one."""
    source = (PIPELINE_DIR / "_l40s_task_body.sh").read_text(encoding="utf-8")
    match = re.search(
        r'export SLM_MAX_SEQ_LENGTH="\$\{SLM_MAX_SEQ_LENGTH:-(\d+)\}"', source
    )
    assert match is not None, (
        "_l40s_task_body.sh no longer exports a defaulted SLM_MAX_SEQ_LENGTH; the invariant below "
        "depends on knowing what a launcher inherits when it stays silent"
    )
    return int(match.group(1))


def test_no_launcher_silently_clamps_its_task_context_below_the_spec():
    """A task whose spec needs more context than the body's default MUST say so in its launcher.

    THE BUG THIS EXISTS FOR (2026-08-24). `_l40s_task_body.sh` defaults SLM_MAX_SEQ_LENGTH to 4096,
    and `training.slm_helpers.task_max_seq_length` gives that env var precedence OVER
    `TaskSpec.max_seq_length`. So a task that declares 8192 — with measurements in its spec
    justifying it — silently ran at 4096. `toolbench` job 38812203 died twice from it: once in eval
    ("4020 tokens, exceeding input budget 2560 ... inside configured max sequence length 4096") and
    once in training ("row 43 contains 4110 tokens, exceeding configured context 4096").

    It is exactly the failure mode the task registry was built to remove — a global default
    overriding a per-task decision — except that this one lives in shell rather than Python, which is
    why no existing test caught it. The spec is the authority on how much context a task needs; a
    launcher may raise the ceiling but must never leave it below what the spec declares.
    """
    import tasks

    default = _body_default_max_seq_length()
    launchers: dict[str, list[str]] = {}
    for filename, key in {**L40S_BENCHMARK_SCRIPTS, **CSE_BENCHMARK_SCRIPTS}.items():
        launchers.setdefault(key, []).append(filename)
    # Verification launchers pin a task too, and inherit the same default.
    for filename in VERIFICATION_SCRIPTS:
        source = (PIPELINE_DIR / filename).read_text(encoding="utf-8")
        match = re.search(r"export SLM_BENCHMARK_TASK=(\S+)", source)
        if match:
            launchers.setdefault(match.group(1), []).append(filename)

    offenders = []
    for key, filenames in sorted(launchers.items()):
        needed = tasks.get_task(key).max_seq_length
        if needed <= default:
            continue
        for filename in sorted(filenames):
            source = (PIPELINE_DIR / filename).read_text(encoding="utf-8")
            match = re.search(r"^export SLM_MAX_SEQ_LENGTH=(\d+)", source, re.MULTILINE)
            declared = int(match.group(1)) if match else default
            if declared < needed:
                offenders.append(
                    f"{filename}: task {key} declares max_seq_length={needed} but the launcher "
                    f"provides {declared} (body default {default})"
                )
    assert offenders == [], (
        "a launcher would run its task at less context than the task's spec requires, which "
        "truncates prompts or aborts training:\n  " + "\n  ".join(offenders)
    )


def test_an_explicit_context_override_is_set_before_the_body_is_sourced():
    """`${SLM_MAX_SEQ_LENGTH:-4096}` only honours an override that already exists.

    Exporting it AFTER the `source` line would parse fine, run fine, and have no effect — the body
    would have already frozen the default. Ordering is the whole mechanism, so it is asserted.
    """
    for path in sorted(PIPELINE_DIR.glob("run_*.slurm")):
        source = path.read_text(encoding="utf-8")
        if "SLM_MAX_SEQ_LENGTH" not in source or "_l40s_task_body.sh" not in source:
            continue
        assert source.index("export SLM_MAX_SEQ_LENGTH") < source.index(
            "source /mmfs1"
        ), f"{path.name} sets SLM_MAX_SEQ_LENGTH after sourcing the body, where it is a no-op"


def test_task_body_writes_synth_log_to_gpu_setup_and_creates_the_dir():
    source = (PIPELINE_DIR / "_l40s_task_body.sh").read_text()
    assert 'SYNTH_LOG="$PROJ/logs/gpu_setup/' in source
    assert 'mkdir -p "$(dirname "$SYNTH_LOG")"' in source, (
        "the redirect fails outright if logs/gpu_setup/ does not exist"
    )


def test_setup_gpu_env_creates_both_log_dirs():
    source = (REPO_ROOT / "scripts" / "setup_gpu_env.sh").read_text()
    assert "mkdir -p logs/slurm logs/gpu_setup" in source


def test_task_body_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(PIPELINE_DIR / "_l40s_task_body.sh")], check=True)
