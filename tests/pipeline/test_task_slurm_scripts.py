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
}

L40S_TASK_SCRIPTS = tuple(L40S_BENCHMARK_SCRIPTS)

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


def test_cse_scripts_match_the_registry_and_use_the_24h_requeue_contract():
    import tasks

    on_disk = {p.name for p in PIPELINE_DIR.glob("run_*_cse.slurm")}
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

    What still must not reappear is a third configuration: a different GPU type, a different
    partition, a preemptible ckpt account, or a script nobody has accounted for.
    """
    scripts = sorted(REPO_ROOT.glob("**/*.slurm"))
    accounted = set(L40S_TASK_SCRIPTS) | set(CSE_BENCHMARK_SCRIPTS) | PROBE_SCRIPTS
    assert {p.name for p in scripts} == accounted
    for path in scripts:
        source = path.read_text(encoding="utf-8")
        directives = _sbatch_directives(source)
        assert directives["account"] in {
            "gpu-l40s-intelligentsystems", "gpu-l40s-cse"}, path.name
        assert directives["partition"] == "gpu-l40s", path.name
        assert directives["gres"] == "gpu:l40s:2", path.name
        assert "gpu:a40" not in source.lower(), path.name
        assert "gpu:a100" not in source.lower(), path.name
        assert "partition=ckpt" not in source, path.name
        if path.name in PROBE_SCRIPTS:
            # A probe measures and exits, so it must NOT hold a pipeline-sized reservation — asking
            # for days to do an hour of work is how a shared queue gets starved. Only the allocation
            # shape above is shared with pipeline runs; the wall clock deliberately is not.
            assert directives["time"] < "24:00:00", (
                f"{path.name} is a probe and must request less than a full pipeline day"
            )
            continue
        # The CSE account is the only place a 24h cap is legitimate; a weeklong request there is
        # rejected at submit time, and a 24h request on the dedicated account is a mistake.
        expected_wall = "24:00:00" if directives["account"].endswith("-cse") else "7-00:00:00"
        assert directives["time"] == expected_wall, path.name


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
