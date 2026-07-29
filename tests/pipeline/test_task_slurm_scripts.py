import ast
import os
import re
import subprocess
from pathlib import Path


PIPELINE_DIR = Path(__file__).parent
TASK_SCRIPTS = (
    "run_code_l40s.slurm",
    "run_math_l40s.slurm",
    "run_ner_l40s.slurm",
    "run_generation_l40s.slurm",
)


def _embedded_python(source: str) -> str:
    match = re.search(r"<<'PY'\n(?P<body>.*?)\nPY(?:\n|$)", source, re.DOTALL)
    assert match is not None
    body = match.group("body")
    ast.parse(body)
    return body


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


def test_code_l40s_script_targets_apps_introductory():
    source = (PIPELINE_DIR / "run_code_l40s.slurm").read_text(encoding="utf-8")
    task_line = next(line for line in source.splitlines() if line.startswith("export TASK="))

    assert "APPS introductory" in task_line
    assert "MBPP" not in task_line
    assert "HumanEval" not in source
    assert "checksum-verified offline APPS introductory bundle" in source
    assert "call-based and stdin/stdout" in source
    assert "SLM_MAX_SEQ_LENGTH=4096" in source
    assert "SLM_APPS_MAX_CASES" not in source
    assert "Samsung Galaxy S24 Ultra" in source
    assert "#SBATCH --gres=gpu:l40s:4" in source
    assert "Sonnet-1M" in source
    assert "Qwen3.6-primary CoT" in source
    assert "SLM_QUANT_EVAL=1" in source
    assert "SLM_CUDA_ISOLATION=1" in source


def test_math_l40s_comments_match_current_cot_and_token_routing():
    source = (PIPELINE_DIR / "run_math_l40s.slurm").read_text(encoding="utf-8")

    assert "Qwen3.6-primary CoT" in source
    assert "DeepSeek then OpenAI" in source
    assert "512-token" in source
    assert "Sonnet teacher" not in source
    assert "256-token" not in source


def test_shared_l40s_body_exercises_agent_discovery_before_local_fallback():
    source = (PIPELINE_DIR / "_l40s_task_body.sh").read_text(encoding="utf-8")

    assert "SLM_AGENT_FIRST_DATASET_DISCOVERY=1" in source
    assert "cost-observability" in source
    assert "local fallback" in source


def test_shared_l40s_body_sets_nonzero_context_and_task_output_reserves():
    source = (PIPELINE_DIR / "_l40s_task_body.sh").read_text(encoding="utf-8")

    assert 'SLM_MAX_SEQ_LENGTH="${SLM_MAX_SEQ_LENGTH:-4096}"' in source
    expected = {
        "SLM_EVAL_MAX_NEW_TOKENS_CLASSIFICATION": 50,
        "SLM_EVAL_MAX_NEW_TOKENS_NER": 512,
        "SLM_EVAL_MAX_NEW_TOKENS_MATH": 512,
        "SLM_EVAL_MAX_NEW_TOKENS_GENERATION": 512,
        "SLM_EVAL_MAX_NEW_TOKENS_APPS": 1024,
    }
    for name, reserve in expected.items():
        assert f'{name}="${{{name}:-{reserve}}}"' in source
        assert 0 < reserve < 4096
    assert (
        'SLM_APPS_PROBLEM_TIMEOUT_S="${SLM_APPS_PROBLEM_TIMEOUT_S:-6}"'
        in source
    )


def test_two_gpu_profile_isolates_synth_and_pipeline_devices():
    completed = _run_gpu_profile(SLURM_GPUS_ON_NODE="2")

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "auto-2gpu",
        "2",
        "0",
        "1",
        "0.82",
        "16",
        "8",
        "1",
    ]
    source = (PIPELINE_DIR / "_l40s_task_body.sh").read_text(encoding="utf-8")
    assert 'CUDA_VISIBLE_DEVICES="$SLM_SYNTH_GPU_IDS"' in source
    assert '--tensor-parallel-size "$SLM_SYNTH_TP"' in source
    assert 'export CUDA_VISIBLE_DEVICES="$SLM_PIPELINE_GPU_ID"' in source


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


def test_four_task_scripts_have_weeklong_requeue_contract():
    for filename in TASK_SCRIPTS:
        source = (PIPELINE_DIR / filename).read_text(encoding="utf-8")
        directives = _sbatch_directives(source)
        assert directives["gres"] == "gpu:l40s:4"
        assert directives["mem"] == "224G"
        assert directives["time"] == "7-00:00:00"
        assert "#SBATCH --requeue" in source
        assert "#SBATCH --signal=B:USR1@7200" in source
        assert "export SLM_CUDA_ISOLATION=1" in source
        assert "gpu:a40" not in source.lower()
        assert "gpu:a100" not in source.lower()


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
    for filename in (*TASK_SCRIPTS, "_l40s_task_body.sh"):
        subprocess.run(["bash", "-n", str(PIPELINE_DIR / filename)], check=True)


def test_emotion_script_has_same_weeklong_checkpoint_signal_contract():
    path = PIPELINE_DIR / "run_emotion_orch_full_l40s.slurm"
    source = path.read_text(encoding="utf-8")
    directives = _sbatch_directives(source)

    assert directives["time"] == "7-00:00:00"
    assert "#SBATCH --requeue" in source
    assert "#SBATCH --signal=B:USR1@7200" in source
    assert "SLM_MAX_WALLCLOCK_S=0" in source
    assert "aggregate wall-clock auto-termination" in source
    assert "SLM_MAX_SEQ_LENGTH=4096" in source
    assert "SLM_EVAL_MAX_NEW_TOKENS_CLASSIFICATION=50" in source
    assert "SLM_CUDA_ISOLATION=1" in source
    assert "SLM_RUN_DIR" in source
    assert "--resume" in source
    assert "scontrol requeue" in source
    assert "durable_resume_available" in source
    _assert_bounded_term_contract(source)
    subprocess.run(["bash", "-n", str(path)], check=True)


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


def test_manual_qwen35_readiness_script_is_short_isolated_and_unsubmitted():
    source = (
        PIPELINE_DIR / "manual_qwen35_readiness_l40s.slurm"
    ).read_text(encoding="utf-8")

    assert "MANUAL / UNSUBMITTED" in source
    assert "PASS CRITERIA" in source
    assert "TIMING CRITERIA" in source
    assert "VRAM CRITERIA" in source
    directives = _sbatch_directives(source)
    assert directives["gres"] == "gpu:l40s:1"
    assert directives["time"] == "00:45:00"
    assert "SLM_CUDA_ISOLATION=1" in source
    assert "Qwen/Qwen3.5-0.8B" in source
    assert "SLM_QUANT_EVAL=1" in source
    assert "SLM_QUANT_EVAL=0" not in source
    assert "nr_epochs=1" in source
    assert "timeout --signal=TERM 35m" in source
    assert "convert_hf_to_gguf" in source
    assert "llama-quantize" in source
    assert "llama-cpp-python" in source
    assert "nvidia-smi" in source
    assert "MAX_PEAK_USED_MIB=" in source
    assert "MAX_POST_WORKER_DELTA_MIB=" in source
    assert "peak_gpu_used_mib" in source
    assert "post_worker_delta_mib" in source

    python = _embedded_python(source)
    tree = ast.parse(python)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    call_names = [node.func.id for node in calls]
    assert "train" in call_names
    assert "infer" in call_names
    assert "infer_batch" in call_names
    assert "_build_gguf_for_eval" in call_names
    batch_call = next(node for node in calls if node.func.id == "infer_batch")
    batch_prompts = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "batch_prompts"
            for target in node.targets
        )
    )
    assert isinstance(batch_prompts, ast.List)
    prompt_values = [
        element.value
        for element in batch_prompts.elts
        if isinstance(element, ast.Constant)
        and isinstance(element.value, str)
    ]
    assert len(prompt_values) > 1
    assert len({len(prompt) for prompt in prompt_values}) > 1
    assert isinstance(batch_call.args[0], ast.Name)
    assert batch_call.args[0].id == "batch_prompts"
    deployment_evals = [
        node
        for node in calls
        if node.func.id == "run_eval"
        and any(
            keyword.arg == "gguf_path"
            and isinstance(keyword.value, ast.Name)
            for keyword in node.keywords
        )
    ]
    assert len(deployment_evals) == 1
    quant_keyword = next(
        keyword
        for keyword in deployment_evals[0].keywords
        if keyword.arg == "quant"
    )
    assert isinstance(quant_keyword.value, ast.Constant)
    assert quant_keyword.value.value == "Q4_K_M"


def test_manual_qwen36_tp4_colocation_smoke_is_bounded_local_and_unsubmitted():
    path = PIPELINE_DIR / "manual_qwen36_tp4_colocation_l40s.slurm"
    source = path.read_text(encoding="utf-8")

    assert "MANUAL / UNSUBMITTED" in source
    assert "PASS CRITERIA" in source
    assert "No cloud keys or calls" in source
    directives = _sbatch_directives(source)
    assert directives["gres"] == "gpu:l40s:4"
    assert directives["time"] == "01:30:00"
    assert "Qwen/Qwen3.6-35B-A3B" in source
    assert "Qwen/Qwen3.5-4B" in source
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in source
    assert "--tensor-parallel-size 4" in source
    assert "--quantization fp8" in source
    utilization = re.search(
        r"--gpu-memory-utilization\s+([0-9.]+)",
        source,
    )
    assert utilization is not None
    assert float(utilization.group(1)) <= 0.50
    assert "http://127.0.0.1:" in source
    assert "LocalJudgeClient" in source
    assert ".preflight()" in source
    assert ".score_many(" in source
    assert "SLM_CUDA_ISOLATION=1" in source
    assert "export CUDA_VISIBLE_DEVICES=0" in source
    assert "SLM_MAX_SEQ_LENGTH=4096" in source
    assert "SLM_EVAL_BATCH_SIZE=4" in source
    assert "generation eval worker keeps its model resident" in source
    assert "APPS 4096/1024 batch pressure" in source
    assert "timeout --signal=TERM 60m" in source
    assert "nvidia-smi" in source
    assert "gpu-memory-snapshots.csv" in source
    assert "OutOfMemoryError" in source
    assert "st_size > 0" in source
    assert "SLM_JUDGE_ALLOW_REMOTE" not in source

    python = _embedded_python(source)
    tree = ast.parse(python)
    call_names = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "train" in call_names
    assert "infer_batch" in call_names
    assert "_build_gguf_for_eval" in call_names
    assert "run_eval" in call_names

    infer_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "infer_batch"
    ]
    apps_pressure = [
        node for node in infer_calls
        if any(
            keyword.arg == "max_new_tokens"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == 1024
            for keyword in node.keywords
        )
        and any(
            keyword.arg == "task_type"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "code_generation"
            for keyword in node.keywords
        )
    ]
    assert len(apps_pressure) == 1

    generation_eval_sets = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "EvalSet"
        and any(
            keyword.arg == "task_type"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "generation"
            for keyword in node.value.keywords
        )
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    generation_eval_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_eval"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in generation_eval_sets
    ]
    assert len(generation_eval_calls) == 1


def test_manual_qwen36_cot_smoke_is_colocated_single_call_and_unsubmitted():
    source = (
        PIPELINE_DIR / "manual_qwen36_cot_smoke_l40s.slurm"
    ).read_text(encoding="utf-8")

    assert "MANUAL / UNSUBMITTED" in source
    assert "PASS CRITERIA" in source
    directives = _sbatch_directives(source)
    assert directives["gres"] == "gpu:l40s:1"
    assert directives["time"] == "00:45:00"
    assert "Qwen/Qwen3.6-35B-A3B" in source
    assert "vllm serve" in source
    assert "--max-num-seqs 1" in source
    assert "http://127.0.0.1:" in source
    assert "MAX_GENERATION_CALLS=1" in source
    assert "ANTHROPIC_API_KEY=manual-readiness-no-call" in source
    assert "EXA_API_KEY=manual-readiness-no-call" in source
    assert "timeout --signal=TERM 240s" in source
    assert "request_timeout=210.0" in source
    assert "seq 1 120" not in source
    assert "does not create cost.json or timings.json" in source
    assert "SLURM output" in source
    assert "vLLM server log" in source

    timeout_match = re.search(r"^READINESS_TIMEOUT_S=(\d+)$", source, re.MULTILINE)
    assert timeout_match is not None
    assert int(timeout_match.group(1)) == 900
    assert re.search(
        r"CUDA_MOD=\$\((?s:.*?)\)\s*\|\| true",
        source,
    )
    assert "READINESS_DEADLINE=$((SECONDS + READINESS_TIMEOUT_S))" in source
    assert '--max-time "$CURL_TIMEOUT_S"' in source
    assert "--noproxy '*'" in source
    assert 'sleep "$SLEEP_S"' in source

    python = _embedded_python(source)
    tree = ast.parse(python)
    generation_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "generate"
    ]
    assert len(generation_calls) == 1
    max_tokens = next(
        keyword.value.value
        for keyword in generation_calls[0].keywords
        if keyword.arg == "max_tokens"
        and isinstance(keyword.value, ast.Constant)
    )
    assert max_tokens == 96


def test_manual_readiness_scripts_have_valid_bash_syntax():
    for path in (
        PIPELINE_DIR / "manual_qwen35_readiness_l40s.slurm",
        PIPELINE_DIR / "manual_qwen36_cot_smoke_l40s.slurm",
        PIPELINE_DIR / "manual_qwen36_tp4_colocation_l40s.slurm",
    ):
        subprocess.run(["bash", "-n", str(path)], check=True)


# --- Log directory split: pipeline runs vs GPU infrastructure -------------------
# logs/slurm/ holds pipeline run logs; logs/gpu_setup/ holds the vLLM synth server and
# GPU toolchain builds. Slurm does not create an --output directory and fails the job
# when it is missing, so the paths and the mkdir guards are both load-bearing.

REPO_ROOT = PIPELINE_DIR.parent.parent


def test_pipeline_task_scripts_log_to_logs_slurm():
    for name in TASK_SCRIPTS + ("run_emotion_orch_full_l40s.slurm",):
        directives = _sbatch_directives((PIPELINE_DIR / name).read_text())
        output = directives["output"]
        assert "/logs/slurm/" in output, f"{name} must log to logs/slurm/, got {output}"


def test_synth_server_logs_to_gpu_setup():
    directives = _sbatch_directives((REPO_ROOT / "scripts" / "serve_synth.slurm").read_text())
    assert "/logs/gpu_setup/" in directives["output"]


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
