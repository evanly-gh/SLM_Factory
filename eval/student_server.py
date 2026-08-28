"""Evaluate the student through a short-lived vLLM server instead of padded HuggingFace batches.

WHY THIS EXISTS
    Toolbench eval costs 41 minutes a turn, 5.5 hours a run. The reason is not the model, it is the
    scheduler: 760 rows x 1,536 output tokens, generated in padded batches of 4-16, is ~74,000
    sequential forward passes in which most rows in a batch are already finished and padding. A
    continuous-batching engine retires a finished sequence and admits the next one instead of waiting
    for the slowest row in the batch, which is the entire difference on a workload whose output
    lengths vary by 10x.

WHAT THE 08-26 POSTMORTEM PROPOSED, AND WHY THIS IS NOT THAT
    That note said "route eval through the idle vLLM server". It cannot be done as written, for two
    independent reasons, and both are worth recording so the idea is not re-proposed:

      * The teacher's server is serving Qwen3.6-35B-A3B. A LoRA adapter only applies to the base model
        it was trained against, so a SmolLM2 or Gemma adapter cannot be loaded into it at any price.
      * That GPU is at 0.90 memory utilization by design (the 2-GPU profile gives the teacher its own
        card and spends the headroom on KV cache), so there is no room to stand a second engine beside
        it either.

    What IS true is the useful half: a GPU is idle. So this module starts its OWN vLLM server for the
    student, on a GPU of the caller's choosing, uses it, and shuts it down. By default that is the
    pipeline GPU, which is genuinely free at eval time whenever `SLM_CUDA_ISOLATION=1` — training ran
    in a disposable worker that has already exited.

PROMPT AND SAMPLING PARITY IS THE WHOLE CORRECTNESS ARGUMENT
    A faster eval that scores differently is not a faster eval, it is a different experiment. So this
    path does not use `/v1/chat/completions`: it renders each prompt with `_render_inference_prompt`,
    the exact function the in-process path calls, and sends the resulting string to `/v1/completions`
    as raw text. Sampling is greedy (`temperature=0`, matching `do_sample=False`) and decoding skips
    special tokens, matching `tokenizer.decode(..., skip_special_tokens=True)`. The prompt bytes and
    the decode rule are therefore identical by construction rather than by inspection.

    That argument covers the BF16/adapter path only. It deliberately does NOT cover `SLM_QUANT_EVAL=1`,
    where the in-process path scores a Q4_K_M GGUF through llama.cpp's own chat template — a different
    artifact under a different template, which is a measurement change and not this module's business.
    `scripts/compare_eval_backends.py` exists to measure agreement before anyone trusts a swap.

OFF BY DEFAULT
    `SLM_EVAL_BACKEND` defaults to `auto`, which means "whatever the run does today". Nothing here
    engages until it is set to `vllm`.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# vLLM is installed in its own virtualenv, not the pipeline's: the two have incompatible torch
# pins, which is why the launcher runs `.venv_vllm/bin/vllm serve` for the teacher. That is the
# reason this module talks HTTP to a subprocess rather than importing vllm.
VLLM_PYTHON = PROJECT_ROOT / ".venv_vllm" / "bin" / "python"

# The name the student is served under. Fixed rather than derived, so a request cannot silently be
# answered by the teacher if the two endpoints are ever confused.
SERVED_MODEL_NAME = "slm-student"


class StudentServerUnavailable(RuntimeError):
    """The student could not be served, so the caller must fall back to in-process inference.

    Raised rather than returned so a misconfiguration is loud, and caught by the harness so a
    misconfiguration is not fatal: a slower eval is a far better outcome than a dead run.
    """


def backend() -> str:
    """Which eval backend to use: `auto` (today's in-process path) or `vllm`."""
    return os.environ.get("SLM_EVAL_BACKEND", "auto").strip().lower()


def _port() -> int:
    """A port distinct from the teacher's. Derived from the job id so two runs sharing a node do
    not collide, which they otherwise would on any fixed default."""
    if explicit := os.environ.get("SLM_STUDENT_PORT"):
        return int(explicit)
    job = os.environ.get("SLURM_JOB_ID", "0")
    digits = "".join(c for c in job if c.isdigit()) or "0"
    return 9000 + (int(digits) % 900)


def _gpu_ids() -> str:
    """Which GPU to serve the student on.

    Defaults to the pipeline's own GPU. That is the counter-intuitive choice — the postmortem's idea
    was to use the teacher's idle card — but it is the correct one: under CUDA isolation the pipeline
    GPU is empty at eval time, while the teacher's card is committed to 0.90 utilization and cannot
    host a second engine. `SLM_STUDENT_GPU_IDS` overrides for anyone who has arranged the headroom.
    """
    if explicit := os.environ.get("SLM_STUDENT_GPU_IDS"):
        return explicit
    # CUDA_VISIBLE_DEVICES is already narrowed to the pipeline's card by the launcher, so within
    # this process the pipeline GPU is always index 0.
    return "0"


def render_prompts(prompts: list[str], weights_ref: str, base_model: str) -> list[str]:
    """Apply the same chat template the in-process path applies, on the CPU.

    Shares `_render_inference_prompt` with the in-process path rather than reimplementing it: the
    template is the one thing that must not drift, and a copy of it here would drift the first time
    someone changed the trainer's.
    """
    from transformers import AutoTokenizer

    from training.slm_helpers import _render_inference_prompt

    # The adapter directory first: training saves the tokenizer alongside the adapter, and if it
    # added tokens or changed the template then THAT is the tokenizer the weights were trained with.
    tokenizer = None
    for source in (weights_ref, base_model):
        if not source:
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(source)
            break
        except Exception as error:  # noqa: BLE001 - try the next source
            logger.debug("tokenizer load failed for %s: %s", source, error)
    if tokenizer is None:
        raise StudentServerUnavailable(
            f"no tokenizer could be loaded from either {weights_ref!r} or {base_model!r}, so the "
            "prompts cannot be rendered the way the in-process path renders them"
        )
    return [_render_inference_prompt(tokenizer, prompt, base_model) for prompt in prompts]


def _wait_for_ready(base_url: str, process: subprocess.Popen, timeout_s: float, log) -> None:
    """Poll /v1/models until the server answers, or the process dies, or the clock runs out.

    Checks the CHILD as well as the socket. A vLLM engine that fails to allocate exits within
    seconds, and polling only the port turns that into a full-timeout wait with no diagnosis.
    """
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if (code := process.poll()) is not None:
            raise StudentServerUnavailable(
                f"the student vLLM server exited with code {code} before becoming ready. The most "
                f"likely cause is GPU memory: it was asked for "
                f"{os.environ.get('SLM_STUDENT_GPU_UTILIZATION', '0.85')} of GPU "
                f"{_gpu_ids()}. See the server log above."
            )
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(3)
    raise StudentServerUnavailable(
        f"the student vLLM server did not become ready within {timeout_s:.0f}s "
        f"(last probe: {last_error or 'no response'})"
    )


class StudentServer:
    """A vLLM server for the student, alive only for the duration of one eval.

    Used as a context manager so the engine cannot outlive the eval and hold a GPU: an orphaned
    engine would make the NEXT training step OOM, turning a speedup into a run failure.
    """

    def __init__(
        self,
        weights_ref: str,
        base_model: str,
        *,
        max_model_len: int,
        max_lora_rank: int = 64,
        log=print,
    ) -> None:
        self.weights_ref = weights_ref
        self.base_model = base_model
        self.max_model_len = max_model_len
        self.max_lora_rank = max_lora_rank
        self.log = log
        self.base_url = f"http://127.0.0.1:{_port()}/v1"
        self._process: subprocess.Popen | None = None
        self._log_file = None

    def _command(self) -> list[str]:
        utilization = os.environ.get("SLM_STUDENT_GPU_UTILIZATION", "0.85")
        command = [
            str(VLLM_PYTHON), "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.base_model,
            "--served-model-name", SERVED_MODEL_NAME,
            "--host", "127.0.0.1",
            "--port", str(_port()),
            "--gpu-memory-utilization", utilization,
            "--max-model-len", str(self.max_model_len),
            "--disable-log-requests",
        ]
        # The student is a LoRA adapter over the base model, so serve it as one rather than merging:
        # a merge writes a full copy of the weights to disk and costs minutes per iteration.
        if self.weights_ref and self.weights_ref != self.base_model:
            command += [
                "--enable-lora",
                "--lora-modules", f"{SERVED_MODEL_NAME}={self.weights_ref}",
                "--max-lora-rank", str(self.max_lora_rank),
            ]
        return command

    def __enter__(self) -> "StudentServer":
        if not VLLM_PYTHON.exists():
            raise StudentServerUnavailable(
                f"{VLLM_PYTHON} does not exist, so there is no vLLM to serve the student with. "
                "The pipeline venv deliberately does not carry vLLM (incompatible torch pin)."
            )
        log_path = Path(os.environ.get("SLM_LOG_DIR", "logs")) / f"student-vllm-{_port()}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = self._command()
        self.log(f"      [student-server] {' '.join(command[2:])}")
        self.log(f"      [student-server] GPU {_gpu_ids()}, log → {log_path}")
        self._log_file = open(log_path, "w")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = _gpu_ids()
        # The server must not inherit the pipeline's proxy: a Squid proxy applied to a 127.0.0.1
        # request returns a 5xx error page, which is how B161 presented.
        for variable in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            environment.pop(variable, None)
        self._process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=environment,
            stdout=self._log_file, stderr=subprocess.STDOUT,
        )
        timeout = float(os.environ.get("SLM_STUDENT_SERVER_WAIT_S", "600"))
        try:
            started = time.perf_counter()
            _wait_for_ready(self.base_url, self._process, timeout, self.log)
            self.log(f"      [student-server] ready in {time.perf_counter() - started:.0f}s")
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        """Stop the server and release the GPU. Safe to call twice."""
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=30)
        self._process = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def complete(self, rendered: list[str], max_new_tokens: int) -> list[str]:
        """Greedy raw-text completions, order preserving.

        `/v1/completions` rather than `/v1/chat/completions` because the template has already been
        applied by `render_prompts`; sending these strings as a chat message would apply it twice.
        """
        from openai import OpenAI
        import httpx

        client = OpenAI(
            base_url=self.base_url,
            api_key="EMPTY",
            timeout=float(os.environ.get("SLM_STUDENT_REQUEST_TIMEOUT_S", "600")),
            http_client=httpx.Client(trust_env=False, limits=httpx.Limits(
                max_connections=256, max_keepalive_connections=256)),
        )
        concurrency = int(os.environ.get("SLM_STUDENT_CONCURRENCY", "64"))
        failures: list[str] = []

        def one(prompt: str) -> str:
            try:
                response = client.completions.create(
                    model=SERVED_MODEL_NAME,
                    prompt=prompt,
                    # Greedy, matching `do_sample=False` on the in-process path. Any other value
                    # makes this backend a different experiment rather than a faster one.
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                )
                return response.choices[0].text or ""
            except Exception as error:  # noqa: BLE001 - one row must not lose the whole eval
                failures.append(f"{type(error).__name__}: {error}")
                return ""

        workers = max(1, min(concurrency, len(rendered) or 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(one, rendered))
        if failures:
            # Reported, not raised: an empty output scores as a format failure, which is the honest
            # reading of a row the model did not answer. But a HIGH failure rate means the score
            # describes the transport, so the caller is told the count and the reason.
            from collections import Counter

            common = Counter(failures).most_common(2)
            self.log(f"      [student-server] {len(failures)}/{len(rendered)} request(s) failed: "
                     + "; ".join(f"{message} (x{count})" for message, count in common))
            if len(failures) / max(1, len(rendered)) > 0.05:
                raise StudentServerUnavailable(
                    f"{len(failures)} of {len(rendered)} completion requests failed, so the score "
                    "would describe the transport rather than the model"
                )
        return outputs


def infer_batch_served(
    prompts: list[str],
    weights_ref: str,
    base_model: str,
    *,
    max_new_tokens: int,
    max_model_len: int,
    log=print,
) -> list[str]:
    """One eval pass through a purpose-started vLLM server. Raises StudentServerUnavailable."""
    if not prompts:
        return []
    rendered = render_prompts(prompts, weights_ref, base_model)
    with StudentServer(weights_ref, base_model, max_model_len=max_model_len, log=log) as server:
        started = time.perf_counter()
        outputs = server.complete(rendered, max_new_tokens)
        elapsed = time.perf_counter() - started
        log(f"      [student-server] {len(prompts)} row(s) in {elapsed:.0f}s "
            f"({len(prompts) / max(elapsed, 1e-9):.2f} rows/s)")
    return outputs
