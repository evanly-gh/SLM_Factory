# tests/training/test_gguf_eval_concurrency.py
"""
`infer_batch_gguf` used to score prompts in a `for` loop; it now dispatches them across a pool of
independent GGUF contexts (`_run_gguf_concurrently`).

The reason this needs its own file is that the change is easy to mistake for tensor batching, and
the two have completely different risk profiles. llama-cpp-python 0.3.34 wires its multi-sequence
`LlamaBatch` only into `embed()`; `create_chat_completion` decodes ONE sequence per call, and
`n_batch`/`n_ubatch` size the prefill within that single sequence. So this is request CONCURRENCY:
every worker calls exactly the completion API the sequential loop called, which makes "per-row
output is unchanged" a property that can be pinned rather than hoped for. These tests pin it, plus
the four things that can go wrong around it — row/result misalignment, an unbounded number of
weight copies, an OOM being swallowed into a degraded score, and the cached primary context being
closed out from under the module-level cache.

Nothing here imports llama.cpp or loads a model: `_load_gguf_context` and `_eval_batch_size` are
replaced on the live module and the scoring callable is injected, so the scheduler is exercised on
its own.
"""
import importlib
import importlib.util
import time
from unittest.mock import MagicMock, patch

import pytest


def _helpers():
    """Resolve the module at call time — `importlib.reload` elsewhere in the suite rebinds it."""
    return importlib.import_module("training.slm_helpers")


class _FakeContext:
    """Stand-in for a `llama_cpp.Llama`: an identity plus close bookkeeping, no inference."""

    def __init__(self, name: str, *, on_close=None, close_raises: bool = False):
        self.name = name
        self.closes = 0
        self.on_close = on_close
        self.close_raises = close_raises

    def close(self):
        self.closes += 1
        if self.on_close is not None:
            self.on_close()
        if self.close_raises:
            raise RuntimeError(f"{self.name} refused to close")

    def __repr__(self):
        return f"_FakeContext({self.name!r})"


class _FakePool:
    """Fake context factory with a memory budget, so an over-large attempt fails like a GPU would.

    `capacity` counts contexts resident at once INCLUDING the primary, and a load that would exceed
    it raises without allocating — the same shape as llama.cpp failing a `cudaMalloc` and leaving
    nothing behind for the caller to free. Closing a context gives its budget back, so a retry can
    only succeed if the production code really did release the previous attempt.
    """

    def __init__(self, *, capacity: int | None = None, close_raises: bool = False):
        self.capacity = capacity
        self.close_raises = close_raises
        self.primary = _FakeContext("primary")
        self.created: list[_FakeContext] = []
        self.load_calls: list[tuple[str, int]] = []
        self.live = 1  # the primary is already resident

    def load(self, gguf_path: str, max_seq_length: int):
        self.load_calls.append((gguf_path, max_seq_length))
        if self.capacity is not None and self.live + 1 > self.capacity:
            raise RuntimeError(
                "ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory"
            )
        self.live += 1
        context = _FakeContext(
            f"extra{len(self.created)}",
            on_close=self._released,
            close_raises=self.close_raises,
        )
        self.created.append(context)
        return context

    def _released(self):
        self.live -= 1


def _install(monkeypatch, *, eval_batch_size, pool=None, ceiling=8):
    """Point the scheduler at fakes: no llama.cpp, no task registry, no model load."""
    helpers = _helpers()
    pool = pool if pool is not None else _FakePool()

    def _batch_size(task):
        if isinstance(eval_batch_size, BaseException):
            raise eval_batch_size
        return eval_batch_size

    monkeypatch.setattr(helpers, "_load_gguf_context", pool.load)
    monkeypatch.setattr(helpers, "_eval_batch_size", _batch_size)
    monkeypatch.setattr(helpers, "MAX_GGUF_EVAL_CONCURRENCY", ceiling)
    return helpers, pool


def _run(helpers, pool, prompts, score_one, *, task="clinc150"):
    return helpers._run_gguf_concurrently(
        prompts,
        primary=pool.primary,
        score_one=score_one,
        gguf_path="/fake/model.gguf",
        max_seq_length=4096,
        task=task,
    )


def _echo(context, prompt, index):
    return f"{prompt}#{index}"


def _module_copy():
    """A throwaway execution of `slm_helpers` so module-scope env reads can be checked.

    `importlib.reload` would rebind the module that every other test and every production caller
    already holds a reference to; a private copy under its own name reads the environment fresh
    while leaving `sys.modules` alone.
    """
    spec = importlib.util.spec_from_file_location(
        "_slm_helpers_env_probe", _helpers().__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPerRowOutputIsUnchanged:
    """The whole justification for request concurrency over hand-rolled multi-sequence decode."""

    def test_concurrent_scoring_returns_exactly_what_the_sequential_loop_returned(
        self, monkeypatch
    ):
        """Accuracy numbers must not move. Scheduling is the only thing this change may alter."""
        prompts = [f"p{index}" for index in range(17)]
        sequential = [_echo(None, prompt, index) for index, prompt in enumerate(prompts)]

        helpers, pool = _install(monkeypatch, eval_batch_size=8)
        concurrent = _run(helpers, pool, prompts, _echo)

        assert concurrent == sequential

    def test_every_prompt_is_scored_once_under_its_own_index(self, monkeypatch):
        """A pool must not drop, duplicate or renumber rows — index feeds the budget diagnostic."""
        prompts = [f"p{index}" for index in range(17)]
        seen: list[tuple[int, str]] = []

        def score_one(context, prompt, index):
            seen.append((index, prompt))
            return _echo(context, prompt, index)

        helpers, pool = _install(monkeypatch, eval_batch_size=8)
        _run(helpers, pool, prompts, score_one)

        assert sorted(seen) == list(enumerate(prompts))

    def test_a_prompt_never_migrates_between_contexts(self, monkeypatch):
        """Assignment is by position, so one row is generated start-to-finish on one context."""
        prompts = [f"p{index}" for index in range(12)]
        owners: dict[int, list[str]] = {}

        def score_one(context, prompt, index):
            owners.setdefault(index, []).append(context.name)
            return _echo(context, prompt, index)

        helpers, pool = _install(monkeypatch, eval_batch_size=4)
        _run(helpers, pool, prompts, score_one)

        assert sorted(owners) == list(range(12))
        assert all(len(names) == 1 for names in owners.values())
        assert len({names[0] for names in owners.values()}) == 4


class TestConcurrencyIsBoundedByMemoryNotByTheTask:
    def test_the_tasks_batch_size_is_clamped_to_the_ceiling(self, monkeypatch):
        """Each context holds its own copy of the weights, so xlam's 32 would ask for 50GB+."""
        prompts = [f"p{index}" for index in range(40)]
        helpers, pool = _install(monkeypatch, eval_batch_size=32, ceiling=8)

        result = _run(helpers, pool, prompts, _echo)

        assert len(pool.created) == 7  # 8 workers, one of which is the already-loaded primary
        assert result == [_echo(None, p, i) for i, p in enumerate(prompts)]

    def test_a_task_below_the_ceiling_gets_exactly_what_it_asked_for(self, monkeypatch):
        """The clamp is a ceiling, not a target — a modest task must not be inflated to it."""
        helpers, pool = _install(monkeypatch, eval_batch_size=3, ceiling=8)

        _run(helpers, pool, [f"p{index}" for index in range(6)], _echo)

        assert len(pool.created) == 2

    def test_the_ceiling_constant_is_what_actually_bounds_the_pool(self, monkeypatch):
        """Pins the constant to the behaviour, so the env override below is not decorative."""
        helpers, pool = _install(monkeypatch, eval_batch_size=32, ceiling=2)

        _run(helpers, pool, [f"p{index}" for index in range(5)], _echo)

        assert len(pool.created) == 1

    def test_the_default_is_off_and_the_env_override_turns_it_on(self, monkeypatch):
        """Ships at 1, i.e. sequential, and the override exists to opt back in deliberately.

        Enabled at 8 on runs 38734202/38734203 it scored two evals and then killed the run: the eval
        CUDA worker exited -6 with "worker produced no response". SIGABRT is how llama.cpp reports a
        failed device allocation — GGML_ASSERT calls abort() rather than raising — so the process is
        gone before `_looks_like_oom` or the halving retry can run, and a 7-day job dies with no
        diagnosis. The measured gain was ~20% (125s sequential vs 101s at 8-way over 535 prompts),
        because eight contexts submitting to one device serialise on it. Not a trade worth a crash
        that cannot be caught (B323).
        """
        monkeypatch.delenv("SLM_GGUF_EVAL_CONCURRENCY", raising=False)
        default_module = _module_copy()
        assert default_module.MAX_GGUF_EVAL_CONCURRENCY == 1
        assert default_module.MIN_GGUF_EVAL_CONCURRENCY == 1

        monkeypatch.setenv("SLM_GGUF_EVAL_CONCURRENCY", "3")
        assert _module_copy().MAX_GGUF_EVAL_CONCURRENCY == 3

    def test_the_default_loads_no_extra_contexts_at_all(self, monkeypatch):
        """At the shipped default the pool is the primary context and nothing else, so the path is
        byte-identical to the sequential one it replaced — no duplicated weights, nothing to abort."""
        helpers, pool = _install(monkeypatch, eval_batch_size=32, ceiling=1)

        results = _run(helpers, pool, [f"p{index}" for index in range(6)], _echo)

        assert pool.created == []
        assert results == [f"p{index}#{index}" for index in range(6)]


class TestOrderIsPreservedRegardlessOfCompletionOrder:
    def test_out_of_order_completions_still_return_in_prompt_order(self, monkeypatch):
        """
        Results are written by index, not appended. A pool that appended would reorder rows
        whenever generation lengths differ — which they always do — and the eval harness pairs row
        i with reference i, so the accuracy report would be scrambled rather than merely noisy.
        """
        prompts = [f"p{index}" for index in range(8)]
        finished: list[int] = []

        def score_one(context, prompt, index):
            # Later prompts finish first, so completion order is the reverse of prompt order.
            time.sleep(0.005 * (len(prompts) - index))
            finished.append(index)
            return _echo(context, prompt, index)

        helpers, pool = _install(monkeypatch, eval_batch_size=8)
        result = _run(helpers, pool, prompts, score_one)

        assert result == [_echo(None, p, i) for i, p in enumerate(prompts)]
        assert finished[-1] == 0  # the slowest row finished last and still came back first
        assert finished != sorted(finished)


class TestConcurrencyOfOneIsTheOldSequentialPath:
    def test_one_context_loads_nothing_extra_and_uses_the_cached_primary(self, monkeypatch):
        """A concurrency of 1 must be identical to the loop it replaced, and cost no extra load."""
        prompts = [f"p{index}" for index in range(4)]
        contexts: list[str] = []

        def score_one(context, prompt, index):
            contexts.append(context.name)
            return _echo(context, prompt, index)

        helpers, pool = _install(monkeypatch, eval_batch_size=1)
        result = _run(helpers, pool, prompts, score_one)

        assert pool.load_calls == []
        assert contexts == ["primary"] * 4
        assert result == [_echo(None, p, i) for i, p in enumerate(prompts)]

    def test_the_primary_is_worker_zero_of_a_larger_pool(self, monkeypatch):
        """Reusing the resident context is what keeps the pool one load cheaper than its size."""
        contexts: list[str] = []

        def score_one(context, prompt, index):
            contexts.append(context.name)
            return _echo(context, prompt, index)

        helpers, pool = _install(monkeypatch, eval_batch_size=4)
        _run(helpers, pool, ["p0"], score_one)

        assert contexts == ["primary"]
        assert len(pool.created) == 3


class TestOutOfMemoryHalvesAndRetries:
    def test_a_load_that_runs_out_of_memory_halves_until_the_pool_fits(self, monkeypatch):
        """
        The eval must survive a card that cannot hold the requested number of weight copies, the
        same way the bf16 path halves its batch — otherwise one unlucky pool size ends a run that
        has already spent an hour training, instead of costing a retry.
        """
        prompts = [f"p{index}" for index in range(6)]
        used: list[str] = []

        def score_one(context, prompt, index):
            used.append(context.name)
            return _echo(context, prompt, index)

        helpers, pool = _install(
            monkeypatch, eval_batch_size=8, pool=_FakePool(capacity=2)
        )
        result = _run(helpers, pool, prompts, score_one)

        assert result == [_echo(None, p, i) for i, p in enumerate(prompts)]
        assert len(set(used)) == 2  # 8 -> 4 -> 2, the first size the fake budget allows
        assert len(pool.load_calls) > 1  # it really retried rather than giving up

    def test_an_out_of_memory_mid_batch_rescores_every_prompt(self, monkeypatch):
        """
        The retry re-runs the WHOLE batch. Rows scored before the failure are recomputed rather
        than reused, because a partially populated result array plus a smaller pool is exactly how
        rows and references drift apart.
        """
        prompts = [f"p{index}" for index in range(6)]
        scored: list[str] = []
        already_failed: list[bool] = []

        def score_one(context, prompt, index):
            scored.append(prompt)
            if prompt == "p3" and not already_failed:
                already_failed.append(True)
                raise RuntimeError("CUDA error: out of memory during decode")
            return _echo(context, prompt, index)

        helpers, pool = _install(monkeypatch, eval_batch_size=4)
        result = _run(helpers, pool, prompts, score_one)

        assert result == [_echo(None, p, i) for i, p in enumerate(prompts)]
        assert scored.count("p0") == 2
        assert sorted(set(scored)) == prompts

    def test_halving_stops_at_one_and_then_the_failure_surfaces(self, monkeypatch):
        """
        A floor of 1 is what makes the retry loop terminate. Below it there is nothing left to
        shrink, so the OOM has to be reported rather than retried forever.
        """
        attempts: list[int] = []

        def score_one(context, prompt, index):
            attempts.append(index)
            raise RuntimeError("CUDA error: out of memory")

        helpers, pool = _install(monkeypatch, eval_batch_size=4)
        with pytest.raises(RuntimeError, match="out of memory"):
            _run(helpers, pool, ["p0"], score_one)

        assert len(attempts) == 3  # concurrency 4, then 2, then 1
        assert len(pool.load_calls) == 4  # 3 extras, then 1, then none

    def test_a_non_memory_failure_propagates_instead_of_becoming_a_degraded_score(
        self, monkeypatch
    ):
        """
        Only an OOM is recoverable. An over-budget prompt or a malformed completion is a real
        defect, and swallowing it would turn a crash into a silently wrong accuracy number — the
        one failure mode this project cannot detect after the fact.
        """
        calls: list[str] = []

        def score_one(context, prompt, index):
            calls.append(prompt)
            raise ValueError(
                f"Rendered GGUF prompt index {index} exceeds the input budget"
            )

        helpers, pool = _install(monkeypatch, eval_batch_size=4)
        with pytest.raises(ValueError, match="exceeds the input budget"):
            _run(helpers, pool, ["p0"], score_one)

        assert calls == ["p0"]  # no retry, no halving
        assert len(pool.load_calls) == 3

    def test_a_failed_row_is_never_blanked_into_an_empty_answer(self, monkeypatch):
        """The `None`-to-`""` fill is for absent rows; a raising row must not be scored as wrong."""
        helpers, pool = _install(monkeypatch, eval_batch_size=2)

        def score_one(context, prompt, index):
            if index == 1:
                raise KeyError("choices")
            return _echo(context, prompt, index)

        with pytest.raises(KeyError):
            _run(helpers, pool, ["p0", "p1"], score_one)


class TestOomDetection:
    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("CUDA error: out of memory"),
            MemoryError("cudaMalloc failed for KV cache"),
            RuntimeError(
                "ggml_backend_cuda_buffer_type_alloc_buffer: failed to allocate 2048.00 MiB"
            ),
            RuntimeError("not enough memory to load the model"),
            RuntimeError("insufficient device memory for context"),
            Exception("OOM"),
        ],
    )
    def test_llama_cpp_style_allocation_failures_are_recognised(self, error):
        """
        Matched on the message because llama.cpp surfaces allocation failures as generic
        exceptions from the C library — there is no typed OOM to catch, so `except OSError` or an
        `isinstance` check would let every one of these through to kill the run.
        """
        assert _helpers()._looks_like_oom(error) is True

    def test_a_typed_out_of_memory_error_is_recognised_by_its_class_name(self):
        """`torch.cuda.OutOfMemoryError` can carry an empty message; the type name still says it."""

        class OutOfMemoryError(RuntimeError):
            pass

        assert _helpers()._looks_like_oom(OutOfMemoryError()) is True

    @pytest.mark.parametrize(
        "error",
        [
            ValueError("Rendered GGUF prompt index 3 exceeds the input budget"),
            KeyError("choices"),
            RuntimeError("llama_decode returned 1"),
            FileNotFoundError("/fake/model.gguf"),
            RuntimeError(
                "Installed llama-cpp-python cannot enforce non-thinking chat-template kwargs"
            ),
        ],
    )
    def test_ordinary_defects_are_not_mistaken_for_memory_pressure(self, error):
        """A false positive here retries a deterministic bug three times and then reports it late."""
        assert _helpers()._looks_like_oom(error) is False


class TestContextLifetimes:
    def test_extras_are_closed_and_the_cached_primary_is_not(self, monkeypatch):
        """
        `primary` belongs to the module-level `_gguf_cache`. Closing it here would make the next
        eval reload the model from disk, which is the cost this whole change exists to avoid.
        """
        helpers, pool = _install(monkeypatch, eval_batch_size=4)

        _run(helpers, pool, [f"p{index}" for index in range(4)], _echo)

        assert [context.closes for context in pool.created] == [1, 1, 1]
        assert pool.primary.closes == 0

    def test_a_failed_attempts_contexts_are_released_before_the_retry(self, monkeypatch):
        """Retrying smaller only helps if the oversized attempt actually gave its memory back."""
        helpers, pool = _install(
            monkeypatch, eval_batch_size=8, pool=_FakePool(capacity=3)
        )

        _run(helpers, pool, [f"p{index}" for index in range(4)], _echo)

        assert len(pool.created) > 1  # more than one attempt allocated
        assert all(context.closes == 1 for context in pool.created)
        assert pool.primary.closes == 0

    def test_extras_are_closed_even_when_the_batch_raises(self, monkeypatch):
        """A propagating non-OOM failure must not leak the weight copies it had loaded."""
        helpers, pool = _install(monkeypatch, eval_batch_size=4)

        def score_one(context, prompt, index):
            raise ValueError("malformed completion")

        with pytest.raises(ValueError, match="malformed completion"):
            _run(helpers, pool, ["p0"], score_one)

        assert [context.closes for context in pool.created] == [1, 1, 1]
        assert pool.primary.closes == 0

    def test_a_context_that_refuses_to_close_does_not_lose_the_results(self, monkeypatch):
        """Teardown is best-effort: a failed free must not discard an eval that already succeeded."""
        prompts = [f"p{index}" for index in range(4)]
        helpers, pool = _install(
            monkeypatch, eval_batch_size=4, pool=_FakePool(close_raises=True)
        )

        result = _run(helpers, pool, prompts, _echo)

        assert result == [_echo(None, p, i) for i, p in enumerate(prompts)]
        assert [context.closes for context in pool.created] == [1, 1, 1]


class TestDegenerateInputs:
    def test_an_unresolvable_task_falls_back_to_a_single_context(self, monkeypatch):
        """
        `infer_batch_gguf` defaults `task=""`, and callers outside the agent loop use it that way.
        An unknown task must lose the speedup, not the eval.
        """
        contexts: list[str] = []

        def score_one(context, prompt, index):
            contexts.append(context.name)
            return _echo(context, prompt, index)

        helpers, pool = _install(
            monkeypatch, eval_batch_size=KeyError("unknown task ''")
        )
        result = _run(helpers, pool, ["p0", "p1"], score_one, task="")

        assert pool.load_calls == []
        assert contexts == ["primary", "primary"]
        assert result == ["p0#0", "p1#1"]

    def test_a_declared_batch_size_below_the_floor_still_yields_one_worker(self, monkeypatch):
        """A zero or negative declared batch size must not produce a pool with no workers in it."""
        helpers, pool = _install(monkeypatch, eval_batch_size=0)

        result = _run(helpers, pool, ["p0"], _echo)

        assert result == ["p0#0"]
        assert pool.load_calls == []

    def test_no_prompts_returns_no_rows(self, monkeypatch):
        """An empty shard must return an empty list rather than tripping over `len(prompts)`."""
        helpers, pool = _install(monkeypatch, eval_batch_size=8)

        assert _run(helpers, pool, [], _echo) == []

    def test_a_missing_completion_becomes_an_empty_string_not_a_none(self, monkeypatch):
        """Downstream metrics do string work on every row, so `None` would raise far from here."""
        helpers, pool = _install(monkeypatch, eval_batch_size=2)

        result = _run(helpers, pool, ["p0", "p1"], lambda c, p, i: None)

        assert result == ["", ""]


class TestExtraContextLoading:
    """`_load_gguf_context` is the only place the pool touches llama.cpp; it is faked, not run."""

    def test_an_extra_context_is_gpu_offloaded_on_the_primarys_terms(self, monkeypatch):
        """A pool of CPU contexts would be slower than the sequential GPU path it replaced."""
        monkeypatch.delenv("SLM_GGUF_GPU_LAYERS", raising=False)
        llama_cls = MagicMock(return_value=_FakeContext("gpu"))

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=llama_cls)}):
            _helpers()._load_gguf_context("/fake/model.gguf", 8192)

        assert llama_cls.call_args.kwargs == {
            "model_path": "/fake/model.gguf",
            "n_ctx": 8192,
            "n_gpu_layers": -1,
            "verbose": False,
        }

    def test_the_gpu_layer_override_reaches_the_extra_contexts_too(self, monkeypatch):
        """
        SLM_GGUF_GPU_LAYERS=0 exists to mirror device compute exactly. A pool that ignored it
        would report GPU numbers under a flag that asked for CPU ones.
        """
        monkeypatch.setenv("SLM_GGUF_GPU_LAYERS", "0")
        llama_cls = MagicMock(return_value=_FakeContext("cpu"))

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=llama_cls)}):
            _helpers()._load_gguf_context("/fake/model.gguf", 4096)

        assert llama_cls.call_args.kwargs["n_gpu_layers"] == 0

    def test_a_cpu_only_wheel_falls_back_instead_of_failing_the_eval(self, monkeypatch):
        """Same fallback the primary load has, so a CPU-only build still gets a pool."""
        monkeypatch.delenv("SLM_GGUF_GPU_LAYERS", raising=False)
        fallback = _FakeContext("cpu")
        llama_cls = MagicMock(
            side_effect=[RuntimeError("no CUDA devices available"), fallback]
        )

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=llama_cls)}):
            context = _helpers()._load_gguf_context("/fake/model.gguf", 4096)

        assert context is fallback
        assert [call.kwargs["n_gpu_layers"] for call in llama_cls.call_args_list] == [-1, 0]


class TestInferBatchGgufOverAPool:
    def test_a_pooled_eval_returns_each_prompts_own_completion_in_order(self, monkeypatch):
        """
        The whole path against a fake `llama_cpp`: primary from the cache, extras from
        `_load_gguf_context`, one completion call per prompt carrying the sequential path's
        arguments. Row i must hold prompt i's answer — a pool that returned the right answers in
        the wrong order would read as a noisy score rather than as a bug.
        """
        monkeypatch.setenv("SLM_EVAL_BATCH_SIZE", "4")
        monkeypatch.delenv("SLM_MAX_SEQ_LENGTH", raising=False)
        monkeypatch.delenv("SLM_GGUF_GPU_LAYERS", raising=False)
        instances: list = []

        class EchoingLlama:
            """Answers with the prompt it was given, so a swapped row shows up in the result."""

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.calls: list[dict] = []
                self.closes = 0
                instances.append(self)

            def create_chat_completion(
                self, messages, *, max_tokens, temperature, chat_template_kwargs=None
            ):
                self.calls.append(
                    {
                        "content": messages[0]["content"],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "chat_template_kwargs": chat_template_kwargs,
                    }
                )
                return {
                    "choices": [
                        {"message": {"content": f"answer:{messages[0]['content']}"}}
                    ]
                }

            def close(self):
                self.closes += 1

        prompts = [f"prompt{index}" for index in range(9)]
        helpers = _helpers()
        monkeypatch.setattr(helpers, "MAX_GGUF_EVAL_CONCURRENCY", 8)
        with (
            patch.object(helpers, "_gguf_cache", {}),
            patch.object(helpers, "_gguf_cache_order", []),
            patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=EchoingLlama)}),
        ):
            result = helpers.infer_batch_gguf(
                prompts,
                "/fake/model.gguf",
                max_new_tokens=50,
                base_model="Qwen/Qwen3-1.7B",
            )

        assert result == [f"answer:{prompt}" for prompt in prompts]
        assert len(instances) == 4  # the cached primary plus three extras
        assert sum(len(instance.calls) for instance in instances) == len(prompts)
        assert all(
            call["chat_template_kwargs"] == {"enable_thinking": False}
            and call["temperature"] == 0.0
            and call["max_tokens"] == 50
            for instance in instances
            for call in instance.calls
        )
        assert [instance.closes for instance in instances] == [0, 1, 1, 1]
        assert instances[0].kwargs["n_ctx"] == helpers._DEFAULT_INFERENCE_SEQ_LENGTH
        assert all(
            instance.kwargs["n_ctx"] == helpers._DEFAULT_INFERENCE_SEQ_LENGTH
            for instance in instances
        )
