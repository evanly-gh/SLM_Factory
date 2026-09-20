"""API teacher mode: `SLM_SYNTH_API_MODE=1` swaps the local vLLM Qwen3.6 for the DeepSeek API.

WHAT THE MODE IS FOR
    The teacher does five jobs — curriculum synthesis, CoT annotation, generated-row verification,
    the fitness/accuracy-goal measurement, and eval judging — and by default all five run on a
    35B served by a vLLM process that occupies a whole GPU and takes ~40 minutes to warm up. API
    mode routes them to a hosted endpoint instead, so a run needs one GPU rather than two and
    starts immediately.

WHAT THESE TESTS PIN
    Every place the two backends must behave DIFFERENTLY, because each of them is a silent failure
    if it regresses rather than a loud one:

      pricing        An unpriced paid provider is recorded at $0.00, so a whole run's synthesis
                     bill would vanish from the ledger while the ledger still looked complete.
      provider tag   Same failure by a different route: a DeepSeek call filed as "local" is
                     charged $0 by definition.
      proxy          `trust_env` is INVERTED between backends. Loopback traffic must bypass the
                     node's Squid proxy; api.deepseek.com is only reachable through it. Either
                     one wrong means every teacher call fails.
      request shape  `top_k` and `chat_template_kwargs` are vLLM extensions. DeepSeek rejects
                     unknown fields, so sending them 400s every call.
      self-check     Disabled in API mode, because it is one paid call per row to ask the teacher
                     whether it agrees with itself. The EXACT programmatic verifiers must survive.
      no waiting     The local preflight blocks up to 40 minutes for a server that is genuinely
                     loading weights. An API that does not answer will not start answering.

    Nothing here touches the network: clients are stubbed and only the arguments they were built
    and called with are inspected.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]


# --------------------------------------------------------------------------
# Config resolution — in a subprocess, because config reads the environment at import
# --------------------------------------------------------------------------


def _config_probe(expression: str, **env_overrides: str) -> subprocess.CompletedProcess:
    """Import config.config in a fresh interpreter and print one expression."""
    env = dict(os.environ)
    for name in ("SLM_SYNTH_API_MODE", "SLM_SYNTH_API_MODEL", "SLM_SYNTH_ENDPOINT",
                 "SLM_SYNTH_MODEL", "SLM_JUDGE_MODEL", "SLM_JUDGE_ENDPOINT",
                 "SLM_SYNTH_MAX_MODEL_LEN", "SLM_SYNTH_API_KEY", "SLM_CHEAP"):
        env.pop(name, None)
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c",
         f"import config.config as c; print({expression})"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )


def test_local_mode_is_the_default_and_leaves_the_teacher_constants_alone():
    probe = _config_probe(
        "(c.SYNTH_API_MODE, c.SYNTH_MODEL, c.JUDGE_MODEL, c.SYNTH_MAX_MODEL_LEN)",
        DEEPSEEK_API_KEY="unused",
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == (
        "(False, 'Qwen/Qwen3.6-35B-A3B', 'Qwen/Qwen3.6-35B-A3B', 8192)"
    )


def test_api_mode_redirects_synth_and_judge_to_deepseek():
    """One flag moves BOTH. The judge shares the teacher's endpoint by default, so leaving it on a
    local URL while the teacher moved would point it at a vLLM server that was never started."""
    probe = _config_probe(
        "(c.SYNTH_ENDPOINT, c.SYNTH_MODEL, c.SYNTH_API_KEY, c.JUDGE_ENDPOINT, c.JUDGE_MODEL)",
        SLM_SYNTH_API_MODE="1",
        DEEPSEEK_API_KEY="dsk-test",
    )
    assert probe.returncode == 0, probe.stderr
    endpoint, model, key, judge_endpoint, judge_model = eval(probe.stdout)
    assert endpoint == "https://api.deepseek.com"
    assert model == "deepseek-v4-flash"
    assert key == "dsk-test"
    assert judge_endpoint == endpoint
    assert judge_model == model


def test_api_mode_raises_at_import_without_a_key():
    """Fail at import, not at the first synthesis call.

    The alternative is a run that allocates GPUs, loads a dataset, builds an eval set and only
    then discovers it cannot reach its teacher — with the honest degradation path being
    "gold-only", so it might not discover it at all.
    """
    probe = _config_probe("c.SYNTH_API_MODE", SLM_SYNTH_API_MODE="1", DEEPSEEK_API_KEY="")
    assert probe.returncode != 0
    assert "DEEPSEEK_API_KEY" in probe.stderr


def test_api_mode_context_is_the_api_models_not_the_vllm_serve_figure():
    """8192 is what `_l40s_task_body.sh` chose to serve on one L40S, not a property of any model.

    `agent/teacher_fitness._prefix_fits` bounds its demonstration block by this number and SKIPS
    the k-shot measurement when five demonstrations do not fit — which on toolbench is the
    difference between a five-shot gate and a zero-shot one. Carrying the local KV-cache
    compromise onto an endpoint with a 1M window would discard demonstrations that fit fine.
    """
    probe = _config_probe("c.SYNTH_MAX_MODEL_LEN",
                          SLM_SYNTH_API_MODE="1", DEEPSEEK_API_KEY="dsk-test")
    assert probe.returncode == 0, probe.stderr
    assert int(probe.stdout) > 8192


def test_the_api_model_is_overridable():
    probe = _config_probe("(c.SYNTH_MODEL, c.JUDGE_MODEL)",
                          SLM_SYNTH_API_MODE="1", DEEPSEEK_API_KEY="dsk-test",
                          SLM_SYNTH_API_MODEL="deepseek-v4-pro")
    assert probe.returncode == 0, probe.stderr
    assert eval(probe.stdout) == ("deepseek-v4-pro", "deepseek-v4-pro")


# --------------------------------------------------------------------------
# Cost: the paid teacher must actually be priced
# --------------------------------------------------------------------------


@pytest.fixture
def api_mode(monkeypatch):
    """Flip the mode on the already-imported config module.

    Every consumer imports these names INSIDE the function that uses them, so patching the module
    attributes moves all of them without a reload — which would be unsafe here, since other
    modules hold references to the objects config exports.
    """
    import config.config as config

    monkeypatch.setattr(config, "SYNTH_API_MODE", True)
    monkeypatch.setattr(config, "SYNTH_ENDPOINT", "https://api.deepseek.com")
    monkeypatch.setattr(config, "SYNTH_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(config, "SYNTH_API_KEY", "dsk-test")
    monkeypatch.setattr(config, "SYNTH_API_MODEL", "deepseek-v4-flash")
    return config


def test_every_selectable_deepseek_model_has_a_price():
    """An unpriced paid model estimates at $0.00 with only a warning, so the run's biggest line
    item would be missing from a ledger that still reported `pricing_complete`."""
    from agent.cost import estimate_cost_usd, pricing_status

    for model in ("deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4-flash-vision-exp"):
        assert pricing_status("deepseek", model) == "known", model
        assert estimate_cost_usd(
            "deepseek", model, input_tokens=1_000_000, output_tokens=1_000_000
        ) > 0.0, model


def test_deepseek_rates_are_the_published_off_peak_figures():
    from agent.cost import estimate_cost_usd

    assert estimate_cost_usd(
        "deepseek", "deepseek-v4-flash", input_tokens=1_000_000, output_tokens=1_000_000
    ) == pytest.approx(0.22 + 0.66)
    assert estimate_cost_usd(
        "deepseek", "deepseek-v4-pro", input_tokens=1_000_000, output_tokens=1_000_000
    ) == pytest.approx(0.66 + 1.98)


def test_the_vision_model_is_not_priced_off_the_text_model_by_prefix():
    """`deepseek-v4-flash` is a strict prefix of `deepseek-v4-flash-vision-exp`, so alias order in
    `_pricing_key` decides which entry wins. Longest-first, or the vision id silently resolves to
    the text model's key and any future price divergence is billed wrong."""
    from agent.cost import _pricing_key, pricing_registry

    assert _pricing_key(
        "deepseek", "deepseek-v4-flash-vision-exp", pricing_registry()
    ) == "deepseek-v4-flash-vision-exp"


def test_prefix_cache_hits_are_priced_far_below_a_miss():
    """DeepSeek caches prompt prefixes automatically and bills hits ~30x lower. Synthesis prompts
    carry a long fixed prefix (task brief plus five demonstrations), so the hit rate IS the cost
    story — pricing hits at the miss rate would overstate spend by an order of magnitude."""
    from agent.cost import estimate_cost_usd

    miss = estimate_cost_usd("deepseek", "deepseek-v4-flash", input_tokens=1_000_000)
    hit = estimate_cost_usd("deepseek", "deepseek-v4-flash", cache_read_tokens=1_000_000)
    assert hit < miss / 10


def test_deepseek_usage_fields_are_read_from_the_response():
    """DeepSeek reports `prompt_cache_hit_tokens`/`prompt_cache_miss_tokens` rather than the
    OpenAI `prompt_tokens_details.cached_tokens` shape. Reading only the latter would bill every
    cache hit at the miss rate."""
    from agent.cost import _openai_usage

    usage = _openai_usage(
        type("R", (), {"usage": {
            "prompt_tokens": 1000,
            "prompt_cache_hit_tokens": 800,
            "prompt_cache_miss_tokens": 200,
            "completion_tokens": 50,
        }})(),
        "deepseek",
    )
    assert usage["input_tokens"] == 200
    assert usage["cache_read_tokens"] == 800
    assert usage["output_tokens"] == 50


def test_a_deepseek_call_is_never_classified_as_local():
    """Both routes to the bug: an explicit provider tag, and inference from the base URL. "local"
    is hardcoded to $0, so a misclassified call is a free call."""
    from agent.cost import openai_provider

    deepseek_client = type("C", (), {"base_url": "https://api.deepseek.com"})()
    assert openai_provider(deepseek_client, "deepseek-v4-flash") == "deepseek"
    assert openai_provider(deepseek_client, "deepseek-v4-flash", provider="deepseek") == "deepseek"
    # Even a client whose base_url was not readable is caught by the model name.
    assert openai_provider(type("C", (), {})(), "deepseek-v4-flash") == "deepseek"
    # And the local endpoint is unaffected.
    local_client = type("C", (), {"base_url": "http://127.0.0.1:8000/v1"})()
    assert openai_provider(local_client, "Qwen/Qwen3.6-35B-A3B", provider="local") == "local"


def test_a_preflight_call_is_attributed_to_the_provider_that_served_it(tmp_path, monkeypatch):
    """`models.list` costs nothing on either backend, so this is about attribution rather than
    money: a run summary that files DeepSeek preflights under "local" misstates which endpoints
    the run actually talked to."""
    from agent.cost import CostLedger, tracked_local_call

    monkeypatch.setenv("SLM_COST_EVENT_PATH", str(tmp_path / "events.jsonl"))
    tracked_local_call(
        lambda: type("R", (), {"data": []})(),
        stage="synth_preflight",
        model="deepseek-v4-flash",
        operation="models.list",
        provider="deepseek",
    )
    events = CostLedger(tmp_path / "events.jsonl").events()
    assert [e["provider"] for e in events] == ["deepseek"]
    assert events[0]["estimated_usd"] == 0.0


def test_per_row_paid_calls_do_not_echo_to_the_console_but_failures_do():
    """A calendar_json run made 26,513 synthesis calls. One console line each buries the log, and
    that is as true of a paid teacher as of a free one — the total belongs in the summary, which
    already has it. A FAILING call still prints, because that is the one worth seeing."""
    from agent.cost import CostEvent, _should_echo_cost_event

    def event(**kwargs):
        return CostEvent(provider="deepseek", model="deepseek-v4-flash", **kwargs)

    assert not _should_echo_cost_event(event(stage="api_synthesis", status="success"))
    assert not _should_echo_cost_event(event(stage="generation_judge", status="success"))
    assert _should_echo_cost_event(event(stage="api_synthesis", status="error"))
    # A one-off paid call is not high volume and still prints.
    assert _should_echo_cost_event(event(stage="synth_preflight", status="success"))


# --------------------------------------------------------------------------
# The synthesis client
# --------------------------------------------------------------------------


class _RecordingOpenAI:
    """Captures the constructor and chat-completion arguments without any network."""

    last_init: dict = {}
    last_call: dict = {}
    # The judge parses replies strictly (one number in [0,1]) while synthesis takes any text, so
    # the canned reply is settable per test rather than shared.
    reply: str = "generated row"

    def __init__(self, **kwargs):
        type(self).last_init = kwargs
        outer = self

        class _Completions:
            def create(self, **call_kwargs):
                type(outer).last_call = call_kwargs
                message = type("M", (), {"content": type(outer).reply})()
                return type("R", (), {
                    "choices": [type("C", (), {"message": message})()],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                    "model": call_kwargs.get("model"),
                    "id": "resp-1",
                })()

        self.chat = type("Chat", (), {"completions": _Completions()})()
        self.models = type("Models", (), {
            "list": staticmethod(lambda: type("R", (), {"data": []})())
        })()


@pytest.fixture
def recording_client(monkeypatch):
    """Stub `openai.OpenAI` and `httpx.Client` so nothing leaves the process."""
    import httpx
    import openai

    captured = {"httpx": {}}

    class _Client:
        def __init__(self, **kwargs):
            captured["httpx"] = kwargs

    monkeypatch.setattr(openai, "OpenAI", _RecordingOpenAI)
    monkeypatch.setattr(httpx, "Client", _Client)
    monkeypatch.setattr(_RecordingOpenAI, "reply", "generated row")
    return captured


def test_api_mode_enables_the_proxy_and_local_mode_disables_it(api_mode, recording_client,
                                                               monkeypatch):
    """The single most breakable thing in this change, and it fails in BOTH directions.

    Compute nodes export HTTP(S)_PROXY for internet access. Applying it to a loopback request
    makes Squid return a 5xx error page (the B161 crash), so the local client must ignore the
    environment; NOT applying it to api.deepseek.com means the request never leaves the node, so
    the API client must honour it.
    """
    import data.synth_client as synth_client

    synth_client._make_client(30.0)
    assert recording_client["httpx"]["trust_env"] is True

    monkeypatch.setattr(api_mode, "SYNTH_API_MODE", False)
    monkeypatch.setattr(api_mode, "SYNTH_ENDPOINT", "http://127.0.0.1:8000/v1")
    synth_client._make_client(30.0)
    assert recording_client["httpx"]["trust_env"] is False


def test_api_mode_sends_no_vllm_only_request_fields(api_mode, recording_client):
    """`top_k` and `chat_template_kwargs` are vLLM extensions, not OpenAI-spec parameters.
    DeepSeek rejects unknown fields, so leaving them in place 400s every synthesis call."""
    import data.synth_client as synth_client

    generate = synth_client.get_generate_fn(log=lambda *_: None)
    assert generate("write me a row") == "generated row"

    call = _RecordingOpenAI.last_call
    extra = call.get("extra_body") or {}
    assert "top_k" not in extra
    assert "chat_template_kwargs" not in extra
    assert call["model"] == "deepseek-v4-flash"
    assert _RecordingOpenAI.last_init["base_url"] == "https://api.deepseek.com"
    assert _RecordingOpenAI.last_init["api_key"] == "dsk-test"


def test_api_mode_disables_deepseek_thinking(api_mode, recording_client):
    """DeepSeek V4 reasons BY DEFAULT, and this used to send `extra_body=None`.

    The two teachers spell "no thinking" differently — `chat_template_kwargs` on vLLM,
    `thinking={"type": "disabled"}` on DeepSeek — and dropping the vLLM spelling for API mode
    (correct, it 400s) left nothing in its place, so every API-mode call reasoned.

    Measured on run 39562029: 2,480 calls, 10.95M output tokens for ~1,440 rows — roughly 7,100
    output tokens per generated row of "a sentence plus a short JSON list" — and 92% of the arm's
    $22.63 went to synthesis. DeepSeek's docs also put reasoning tokens INSIDE the completion
    budget, so a call can return HTTP 200 with no final answer once max_tokens is spent thinking,
    which is the likeliest cause of that run's format_valid=0.7040 against the local teacher's
    1.0000. One missing field, both symptoms.
    """
    import data.synth_client as synth_client

    generate = synth_client.get_generate_fn(log=lambda *_: None)
    assert generate("write me a row") == "generated row"
    assert _RecordingOpenAI.last_call["extra_body"] == {"thinking": {"type": "disabled"}}


def test_thinking_can_be_re_enabled_deliberately(api_mode, recording_client, monkeypatch):
    """A task that genuinely wants a reasoning teacher can ask for one, loudly."""
    import data.synth_client as synth_client

    monkeypatch.setenv("SLM_SYNTH_API_THINKING", "enabled")
    lines: list[str] = []
    generate = synth_client.get_generate_fn(log=lines.append)
    generate("write me a row")
    assert _RecordingOpenAI.last_call["extra_body"] == {"thinking": {"type": "enabled"}}
    assert any("thinking mode is ENABLED" in line for line in lines)


def test_an_unknown_thinking_value_is_refused_before_any_call(api_mode, recording_client,
                                                              monkeypatch):
    """Better a startup error than a 400 on every synthesis call, hours into a run."""
    import data.synth_client as synth_client

    monkeypatch.setenv("SLM_SYNTH_API_THINKING", "off")
    with pytest.raises(ValueError, match="must be 'disabled' or 'enabled'"):
        synth_client.get_generate_fn(log=lambda *_: None)


def test_a_non_deepseek_api_provider_gets_no_provider_specific_fields(recording_client,
                                                                     monkeypatch):
    """`thinking` is DeepSeek's own field; sending it blindly repeats the original mistake."""
    import config.config as config
    import data.synth_client as synth_client

    monkeypatch.setattr(config, "SYNTH_API_MODE", True)
    monkeypatch.setattr(config, "SYNTH_API_PROVIDER", "someco")
    monkeypatch.setattr(config, "SYNTH_ENDPOINT", "https://api.someco.com")
    monkeypatch.setattr(config, "SYNTH_MODEL", "someco-1")

    generate = synth_client.get_generate_fn(log=lambda *_: None)
    generate("write me a row")
    assert "extra_body" not in _RecordingOpenAI.last_call


def test_local_mode_still_sends_the_non_thinking_sampling_controls(recording_client, monkeypatch):
    """The other half of the same guard: dropping them for everyone would silently switch the
    local teacher into thinking mode, which prepends a <think> block nothing strips."""
    import config.config as config
    import data.synth_client as synth_client

    monkeypatch.setattr(config, "SYNTH_API_MODE", False)
    monkeypatch.setattr(config, "SYNTH_ENDPOINT", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(config, "SYNTH_MODEL", "Qwen/Qwen3.6-35B-A3B")

    generate = synth_client.get_generate_fn(log=lambda *_: None)
    assert generate is not None
    generate("write me a row")

    extra = _RecordingOpenAI.last_call["extra_body"]
    assert extra["top_k"] == 20
    assert extra["chat_template_kwargs"] == {"enable_thinking": False}


def test_api_mode_probes_once_instead_of_blocking_for_a_server_to_boot(api_mode, monkeypatch):
    """`wait_until_available` blocks up to 10 minutes mid-run for a vLLM server that may be
    restarting. An API has no cold start: if it is not answering, the cause is a key, a route or
    an outage, and none of those clear by sleeping — so a run should fail fast and be resubmitted
    with a fix rather than burn its wall-clock reserve waiting."""
    import data.synth_client as synth_client

    slept: list[float] = []
    monkeypatch.setattr(synth_client, "is_available", lambda **_kwargs: False)

    assert synth_client.wait_until_available(
        timeout_s=600.0, log=lambda *_: None, sleep=slept.append
    ) is False
    assert slept == [], "API mode must not poll"


def test_generated_rows_record_which_teacher_wrote_them(api_mode):
    """Provenance is the reason the mode is opt-in. A `synth:deepseek` row came from a model we do
    not own and whose training data we cannot inspect, and a results run has to be able to say so
    — the audit archive copies every field, so the label survives into the evidence file."""
    import data.synth_client as synth_client

    assert synth_client.synth_source_label() == "synth:deepseek"


def test_local_rows_are_still_labelled_vllm():
    import data.synth_client as synth_client

    assert synth_client.synth_source_label() == "synth:vllm"


# --------------------------------------------------------------------------
# The LLM self-check over generated rows
# --------------------------------------------------------------------------


def test_the_self_check_is_on_locally_and_off_on_the_paid_teacher(api_mode, monkeypatch):
    """It costs one paid call per surviving row — about a quarter of synthesis spend on the
    measured runs — to ask a model whether it agrees with itself, which is the weakest signal in
    the pipeline. The exact programmatic verifiers do the real work and are unaffected."""
    import data.curriculum as curriculum

    monkeypatch.delenv("SLM_VERIFY_SYNTH", raising=False)
    assert curriculum._verify_synth_enabled() is False

    monkeypatch.setattr(api_mode, "SYNTH_API_MODE", False)
    assert curriculum._verify_synth_enabled() is True


def test_an_explicit_verify_setting_overrides_the_mode_default_both_ways(api_mode, monkeypatch):
    import data.curriculum as curriculum

    monkeypatch.setenv("SLM_VERIFY_SYNTH", "1")
    assert curriculum._verify_synth_enabled() is True

    monkeypatch.setenv("SLM_VERIFY_SYNTH", "0")
    assert curriculum._verify_synth_enabled() is False

    monkeypatch.setattr(api_mode, "SYNTH_API_MODE", False)
    assert curriculum._verify_synth_enabled() is False


def test_disabling_the_self_check_says_so_once(api_mode, monkeypatch):
    """Announced once per process, not once per call: `synthesize_examples` runs once per targeted
    failure category and a rebuild targets up to five, so a per-call notice repeats itself."""
    import data.curriculum as curriculum

    monkeypatch.delenv("SLM_VERIFY_SYNTH", raising=False)
    monkeypatch.setattr(curriculum, "_VERIFY_DISABLED_ANNOUNCED", [False])
    logs: list[str] = []
    for _ in range(5):
        curriculum._verify_synth_enabled(log=logs.append)
    assert len(logs) == 1
    assert "SLM_VERIFY_SYNTH=1" in logs[0]


def test_the_exact_programmatic_verifiers_are_untouched_by_the_mode(api_mode):
    """These are what actually keep bad rows out: free, unfoolable, and run BEFORE any teacher
    call. Disabling the self-check must not be read as disabling verification."""
    from data.synth_verifiers import verify_calendar_row

    ok, reason = verify_calendar_row({"text": "x", "answer": "not json"})
    assert ok is False
    assert reason


# --------------------------------------------------------------------------
# The eval judge
# --------------------------------------------------------------------------


def test_the_judge_accepts_the_hosted_endpoint_without_a_second_opt_in(api_mode):
    """A hosted judge is remote by construction. Requiring SLM_JUDGE_ALLOW_REMOTE=1 on top of the
    API-mode flag would be two switches for one decision the run has already made."""
    from eval.judge_client import LocalJudgeClient

    judge = LocalJudgeClient(endpoint="https://api.deepseek.com", model="deepseek-v4-flash")
    assert judge.api_mode is True
    assert judge.allow_remote is True
    assert judge.provider == "deepseek"


def test_a_remote_judge_is_still_refused_outside_api_mode():
    """The loopback requirement exists so a judged metric is reproducible. It is lifted by an
    explicit choice, never by an accident of configuration."""
    from eval.judge_client import JudgeInfrastructureError, validate_judge_endpoint

    with pytest.raises(JudgeInfrastructureError):
        validate_judge_endpoint("https://api.deepseek.com", allow_remote=False)
    assert validate_judge_endpoint("http://127.0.0.1:8000/v1", allow_remote=False)


def test_the_judge_also_disables_deepseek_thinking(api_mode, recording_client, monkeypatch):
    """The judge is the second consumer and had the same gap.

    It emits a short verdict, so a reasoning trace in front of it is pure cost — and, as with
    synthesis, reasoning tokens share the completion budget and can exhaust `max_tokens` before the
    verdict appears. `ner_bc5cdr` is judge-free (span_f1 is exact), which is why this went unnoticed
    on the ablation arms; `dialogsum` and `toolbench` are not.
    """
    from data.synth_client import judge_extra_body

    assert judge_extra_body(True) == {"thinking": {"type": "disabled"}}
    # No `top_k`: the judge is scoring, and its sampling is set deliberately by the caller.
    assert judge_extra_body(False) == {"chat_template_kwargs": {"enable_thinking": False}}


def test_the_judge_drops_the_vllm_extra_body_in_api_mode(api_mode, recording_client, monkeypatch):
    from eval.judge_client import LocalJudgeClient

    monkeypatch.setattr(_RecordingOpenAI, "reply", "0.7")
    judge = LocalJudgeClient(endpoint="https://api.deepseek.com", model="deepseek-v4-flash")
    judge._preflight_complete = True
    assert judge._score_uncached({"question": "q", "gold": "g", "prediction": "p"}) == 0.7

    call = _RecordingOpenAI.last_call
    extra = call.get("extra_body") or {}
    # The vLLM-only field must not be sent (DeepSeek 400s it) — but something must be, or the
    # judge reasons by default. Note `max_tokens` here: the judge asks for a bare number, so its
    # budget is single digits. With thinking left ON, reasoning tokens share that budget and the
    # call can return 200 with no verdict at all.
    assert "chat_template_kwargs" not in extra
    assert extra == {"thinking": {"type": "disabled"}}
    assert call["max_tokens"] <= 16, (
        "if this grows, re-check the reasoning-token interaction above"
    )
    assert _RecordingOpenAI.last_call["model"] == "deepseek-v4-flash"


def test_the_judge_still_sends_the_vllm_extra_body_locally(recording_client, monkeypatch):
    from eval.judge_client import LocalJudgeClient

    monkeypatch.setattr(_RecordingOpenAI, "reply", "0.7")
    judge = LocalJudgeClient(endpoint="http://127.0.0.1:8000/v1",
                             model="Qwen/Qwen3.6-35B-A3B", api_mode=False)
    judge._preflight_complete = True
    judge._score_uncached({"question": "q", "gold": "g", "prediction": "p"})

    assert _RecordingOpenAI.last_call["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_the_judge_accepts_a_vendor_resolved_snapshot_name(api_mode):
    """DeepSeek documents `deepseek-v4-flash` as an alias that currently resolves to
    `DeepSeek-V4-Flash-0731` and echoes the resolved name back, so an exact comparison would
    reject every reply the moment the vendor rolls a snapshot. A different family is still
    rejected, which is the failure the check exists for."""
    from eval.judge_client import LocalJudgeClient

    judge = LocalJudgeClient(endpoint="https://api.deepseek.com", model="deepseek-v4-flash")
    assert judge._model_matches("deepseek-v4-flash")
    assert judge._model_matches("deepseek-v4-flash-0731")
    assert not judge._model_matches("deepseek-v4-pro")


def test_a_local_judge_demands_the_exact_model_it_asked_for():
    from eval.judge_client import LocalJudgeClient

    judge = LocalJudgeClient(endpoint="http://127.0.0.1:8000/v1", model="Qwen/Qwen3.6-35B-A3B",
                             api_mode=False)
    assert judge._model_matches("Qwen/Qwen3.6-35B-A3B")
    assert not judge._model_matches("Qwen/Qwen3.6-35B-A3B-FP8")


def test_the_judge_score_cache_key_separates_the_two_backends():
    """Scores judged by different models must never be reused for one another. They are not
    comparable, and a silent reuse would mix two metrics inside one run's trajectory."""
    from eval.judge_client import LocalJudgeClient

    payload = {"question": "q", "gold": "g", "prediction": "p"}
    local = LocalJudgeClient(endpoint="http://127.0.0.1:8000/v1",
                             model="Qwen/Qwen3.6-35B-A3B", api_mode=False)
    api = LocalJudgeClient(endpoint="https://api.deepseek.com",
                           model="deepseek-v4-flash", api_mode=True)
    assert local._cache_key(payload) != api._cache_key(payload)


def test_the_judge_timing_metadata_does_not_claim_a_paid_call_was_free(api_mode):
    from eval.judge_client import LocalJudgeClient

    api = LocalJudgeClient(endpoint="https://api.deepseek.com", model="deepseek-v4-flash")
    metadata = api._timing_metadata("chat.completions.create")
    assert metadata["provider"] == "deepseek"
    assert "estimated_usd" not in metadata

    local = LocalJudgeClient(endpoint="http://127.0.0.1:8000/v1",
                             model="Qwen/Qwen3.6-35B-A3B", api_mode=False)
    assert local._timing_metadata("chat.completions.create")["estimated_usd"] == 0.0


# --------------------------------------------------------------------------
# The SLURM body: no vLLM, no reserved GPU
# --------------------------------------------------------------------------

BODY = (REPO_ROOT / "tests" / "pipeline" / "_l40s_task_body.sh").read_text(encoding="utf-8")


def _run_gpu_profile(**env_overrides: str) -> subprocess.CompletedProcess:
    functions = "\n".join(
        re.search(rf"(?ms)^{re.escape(name)}\(\) \{{\n.*?^\}}\n", BODY).group(0)
        for name in ("_detect_allocated_gpu_count", "_validate_gpu_id",
                     "_configure_gpu_profile")
    )
    harness = (
        f"{functions}\n_configure_gpu_profile || exit $?\n"
        "printf '%s\\n' \"$SLM_GPU_PROFILE\" \"$SLM_GPU_COUNT\" "
        "\"[$SLM_SYNTH_GPU_IDS]\" \"$SLM_SYNTH_CONCURRENCY\" \"$SLM_PIPELINE_GPU_ID\"\n"
    )
    env = dict(os.environ)
    for name in ("SLURM_GPUS_ON_NODE", "CUDA_VISIBLE_DEVICES", "SLM_GPU_PROFILE",
                 "SLM_GPU_COUNT", "SLM_SYNTH_GPU_IDS", "SLM_SYNTH_TP",
                 "SLM_SYNTH_GPU_UTILIZATION", "SLM_SYNTH_MAX_NUM_SEQS",
                 "SLM_SYNTH_CONCURRENCY", "SLM_PIPELINE_GPU_ID", "SLM_SYNTH_API_MODE"):
        env.pop(name, None)
    env.update(env_overrides)
    return subprocess.run(["bash", "-c", harness], env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)


def test_api_mode_runs_on_a_single_gpu():
    """The 2-GPU floor exists solely because the vLLM teacher needs a card of its own. Keeping it
    when no teacher is launched would reserve an idle GPU on a shared queue for the whole run."""
    completed = _run_gpu_profile(SLM_SYNTH_API_MODE="1", SLURM_GPUS_ON_NODE="1")
    assert completed.returncode == 0, completed.stderr
    profile, count, synth_ids, concurrency, pipeline_id = completed.stdout.split()
    assert profile == "api-teacher-1gpu"
    assert count == "1"
    assert synth_ids == "[]", "no GPU is reserved for a teacher that is not running"
    assert pipeline_id == "0"
    assert int(concurrency) > 0, "synthesis still fans out; it is HTTP requests now, not sequences"


def test_local_mode_still_refuses_a_single_gpu():
    completed = _run_gpu_profile(SLURM_GPUS_ON_NODE="1")
    assert completed.returncode == 2
    assert "at least 2 allocated GPUs" in completed.stderr


def test_api_mode_still_validates_the_pipeline_gpu_id():
    completed = _run_gpu_profile(SLM_SYNTH_API_MODE="1", SLURM_GPUS_ON_NODE="1",
                                 SLM_PIPELINE_GPU_ID="3")
    assert completed.returncode == 2
    assert "outside the 1-GPU allocation" in completed.stderr


def test_the_body_skips_the_whole_vllm_block_in_api_mode():
    """Not just `vllm serve` — the CUDA module load, the separate .venv_vllm bootstrap and the
    endpoint export all belong to a server that is not being started. Exporting a stale
    SLM_SYNTH_ENDPOINT would be the worst of them: config resolves the teacher from it."""
    guard = 'if [ "${SLM_SYNTH_API_MODE:-0}" = "1" ]; then'
    assert guard in BODY
    api_branch = BODY[BODY.index("skipping the vLLM synth server"):]
    assert "unset SLM_SYNTH_ENDPOINT" in api_branch[:400]
    assert "export SLM_SYNTH_WAIT_S=0" in api_branch[:400]
    # The endpoint export and the serve invocation are both behind a mode check.
    assert 'if [ "${SLM_SYNTH_API_MODE:-0}" != "1" ]; then\n    export SLM_SYNTH_ENDPOINT=' in BODY
    # A wait the API branch set to 0 must not be stomped by the later default.
    assert 'export SLM_SYNTH_WAIT_S="${SLM_SYNTH_WAIT_S:-2400}"' in BODY


def test_the_exit_trap_tolerates_never_having_started_a_server():
    """`trap _cleanup_task_run EXIT` fires on every path, including the API one where VLLM_PID was
    never assigned. Killing an empty PID is harmless; printing "stopping vLLM ()" is a lie."""
    assert 'VLLM_PID=""' in BODY
    cleanup = re.search(r"(?ms)^_cleanup_task_run\(\) \{\n.*?^\}\n", BODY).group(0)
    assert 'if [ -z "$VLLM_PID" ]; then' in cleanup


# --------------------------------------------------------------------------
# The CUDA toolkit is needed by the TRAINING venv, not only by the vLLM teacher
# --------------------------------------------------------------------------
#
# `module load cuda` and `CUDA_HOME` used to live inside the `else` branch that launches the local
# vLLM server, under a comment asserting that everything in that block existed only to serve a
# teacher the API now serves instead. That was wrong about `nvcc`: Unsloth runs the model through
# torch.compile and TorchInductor shells out to `nvcc` to build kernels.
#
# Run 39361189, the first API-mode run on this launcher, died 21 minutes in — after paying for the
# full 1,000-row teacher measurement — with:
#     torch._inductor.exc.InductorError: PermissionError: [Errno 13] Permission denied: 'nvcc'


def _body_lines() -> list[str]:
    return BODY.splitlines()


def _index_of(needle: str) -> int:
    for index, line in enumerate(_body_lines()):
        if needle in line:
            return index
    raise AssertionError(f"{needle!r} not found in _l40s_task_body.sh")


def test_the_cuda_module_is_loaded_before_the_api_mode_branch():
    """Hoisted out of the vLLM-only branch, so BOTH teacher modes get a usable nvcc."""
    assert _index_of('module load "$CUDA_MOD"') < _index_of('echo "=== API teacher mode')


def test_cuda_home_is_exported_before_the_api_mode_branch():
    assert _index_of('export CUDA_HOME=') < _index_of('echo "=== API teacher mode')


def test_the_vllm_venv_path_stays_inside_the_local_branch():
    """Only the toolkit is shared. The .venv_vllm bin directory belongs to the local teacher."""
    assert _index_of('export PATH="$PROJ/.venv_vllm/bin:$PATH"') > _index_of('echo "=== API teacher mode')


def test_a_missing_nvcc_fails_fast():
    """Failing at setup costs seconds. Failing inside the first compile costs the whole teacher
    measurement — 1,000 paid API calls on run 39361189 before it fell over."""
    assert "ERROR: nvcc is not on PATH" in BODY
    assert _index_of("ERROR: nvcc is not on PATH") < _index_of('echo "=== API teacher mode')
