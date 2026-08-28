from data.curation_log import CurationLog
from eval.harness import EvalResult


def test_default_curation_log_path_uses_explicit_run_environment(
    tmp_path,
    monkeypatch,
):
    expected = tmp_path / "run-a" / "data-curation.md"
    monkeypatch.setenv("SLM_CURATION_LOG_PATH", str(expected))

    assert CurationLog().path == str(expected)


def test_curation_log_persists_complete_dataset_composition(tmp_path):
    path = tmp_path / "data-curation.md"
    log = CurationLog(str(path))
    result = EvalResult(
        f1=0.8,
        per_class={},
        failures=[{
            "text": "raw held-out prompt secret",
            "label": "gold",
            "predicted": "raw held-out prediction secret",
        }],
    )

    log.write_iteration(
        iteration=2,
        task="clinc150",
        dataset_version="v3",
        total_examples=12,
        n_gold=5,
        n_hard=3,
        n_hard_source=2,
        n_hard_generated=3,
        rebuild_plan_identity="plan-abc",
        strategy_composition=[
            {"strategy": "resample_existing", "rows": 5},
            {"strategy": "mine_new_real_source", "rows": 2},
        ],
        source_novelty={
            "requested": 5,
            "novel_rows": 2,
            "novel_fraction": 0.4,
        },
        plan_yield={
            "status": "novel",
            "final_rows": 12,
            "novel_rows": 2,
        },
        confusion_pairs=[
            {"gold": "a", "predicted": "b", "count": 3},
        ],
        label_dist={"a": 6, "b": 6},
        config_a="a",
        config_b="b",
        best_config="a",
        eval_result=result,
        score_band="0.80-0.95",
        next_intervention="hyperparameter",
        hypothesis="test",
        model_id="test/model@Q4_K_M",
        size_mb=100,
        tier=0,
    )

    text = path.read_text()
    assert "- Total examples: 12" in text
    assert "- Initial gold: 5" in text
    assert "- Source anchors: 2" in text
    assert "- Generated hard rows: 3" in text
    assert "- Rebuild plan identity: plan-abc" in text
    assert "- Strategy composition:" in text
    assert "mine_new_real_source" in text
    assert "- Source novelty:" in text
    assert "- Plan yield:" in text
    assert "a→b: 3" in text
    assert "raw held-out prompt secret" not in text
    assert "raw held-out prediction secret" not in text


def _base_kwargs(result):
    return {
        "iteration": 4,
        "task": "clinc150",
        "dataset_version": "v4",
        "n_gold": 5,
        "n_hard": 0,
        "label_dist": {"a": 5},
        "config_a": "a",
        "config_b": "b",
        "best_config": "a",
        "eval_result": result,
        "score_band": "0.80-0.95",
        "next_intervention": "data_rebuild",
        "hypothesis": "test",
        "model_id": "test/model",
        "size_mb": 100,
        "tier": 0,
    }


def test_curation_log_renders_data_sources_section_with_links_and_counts(tmp_path):
    path = tmp_path / "data-curation.md"
    log = CurationLog(str(path))
    result = EvalResult(0.8, {}, [])
    log.write_iteration(
        **_base_kwargs(result),
        source_usage=[
            {"source": "hf:Salesforce/xlam/train",
             "url": "https://huggingface.co/datasets/Salesforce/xlam",
             "rows": 240, "novel_rows": 210},
            {"source": "web:site.example",
             "url": "https://site.example/faq", "rows": 8, "novel_rows": 8},
            {"source": "existing pool", "url": None, "rows": 2752, "novel_rows": 0},
        ],
    )
    text = path.read_text(encoding="utf-8")
    assert "### Data sources" in text
    assert "https://huggingface.co/datasets/Salesforce/xlam" in text
    assert "240 rows" in text
    assert "https://site.example/faq" in text
    assert "existing pool" in text


def test_curation_log_omits_data_sources_section_when_no_external_source(tmp_path):
    path = tmp_path / "data-curation.md"
    log = CurationLog(str(path))
    result = EvalResult(0.8, {}, [])
    log.write_iteration(
        **_base_kwargs(result),
        source_usage=[{"source": "existing pool", "url": None, "rows": 3000, "novel_rows": 0}],
    )
    text = path.read_text(encoding="utf-8")
    assert "### Data sources" not in text


def test_curation_log_renders_eval_firewall_line_when_rows_removed(tmp_path):
    path = tmp_path / "data-curation.md"
    log = CurationLog(str(path))
    result = EvalResult(0.8, {}, [])
    log.write_iteration(
        **_base_kwargs(result),
        eval_firewall={"total": 4, "by_layer": {"train_anchor": 3, "final": 1}},
    )
    text = path.read_text(encoding="utf-8")
    assert "- Eval firewall removed: 4 row(s)" in text
    assert "train_anchor=3" in text
    assert "final=1" in text


def test_curation_log_renders_zero_firewall_line_for_clean_build(tmp_path):
    # A clean build (nothing removed) still emits the line so the firewall is positively confirmed.
    path = tmp_path / "data-curation.md"
    log = CurationLog(str(path))
    result = EvalResult(0.8, {}, [])
    log.write_iteration(
        **_base_kwargs(result),
        eval_firewall={"total": 0, "by_layer": {"train_anchor": 0, "final": 0}},
    )
    text = path.read_text(encoding="utf-8")
    assert "- Eval firewall removed: 0 row(s)" in text


def test_curation_log_omits_firewall_line_when_not_provided(tmp_path):
    path = tmp_path / "data-curation.md"
    log = CurationLog(str(path))
    result = EvalResult(0.8, {}, [])
    log.write_iteration(**_base_kwargs(result))
    text = path.read_text(encoding="utf-8")
    assert "Eval firewall removed" not in text


def test_curation_log_iteration_is_retry_idempotent(tmp_path):
    path = tmp_path / "data-curation.md"
    log = CurationLog(str(path))
    result = EvalResult(0.8, {}, [])
    kwargs = {
        "iteration": 2,
        "task": "clinc150",
        "dataset_version": "v3",
        "n_gold": 5,
        "n_hard": 3,
        "label_dist": {"a": 5, "b": 3},
        "config_a": "a",
        "config_b": "b",
        "best_config": "a",
        "eval_result": result,
        "score_band": "0.80-0.95",
        "next_intervention": "hyperparameter",
        "hypothesis": "test",
        "model_id": "test/model",
        "size_mb": 100,
        "tier": 0,
        "entry_id": "test/model@Q4_K_M:2:/weights",
    }

    log.write_iteration(**kwargs)
    log.write_iteration(**kwargs)

    text = path.read_text(encoding="utf-8")
    assert text.count("## Iteration 2") == 1
    assert text.count("slm-curation-entry:") == 1


# --- Generation output budget (B: the 512-token truncation, 2026-08-28) -------------------
#
# Three runs (38832586, 38985393, 39041380) each reported "0 rows kept" from 1,019 / 133 / 425
# synthesis attempts on toolbench and were read as the VERIFIER rejecting everything. Nothing was
# ever verified: `max_tokens` was hardcoded at 512 for every task, a toolbench row serialises to
# 3,913 characters at the very smallest (~978 tokens), so every reply was cut off mid-object and
# dropped by a bare `except Exception: return None`.


def test_the_output_budget_is_sized_to_the_row_not_a_constant():
    """A budget that cannot hold the smallest row in the task loses 100% of generations."""
    import json

    from data.curriculum import _row_output_budget

    # A toolbench-shaped row: the `text` carries a full ReAct system prompt with its API list.
    big = {"text": "You are AutoGPT. " + ("api_description " * 900),
           "answer": "Thought: go\nAction: x\nAction Input: {}",
           "tools": [{"name": f"api_{i}", "parameters": {}} for i in range(12)]}
    needed = len(json.dumps(big)) / 4.0
    budget = _row_output_budget(big)
    assert budget > 512, "the old hardcoded 512 is exactly the bug"
    assert budget >= needed, (
        f"budget {budget} cannot hold a row needing ~{needed:.0f} tokens, so every generation "
        "would be truncated mid-JSON and lost"
    )


def test_a_small_row_still_gets_generous_room(monkeypatch):
    """A short answer must not be given a SMALL budget just because the row is small.

    This asserted `== 512` while the budget was sized per row. Every teacher call now asks for as
    much as the served context can return, so a short-answer task gets far more room than its reply
    needs — which costs nothing, because generation stops at the EOS token.
    """
    from data.curriculum import _row_output_budget

    monkeypatch.setenv("SLM_SYNTH_MAX_MODEL_LEN", "8192")
    assert _row_output_budget({"text": "WINNER claim now", "answer": "spam"}) >= 512


def test_the_budget_never_exceeds_what_the_served_context_can_return(monkeypatch):
    """Asking for more output than `max_model_len` allows is an HTTP 400, i.e. zero rows again."""
    from data.curriculum import _row_output_budget

    monkeypatch.setenv("SLM_SYNTH_MAX_MODEL_LEN", "8192")
    enormous = {"text": "x" * 500_000}
    assert _row_output_budget(enormous) <= 8192
    # And a long prompt has to come out of the same context.
    assert _row_output_budget(enormous, prompt="p" * 20_000) <= 8192 - (20_000 / 4.0)


def test_the_clamp_does_not_depend_on_unrelated_credentials(monkeypatch):
    """It read the value via `from config.config import ...`, which raises on ANY unset API key, under
    a bare `except: pass` — so a missing EXA_API_KEY silently disabled the clamp. Reading the env var
    directly is what makes the guard actually present."""
    import data.curriculum as curriculum_module

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setenv("SLM_SYNTH_MAX_MODEL_LEN", "4096")
    assert curriculum_module._row_output_budget({"text": "x" * 500_000}) <= 4096


def test_a_truncated_generation_is_reported_as_a_generation_failure_not_a_rejection():
    """The two have opposite fixes — raise the output budget, versus change the generator prompt — so
    a log that calls one the other sends the reader in the wrong direction. It did, three times."""
    from data.curriculum import _note_generation_failure, take_generation_failures

    take_generation_failures()
    _note_generation_failure("unparseable JSON from the generator",
                             ValueError("Unterminated string"),
                             raw='{"text": "You are AutoGPT and the reply stops mid-str')
    failures = take_generation_failures()
    assert len(failures) == 1
    kind, detail = failures[0]
    assert kind == "unparseable JSON from the generator"
    # The tail of the reply is the evidence that distinguishes truncation from malformed output.
    assert "stops mid-str" in detail
    assert take_generation_failures() == [], "taking the sample must clear it"
