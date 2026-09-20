"""ToolBench / ToolEval — plan a whole multi-step API solution path, scored by pass rate.

The task from ``Small Language Models for Efficient Agentic Tool Calling`` (arXiv:2512.15943),
which fine-tunes on ToolLLM's ToolBench corpus (Qin et al., arXiv:2307.16789) and reports ToolEval
pass rate over six named test subsets.

HOW THIS DIFFERS FROM `xlam_bfcl`, THE OTHER TOOL-CALLING TASK
    They look adjacent and share almost no machinery, which is worth stating explicitly because
    under the old `task_type` design both would have been `function_call` and would have inherited
    each other's scorer.

      unit of work   xlam emits ONE call for one request. ToolBench emits a whole PATH: several
                     Thought/Action/Action Input steps ending in `Finish`.
      output format  xlam is a JSON array of calls. ToolBench is ToolLLM's own interleaved
                     `Thought:` / `Action:` / `Action Input:` text.
      callable set   xlam declares a handful of tools per row. ToolBench declares 2-16 drawn from
                     ~16,000 real RapidAPI endpoints, so the API names themselves are the hard part.
      scoring        xlam is exact AST argument match against a canonical gold. ToolBench has NO
                     gold for its test queries — pass rate asks a judge whether the produced answer
                     addresses the query, majority-voted over 3 rounds.
      context        xlam fits in 2,048 tokens. A ToolBench prompt is an API schema list: measured
                     median 2,082 tokens, p99 5,326, max 6,802.

    So this is the suite's only judged structured-output task, and its metric is the only one that
    is not decidable by computation.

TOKEN BUDGETS, FROM MEASUREMENT
    Measured on the real loaded splits (765 eval rows, Qwen2.5 tokenizer, 2026-08-24):

        eval prompt    median 1,248   p95 2,838   p99 3,973   max 9,155
        train target   median   652   p95 1,127   p99 1,410   max 2,445

    8192 with a 1536-token reserve fits 99.6% of eval prompts and 99.3% of train targets, and is
    the knee of the curve: a 1024 reserve would truncate 8% of TARGETS, which is worse than
    truncating a prompt because a clipped target teaches the model to stop mid-path — and the eval
    scores exactly that as `no_finish_call`. Going to 16384 buys the last 0.4% of prompts (3 rows of
    765) for double the KV cache on every row, which is not a trade worth making.

    Worth noting against the paper, which claims an 8,192-token sequence length on
    `facebook/opt-350m`. That checkpoint's `max_position_embeddings` is 2,048, so the claim is not
    possible as stated — and a 2,048 context would truncate the prompt on roughly a third of rows.
"""
from __future__ import annotations

from data import quality_controls as qc
from eval.scorers import toolbench as scorer
from tasks._builders import toolbench_turn
from tasks.spec import MiningSource, TaskSpec


def _load(max_train: int, max_test: int, log=print):
    from data.loaders.toolbench import load_toolbench

    return load_toolbench(max_train=max_train, max_test=max_test, log=log)


def _check(row: dict) -> tuple[bool, str]:
    """The verdict AND the reason. Exposed as `_verify.checker` below."""
    from data.synth_verifiers import verify_toolbench_row

    return verify_toolbench_row(row)


def _verify(row: dict) -> bool:
    return _check(row)[0]


# WHY THE REASON IS PUBLISHED SEPARATELY
#     `TaskSpec.synth_verifier` only has to answer yes/no, so this wrapper used to be
#     `return verify_toolbench_row(row)[0]` and the reason string was thrown away on the spot. `data.curriculum`
#     looks for a `.checker` attribute to recover it, finds nothing, and its
#     "[verify:exact] programmatic verifier rejected N row(s)" block is then unreachable.
#
#     Run 38985393 is what that costs. Synthesis generated 519 rows, the exact verifier rejected all
#     519, and the log recorded only the total — so which of the five checks fired (unparseable path,
#     undeclared API, bad argument schema, no terminal Finish, over the call budget) had to be
#     reverse-engineered afterwards from the vLLM access log. The information existed at the moment of
#     rejection and was discarded one character from where it was needed.
_verify.checker = _check


SPEC = TaskSpec(
    name="toolbench",
    title="ToolBench / ToolEval pass rate (arXiv:2512.15943)",
    category="format_bound",
    family="structured_output",

    load=_load,
    # NOT `answer`. The train rows carry a gold path, but the ToolEval test queries have none —
    # pass rate is reference-free, so requiring `answer` would reject every eval row. `query` is
    # what the scorer actually grades against: it is the text the judge is asked about, and it is
    # present on both splits.
    required_fields=("text", "query"),
    initial_train_cap=5000,
    select_cap=1000,
    eval_sampling="shuffled",
    closed_label_space=False,
    label_definitions={},
    entity_type_vocabulary=(),  # extracts no spans
    verifier_notes="",
    quality_controls=(
        qc.require_fields("text", "query", "answer", "tools"),
        qc.complete_toolbench_path(),
        qc.length_outliers(key="text"),
        # NOT deduplicated, and this is the one task where a surface filter would be actively
        # destructive rather than merely unhelpful. The corpus ships each trajectory as several
        # step-truncated prefixes, so two rows for the same query differ by a single action inside
        # ~8,000 characters of shared API schema — a trigram Jaccard threshold low enough to catch
        # them also catches genuinely distinct trajectories over the same tool. The loader's
        # complete-path rule removes the prefixes structurally instead, which is exact.
    ),

    build_prompts=scorer.build_prompts,
    extract_predictions=scorer.extract_predictions,
    score=scorer.score,
    metric_name="tooleval_pass_rate",
    max_new_tokens=1536,
    max_seq_length=8192,
    # Sixteen rather than the 32 the other structured-output tasks use: a ToolBench prompt is four
    # times the length of an xlam one and the generation is four times longer, so the same batch
    # size is roughly sixteen times the KV cache.
    eval_batch_size=16,
    failure_category=scorer.failure_category_of,
    # The judge decides the metric, so a judge outage must stop the run rather than score every
    # query unsolved — which would look exactly like a model that cannot use tools and would send
    # the loop after a phantom regression.
    needs_judge=True,
    # The overlap warmer in `eval/harness.py` pre-populates the cache using the GENERATION scorer's
    # (text, gold, prediction) triples. ToolEval judges (query, answer, round) payloads under a
    # different rubric, so the warmer would miss every entry and the judging would run twice.
    # Overlapping this task's judge needs a rubric-aware warmer, which does not exist yet.
    judge_overlap=False,
    # ToolBench reasoning is `Thought:` inside the path, not a `<reasoning>` block, and the path is
    # already recorded whole on the failure record by the scorer. The generic attacher would find
    # nothing and add nothing.
    attach_reasoning=False,

    # Selection and reporting coincide: pass rate IS the published ToolEval number. Worth noting
    # this is also the suite's most expensive eval, which is precisely why the report pass is a
    # separate once-per-run script rather than a bigger in-loop draw.
    report_load=None,
    report_score=scorer.score,
    report_metric_name="tooleval_pass_rate",

    build_training_turn=toolbench_turn,

    synth_verifier=_verify,
    cot_annotation=False,

    # The corpus is 187,542 rows and a cold start takes 5,000, so rung 1 of the mining ladder has
    # ~180,000 rows of headroom and should never need paid discovery. The loader reads a growing
    # byte prefix, so a re-read for a deeper slice costs only the difference — which is the
    # capability B303 records `calendar_json`'s placeholder source as lacking.
    mining_sources=(
        MiningSource(
            hf_id="Yhyu13/ToolBench_toolllama_G123_dfs",
            config=None,
            split="train",
            url="https://huggingface.co/datasets/Yhyu13/ToolBench_toolllama_G123_dfs",
            supports_offset=True,
        ),
    ),
    # No other corpus carries ToolBench's API namespace: a solution path is only gradable against
    # the ~16,000 RapidAPI endpoints these prompts declare, and a discovered tool-calling dataset
    # would be mapped onto a different callable surface entirely. With 180,000 rows in reserve
    # there is also nothing discovery could add that rung 1 cannot.
    allow_paid_discovery=False,

    model_ranking_metric=None,
)
