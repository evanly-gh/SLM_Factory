"""xLAM-60k train / BFCL eval — turn a request into a JSON call against declared tools."""
from __future__ import annotations

from data import quality_controls as qc
from eval.scorers import function_call as scorer
from tasks._builders import function_call_turn
from tasks.spec import MiningSource, TaskSpec


def _load(max_train: int, max_test: int, log=print):
    from data.loaders.xlam_bfcl import load_xlam_bfcl

    return load_xlam_bfcl(max_train=max_train, max_test=max_test, log=log)


def _check(row: dict) -> tuple[bool, str]:
    """The verdict AND the reason. Exposed as `_verify.checker` below."""
    from data.synth_verifiers import verify_function_call_row

    return verify_function_call_row(row)


def _verify(row: dict) -> bool:
    return _check(row)[0]


# WHY THE REASON IS PUBLISHED SEPARATELY
#     `TaskSpec.synth_verifier` only has to answer yes/no, so this wrapper used to be
#     `return verify_function_call_row(row)[0]` and the reason string was thrown away on the spot. `data.curriculum`
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
    name="xlam_bfcl",
    title="xLAM-60k / BFCL",
    category="format_bound",
    family="structured_output",

    load=_load,
    required_fields=("text", "answer"),
    initial_train_cap=5000,
    eval_cap=1000,
    eval_sampling="shuffled",
    closed_label_space=False,
    label_definitions={},
    verifier_notes="",
    # These four had never run: `function_call` fell into `apply_quality_controls`' `else` branch
    # and the dataset was returned untouched, so no xlam curriculum was ever filtered (B299).
    # `valid_json_answer` matters most here — a gold answer that does not parse trains the model
    # to emit something the scorer will mark wrong no matter what it predicts.
    quality_controls=(
        qc.require_fields("text", "answer", "tools"),
        qc.valid_json_answer(),
        qc.length_outliers(key="text"),
        qc.dedup_surface(key="text"),
    ),

    build_prompts=scorer.build_prompts,
    extract_predictions=scorer.extract_predictions,
    score=scorer.score,
    metric_name="ast_arg_match",
    max_new_tokens=256,
    max_seq_length=2048,
    eval_batch_size=32,
    failure_category=scorer.failure_category_of,
    needs_judge=False,
    judge_overlap=False,
    attach_reasoning=True,

    build_training_turn=function_call_turn,

    # The exact verifier is the point of synthesizing here: a generated call can be checked for
    # free against the row's own declared tool schema, with no teacher call and no judgement.
    synth_verifier=_verify,
    cot_annotation=False,

    # The canonical corpus, so `acquire` can serve unseen rows from the local cache instead of
    # paying Exa to rediscover broken mirrors of it (B297). xLAM ships ~60,000 rows and the
    # initial curriculum takes the first ~3,250.
    mining_sources=(
        MiningSource(
            hf_id="Salesforce/xlam-function-calling-60k",
            config=None,
            split="train",
            url="https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k",
            supports_offset=True,
        ),
    ),
    allow_paid_discovery=True,

    model_ranking_metric=None,
)
