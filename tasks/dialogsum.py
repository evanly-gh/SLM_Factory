"""DialogSum — summarize a real-life spoken conversation in a sentence or two.

REWORKED 2026-09-06. This task previously trained on DialogSum + SAMSum concatenated and scored
with the LLM judge under `metric_name="judge_mean_0_1"`. Four things changed, and each one is a
property of the data rather than a preference:

  1. SAMSUM IS GONE. Same task, single reference, easier, and it was diluting the training set
     with a second summary style while contributing nothing the metric could use.

  2. THE SOURCE CHANGED, because the old one cannot express the metric. DialogSum's test split has
     THREE independent human summaries per dialogue, and `knkarthick/dialogsum` — the mirror the
     old loader used — flattens them: its `test.csv` is 1,500 rows with a single `summary` column,
     the same dialogue repeated once per annotator. Scoring that would count every dialogue three
     times against one arbitrary reference. `data/loaders/dialogsum.py` now reads the original
     release and REFUSES to load a split whose rows do not carry exactly three references.

  3. THE JUDGE IS NO LONGER THE HEADLINE. A judge score is comparable to no published number, is
     non-deterministic, and costs money on every eval in a loop that evaluates dozens of times.
     `needs_judge` and `judge_overlap` are now False, and `toolbench` is the only judged task —
     which is right, because ToolEval's pass rate is DEFINED as a judged vote.

  4. THE CEILING IS RECORDED. One annotator scored against the other two reaches ROUGE-1 53.35 /
     ROUGE-2 26.72 / ROUGE-L 50.84. The best fine-tuned models sit near 47 ROUGE-1 and
     out-of-the-box models near 36. So the ~11 points fine-tuning buys and the ~6 still on the
     table are both real, and a 47 is about 88% of human rather than 47% of perfect. The scorer
     carries those three numbers so a report cannot lose them.

WHY THIS TASK IS IN THE SUITE
    Summarizing private chat threads into a notification digest. Nobody uploads their message
    history to an API for a one-line preview. It is also harder than SAMSum in the way that
    matters: the conversations are longer and the summaries shorter, so it demands real
    abstraction rather than copying.

WATCH-OUTS
    500 test dialogues is the smallest eval in the suite. That is workable BECAUSE ROUGE is
    continuous per example rather than 0/1 — the confidence interval lands near +/-1.3 points —
    but it is reported rather than glossed over. And DialogSum appears in instruction-tuning
    collections, so an instruct base will have an inflated zero-shot baseline; that compresses the
    apparent delta rather than exaggerating it.
"""
from __future__ import annotations

from data import quality_controls as qc
from eval.scorers import summarization as scorer
from tasks._builders import summarization_turn
from tasks.spec import MiningSource, TaskSpec


def _check(row: dict) -> tuple[bool, str]:
    """The verdict AND the reason. Exposed as `_verify.checker` below."""
    from data.synth_verifiers import verify_summarization_row

    return verify_summarization_row(row)


def _verify(row: dict) -> bool:
    return _check(row)[0]


_verify.checker = _check


def _load(max_train: int, max_test: int, log=print):
    from data.loaders.dialogsum import load_dialogsum

    return load_dialogsum(max_train=max_train, max_test=max_test, log=log)


SPEC = TaskSpec(
    name="dialogsum",
    title="DialogSum (dialogue summarization, 3 references)",
    category="in_distribution",
    family="generation",

    load=_load,
    # `references` is required, not optional. It is the field the scorer grades against, and
    # requiring it is what makes a source that flattens the three summaries fail at load rather
    # than silently reduce the metric to single-reference ROUGE.
    required_fields=("text", "answer", "references"),
    initial_train_cap=5000,
    # The test split is 500 rows in total, so this cap never bites and the loop scores all of
    # them. That is also why `select_cap` is asserted as `<= 1000` rather than `== 1000`: a task
    # with a smaller split is not a task whose cap drifted.
    select_cap=1000,
    eval_sampling="shuffled",
    closed_label_space=False,
    label_definitions={},
    entity_type_vocabulary=(),  # extracts no spans
    verifier_notes=(
        "You are shown a conversation transcript and, as the proposed answer, ONLY its summary — "
        "one to three sentences. Speakers are labelled `#Person1#`, `#Person2#`; that is the "
        "corpus convention and the summary should use the same labels rather than inventing "
        "names.\n"
        "Summaries in this corpus are SHORT relative to the conversation — a median of about 17 "
        "words against a 120-word dialogue — so brevity is correct and a summary that reads like "
        "a condensed transcript is wrong.\n"
        "Judge ONLY whether the summary is accurate and complete for this conversation: whether "
        "it states something the dialogue does not support, and whether it omits the main point. "
        "Do NOT reject it for being terse, for omitting pleasantries, or for wording that "
        "differs from how you would have phrased it."
    ),
    quality_controls=(
        qc.require_fields("text", "answer", "references"),
        qc.length_outliers(key="text"),
        # NOT deduplicated. Chat transcripts share a great deal of surface form — greetings,
        # scheduling small talk — so a trigram-Jaccard filter removes genuinely distinct
        # conversations. This conclusion carried over from the previous version of the task, where
        # it was reached about the same corpus.
    ),

    build_prompts=scorer.build_prompts,
    extract_predictions=scorer.extract_predictions,
    score=scorer.score,
    metric_name="rouge_l",
    max_new_tokens=512,
    max_seq_length=2048,
    eval_batch_size=16,
    failure_category=scorer.failure_category_of,
    # No longer judged. The judge machinery is untouched and still serves `toolbench`; what
    # changed is that a summary's quality is now measured against three human references rather
    # than by asking a model its opinion.
    needs_judge=False,
    judge_overlap=False,
    attach_reasoning=True,

    # Selection is ROUGE-L alone: cheap, deterministic, and a sound ranking signal. The report
    # adds ROUGE-1, ROUGE-2 and BERTScore — ROUGE punishes a correct summary worded differently,
    # and BERTScore is what catches those. BERTScore is NOT the headline: its floor is high
    # (unrelated text scores ~0.87 with roberta-large) and model-dependent, so it discriminates
    # poorly alone and is not comparable across papers.
    report_load=None,
    report_score=scorer.score_report,
    report_metric_name="rouge_1_2_l_bertscore",

    build_training_turn=summarization_turn,

    # ADDED 2026-09-08, having first been declared unnecessary. Whether a summary is FAITHFUL is
    # indeed a judgement — but an audit of 1,161 synthetic rows found three defects that are
    # decidable and were going uncaught: fabricated extra "human references" (the count came out
    # {1: 877, 2: 127, 3: 157} for rows with one author), 76 rows whose trained target
    # `references[0]` disagreed with their stated `answer`, and 18 whose "summary" carried a
    # transcript turn label — the B250 continuation failure, in generated training data.
    #
    # It still cannot check faithfulness, so the teacher pass still runs.
    synth_verifier=_verify,
    # No chain-of-thought. A summary is not reached by reasoning steps, and prepending a
    # `<reasoning>` block to the target teaches the model to emit text that then gets scored as
    # part of the summary.
    cot_annotation=False,

    mining_sources=(
        MiningSource(
            hf_id="cylnlp/dialogsum",
            config=None,
            split="train",
            url="https://github.com/cylnlp/dialogsum",
            supports_offset=True,
        ),
    ),
    # SAMSum is deliberately NOT a mining source any more, and paid discovery is off so it cannot
    # come back by the side door: it is single-reference, so rows mined from it could not be
    # scored under this task's metric, and mixing them in would quietly return the training set to
    # the two-corpus blend the rework removed.
    allow_paid_discovery=False,

    model_ranking_metric=None,
)
