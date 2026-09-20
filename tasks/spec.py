"""`TaskSpec` — one benchmark task's complete, explicit behaviour.

WHY THIS EXISTS
    The pipeline used to route every benchmark through an abstract `task_type` channel
    (`classification`, `NER`, `function_call`, `generation`, ...). Eight concrete tasks shared five
    channels, so behaviour was decided by `if task_type == ...` chains — **47 of them** across eval,
    data, synthesis, orchestration and training. A task inherited whatever the chain's `else` branch
    happened to do, and three production bugs came from exactly that:

      * B291 — `function_call` matched neither synthesis branch, so `synthesize_examples` returned
        `[]`. Six rebuilds on xlam announced 250-500 rows and produced none, silently, and the exact
        verifiers written for that path had never executed.
      * B296 — every non-classification failure was reported as the constant confusion pair
        `gold_verifier -> incorrect`, and the orchestrator wrote pages of reasoning about it.
      * B299 — quality control's `else` returned the dataset untouched, so xlam and calendar were
        never filtered at all, and gsm8k/dialogsum filtered on a `"prompt"` key their rows lack.
        Four of eight tasks, no quality control, nothing logged.

    Meanwhile per-task behaviour that the channel could not express had already leaked into five
    ad-hoc side registries (`synth_verifiers._BY_BENCHMARK`, `label_space._LABEL_DEFINITIONS`,
    `_BENCHMARK_ALIASES`, `NAMED_BENCHMARK_TASK_TYPES`, the `_instruction` row field) — each added
    reactively after a bug.

THE RULE
    **No field on this dataclass has a default.** Python then refuses to construct a `TaskSpec` that
    does not state every decision, so "we never considered this for that task" becomes an import-time
    error instead of a silent runtime fallthrough three hours into a run. A task that genuinely wants
    nothing writes `synthesize=None` or `quality_controls=()` — an explicit choice a reader can see
    and a reviewer can challenge.

    `family` survives only as a descriptive tag for reports and model-selection hints. It must never
    be used as a dispatch key again; `tests/test_task_registry.py` enforces that.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields

from data.quality_controls import QCStep

# Descriptive groupings from the benchmark suite design. Reporting only.
CATEGORIES = ("in_distribution", "format_bound", "out_of_distribution")

# Descriptive shape tag. Used for report labelling and the orchestrator's model-selection hint —
# NEVER to choose behaviour. Every behavioural choice is a field on the spec.
FAMILIES = ("classification", "extraction", "generation", "structured_output")

EVAL_SAMPLING = ("label_balanced", "shuffled")


@dataclass(frozen=True)
class MiningSource:
    """A corpus that can supply MORE real rows for this task, mid-run.

    `offset_field` is what makes mining useful on a curated benchmark: the initial curriculum takes
    the first N rows, so a later `acquire` must start where that left off or it re-offers rows the
    pool already has. Without this, xlam's `acquire` could only pay Exa to rediscover mirrors of a
    corpus already sitting in the local cache while ~57,000 unused rows went unreachable (B297).
    """

    hf_id: str
    config: str | None
    split: str
    url: str
    supports_offset: bool


@dataclass(frozen=True)
class TaskSpec:
    """Everything the pipeline needs to know about one benchmark task.

    Field order groups by subsystem. None of them has a default — see the module docstring.
    """

    # ---- identity -------------------------------------------------------------------
    name: str
    """Registry key. Equals `SLM_BENCHMARK_TASK` and the slurm script suffix."""
    title: str
    """Human label for reports and charts."""
    category: str
    """One of CATEGORIES. Reporting only."""
    family: str
    """One of FAMILIES. Reporting and model-selection hints only — never dispatch."""

    # ---- data -----------------------------------------------------------------------
    load: Callable[..., tuple[list[dict], list[dict]]]
    """`load(max_train, max_test, log=...) -> (train_rows, eval_rows)`."""
    required_fields: tuple[str, ...]
    """Fields every row of this task must carry, checked at load and after mining."""
    initial_train_cap: int
    """Gold rows to request at cold start. The loader returns as many as it has, up to this.

    Not a fraction of anything. It used to be `curriculum_size_target × 0.65`, which is where the
    mystery 3,250 came from: a 5,000-row "target" the curriculum was then never allowed to reach.
    """
    select_cap: int
    """Maximum held-out rows for the IN-LOOP eval. The loader returns as many as it has, up to this.

    This number sizes CHECKPOINT SELECTION, not the published result, and the two want opposite
    things. It used to be called `eval_cap` and was doing both jobs at once, which made the 1,000
    indefensible in either direction: too small to report (the 95% CI half-width at n=1,000 and 80%
    accuracy is +/-2.5 points) and too large to want to pay for on every one of a run's iterations.

    1,000 is right for selection specifically because the comparison is PAIRED — the same fixed
    rows scored against successive checkpoints — so most of the sampling error is common-mode and
    cancels out of the ranking. It is not right for a number in a paper, and nothing here should
    ever publish it. `report_load` / `report_score` are the reporting half, run once by
    `scripts/report_eval.py`.
    """
    eval_sampling: str
    """One of EVAL_SAMPLING. `label_balanced` round-robins across classes."""
    closed_label_space: bool
    """True when `label` is a real class to predict and the vocabulary is pinned from the eval set."""
    label_definitions: Mapping[str, str]
    """What each class MEANS, for the teacher's prompts. Empty when the label is self-describing."""
    entity_type_vocabulary: tuple[str, ...]
    """The AUTHORITATIVE entity-type taxonomy, for span tasks. `()` when the task extracts no spans.

    The NER counterpart of `label_definitions`, and it exists because deriving this list from the
    rows in front of the teacher is only safe when the taxonomy is small. `_verifier_context_block`
    used to do exactly that, telling the teacher that the observed types were "the only ones that
    may appear". True for BC5CDR, whose two types both show up in any sample. False for MultiCoNER:
    a 40-row anchor sample contains 5 of its 33 types, so that sentence declared the other 28
    invalid, and on the 2026-09-09 synthesis audit the teacher duly rejected rows typed `OtherLOC`
    — a real type — with "OtherLOC is not a valid type for this task."

    A sample is evidence of what a type space CONTAINS and never evidence of its BOUNDARY. Only the
    task knows the boundary, so the task has to say it.
    """
    verifier_notes: str
    """CONVENTIONS the teacher must be told before it judges a generated row. `""` is explicit.

    A closed label space is not the only task-level fact a verifier needs. `calendar_json` encodes
    four conventions that are decidable, documented, and invisible in any single row — a 60-minute
    default duration, `tonight` meaning 20:00, a bare date meaning 09:00, and a date already past
    rolling forward a year. `verify_calendar_row` checks all four exactly, by re-resolving the
    request with the loader's own grammar.

    The teacher pass then ran AFTER that exact check and overruled it. Measured on run 38832587:
    22.6% of generated rows rejected (1,965 of 8,704), including
    `"7pm start plus 60 mins is 8pm, not 20:00"` — 8pm IS 20:00 — and six rejections of the
    year-rollover the gold itself uses. Every one of those rows had already passed the programmatic
    verifier. A model asked to judge a convention nobody told it about invents one, which is B267/
    B269/B314 one level up from the tools list.

    So this is the task-level counterpart of `label_definitions`: state the conventions, or the
    teacher will make some up.
    """
    quality_controls: tuple[QCStep, ...]
    """Ordered QC steps. `()` is a legal, explicit choice."""

    # ---- eval -----------------------------------------------------------------------
    build_prompts: Callable[[object], list[str]]
    extract_predictions: Callable[[list[str], object], Sequence]
    score: Callable[[object, Sequence], dict]
    metric_name: str
    """What the comparison scalar actually measures. Per TASK, so binary and multi-class
    classification are not both mislabelled `macro_f1`."""
    max_new_tokens: int
    max_seq_length: int
    eval_batch_size: int
    failure_category: Callable[[dict], str] | None
    """Maps a failure record to an actionable category for the orchestrator's confusion pairs.
    `None` means this task reports failures without a taxonomy."""
    needs_judge: bool
    """True when scoring calls the LLM judge, so a judge outage must fail loudly not score zero."""
    judge_overlap: bool
    """Overlap judging of finished chunks with the next generation batch."""
    attach_reasoning: bool
    """Record the model's `<reasoning>` block on failure records."""

    # ---- reporting ------------------------------------------------------------------
    # The eval above runs EVERY iteration and exists to rank checkpoints. These three run ONCE,
    # from `scripts/report_eval.py`, and produce the number that gets published. Separating them
    # is what makes both defensible: selection can stay small and cheap because it is a paired
    # comparison, and the report can be large and slow because it happens once.
    report_load: Callable[..., list[dict]] | None
    """`report_load(log=...) -> rows` for the FULL report split. `None` reuses `load`'s eval rows.

    Not merely a bigger `select_cap`: for two tasks the report is a DIFFERENT SPLIT, not a larger
    draw from the same one. `multiconer` selects on the official 871-row dev — which cannot support
    a 33-class macro-F1 at all — and reports on a fixed stratified slice of the 249,980-row test.
    A single `load` cannot express that, so a task that needs it says so here.
    """
    report_score: Callable[[object, Sequence], dict]
    """The scorer for the report pass. Often the same callable as `score`; deliberately separate.

    Selection and reporting metrics SHOULD differ where the honest headline is unusable as a
    per-iteration signal:
      * `multiconer` — micro-F1 selects, macro-F1 reports. A 1,000-row draw can contain zero
        examples of a class that is 0.18% of entities, which makes macro-F1 undefined or wildly
        noisy as a ranking signal while remaining the right thing to publish.
      * `goemotions` — Ekman-7 macro-F1 selects, threshold-free macro AUPRC reports. Macro-F1 over
        28 labels is a thresholding artifact: the same model swings several points between a fixed
        0.5, a fixed 0.3, and a dev-tuned sweep.
    Where they genuinely coincide a task names the same function twice, which is an explicit
    statement that the question was asked rather than a default nobody chose.
    """
    report_metric_name: str
    """What the reported number is. Equals `metric_name` when the report metric is the same."""

    # ---- training -------------------------------------------------------------------
    build_training_turn: Callable[[dict, object], tuple[str, str]]
    """`(row, spec) -> (prompt, target)` for completion-only SFT. Must produce the SAME prompt the
    eval harness sends, or training teaches a prefix inference never supplies (B290)."""

    # ---- synthesis ------------------------------------------------------------------
    synth_verifier: Callable[[dict], bool] | None
    """EXACT programmatic check on a generated row, run before any teacher call.

    Set for the format-bound tasks, where correctness of FORM is decidable by computation: parse the
    output, confirm it targets a declared tool with arguments its schema has, confirm spans are
    real substrings of the text. `None` for the rest, where the only available check is the
    teacher's judgement of its own output — which is weaker, and is why this field is required
    rather than defaulted, so the weakness is visible at the point the decision was made.
    """
    cot_annotation: bool
    """Annotate rows with chain-of-thought before training."""

    # ---- mining ---------------------------------------------------------------------
    mining_sources: tuple[MiningSource, ...]
    """Canonical corpora that can supply more real rows. `()` means paid discovery only."""
    allow_paid_discovery: bool
    """Whether `acquire` may fall through to Exa + orchestrator dataset discovery."""

    # ---- model selection ------------------------------------------------------------
    model_ranking_metric: str | None
    """Published benchmark used to rank candidate models for this task, or `None` for no ranking."""

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"task name must be an identifier-like string, got {self.name!r}")
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.name}: category must be one of {CATEGORIES}")
        if self.family not in FAMILIES:
            raise ValueError(f"{self.name}: family must be one of {FAMILIES}")
        if self.eval_sampling not in EVAL_SAMPLING:
            raise ValueError(f"{self.name}: eval_sampling must be one of {EVAL_SAMPLING}")
        if not self.required_fields:
            raise ValueError(f"{self.name}: required_fields must not be empty")
        for number in ("initial_train_cap", "select_cap", "max_new_tokens", "max_seq_length",
                       "eval_batch_size"):
            if int(getattr(self, number)) < 1:
                raise ValueError(f"{self.name}: {number} must be positive")
        if self.max_new_tokens >= self.max_seq_length:
            raise ValueError(
                f"{self.name}: max_new_tokens ({self.max_new_tokens}) leaves no prompt budget "
                f"inside max_seq_length ({self.max_seq_length})"
            )
        if not self.metric_name:
            raise ValueError(f"{self.name}: metric_name must be a non-empty string")
        if not self.report_metric_name:
            raise ValueError(f"{self.name}: report_metric_name must be a non-empty string")
        for callable_field in ("load", "build_prompts", "extract_predictions", "score",
                               "build_training_turn", "report_score"):
            if not callable(getattr(self, callable_field)):
                raise ValueError(f"{self.name}: {callable_field} must be callable")
        if self.report_load is not None and not callable(self.report_load):
            raise ValueError(f"{self.name}: report_load must be callable or None")
        if self.label_definitions and not self.closed_label_space:
            raise ValueError(
                f"{self.name}: label_definitions only mean something for a closed label space"
            )
        if len(set(self.entity_type_vocabulary)) != len(self.entity_type_vocabulary):
            raise ValueError(f"{self.name}: entity_type_vocabulary repeats a type")

    def qc_context_labels(self, eval_set) -> set[str] | None:
        """The closed class vocabulary for this task, read from the frozen eval set."""
        if not self.closed_label_space:
            return None
        rows = getattr(eval_set, "all", None) or []
        labels = {
            str(row.get("label")) for row in rows
            if isinstance(row, dict) and row.get("label") is not None
        }
        return labels or None


def spec_field_names() -> tuple[str, ...]:
    """Every field a task must declare. Used by the registry completeness test."""
    return tuple(f.name for f in fields(TaskSpec))
