# Long-run prompt audit

Effective date: **2026-07-21**

This is a code-referenced inventory of prompt text reachable from the cold-start and
production graphs, including orchestrator, teacher/local-Qwen, judge, train/eval SLM, and
production inference prompts. The audit is static: it made no model API calls.

“Active” means reachable in a configured runtime path, including optional fallbacks. Symbols
are included with paths so this remains useful when line numbers move.

## 1. Orchestrator prompts

### 1.1 Task analysis and run planning

- **Code:** `agent/task_planner.py::_PLANNER_PROMPT`, formatted by `plan_task`.
- **Purpose:** classify the requested task, choose labels/flags/search queries, propose
  curriculum and eval sizes, and calibrate the stop threshold.
- **Inputs:** free-text task description, parameter range, and one deduplicated summary per
  pool model. Capability values now retain explicit metric names, missing values say
  `not reported`, and source URLs are included.
- **Expected output:** one JSON object with task type/name, labels, flags, Exa queries,
  benchmark, stop threshold, data sizes, and rationale.
- **Parser/validation:** `_extract_json` accepts a whole JSON reply or the first greedy
  `{...}` block. `plan_task` validates only `task_type`, supplies defaults for missing keys,
  and defers size clamping to
  `agent/nodes/cold_start/task_analysis.py::_apply_data_targets`.
- **Mode/backend:** autonomous cold start; `ORCHESTRATOR_MODEL`.
- **Critique:** the contract is detailed, but a single call controls task schema, data
  acquisition, and the termination target. Numeric fields, labels, and query-map shape are
  not schema-validated. The prompt still asks the model to use its leaderboard knowledge,
  which may be stale even though pool metrics are now sourced.
- **Recommended improvement:** validate with a typed JSON schema; split task/schema
  inference from threshold calibration; derive known-benchmark thresholds from a versioned
  local registry and ask the LLM only for an adjustment with cited evidence.

### 1.2 Hardware resolution

- **Code:** `agent/nodes/cold_start/hardware_research.py::_PROMPT`, formatted by
  `research_device`.
- **Purpose:** convert a device description plus local-DB or Exa snippets into usable model
  RAM, model-file storage budget, and a reference chip.
- **Inputs:** user description, up to three device records/snippets, and allowed
  `REFERENCE_CHIPS`.
- **Expected output:** strict JSON with device/chipset/total capacities, usable RAM,
  storage budget, reference chip, and rationale.
- **Parser/validation:** `_extract_json` parses a whole reply or greedy object.
  `reference_chip` is allow-listed; numeric fields are cast with defaults. There is no
  cross-check that usable RAM is below total RAM or that storage is physically plausible.
- **Mode/backend:** cold start; local DB first, Exa fallback, then `ORCHESTRATOR_MODEL`.
- **Critique:** deterministic records are routed through an LLM for arithmetic that code
  could perform. Prompt-injected web snippets could influence output.
- **Recommended improvement:** compute budgets deterministically for complete DB rows,
  sanitize/quote external snippets, and add consistency/range validation before accepting
  LLM-resolved values.

### 1.3 Dataset schema mapping

- **Code:** `data/loaders/web_acquire.py::_llm_map_dataset`.
- **Purpose:** decide whether a Hugging Face dataset fits and map its columns/splits to the
  internal schema.
- **Inputs:** task plan, repository/config, advertised splits/columns, and one sample row.
- **Expected output:** task-specific mapping JSON or `{"suitable": false}`.
- **Parser/validation:** extracts the first greedy JSON object and requires one of
  `text_col`, `question_col`, or `tokens_col`. Later materialization accesses the proposed
  names but there is no complete preflight schema check.
- **Mode/backend:** cold-start acquisition fallback; `ORCHESTRATOR_MODEL`.
- **Critique:** one row may not expose nullable or alternate schemas, and repository data
  is untrusted prompt content. Suitability and mapping are conflated.
- **Recommended improvement:** validate every proposed split/column against dataset
  metadata, inspect several bounded rows, and separate suitability classification from a
  deterministic mapping validator.

### 1.4 Last-resort seed synthesis

- **Code:** `data/loaders/web_acquire.py::synthesize_seed_examples`.
- **Purpose:** generate minimally viable training examples only when real/local/web data
  cannot meet the floor.
- **Inputs:** task name/type, allowed labels/entity types, and requested count.
- **Expected output:** strict `{"examples": [...]}` JSON using a task-family schema.
- **Parser/validation:** extracts one object; deduplicates text; checks classification
  labels, exact-substring NER spans, or non-empty generation answers. It does not verify
  factual correctness.
- **Mode/backend:** cold-start emergency fallback; `ORCHESTRATOR_MODEL`, temperature 1.0.
- **Critique:** the same model plans the task and fabricates “gold,” creating correlated
  errors. A requested large count is placed in one response and may truncate.
- **Recommended improvement:** generate in bounded batches, require independent
  verification, retain per-example provenance/confidence, and exclude unverifiable
  synthetic gold from held-out evaluation.

### 1.5 Web-acquired NER annotation

- **Code:** `data/loaders/web_acquire.py::_annotate_ner_entities`.
- **Purpose:** label entity spans in unannotated web passages.
- **Inputs:** the first 500 characters of each passage.
- **Expected output:** JSON list of exact-span/type objects, or `[]`.
- **Parser/validation:** greedy list extraction and presence checks for `text`/`type`.
  Contrary to the prompt, returned spans are not rechecked as exact substrings here and
  types are not allow-listed.
- **Mode/backend:** cold-start acquisition; one `ORCHESTRATOR_MODEL` call per passage.
- **Critique:** silent parse/call failure becomes an empty-entity gold example, which is
  indistinguishable from a genuine negative. Truncation can remove entity context.
- **Recommended improvement:** validate spans/types, mark annotation failures instead of
  converting them to negatives, batch requests, and retry only malformed items.

### 1.6 Initial orchestrator model choice

- **Code:** `agent/nodes/cold_start/model_selection/orchestrator_choice.py::_CHOOSE_SYSTEM`
  and `orchestrator_choice_node`.
- **Purpose:** choose the least-resource candidate that can plausibly meet the goal after
  LoRA.
- **Inputs:** task description/type/schema, goal, hardware budget, sourced
  `capability_sections`, exact variant selectors, quant/size/peak/decode/notes, and
  measurements carrying metric/artifact/mode/protocol/source.
- **Expected output:** strict JSON with exact `selector` and one-sentence reason.
- **Parser/validation:** `_parse_choice` accepts JSON, a greedy object, or bare text; the
  selector must match a candidate. Failure/unknown output selects the lowest-peak-RAM
  feasible variant.
- **Mode/backend:** cold start only when strategy is `orchestrator_choice`;
  `ORCHESTRATOR_MODEL`.
- **Critique:** exact selector ambiguity is resolved. Legacy bare IDs remain accepted with a
  documented lowest-peak-RAM default. Failure/unknown output now uses the same
  lowest-peak-RAM resource-safe rule.
- **Recommended improvement:** calibrate whether that conservative fallback can meet the
  goal without adding another model call.

### 1.7 Upward escalation and downward candidate choice

- **Code:** `agent/nodes/escalate.py::_CHOOSE_SYSTEM` and `_llm_choose_model`; called by
  `escalate_node` and `agent/nodes/downward_probe.py::downward_probe_node`.
- **Purpose:** pick a candidate within the next higher or lower peak-RAM tier.
- **Inputs:** task type/name/labels, current best score, quant/size/peak/notes, explicitly
  named optional metrics, and sourced `capability_sections`.
- **Expected output:** exact deployment selector plus reason in strict JSON.
- **Parser/validation:** same tolerant JSON/bare-ID parsing and membership check. Up/down
  direction and target tier are explicit; failure chooses the lowest-peak-RAM exact variant
  already inside that direction-appropriate tier.
- **Mode/backend:** shared cold-start loop; `ORCHESTRATOR_MODEL`. This is the capability
  injection point for both escalation and downward model choice.
- **Critique:** exact quant siblings and resource-safe direction-aware fallbacks are
  resolved. Capability prediction remains qualitative when comparable evidence is absent.
- **Recommended improvement:** add measured task-local evidence when available.

### 1.8 Downward re-exploration decision

- **Code:** `agent/nodes/downward_probe.py::_should_reexplore_downward`.
- **Purpose:** decide whether the resource savings justify probing another lower tier
  after convergence.
- **Inputs:** winning score, goal/margin, iterations, number of models/tiers tried, and
  number of untried lower tiers.
- **Expected output:** `{"reexplore": boolean, "reason": string}`.
- **Parser/validation:** extracts a greedy object and applies `bool(...)` to `reexplore`.
  Any failure falls back to `margin >= 0.03`.
- **Mode/backend:** post-convergence path for `interpolation` and
  `orchestrator_choice`; `ORCHESTRATOR_MODEL`.
- **Critique:** JSON string `"false"` would become true; the prompt omits estimated probe
  cost and potential RAM savings, so the decision is not actually cost-aware.
- **Recommended improvement:** require a real JSON boolean, include candidate resource
  deltas and remaining wall-clock budget, or replace this call with an explicit policy.

### 1.9 Iteration/EXPAND decision

- **Code:** `agent/nodes/iterate.py::_ITERATE_SYSTEM` and `_llm_iterate`.
- **Purpose:** diagnose the score trajectory and choose `data_rebuild`,
  or `hyperparameter`, with a bounded rebuild plan or LoRA settings.
- **Inputs:** compacted curation trajectory, model/iteration/scores/threshold, test-agent
  aggregate difficulty/confusion report, tried hyperparameters and rebuild identities,
  prior hypothesis, source novelty/yield, and remaining budgets. Raw eval rows are excluded.
- **Expected output:** one decision JSON object with intervention, hypothesis,
  hyperparameters or a declarative rebuild plan, and optional threshold adjustment.
- **Parser/validation:** `_parse_decision_json` handles text blocks, fences, embedded JSON,
  and Python literal dicts. `iterate_node` validates intervention membership; training
  later clamps/coerces only part of the hyperparameter schema. Threshold can only decrease
  to the immutable initial floor, making the prompt’s “lower” path effectively inert when
  current threshold equals that floor.
- **Mode/backend:** shared loop; one tool-free `ORCHESTRATOR_MODEL` call. It receives only
  the bounded trajectory and aggregate report already assembled by the node.
- **Validation:** strict top-level/nested JSON, intervention-specific payload exclusivity,
  bounded scalar types, finite threshold adjustment, recursive held-out-text rejection,
  and deterministic plan identities.
- **Cost bound:** one tracked `iterate` call plus at most one tracked
  `iterate_json_reask`; tool-use blocks are never executed or reflected back.
- **Remaining critique:** the broad score-band guidance can still anchor diagnosis, and
  threshold semantics remain constrained by the immutable calibrated floor.

### 1.10 Tool-free JSON re-ask

- **Code:** `agent/nodes/iterate.py::_reask_json_only`.
- **Purpose:** recover a final decision after prose, malformed JSON, or an attempted
  tool-use block.
- **Inputs:** only the original bounded context, followed by a short “JSON only, no tools”
  instruction. The malformed first response is not replayed.
- **Expected output/parser:** the same iteration JSON, parsed by `_parse_decision_json`.
- **Mode/backend:** fallback inside shared iterate flow; fresh non-tool-bound
  `ORCHESTRATOR_MODEL`.
- **Critique:** this remains a second paid call when the first response is malformed.
- **Recommended improvement:** enforce provider-side structured output when available.

## 2. Teacher and local-Qwen prompts

`data/synth_client.py::get_generate_fn` sends all local-Qwen prompts in non-thinking mode
with `top_p=0.8`, `top_k=20`, caller-supplied temperature, and caller-supplied token limit.
`agent/nodes/curate.py` uses local `SYNTH_MODEL` first and does not fall back to Claude for
hard negatives. CoT can fall back to task-routed DeepSeek/OpenAI models.

### 2.1 CoT/implementation-reasoning annotation

- **Code:** `data/curriculum.py::annotate_cot::_build_prompt`.
- **Purpose:** add reasoning to generation-family gold examples that lack existing CoT.
- **Inputs:** problem/prompt and correct answer/solution.
- **Expected output:** reasoning steps only. Code tasks request an implementation plan
  without code; other tasks request step-by-step reasoning without the final answer.
- **Parser/validation:** non-empty stripped text is accepted verbatim as `cot_reasoning`.
  Existing CoT is preserved. Backend failures advance local Qwen → configured cloud
  fallbacks; total failure leaves the example unchanged.
- **Mode/backend:** curate for math/code/generation; local Qwen3.6 non-thinking first,
  then DeepSeek thinking for math/science or GPT-first for code/general.
- **Critique:** no check that reasoning is consistent with the supplied gold, does not leak
  the final answer, or fits the configured 4096-token training context after formatting.
- **Recommended improvement:** verify answer consistency, reject final-answer leakage when
  requested, and preserve backend/model/prompt-version metadata per annotation.

### 2.2 Classification hard-negative generation

- **Code:** `data/curriculum.py::synthesize_hard_negatives`, classification branch.
- **Purpose:** create text that resembles a source class but belongs to another class;
  the positive-synthesis strategy includes an aggregate pattern hint.
- **Inputs:** source text/label, first alternate target label, aggregate confusion hint.
- **Expected output:** raw text only for the target class.
- **Parser/validation:** any non-empty text is assigned the target label by code; later
  quality controls deduplicate and cap label imbalance. The 2-for-1 result includes a real
  source anchor and a generated row; curate reports `n_hard_source` and
  `n_hard_generated` separately.
- **Mode/backend:** classification data rebuild; local Qwen; legacy direct
  Claude path exists only when a caller supplies no `generate_fn`.
- **Critique:** semantic membership in the target class is not verified. For multiclass
  tasks, choosing the first alternate label can overproduce one class.
- **Recommended improvement:** choose targets from the observed confusion pair, require an
  independent label verifier, and reject near-copies before adding them as gold targets.

### 2.3 NER hard-example generation

- **Code:** `data/curriculum.py::synthesize_hard_negatives`, NER branch.
- **Purpose:** rewrite a passage into a more ambiguous context while preserving correct
  entity types.
- **Inputs:** original passage and up to five entity text/type pairs.
- **Expected output:** JSON object containing rewritten text and typed entity spans.
- **Parser/validation:** greedy object parsing followed by strict validation. Malformed
  JSON, empty entity lists, spans absent from rewritten text, and types absent from the
  source row are discarded. Source anchors and generated rewrites are accounted separately.
- **Mode/backend:** NER positive-synthesis data rebuild; local Qwen.
- **Critique:** exact span/type integrity is checked, but no independent semantic verifier
  confirms that the rewritten context preserves the intended aggregate difficulty.
- **Recommended improvement:** add a task-level semantic verifier beyond exact span/type
  integrity.

### 2.4 Open-generation augmentation policy

- **Code:** `data/curriculum.py::synthesize_hard_negatives`, generation branch.
- **Purpose/output:** no rejected-answer prompt is sent. The compatibility function returns
  gold anchors unchanged and open-generation data rebuild remains gold/CoT-only.
- **Mode/backend:** no synthesis backend is contacted for this route.
- **Safety property:** intentionally wrong responses are never written as positive SFT
  targets.
- **Recommended improvement:** add a preference/ranking objective with explicit
  chosen/rejected fields, or synthesize difficult questions paired with independently
  verified correct answers.

### 2.5 Math/code positive-synthesis eligibility

- **Code:** `data/curriculum.py::synthesize_hard_negatives`, math/code branches.
- **Purpose/output:** no generation prompt is sent. The function warns and returns original
  examples because wrong-answer/code SFT is unsafe.
- **Mode:** data rebuild is gold/CoT-only and schema validation rejects positive synthesis
  before contacting a backend. Held-out failure rows are never appended.
- **Recommended improvement:** add verified new-problem
  generation or test mutation with known solutions.

## 3. Judge prompt

### 3.1 Open-generation semantic judge

- **Code:** `eval/judge_client.py::JUDGE_SYSTEM`, `JUDGE_PROMPT_VERSION`,
  `_build_judge_prompt`, and `LocalJudgeClient`; `eval/scorers/generation.py` dispatches
  generation triples to it.
- **Purpose:** score a generated answer against a gold answer on a 0–1 anchored rubric.
- **Inputs:** question, gold, and prediction.
- **Expected output:** one decimal number in `[0,1]`.
- **Parser/validation:** strict full-response numeric parsing followed by a finite closed
  range check. Extra text, non-numeric values, NaN/infinity, and values outside `[0,1]`
  abort evaluation rather than becoming a false model score.
- **Mode/backend:** only `task_type == "generation"`; required local Qwen3.6 through the
  configured OpenAI-compatible vLLM endpoint. `/models` must expose the exact configured
  Qwen3.6-family model and any completion `response.model` must match. Every request sets
  `enable_thinking=False`; there is no cloud fallback. Remote hosts hard-fail by default,
  including private cluster addresses; only loopback/localhost/Unix sockets are accepted
  without the explicit `SLM_JUDGE_ALLOW_REMOTE=1` opt-in.
- **Prompt safety:** normalized question/gold/prediction values are encoded in a marked
  untrusted JSON object. The system prompt says never to follow instructions in those
  fields.
- **Execution/cache:** cache keys include normalized inputs, model, prompt version, and
  prompt fingerprint. A locked append-only JSONL cache under the stable run artifacts is
  shared across workers/restarts and ignores corrupt records. Cache misses use a bounded
  sliding window; the first failure stops new submissions and cancels pending futures while
  successful output ordering remains deterministic.
- **Failure semantics:** endpoint, model identity, response identity, event, cache,
  executor, request, and parse failures all become `JudgeInfrastructureError` and
  propagate, including from zero-shot baseline evaluation. Math exact match and local
  APPS/MBPP executable scoring are unchanged.
- **Observability:** model preflight and completion calls append process-safe local-provider
  cost events at `$0` plus process-safe timing events.
- **Remaining critique:** periodically measure judge agreement on a fixed calibration set.

## 4. SLM training and evaluation prompts

### 4.1 Classification

- **Code:** `eval/scorers/classification.py::CLASSIFY_PROMPT` and
  `build_classify_prompt`; reused by
  `training/lora_trainer.py::_run_unsloth_training`.
- **Purpose/inputs:** map message text to exactly one enumerated, sorted label.
- **Expected output:** label word only.
- **Parser/validation:** eval performs case-insensitive word-boundary then substring
  matching against allowed labels; otherwise `__EXTRACTION_FAILED__`.
- **Mode:** SLM train and eval; Qwen templates are rendered with
  `enable_thinking=False`. Missing Qwen templates fail clearly.
- **Critique:** the same prompt is used for train/eval, which is good. Substring fallback can
  accept a label embedded in unwanted prose or another token.
- **Recommended improvement:** use constrained decoding or exact normalized output before a
  carefully delimited fallback; add an explicit input delimiter resistant to instruction
  text inside the message.

### 4.2 NER

- **Code:** eval prompt `eval/scorers/ner.py::NER_PROMPT`; training prompt duplicated in
  `training/lora_trainer.py`, NER `format_example`.
- **Purpose/inputs:** extract typed spans from text as a JSON list.
- **Expected output:** list of `{"text","type"}`; eval explicitly says `[]` for none.
- **Parser/validation:** whole-list JSON or greedy list extraction; retains objects with both
  keys. No exact-substring/type allow-list validation at prediction extraction.
- **Mode:** SLM train/eval.
- **Critique:** train and eval strings are not identical: training omits the explicit `[]`
  instruction and uses slightly different punctuation. Entity type vocabulary is not listed.
- **Recommended improvement:** share one prompt builder for train/eval, enumerate allowed
  types/schema, and validate exact spans.

### 4.3 General generation and math

- **Code:** eval `eval/scorers/generation.py::GENERATE_PROMPT`; training generation branch
  in `training/lora_trainer.py`.
- **Purpose/inputs:** answer a question/problem; training may prepend `<reasoning>` content
  to the assistant target.
- **Expected output/parser:** general generation is free text judged by an LLM; math extracts
  explicit final-answer markers or the last number for exact match.
- **Mode:** SLM train/eval.
- **Critique:** eval prepends `Answer the following question`, while training sends the raw
  prompt as the user turn. Math has no explicit final-answer format request even though its
  parser depends on one.
- **Recommended improvement:** centralize task-specific builders and require a stable math
  final-answer marker while keeping reasoning optional and parseable.

### 4.4 Code generation

- **Code:** eval `eval/scorers/generation.py::CODE_GENERATE_PROMPT` and
  `build_code_prompt`; reused by the code branch in `training/lora_trainer.py`.
- **Purpose/inputs:** produce executable Python for APPS introductory, preserving starter
  code, required signature/entry-point metadata, and call-based versus stdin/stdout
  instructions without exposing the gold body.
- **Expected output:** code only, no fences/explanation.
- **Parser/validation:** strips Python/empty Markdown fences, then executes every preserved
  APPS case through an isolated candidate worker with a per-case timeout and a bounded
  per-problem total deadline. The trusted controller retains expected outputs and records
  cases executed/total; MBPP assertions remain the CI smoke.
- **Mode:** SLM train/eval.
- **Critique:** train/eval prompt parity is explicit, and optional code reasoning is rendered
  as executable Python comments. Candidate workers inherit no success FD or expected-output
  payload. The process boundary and limits reduce accidental damage but are not seccomp or a
  hostile-code sandbox.
- **Recommended improvement:** keep full executable-test validation authoritative.

### 4.5 Inference chat wrapping

- **Code:** `training/slm_helpers.py::infer` and `infer_batch_gguf`.
- **Purpose:** apply the model/GGUF chat template around each already-built task prompt.
- **Inputs/expected output:** one user turn; assistant generation continuation.
- **Parser/validation:** delegated to the task scorer. HF inference is greedy and passes
  `enable_thinking=False`. GGUF uses `chat_template_kwargs` when supported. The installed
  llama-cpp-python 0.3.34 signature lacks it. Hybrid Qwen3/Qwen3.5 use the verified
  empty-think ChatML prefix; non-thinking-only Qwen3-4B-Instruct-2507 uses its plain
  assistant prefix. Unknown templates fail instead of silently changing mode.
- **Mode:** all SLM eval and production re-inference.
- **Quant identity:** Q4/Q8 baselines and interpolation/downward probes build/reuse their
  exact GGUF and pass `gguf_path`; histories retain `model_id@quant` selectors.
- **Critique:** mode mixing is resolved, but the manual Qwen fallback depends on the official
  ChatML template contract and should be revisited when llama-cpp-python is upgraded.
- **Recommended improvement:** record the template path/version with eval artifacts and
  remove the manual path once the deployed llama-cpp-python accepts template kwargs.

## 5. Production prompts

### 5.1 Failure taxonomy

- **Code:** `agent/nodes/production/taxonomy.py::taxonomy_construct_node` inline system and
  user prompts.
- **Purpose:** cluster up to 40 failed traces and label each cluster fixable by training data
  or external.
- **Inputs:** total/sample count and full sampled trace JSON with stable indices.
- **Expected output:** strict JSON with clusters, counts, root causes, boolean fixability,
  trace indices, and summary.
- **Parser/validation:** greedy object parse or raw-summary fallback. Code does not enforce
  unique/complete indices, recompute counts, type-check `fixable`, or validate cluster names.
- **Mode/backend:** production graph; `ORCHESTRATOR_MODEL`.
- **Critique:** raw deployed inputs/outputs are untrusted prompt content. The prompt demands
  a partition but the implementation accepts overlaps and omissions.
- **Recommended improvement:** schema-validate, deterministically repair/reject invalid
  partitions, recompute counts from indices, and delimit traces as untrusted data.

### 5.2 Live confirmation prompt

- **Code:** `agent/nodes/production/live_confirm.py::live_confirm_node`.
- **Purpose:** re-run each prescreened failure through deployed model M0.
- **Inputs/expected output:** sends `trace["input"]` directly, without a task prompt; compares
  stripped model output to `corrected_output` by exact string.
- **Parser/validation:** no task-specific parser. Inference errors are conservatively counted
  as confirmed failures.
- **Mode/backend:** production graph; deployed SLM.
- **Critique:** this may not reproduce the original serving prompt/template and exact string
  equality is wrong for many classification, NER, and generation outputs.
- **Recommended improvement:** store original rendered prompt and task/parser metadata in
  traces, then replay through the same scorer and distinguish infrastructure errors.

### 5.3 Production training prompts

- **Code:** production graph in `agent/graph.py::build_graph` routes confirmed examples through
  shared `curate`, `train`, `evaluate`, and `iterate` nodes.
- **Purpose/contract:** uses the same SLM and teacher prompts audited above.
- **Parser/validation:** task-specific shared paths, provided `eval_set` and task metadata are
  present.
- **Critique:** `curate_node` explicitly rejects production `data_rebuild` when `eval_set` is
  `None`, while the production graph itself does not create an eval set. Production startup
  therefore depends on caller-populated state not expressed as a prompt contract.
- **Recommended improvement:** define and validate a production-state schema at graph entry,
  including task type, eval/regression sets, deployed model, and existing dataset.

## 6. Present but not active

- **`agent/tools/delegate_task.py::delegate_task`:** contains a generic no-tools sub-agent
  system prompt and forwards an arbitrary task description, but there are no production
  call sites in this repository as of the effective date.
- **Legacy Claude hard-negative backend:** the fallback inside
  `data/curriculum.py::synthesize_hard_negatives` is reachable only to external/legacy callers
  that omit `generate_fn`; `curate_node` supplies local Qwen or skips synthesis.
- **Exa queries:** dynamic search strings and `use_autoprompt=True` are search inputs, not
  orchestrator/teacher/judge/SLM generation prompts, so they are outside this prompt inventory.

## 7. Highest-priority prompt improvements

1. Unify train/eval builders for NER, general/math generation, and code generation.
2. Add typed JSON-schema validation to planner, hardware, iterate, downward, acquisition,
   and taxonomy outputs.
3. Add verified-positive or preference-objective augmentation for open generation.
4. Reproduce the original production serving prompt/parser during live confirmation.
5. Treat external dataset rows, web snippets, eval failures, and production traces as
   untrusted data when tool-enabled or decision prompts consume them.
