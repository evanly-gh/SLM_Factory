# SLM Factory — Phase 1 Design Document

**Date:** 2026-06-19
**Scope:** Phase 1 (Weeks 1–4) — Cold-start agentic fine-tuning loop, no hardware gating yet
**Reference:** Pioneer Agent (arXiv:2604.09791, Fastino Labs, April 2026)
**Goal:** Replicate Pioneer Agent's cold-start loop with the model search space bounded to Android-deployable sizes, demonstrably improving accuracy over 3–4 iterations. Phase 1 uses SMS Spam as the first concrete task to prove the loop works; the system is designed to generalize across all three task types from day one.

---

## 1. System Overview & Goals

SLM Factory is an agentic fine-tuning loop that replicates Pioneer Agent's architecture, with the model search space constrained to Android-deployable sizes (≤~1.5GB INT4). The overarching goal is to optimize accuracy under user-specified hardware constraints (latency, power, storage, memory) that define the feasibility boundary for on-device Android deployment.

**Phase 1 goals (Weeks 1–4):**

- Weeks 1–2: One working fine-tune-and-eval pass on SMS Spam (classification) with a base model from the Android pool. Prove the pipeline connects end-to-end.
- Weeks 3–4: Add the iteration loop. Agent inspects failures, generates targeted training data, retrains, re-evaluates. 3–4 rounds. No hardware gating yet.
- Week 4 checkpoint: a fine-tuned model on one task, demonstrably improving across ≥3 consecutive rounds.

**Task abstraction — three task types:**

The system classifies every task into one of three types. This decision, made in `task_analysis`, routes the rest of the pipeline to type-appropriate logic:

| Task type | Paper benchmarks covered | Supervision format | Eval metric | Curation strategy |
|---|---|---|---|---|
| `classification` | SMS Spam, CLINC150 | direct labels | accuracy / F1 | hard negatives at label boundaries |
| `NER` | CoNLL-2003 | direct labels | entity F1 | entity diversification, no entity value >2–3× |
| `generation` | ARC-Challenge, GSM8K, TriviaQA, HumanEval, XSum, SAMSum | chain-of-thought | LLM-as-judge [0,1] | teacher model annotation (DeepSeek-R1 / GPT-4.1) |

Phase 1 proves the loop with SMS Spam (`classification`). The same code runs any of the nine paper benchmarks by passing a different `description` — no restructuring required.

**Phase 1 first task:** SMS Spam binary classification (UCI SMS Spam Collection). Cold-start mode — agent downloads the dataset, builds the eval set, trains from scratch. We use decoder models from the Android pool (the paper used GLiNER2 encoder, which is not in our Android pool).

**Success criteria:**
- Metric improves across ≥3 consecutive rounds on the Phase 1 task
- At least one rollback occurs and is handled correctly (demonstrates the gate works)
- Full run completes in <8 hours at <$50 total cost
- All selected models fit within the Android pool bounds
- `data-curation.md` provides a complete lineage trace of every intervention
- Passing a different `description` (e.g., "fine-tune on ARC-Challenge reasoning") routes correctly through the same graph without code changes

**Explicitly out of scope for Phase 1:**
- On-device hardware gating (latency/power/memory thresholds as hard constraints)
- Quantization / ONNX / QNN export pipelines
- Production mode (judged inference traces, regression gate, trace database)
- Multi-task fine-tuning
- HRM-Text on-device deployment (custom runtime not upstream in llama.cpp)
- AdaptFT-Bench noise injection evaluation

---

## 2. Agent Architecture

### 2.1 Orchestrator

LangGraph state machine with Claude Sonnet 4.6 (1M context window, 32K extended thinking budget). Runs in an isolated environment with pre-loaded helpers. Up to 1,500 LangGraph turns for cold-start mode. Each turn = one LLM reasoning step + associated tool calls.

### 2.2 Hierarchical Agent Structure

**Main agent** — orchestrates the full pipeline across all state machine nodes.

**Independent sub-agents** via `delegate_task` — spawned for parallel work (e.g., synthesize next dataset while a training job runs). Sub-agents run the same Claude Sonnet 4.6 model, capped at 1,000 turns. Sub-agents share the filesystem with the main agent — they write summaries to disk; the main agent reads those files without loading raw data into its own context window. This is the core context isolation pattern.

**Trace Analyzer sub-agent** — production mode only, not used in Phase 1.

**Context Manager** — monitors conversation state and selectively compacts older turns while preserving key decisions, evaluation results, and dataset lineage. The durable complement is `data-curation.md`.

**`data-curation.md`** — a structured markdown file written to disk every iteration. Records dataset versions, composition ratios, quality-control decisions, and per-iteration evaluation results. This is the primary lineage artifact. It survives compaction cycles and can be re-read by the agent at any point. It is what makes multi-round iteration reliable — not the context window alone.

### 2.3 Tools (Cold-Start Mode)

Four named tools:

| Tool | Purpose |
|---|---|
| `web_search` | Exa deep research API — data acquisition and baseline survey only |
| `bash` | Unrestricted shell with pre-loaded `slm_helpers.py` |
| `read_file` | Datasets, configs, eval results, `data-curation.md` |
| `edit_file` | Training scripts, configs, dataset files |

`delegate_task` is a separate sub-agent spawning mechanism — not one of the four named tools.

Production mode adds `query_traces` and `trace_analysis_subagent` and removes `web_search`. Not relevant to Phase 1.

### 2.4 State Machine Nodes

**Input:** Task specification triple:
```
τ = (description, M, constraints)

description  = "fine-tune on SMS Spam binary classification"
M            = Android_pool
constraints  = {
  target_metric: "F1",
  hardware_thresholds: {        # seeded but not gating in Phase 1
    storage_mb:      800,
    memory_mb:       1200,
    latency_ttft_ms: 2000,
    power_watts:     5.0,
  },
  target_chip: "snapdragon_778g"
}
```

**Node 1: `task_analysis`**

Three sequential sub-stages executed before any data is touched:

*(1) Task classification* — parses `description` to determine task type `t ∈ {classification, NER, generation}`. This single decision locks in three downstream choices for the entire run:

| Task type | Supervision format | Eval metric | Curation strategy |
|---|---|---|---|
| `classification` | direct labels | accuracy or F1 | hard negatives at label boundaries |
| `NER` | direct labels | entity F1 | entity diversification (no entity value >2–3×) |
| `generation` | chain-of-thought | LLM-as-judge [0,1] | teacher model (DeepSeek-R1 for math/science; GPT-4.1 for code/QA) |

Also selects the starting model from the Android pool (always Tier 1 first — smallest feasible model).

*(2) Data acquisition* — runs `web_search` to locate the relevant dataset. If a known benchmark is named (e.g., UCI SMS Spam, CoNLL-2003, GSM8K, ARC-Challenge), download actual benchmark data — do not synthesize. For custom tasks without public datasets, use a teacher model to synthesize seed examples.

*(3) Baseline survey* — runs `web_search` to find published SOTA for the target task at the target model size class. Calibrates the accuracy target relative to published numbers rather than using the 0.96 default blindly. Identifies known challenges that inform data curation before any training begins (e.g., for SMS Spam: 87:13 class imbalance; for ARC-Challenge: format learning required for base models; for CoNLL-2003: precision vs. recall asymmetry).

Writes all decisions — including task type classification and its downstream implications — to `data-curation.md` before any training.

**Node 2: `eval_setup`**

Constructs the held-out evaluation set `E` before any training. Fixed throughout all iterations, never included in training data:

```
E = Epos ∪ Eneg ∪ Eboundary
```

The three slices are always disjoint. Their concrete meaning varies by task type:

| Slice | classification (e.g. SMS Spam) | NER (e.g. CoNLL-2003) | generation (e.g. ARC-Challenge) |
|---|---|---|---|
| `Epos` | Clear positive-class examples (genuine spam) | Entity-rich passages with gold annotations | Well-formed problems with unambiguous answers |
| `Eneg` | Clear negative-class examples (legit messages) | Entity-free passages (hallucination test) | Adversarial / ill-posed inputs the model should decline or flag |
| `Eboundary` | Confusable pairs at the class boundary | Near-miss spans, overlapping entity types | Multi-step or edge-case problems requiring fine-grained reasoning |

For generation tasks, `Eneg` is replaced by adversarial inputs (ill-posed questions, out-of-domain prompts) and `Eboundary` contains problems that require multi-step reasoning or edge-case handling.

Default stopping criterion: `f(π) ≥ 0.96` on `E`. Agent may lower this if it detects a genuine capacity plateau (remaining failures reflect model capacity limits or irreducible ambiguity, not fixable training gaps).

**Node 3: `train`**

Fires **at least two configurations in parallel** every iteration — e.g., full fine-tune vs. LoRA, or two different base models. This is not optional; the paper does this on every iteration.

Training always restarts from the base foundation model, never from a prior fine-tuned checkpoint. This makes rollback clean: reverting to a previous dataset fully restores the corresponding model behavior without needing to untangle accumulated fine-tuning steps.

**Node 4: `evaluate`**

Scores each trained configuration against held-out `E`. Records per-class F1 breakdown. Logs the node to the lineage DAG as `v = (π, f(π))` where `π = (D, H, S)`. Selects the best-scoring configuration from parallel runs. Writes iteration result to `data-curation.md`. Logs hardware profile (theoretical in Phase 1).

**Node 5: `iterate`**

Applies the shared iteration policy based on `f(π)`:

| Score band | Diagnosis | Agent action |
|---|---|---|
| `< 0.80` | Data problem | Rebuild dataset — analyze failures, identify gaps (missing label coverage, distribution mismatch, insufficient hard negatives) |
| `0.80–0.95` | Optimization problem | Tune hyperparameters (epochs, LR, LoRA rank, base model) — hold dataset fixed |
| `≥ 0.95` | Capacity / surgical | Add 2–3 targeted examples per remaining failure pattern only |
| `f(πi+1) < f(πi)` | Regression | Roll back to previous configuration immediately — do not compensate |

After each iteration the agent analyzes remaining failures on `E` to determine whether the next intervention targets `D`, `H`, or `S`.

**Node 6: `curate`**

Builds `Dcold = Dgold ∪ Dhard` at 65:35 ratio. No replay buffer in cold-start (Dparent = ∅; replay allocation redistributed to gold examples).

Five quality controls applied to every dataset version, with task-type-specific behaviour:

| Quality control | classification | NER | generation |
|---|---|---|---|
| **2-for-1 rule** | For each boundary example: one gold + one hard negative (similar input, different correct label) | For each entity boundary case: one gold span + one near-miss negative | For each problem: one gold answer + one plausible wrong answer |
| **Label balancing** | No label exceeds 3× count of any other | No entity type exceeds 3× count of any other | N/A |
| **Context-length matching** | Training lengths match realistic input distribution | Training lengths match realistic document lengths | Training lengths match realistic query lengths |
| **Entity diversification** | N/A | No single entity value appears >2–3×; replace with synthetic equivalents | N/A |
| **CoT annotation** | N/A (direct labels) | N/A (direct labels) | Teacher model generates step-by-step reasoning chains (DeepSeek-R1 for math/science; GPT-4.1 for code/QA) |

Each label/entity type requires 3–5 distinct surface-text patterns. Target dataset sizes: 100–200 examples for classification/NER; 500–3,000 for generation depending on task complexity.

**Node 7: `rollback_check`**

Cold-start uses simple rollback only — no dual-gate (that is production mode). If `f(πi+1) < f(πi)`, immediately revert to previous dataset and configuration. Do not compensate with more data. Cascading fixes that each introduce new failure modes are the primary cause of stagnation.

**Node 8: `escalate_or_terminate`**

- **Terminate** if `f(π) ≥ 0.96` (or agent-adjusted threshold)
- **Terminate early** if agent diagnoses a genuine plateau (capacity ceiling or irreducible ambiguity)
- **Escalate model** if plateau persists >2 rounds without capacity diagnosis → move to next Android pool tier, restart `task_analysis` with new model, carry forward best dataset version
- **Terminate** if 1,500 LangGraph turns exhausted

Escalation is hardware-aware even in Phase 1: agent checks theoretical latency/memory for the next tier model against `constraints.hardware_thresholds` before escalating. In Phase 1 this check is informational (logged but not blocking). In Phase 2 it becomes a hard gate.

### 2.5 Search Procedure

**Formal definitions:**

Scoring function `f: Π → R` — maps pipeline `π = (D, H, S)` to held-out validation performance by executing the full train-then-evaluate loop. For SMS Spam: binary F1. No surrogates — every node score requires actual training and inference on `E`.

Regression function `r: Π × 2^X → Z≥0` — counts previously correct examples the new model answers incorrectly on held-out regression set `R ⊂ X`.

Optimization objectives:
```
Cold-start (Phase 1):   π* = argmax_{π∈Π} f(π)             [unconstrained]

Production (Phase 2+):  π* = argmax_{π∈Π} f(π)
                             subject to r(π; R) ≤ ε          [ε = 2]
```

The regression constraint (ε = 2, absolute count not relative rate) is production mode only. Phase 1 uses simple rollback instead.

**DAG structure:**

Phase 1 uses sequential greedy iteration — not full MCGS. Each iteration is a node:
```
v_i = (π_i, f(π_i))
edge (v_i, v_j): π_j derived from π_i via a targeted modification
```

The agent selects the best current node, proposes a hypothesis-driven modification (EXPAND operator), trains it, evaluates it. The DAG records lineage so the agent can attribute accuracy changes to specific interventions. Full MCGS (branching, UCT selection, cross-branch fusion) is deferred to Phase 2+ for harder tasks.

**Convergence criteria (in priority order):**
1. `f(π) ≥ 0.96` on `E` → promote
2. Irreducible plateau detected → lower target, terminate early
3. 1,500 LangGraph turns exhausted → terminate with best checkpoint
4. All Android pool tiers exhausted without meeting target → report capacity ceiling

---

## 3. Tool Layer & Training Backend

### 3.1 `slm_helpers.py`

Our equivalent of the paper's `tinker_helpers.py`. Three core functions pre-loaded into the agent's `bash` environment:

```python
train(
    dataset_path,   # path to Dcold JSONL on disk
    base_model,     # model ID from Android pool
    nr_epochs,      # agent-selected
    learning_rate,  # agent-selected
    batch_size,     # agent-selected
    lora_rank       # agent-selected; None for full fine-tune
)
# Executes full LoRA loop via Unsloth (decoder) or HF PEFT (fallback)
# Tokenizes with apply_chat_template for train/serve parity
# Loss computed on assistant tokens only via per-token loss weights
# EOS truncation handled for base models that don't stop at end-of-turn
# Returns weights_ref string for immediate inference

infer(prompt, weights_ref, base_model, ...)
# Loads checkpoint, generates text — no deployment step required

infer_batch(prompts, weights_ref, ..., max_workers=20)
# Parallel inference via ThreadPoolExecutor
# Used by eval harness: runs all held-out E examples concurrently
```

### 3.2 Training Backend

| Framework | Used for | Cluster resource |
|---|---|---|
| Unsloth | Decoder LoRA (Qwen3, Llama3.2, MiniCPM, Gemma3) | L40 for Phase 1 |
| HF PEFT + Transformers | Fallback; HRM-Text (custom arch) | L40 |
| Full fine-tune path | When agent selects over LoRA | A100 if needed |

Target training time: 10–30 min per run for decoder models. Use L40 for Phase 1 to preserve H200s/A100s for larger jobs.

LoRA hyperparameter search space the agent explores:
- `lora_rank`: 4, 8, 16, 32, 64
- `learning_rate`: 1e-4 (knowledge-intensive) or 2e-4 (format-learning) — agent discovers which
- `nr_epochs`: 1–8; agent monitors train vs. validation divergence
- `batch_size`: agent-selected; paper found 16 optimal for some tasks
- `target_modules`: all-linear for decoders

### 3.3 Eval Harness

The agent writes its own eval code per task (per paper Appendix D.2). The harness is task-type-aware:

| Task type | Metric | Extraction method |
|---|---|---|
| `classification` | binary F1 or accuracy | exact match on predicted label string |
| `NER` | entity F1 | span extraction + entity-type match |
| `generation` | LLM-as-judge score [0,1] | Claude API judges prediction against gold answer |

All three types output: overall metric score, per-class/per-entity breakdown, per-slice scores (Epos / Eneg / Eboundary). The agent writes the extraction function at runtime for the specific task, wired to `infer_batch`.

---

## 4. Data Pipeline

### 4.1 Held-Out Eval Set Construction

`E = Epos ∪ Eneg ∪ Eboundary` — built before any training, fixed throughout. Always three disjoint slices. Construction varies by task type:

**Classification (e.g. SMS Spam):**
- `Epos`: genuine spam stratified across surface-form range — UCI test split
- `Eneg`: clear legitimate messages, no label ambiguity — UCI ham split
- `Eboundary`: confusable pairs at the boundary — promotional messages, marketing, appointment reminders with links

**NER (e.g. CoNLL-2003):**
- `Epos`: entity-rich passages with gold span annotations
- `Eneg`: entity-free passages (tests hallucination suppression)
- `Eboundary`: passages with overlapping or near-miss entity types

**Generation (e.g. ARC-Challenge, GSM8K):**
- `Epos`: well-formed problems with unambiguous correct answers
- `Eneg`: adversarial inputs — ill-posed questions, out-of-domain prompts, jailbreak-style inputs
- `Eboundary`: multi-step or edge-case problems requiring fine-grained reasoning

### 4.2 Curriculum Synthesis

`Dcold = Dgold ∪ Dhard`, target ratio 65:35. No replay buffer (Dparent = ∅ in cold-start).

`Dgold` — correct labeled examples from downloaded benchmark data:
- Balanced across labels/entity types (no label >3× any other)
- 3–5 distinct surface-text patterns per label

`Dhard` — hard negatives generated by agent using 2-for-1 rule. Method varies by task type:
- **Classification:** generate counterexample with similar surface form but opposite label — via Claude API directly (no CoT needed)
- **NER:** generate near-miss span with same surface form but different or absent entity type — via Claude API directly
- **Generation:** generate plausible but incorrect answer (wrong reasoning path) — via DeepSeek-R1 for math/science, GPT-4.1 for code/QA

Target sizes: 100–200 examples for classification/NER; 500–3,000 for generation. Grows via surgical augmentation across iterations.

### 4.3 `data-curation.md` Schema

Written every iteration. The agent re-reads this at the start of each round to reconstruct full prior context.

```markdown
## Iteration N — [timestamp]

### Dataset
- Version: v{N}
- Total examples: {X}
- Dgold: {X} ({%}) — source: UCI download / Claude synthesis
- Dhard: {X} ({%}) — generated via 2-for-1 rule
- Label distribution: spam {X}, ham {X}

### Training config (π_N)
- Base model: {model_id}
- LoRA rank: {r} | LR: {lr} | Epochs: {n} | Batch size: {b}
- Config A: {description}
- Config B: {description}
- Best config selected: {A/B}

### Eval results
- Task type: {classification / NER / generation}
- Metric: {F1 / entity_F1 / judge_score}
- f(π_N): {score} on E
- Per-class breakdown: {label/entity: score, ...}
- Epos: {score} | Eneg: {score} | Eboundary: {score}
- Remaining failures: {N}
- Failure taxonomy: {description}

### Iteration policy decision
- Score band: {<0.80 / 0.80–0.95 / ≥0.95 / regression}
- Next intervention: {data rebuild / hyperparameter tuning / surgical / rollback}
- Hypothesis: {agent's causal reasoning for the next modification}

### Hardware profile (Phase 1: theoretical)
- Model: {id} | INT4 size: {MB} | Tier: {A/B/C}
- Reference chip: {target_chip}
- Storage: {MB} / {S_max}MB — {PASS/FAIL}
- Peak memory: {MB} / {M_max}MB — {PASS/FAIL}
- TTFT: {ms} / {L_max}ms — {PASS/FAIL}
- Power: {W} / {P_max}W — {PASS/FAIL}
- Escalation available: {yes/no} | Next model: {id} | Would pass constraints: {yes/no}
```

---

## 5. Iteration Loop & Convergence

### 5.1 Formal Objective

```
π* = argmax_{π∈Π} f(π)     [cold-start, unconstrained]
```

`f(π)` requires full train-then-evaluate execution — no surrogate models. Node scores reflect true task performance.

### 5.2 Expected Iteration Trajectory

From the paper: the first iteration captures only 40–70% of achievable improvement. Remaining gains come from diagnosing failure modes and applying targeted interventions. The diagnosis phase consumes more agent reasoning turns than the training phase.

Expected SMS Spam trajectory for Phase 1:

| Round | Expected F1 | Band | Agent action |
|---|---|---|---|
| 0 | ~0.70–0.85 | <0.80 or 0.80–0.95 | 2+ parallel initial configs; data rebuild or hyperparameter tuning |
| 1 | ~0.95–0.98 | 0.80–0.95 or ≥0.95 | Surgical: identify recall vs. precision failures |
| 2 | ~0.98–0.99 | ≥0.95 | Surgical: precision repair via hard ham examples |
| 3 | ~0.997 | ≥0.95 | Surgical: targeted augmentation; regression may trigger rollback |
| Final | ≥0.96 or rollback | — | Promote V1 or roll back from V2 regression |

### 5.3 Rollback Behavior

Simple rollback in cold-start: if `f(πi+1) < f(πi)`, revert immediately. No compensation. The paper's SMS Spam run demonstrates this exactly: V2 (17 additional examples) regressed from 0.9834 to a lower score; system rolled back to V1. This demonstrates the system is working correctly, not that it failed.

### 5.4 Stagnation Handling

Sequential greedy DAG for Phase 1. Stagnation detected when 2+ consecutive rounds show no improvement. On stagnation:
1. **Terminate early** — if agent diagnoses capacity ceiling (e.g., "remaining failures reflect model size limitations")
2. **Model escalation** — upgrade to next Android pool tier, restart `task_analysis` with new model, carry forward best dataset version

### 5.5 Phase 1 Success Definition

The Week 4 checkpoint is successful if:
- F1 improves across ≥3 consecutive rounds
- At least one rollback occurs and is handled correctly
- Full run completes in <8 hours at <$50
- All models fit within Android pool bounds
- `data-curation.md` provides complete lineage

---

## 6. Hardware Constraint Layer

### 6.1 Objective (Full System — Phase 2+)

```
π* = argmax f(π)
     subject to:
       storage(model)          ≤ S_max    # INT4 file size fits on device
       memory(model, chip)     ≤ M_max    # no OOM; other apps still run
       latency(model, chip)    ≤ L_max    # model feels responsive to user
       power(model, chip)      ≤ P_max    # battery drain acceptable
       r(π; R)                 ≤ ε        # production mode only
```

The hardware constraints define the **feasibility boundary**. Accuracy is the only thing being optimized — the constraints just define what solutions are admissible.

### 6.2 The Four Constraints

**Storage** `S(model) ≤ S_max`
- INT4 quantized file size on disk
- Measurable before inference: `S ≈ param_count × 4bits / 8`
- Enforced at model selection time — infeasible models excluded before the run begins

**Memory** `M(model, chip) ≤ M_max`
- Peak RAM during inference: `M ≈ INT4_size × 1.2 + KV_cache(context_length)`
- Android OOM = hard failure, not soft degradation
- `M_max` = device RAM minus OS overhead (~1.5GB reserved on a 4GB device)
- Enforced at model selection time

**Latency** `L(model, chip) ≤ L_max`
- Time to first token (TTFT) for interactive use; tok/s for batch/background use
- Varies most across chip generations — a model at 25 tok/s on Snapdragon 8 Gen 3 may run at 6 tok/s on Snapdragon 660
- Common acceptability threshold: TTFT ≤ 2 seconds for interactive use
- Phase 1: theoretical from benchmark data; Phase 2: measured via Qualcomm AI Hub or physical device

**Power** `P(model, chip) ≤ P_max`
- Average watts during sustained inference
- Determines battery life impact
- Phase 1: proxy from model size; Phase 2: measured via Android Battery Historian or Qualcomm power estimation tools

### 6.3 How Constraints Enter the System

Via `constraints.hardware_thresholds` in the task specification. The `target_chip` is the **worst-case reference device** — least powerful chip in the target range. If acceptable there, acceptable everywhere.

**At model selection (Node 1):** filter Android pool by `S_max` and theoretical `M_max` before the run begins.

**At model escalation (Node 8):** before escalating to next tier, check theoretical latency and power for that model on the reference chip. In Phase 1: informational. In Phase 2: hard gate — cannot escalate if any constraint would be violated.

**At checkpoint promotion (Phase 2):** quantize to INT4, measure all four constraints on reference chip. Checkpoint only promoted if `f(π) > f(π_prev)` AND all constraints satisfied.

### 6.4 Android Model Pool

The feasible model search space, bounded by "generous but realistic" Android deployment. Agent starts at Tier 1 and escalates only if accuracy target is unmet and hardware constraints permit.

**Tier 1 — fits anywhere (≤800MB INT4):**

| Model | INT4 Size | Notes |
|---|---|---|
| Qwen3-0.6B | ~397MB | Fastest iteration baseline |
| Qwen3.5-0.8B | ~500MB | DeltaNet hybrid arch, newer |
| MiniCPM5-1B | ~500MB | Best sub-2B benchmark score (May 2026) |
| MiniCPM4-0.5B | ~300MB | 7× faster decode than Qwen3-0.6B |
| Gemma3-1B | ~529MB (QAT) | LiteRT+QNN NPU path |
| Llama3.2-1B | 808MB | Deepest Qualcomm/ExecuTorch integration |
| HRM-Text-1B | ~600MB | Research candidate; custom runtime, no llama.cpp support yet |

**Tier 2 — fits most Android devices (800MB–1.5GB INT4):**

| Model | INT4 Size | Notes |
|---|---|---|
| SmolLM2-1.7B | ~1.06GB | HuggingFace's explicit on-device model |
| Qwen3-1.7B | ~1.19GB | Good capability step-up |
| Qwen3.5-2B | ~1.2GB | Multimodal, 201 languages |
| DeepSeek-R1-Distill-1.5B | ~1.29GB | Strong reasoning at this size |
| Gemma3n-E2B | ~1.3GB | MatFormer arch, 50–80 tok/s on NPU |

**Tier 3 — generous upper bound (recent/flagship only):**

| Model | INT4 Size | Notes |
|---|---|---|
| Llama3.2-3B (Q3_K_M) | ~1.4GB | Qualcomm Hexagon NPU optimized |
| Ministral-3B | ~<2GB | 256K context, strong classification |

### 6.5 Phase 1 vs Phase 2 Hardware Implementation

| Constraint | Phase 1 | Phase 2 |
|---|---|---|
| Storage | Calculated — gates model selection | Same |
| Memory | Estimated — gates model selection | Same + measured on device |
| Latency | Theoretical from benchmarks — logged, not gating | Measured via AI Hub / device — hard gate |
| Power | Proxy from model size — logged, not gating | Measured via power profiling — hard gate |

Phase 1 seeds all four values in `data-curation.md` every iteration. Phase 2 replaces theoretical with measured and activates the gates.

---

## 7. Phase 1 → Phase 2 Transition *(Tangential — for reference only, not Phase 1 scope)*

Notes on what Phase 1 must produce for Phase 2 to start cleanly, to avoid creating debt in Phase 1 decisions.

**Artifacts Phase 1 must hand off:**

1. **Deployable model artifact** — best checkpoint from final Phase 1 run, quantized to INT4. This becomes Phase 2's "deployed model M₀". Represented as `(best_dataset_version, base_model_id, training_config)` — not just weights — because Phase 2 retrains from the base model using this recipe.

2. **Complete `data-curation.md` lineage** — Phase 2's parent model awareness stage (step 4 of production failure diagnosis) reads this to recover `D_parent`. Without it, the production agent cannot construct a complementary dataset and would duplicate existing supervision.

3. **Inference logging infrastructure** — Phase 2 requires a production inference database (PostgreSQL / Supabase with per-user RLS). Each record: `(input, prediction, corrected_output, verdict, judge_reasoning, judge_metadata)`. Schema should be designed alongside Phase 1 even though Phase 1 doesn't use it.

4. **LLM-as-judge config** — for SMS Spam, exact match is sufficient. For harder Phase 2 tasks, an LLM judge with a fixed prompt template and criteria stored in inference table metadata is required. Judge config must be reproducible — the production agent needs to reproduce verdicts consistently.

5. **Hardware profiling baseline** — after Week 4, run the Phase 1 best checkpoint through Qualcomm AI Hub profiling on the reference chip to get real latency, memory, and power numbers. These replace Phase 1 theoretical values and become Phase 2 constraint baselines.

**What Phase 1 explicitly does not need to build:**
- `query_traces` tool
- `trace_analysis_subagent`
- Regression set construction
- Confidence calibration mechanism
- AdaptFT-Bench noise injection pipeline

---

## Appendix A: Key Paper Deviations

Decisions where we diverge from Pioneer Agent and why:

| Decision | Paper | SLM Factory | Reason |
|---|---|---|---|
| Execution environment | Modal sandboxes (16GB, 24hr) | GPU cluster (H200/A100/L40) | Existing infrastructure |
| Training SDK | Tinker SDK (proprietary) | Unsloth + HF PEFT | Open source equivalents |
| Model families | GLiNER2 + Qwen/Llama | Android pool (12 models) | Hardware constraint requires bounded search |
| Cold-start task | Any of 9 benchmarks | Same 9 benchmarks, same `t ∈ {classification, NER, generation}` abstraction | |
| Phase 1 first task | SMS Spam (GLiNER2 encoder) | SMS Spam (decoder from Android pool) | GLiNER2 not in Android pool |
| MCGS | Used for ARC-Challenge | Deferred to Phase 2+ | Phase 1 uses sequential greedy; MCGS adds complexity without benefit on simpler tasks |
| Trace database | PostgreSQL (Supabase) | Deferred to Phase 2 | Not needed in cold-start |
| Hardware gating | Not in paper | Added as Phase 2 constraint | Core project goal |

## Appendix B: Cost Estimates

Based on paper's Table 9 (cost breakdown per benchmark run):

| Component | Estimate |
|---|---|
| Claude Sonnet 4.6 API (4M tokens per 12hr run, 75/25 input/output) | ~$24 per full run |
| Training compute (L40, 4–5 training runs per full loop) | ~$5–15 |
| Exa web_search API | ~$1–2 |
| **Total per Phase 1 run** | **~$30–40** |

SMS Spam is a simpler task than ARC-Challenge (paper: $36 for ARC, $3.48 for CLINC150 production run). Phase 1 should land closer to $30 than $50.
