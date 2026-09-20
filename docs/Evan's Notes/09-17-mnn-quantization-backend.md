# 09-17 — A second quantization backend: MNN alongside llama.cpp

The loop could only ever ship one on-device format. Everything from `evaluate_node` down assumed a
GGUF: one file, built by `convert_hf_to_gguf` + `llama-quantize`, scored by `llama-cpp-python`. This
adds MNN as a second, selectable backend — the runtime MNN-Chat and the Qwen-family mobile stack
actually run — ported from the reference harness in
[Krishna-Deshpande1/SLM_Factory](https://github.com/Krishna-Deshpande1/SLM_Factory)
(`mnn-benchmark-harness/agent_mnn_quantize.py`), which drives MNN's `llmexport.py` with
`--quant_bit` / `--quant_block` / `--mnnconvert` and pushes the result to a phone.

llama.cpp remains the default. Nothing about an existing launcher, checkpoint or number changes
unless the flag is set.

```
python tests/pipeline/run.py --quant-backend mnn "<task description>"     # or SLM_QUANT_BACKEND=mnn
bash scripts/setup_mnn_env.sh                                            # one-time toolchain build
```

---

## What the two backends actually are

| | `llama_cpp` (default) | `mnn` |
|---|---|---|
| Builder | `convert_hf_to_gguf` + `llama-quantize` | `llmexport.py` + locally built `MNNConvert` |
| Artifact | one file, `model-<method>.gguf` | a **directory**: `llm.mnn`, `llm.mnn.weight`, `config.json`, `llm_config.json`, `tokenizer.mtok` |
| Quant control | named presets `Q4_K_M` / `Q8_0` | `--quant_bit` 4 or 8, `--quant_block` 64, `--lm_quant_bit` 8 |
| Scorer | `llama-cpp-python`, layers offloaded to the GPU | `pymnn`'s LLM API (`MNN.llm`), CPU |
| Cache tree | `artifacts/gguf/` | `artifacts/mnn/` |

The pool's selectors are shared, so `@Q4_K_M` means "the 4-bit build for this run's backend" and the
model ladder, tiering, reporting and hardware gates are untouched by the choice. `Q4_K_M`'s k-quant
per-tensor choices have no exact MNN equivalent — the closest match, keeping the lm_head above the
body's width, is what the default does — so **a llama.cpp-vs-MNN comparison is a bit-width
comparison**, and that is the only thing the two formats can honestly be compared on.

One backend per run. `SLM_QUANT_BACKEND` is in the resume fingerprint, so a checkpoint cannot be
continued under the other backend and have its old scores compared against new ones.

---

## The measurement

Job `40268007` (the uniform-4-bit row is the earlier `40258512`). Same adapter, same rows, same
rendered prompts; only the toolchain and the engine differ. The adapter is iteration 30 of clinc150
run `40175895` — the best SmolLM2-360M checkpoint that run produced, **0.8047 macro_f1 through
llama.cpp/GGUF over all 1,000 eval rows**. Here both arms score the first 150 of those same rows.

| backend | quant | macro_f1 | format_valid | on-disk MB | quantize | eval (150 rows) |
|---|---|---|---|---|---|---|
| llama.cpp/GGUF | Q4_K_M | **0.7511** | 0.9333 | 258.1 | 50s | 9s (GPU) |
| MNN | 4-bit body, 8-bit lm_head, block 64 | **0.7367** | 0.9733 | 217.7 | 190s | 523s (CPU) |
| MNN | 4-bit throughout, block 64 | 0.7278 | 0.9400 | 195.2 | 105s | 323s (CPU) |

**MNN is 0.0144 behind (−1.9% relative) and 16% smaller on disk.** `format_valid` is *higher* on
MNN, which is the useful control: whatever differs, it is not the prompt or the chat template,
because the model still emits well-formed intent labels at least as often.

The third row is why the shipped default keeps the lm_head at 8 bits. `Q4_K_M` is a
*mixed*-precision recipe — it quantizes the body to 4-bit and keeps `output.weight` at Q6_K — so an
MNN export at a uniform 4 bits is not the artifact `@Q4_K_M` names, and comparing the two charges
MNN for a difference in recipe rather than in runtime. Matching it costs 22 MB and recovers 0.0089
macro_f1, cutting the gap to llama.cpp roughly in half.

Sanity of the output itself, not just the score — a 4-bit SmolLM2-135M export answering through the
same path:

```
prompt: 'What is the capital of France?'  → 'The capital of France is Paris.'
prompt: 'Say hello.'                      → 'Hello there!'
```

---

## 09-18 — Putting MNN eval on the GPU

MNN eval was 58x slower than llama.cpp's. It is now 6.75x, and the whole difference is that MNN's
CUDA backend does the prefill. Same adapter, same 150 clinc150 rows, same rendered prompts:

| path | eval time | rows/s | macro_f1 | format_valid |
|---|---|---|---|---|
| llama.cpp/GGUF, layers on GPU | **8s** | 18.8 | 0.7511 | 0.9333 |
| MNN, CUDA | **54s** | 2.80 | 0.7256 | 0.9533 |
| MNN, CPU (what shipped yesterday) | 523s | 0.29 | 0.7367 | 0.9733 |

A 1,000-row clinc150 eval therefore goes from ~35 minutes to ~6, against llama.cpp's ~53 seconds.
Accuracy moves a little on the GPU — 0.7256 against the CPU's 0.7367, with two points less format
validity — which is the fp16 accumulate showing up and is worth knowing when comparing a GPU-scored
MNN run against a CPU-scored one.

Where it comes from, measured with `llm_demo` on one artifact:

| | prefill | decode |
|---|---|---|
| CPU | 671 tok/s | 141 tok/s |
| CUDA | **6,867 tok/s** (10.2x) | 27–36 tok/s (0.2x) |

**The GPU wins prefill by 10x and loses decode by 4x**, and which one dominates depends entirely on
the model. A fine-tuned classifier answers with one label, so a ~1,385-token prompt is ~99% of the
work and the GPU is ~10x ahead. An untrained baseline rambles to the 50-token cap, so decode
dominates and the GPU is barely ahead at all (measured 0.51 rows/s on an untrained 135M, where
prefill was 116ms and decode 1,856ms). That is the remaining lever if this needs to get faster:
MNN's CUDA decode at batch 1 is kernel-launch-bound for models this small.

**Concurrency is unchanged and deliberately so.** MNN scores one row at a time, which is parity
with the GGUF path rather than a shortfall — `MAX_GGUF_EVAL_CONCURRENCY` defaults to 1 there too,
for measured reasons recorded in `slm_helpers`. `SLM_MNN_EVAL_CONCURRENCY` above 1 is refused
rather than ignored.

### The trap that cost four measurements

The first four CUDA numbers I took were **wrong, and looked like a slow, broken GPU backend**: no
speedup at all, and output like `'accept<|endoftext|><|endoftext|>...'` where the CPU said
`'accept_reservations'`. I nearly concluded MNN's CUDA path was unusable.

What was actually happening: `Can't Find type=2 backend, use 0 instead`, printed to stdout and
nowhere else. Type 2 is CUDA, type 0 is CPU. GPU utilisation was 0% and VRAM 0 MiB the whole time.
MNN had fallen back to the CPU — *and then configured itself as though it were on a GPU*, skipping
the CPU blockwise-quant setup, which is where the garbage came from.

The cause is a linking detail with no runtime symptom. MNN's CUDA backend registers through a
file-scope initializer in `source/backend/cuda/Register.cpp`, compiled into `libMNN` as an OBJECT
library. pymnn's own `build_deps.py` builds **static** on Linux, and a static archive contributes
only the objects needed to resolve referenced symbols — nothing references that initializer, so the
linker drops it. The fix is `-DMNN_BUILD_SHARED_LIBS=ON` plus staging `libMNN.so` and
`libMNN_Cuda_Main.so`; `scripts/setup_mnn_env.sh` now does both and verifies the link.

Two more things made it hard to see, and both are now handled:
- **setuptools silently reinstalled nothing.** Two "CUDA installs" in a row reported success while
  leaving the CPU-only extension in place, because `pip_package/build/` looked current. The setup
  script now removes that tree and the installed `.so` before installing.
- **`ldd` satisfaction is not registration.** The only way to know the backend registered is to
  read MNN's own stdout, so `load_mnn_llm` now CAPTURES it rather than discarding it, and
  `_assert_backend_honoured` fails the load on a fallback. A fallback defeats the entire purpose of
  the GPU path and corrupts the output, so it is fatal rather than a warning.

### The GPU ran the whole loop to convergence

Job `40298521`, `run_clinc150_mnn_verify.slurm` unchanged except that MNN now scores on CUDA:

| | GPU (job 40298521) | CPU (job 40268008) |
|---|---|---|
| wall clock | 3h11m, **CONVERGED** | 4h40m, wall-clock stop |
| best macro_f1 | **0.8965** (format_valid 1.0000) | 0.7049 |
| tiers reached | 1 → 2 → 3 → **4** | 1 → 2 |
| train→eval iterations | 7 | 4 |
| 1,000-row eval | **325s** | ~2,100s |

It cleared the teacher-calibrated goal of 0.8856 on `Qwen3-4B-Instruct-2507@Q4_K_M` and terminated
as converged, with no tracebacks and every artifact cross-checked. The same bounded box that
previously bought two tiers now buys four, which is the whole point of the speedup: the eval is no
longer the thing limiting how much search a run can do.

### Every tier, every precision, both devices — 27/27

`sbatch tests/pipeline/verify_mnn_matrix.slurm` (job `40305686`, 4h48m): all 9 pool models across
Q4_K_M / Q8_0 / FP16, each exported, load-validated and scored on the same 100 clinc150 rows on
CUDA *and* on the CPU. One subprocess per measurement. Full data in
`logs/probes/mnn-backend-matrix-40305686.json`.

| tier | model | GPU speedup (q4 / q8 / fp16) | max \|GPU−CPU f1 gap\| |
|---|---|---|---|
| 1 | SmolLM2-135M-Instruct | 0.6x / 0.5x / 0.7x | 0.0101 |
| 1 | SmolLM2-360M-Instruct | 1.9x / 1.8x / 2.6x | 0.0283 |
| 1 | gemma-3-270m-it | 0.7x / 0.8x / 1.1x | 0.0000 |
| 2 | Qwen3-0.6B | 5.5x / 5.4x / 6.5x | 0.0833 |
| 2 | Qwen3.5-0.8B | 10.9x / 7.9x / 7.2x | 0.0226 |
| 3 | Qwen3-1.7B | 7.3x / 3.5x / 6.4x | 0.0994 |
| 3 | Qwen3.5-2B | 13.7x / 4.8x / 6.9x | 0.0525 |
| 4 | Qwen3-4B-Instruct-2507 | 3.5x / 4.8x / 9.4x | 0.0133 |
| 5 | Qwen3.5-4B | 6.9x / 6.1x / 5.6x | 0.0566 |

Two things this establishes and one caveat:

**Every pool entry works on the GPU, at all three precisions.** No cell failed to export, load or
score, and MNN's `llmexport.py` handled all three architecture families (SmolLM2/Llama, Gemma-3,
Qwen3 and Qwen3.5).

**The GPU computes what the CPU computes.** Median |GPU−CPU| macro_f1 gap across 27 cells is
**0.0101**, worst case 0.0994 on a cell whose CPU format_valid is 0.90 — i.e. the disagreements
live where predictions are near-ties, which is what fp16 accumulation does. Nothing here looks like
a different model.

**The GPU is 3.5–13.7x faster for every tier the loop escalates to, and NOT faster at tier 1.** The
two smallest models (135M, gemma-270m) are 0.5–1.1x, because an untrained model that rambles to the
50-token cap is decode-bound and MNN's CUDA decode is slower than its CPU decode at that size. That
is the honest shape of the win: it arrives exactly where the eval is expensive.

**The caveat.** Two of the 27 cells show the GPU producing materially fewer parseable outputs than
the CPU: `Qwen3-0.6B` at Q4_K_M (format_valid 0.66 against 0.82, though its macro_f1 was *higher*,
0.1308 against 0.1091) and at Q8_0 (0.09 against 0.20, both at the floor). Both are untrained base
models that barely format at all on either device, so I cannot separate fp16 noise from something
real there. The case that matters for the loop is the fine-tuned one, and there the gap is small and
measured twice: 0.7256 GPU against 0.7367 CPU at format_valid 0.9533 against 0.9733. Worth
re-checking if `Qwen3-0.6B` ever becomes a tier a run settles on.

### MNN's CUDA int8 kernel is broken; its int4 kernel is not

The one finding that changes the shipped configuration. Measured on SmolLM2-360M @ 8-bit, clinc150:

| artifact | backend | memory | macro_f1 | format_valid | rows/s |
|---|---|---|---|---|---|
| 8-bit | cpu | low | 0.2667 | 0.2667 | 0.14 |
| 8-bit | cuda | low | **0.0000** | 0.0000 | 0.36 |
| 8-bit | cuda | **normal** | **0.2667** | 0.2667 | **0.59** |
| 4-bit | cuda | low | works | works | 2.86 |

`memory=low` keeps the quantized weights packed and dequantizes inside the GEMM — MNN's
weight-only-quant kernels. On CUDA the int4 one is sound and the int8 one returns `<|endoftext|>`
on every row. `memory=normal` dequantizes the weights into fp16 device memory once and runs dense
GEMMs, which bypasses the broken kernel and reproduces the CPU score EXACTLY (0.2667 both) at 4x
the CPU's speed. That costs VRAM, not accuracy: the values are still the quantized ones, just
stored wider.

So `SLM_MNN_MEMORY` now defaults to `auto` and is chosen by bit width — `low` for 4-bit, `normal`
for anything wider — and forcing `low` with a wider artifact on CUDA is refused with the numbers
above. Separately, `precision=normal` with `memory=low` on CUDA decodes
`'ordinaryritz Hviations:`~ ...'` at no speed gain and is refused by name.

### MNN's CUDA runtime does not survive a second model load

Loading and freeing 27 artifacts in one process: the first decoded correctly, the second decoded
`<|endoftext|>` repeatedly, and every one after returned an EMPTY STRING — scored 0.0000, nothing
raised, at a nonsensical 878 rows/s because generation was doing no work.

The pipeline was never exposed: `run_eval` and `build_quant_artifact` each run in their own
disposable CUDA worker, so one process loads one artifact — which is why the A/B and the loop run
were correct throughout. `slm_helpers._note_cuda_load` now refuses a second CUDA load outright, so
an out-of-loop sweep gets an error naming the cause instead of a column of zeros.

**The first version of the backend matrix was that column of zeros, and it reported "27/27
passed"** — because its only pass criterion was that scoring had not raised. It now runs one
subprocess per (cell, device), exactly as the pipeline does, and a cell whose output is empty for
most rows is reported as failed.

### Where the time goes on the CPU path

**Prefill dominates, and CLINC150 is the worst case for it.** Last-row telemetry from MNN's own
context: `prompt=1376 tok, generated=2 tok, prefill=2016ms, decode=11ms`. The prompt carries all 151
intent names; the answer is one label. So the 0.46 rows/s is almost entirely the cost of reading the
prompt on CPU, and it scales with the prompt, not with the model's chattiness.

Practical consequences, for whoever runs this next:
- A 1,000-row clinc150 eval is ~36 min at the 360M tier and ~15 min at the 135M tier, against 9s for
  150 rows on the GPU-offloaded GGUF path. **MNN eval is the slow part of an MNN run, not the export.**
- The export is also slower but bounded: 105s for a merged 360M, 417s for a 135M the first time
  (cold `.venv_mnn`, ONNX trace + `MNNConvert` graph rewrite), once per iteration.
- `SLM_MNN_THREADS` (default 8) is the knob that matters; `precision` is already `low`, and the login
  and compute nodes report `fp16: 0`, so there is no half-precision path to buy on x86.

---

## The loop run

`sbatch tests/pipeline/run_clinc150_mnn_verify.slurm` — the full agent loop with
`SLM_QUANT_BACKEND=mnn`, bounded to 4.5h with a 600-row curriculum so several iterations and a tier
change fit inside the box. A plumbing proof, not a results run; the accuracy question is the A/B
above. Job `40268008` is the run with everything fixed; `40260927` is the earlier one that found
the thread bug.

| | MNN 4-bit (job 40268008) | llama.cpp Q4_K_M (run 40175895) |
|---|---|---|
| Tier 1 zero-shot baseline, untrained SmolLM2-360M | 0.0595 (`format_valid` 0.1610) | 0.1142 (`format_valid` 0.3490) |
| Tier 1 iteration 1, fine-tuned | **0.2775** (`format_valid` 0.9610) | 0.2934 (`format_valid` 0.7860) |
| Tier 1 iteration 2, fine-tuned | **0.6056** (`format_valid` 0.9740) | — |
| Tier 2 after escalation, zero-shot Qwen3.5-0.8B | **0.2907** (`format_valid` 0.9680) | 0.3717 (`format_valid` 0.9620) |
| Tier 2 iteration 1, fine-tuned | **0.5954** (`format_valid` 0.9650) | — |
| Tier 2 iteration 2, fine-tuned | **0.7049** (`format_valid` 0.9840) | — |
| Curriculum | 600 rows | 4,998 rows |

Iteration 1 lands in the same band as the llama.cpp run's first fine-tune — on an eighth of the
training data — with higher format validity, and the loop then improves it to 0.6056 through its own
data rebuild. Both tier-1 baselines are floor readings rather than measurements of capability: an
untrained 360M asked to pick one of 151 intents mostly fails to emit a parseable label at all
(`format_valid` 0.16 and 0.35), so the difference between them is formatting noise. The tier-2
zero-shot is the honest backend comparison at this size — 0.2907 against 0.3717, both at
`format_valid` ~0.97 — and it is also the number that was **0.0000 in job 40260927** before the
thread bug was found. Fine-tuning then took that tier to 0.5954 and 0.7049, so across 4h40m the
loop trained, quantized, scored and improved entirely through MNN at both tiers —
0.0595 → 0.6056 at tier 1, then 0.2907 → 0.7049 at tier 2 — and terminated on its own wall-clock
guard with a full report rather than being killed.

What the run demonstrated, item by item:
- the zero-shot baseline is built and scored on a real MNN artifact, with the smoke test decoding
  `'Hello!'` and the thread cross-check passing before anything is scored;
- each iteration merges, exports, bit-width-verifies and scores its own adapter;
- `Reaped non-best MNN artifact: .../model-mnn-q4` — the reaper deletes a directory rather than
  failing on a path that is not a file;
- the escalation rebuilt at the new tier on a **different model family** (SmolLM2 → Qwen3.5-0.8B),
  so the exporter is not SmolLM2-specific;
- the DAG carries `quant_backend` on every node, so these scores cannot later be pooled with
  llama.cpp numbers for the same task.

In-loop costs, from `timing-events.jsonl`: **90s** per `build_quant_artifact` (export + load
validation + cross-check, base 360M) and **~35 min** per 1,000-row eval at 0.42–0.59 rows/s with
8 threads.

---

## What had to change in the loop, and why

`training/quant_backend.py` is the single routing point. It exists rather than an
`if backend == "mnn"` at each call site because the GGUF path had acquired **six** of them —
`evaluate_node`'s baseline, its per-config build, the reaper, the downward probe, the interpolation
probe, and the post-convergence on-device verification — and B161 is the record of what one missed
site costs (a Q8_0 tier silently scoring a Q4_K_M file).

Five things needed real thought rather than a parameter:

**1. The artifact is a directory.** `run_eval`'s `gguf_path` became `quant_artifact` +
`quant_backend`, passed as a pair. A path alone cannot say which engine should load it, and guessing
from the extension is how an MNN directory gets handed to llama.cpp.

**2. "Genuinely quantized" needed a new check.** A GGUF's quantization is in its file format, so a
`Q4_K_M` file cannot be mistaken for `Q8_0`. A 4-bit and an 8-bit MNN export are *the same five
filenames*. `validate_and_record_mnn` therefore re-reads the exporter's own `export_args.json` and
refuses an artifact whose `quant_bit` does not match its label. Verified against the real artifact:

```
{"quant": "Q4_K_M", "quant_bit": 4, "quant_block": 64, "weight_size_mb": 195.22,
 "tool_versions": {"mnn_commit": "47ccf6c6bb5b", "mnn_version": "3.6.1", "pymnn": "3.6.1"}}
correctly refused mislabel: ... was exported at 4-bit but is labelled Q8_0 (8-bit)
```

**3. The cache sidecar hashes every file, not one.** An MNN model is only usable if the graph, the
weights, the tokenizer and both configs agree; hashing `llm.mnn.weight` alone would miss a weight
file swapped under an unchanged graph — the same half-written-artifact case the GGUF sidecar exists
for. The reaper likewise had to learn to delete a directory.

**4. Both backends get the SAME rendered prompt.** MNN runs with `use_template: false` and is handed
the output of `_serving_prompt_prefix`, the same function the GGUF path uses. If each backend applied
its own chat template, a score difference between them could be the quantization, the runtime, or the
template, and nothing would say which — and B290 (two stray `<think>` tags per prediction) is what
that failure looks like when it happens.

**5. MNN's defaults are a chat app's.** `llmexport.py` writes `sampler_type: "mixed"` with
temperature 0.8 and top_k 40 into `config.json`, and a pymnn `Llm` accumulates conversation history
across `response()` calls. Scoring under those would have made every eval a different experiment and
answered row 2 in row 1's context. The eval path sets `greedy`, calls `reset()` before every row, and
raises `max_all_tokens` off MNN's 2048 default to the task's context.

### Four findings worth keeping

**MNN's THREAD COUNT IS A CORRECTNESS SETTING, and this is the one that would have shipped a wrong
number.** The loop run's escalation tier, `Qwen/Qwen3.5-0.8B@Q4_K_M`, scored 0.0000 with
format_valid 0.0000: all 1,000 rows decoded to `%+!!!!!!!!!!`, `feier!!!!!!!!!!` or `оте!!!!!!!!!!`.
The weights were fine. Sweeping the same artifact on the same prompts, one fresh process per
setting:

```
threads   1   2   4   8  10  12  13  14  16
verdict  ok  ok  ok  ok  ok  ok  ok  BAD  ok
```

14 corrupts compute and everything either side of it is correct and agrees token-for-token. Three
properties make this nasty: the logits are **finite, plausibly scaled, and only the argmax is
wrong** (so there is nothing to detect in the numbers); it needs a **long prompt** (the same
artifact answers a 16-token `Hello` perfectly and fails at ~1,050 tokens, which is exactly why the
build's smoke test passed); and MNN's thread pool is a **process-global, first-writer-wins** object
keyed by CPU mask, so an in-process sweep that starts low silently measures the low setting over
and over — my first sweep "cleared" every thread count that way.

The guard is now a determinism invariant rather than a quality heuristic: every artifact build
decodes a 512-token prompt at the configured thread count and at a known-good reference count, and
**refuses the artifact if they disagree**. Greedy decoding cannot depend on how the work was split.
Disagreement is the right condition and that is measured, not stylistic — on synthetic filler the
bad setting still produces readable text, so a "looks like garbage" test passes it, while 8, 10, 12,
13 and 16 threads each agree with the reference exactly and only 14 differs. Verified against the
real artifact: refused at 14, passes at 8. The default is 8, and the launcher that used 14 now says
why it does not.

### Three earlier findings

**The first attempt of the A/B died on a real defect, in llama.cpp's arm.** `Rendered GGUF prompt
index 0 contains 1410 tokens, exceeding input budget 974 inside configured max sequence length 1024`
— CLINC150's spec declares `max_seq_length=1024`, and every real clinc150 run has silently overridden
it to 4096 through `_l40s_task_body.sh`, because `task_max_seq_length` gives the env var precedence
over the spec. A standalone script that does not export it cannot reproduce the runs. The verify
launcher now sets it explicitly, and says why.

**The MNN path needed the same budget guard.** MNN sizes its KV cache from `max_all_tokens` and drops
what does not fit, so an over-long prompt yields a plausible answer to a question the model never
saw. `_validate_mnn_budget` counts with the artifact's own `tokenizer.mtok` and raises — outside the
per-row error handler, because a prompt that does not fit is a configuration fault affecting every
row, not one unscoreable row to absorb into a score of zero.

**Relative output paths do not survive a change of working directory.** `llmexport.py` has to be
launched from its own source tree, and `evaluate_node` passes `artifacts/mnn/<model>/<key>` relative
to the project root — so the first loop run (`40260162`) wrote a complete 199 MB model into
`<MNN>/transformers/llm/export/artifacts/` and the exporter exited 0 having done it. The only
symptom was `exited 0 but the MNN artifact is incomplete` pointing at an empty directory. Every path
handed to the exporter is now absolute, and the completeness check — which is what turned a silent
misplacement into a legible error at the right moment — stays.

---

## Toolchain

`bash scripts/setup_mnn_env.sh` is idempotent and builds four things, for the reasons the script
states in full: `MNNConvert` + `llm_demo` (MNN 3.6.1, CPU, AVX-512 kernels, `MNN_BUILD_LLM=ON`), a
`.venv_mnn` for the exporter's torch/onnx stack kept out of `.venv_gpu`, a second static MNN build for
the Python extension, and pymnn with the LLM API installed into `.venv_gpu` — the interpreter the eval
CUDA worker runs in.

`--mnnconvert` is mandatory, not an optimisation: without it `llmexport.py` falls back to the pymnn
bindings, which the reference harness recorded crashing with a bus error rather than failing cleanly.
`MNN.llm` also has to be imported as a submodule rather than read off the `MNN` package — the
extension exposes an `llm` attribute of its own whose `set_config` takes a JSON string where the
wrapper's takes a dict, and getting the wrong one surfaces as
`SystemError: <method 'set_config' of 'LLM' objects> returned a result with an exception set`.

## Where the evidence is

| What | Where |
|---|---|
| Backend A/B, both arms | `logs/slurm/slm-mnn-backend-verify-40268007.out`, `logs/quant_eval/mnn-verify-40268007-{llama_cpp,mnn}/` |
| Full-loop MNN run (bounded) | `logs/slurm/slm-clinc150-mnn-verify-40268008.out`, `logs/runs/slm-clinc150-mnn-verify-40268008/` |
| The run that found the thread bug | `logs/slurm/slm-clinc150-mnn-verify-40260927.out` (tier 2 scored 0.0000 at `thread_num=14`) |
| A/B before the lm_head change | `logs/slurm/slm-mnn-backend-verify-40258512.out` (MNN 0.7278 at a uniform 4 bits) |
| **GPU A/B vs llama.cpp** | `logs/slurm/slm-mnn-backend-verify-40296584.out` (MNN/CUDA 54s against llama.cpp's 8s) |
| **GPU loop run, converged** | `logs/slurm/slm-clinc150-mnn-verify-40298521.out` (0.8965 vs goal 0.8856, 4 tiers, 3h11m) |
| **Backend matrix, 27/27** | `logs/slurm/slm-mnn-matrix-40305686.out`, `logs/probes/mnn-backend-matrix-40305686.json` |
| The matrix's own first version | `logs/probes/mnn-backend-matrix-40296583-INVALID-cuda-column.json` (an all-empty CUDA column reported as 27/27) |
| Re-runnable matrix | `sbatch tests/pipeline/verify_mnn_matrix.slurm` |
| Re-runnable A/B | `sbatch tests/pipeline/verify_mnn_backend.slurm` |
| Re-runnable loop proof | `sbatch tests/pipeline/run_clinc150_mnn_verify.slurm` |
| Tests | `tests/training/test_quant_backend.py`, `test_quantize_mnn.py`, `test_mnn_eval.py` |
