# Hypothesis length, the output budget, and what "extraction failed" means (2026-08-05)

Fifth follow-up. Written after the first re-run of CLINC150 (`slm-clinc150-cse-38179864`) was
cancelled 46 minutes in. **You were right on both counts** — the length cap was the wrong
mechanism, and I should have been prompting rather than truncating.

---

## Q1. "Why are you capping the orchestrator's hypothesis length at all? I thought a longer, more thought-out response is better, since the orchestrator adds that to its context for future runs?"

**You are right, and the cap was the wrong tool.** A hard character limit is a bad way to control
length for exactly the reason you give: the hypothesis is not disposable output, it is *written
into the run's memory and read back later*. Cutting it does not make the model more concise — it
makes the model's own past reasoning unreadable to itself. This morning's B238 fix established
that; capping the text after the fact was still a residue of the same mistake.

The cap should exist for one purpose only: stopping a pathological response from blowing up the
context. It should sit far above any legitimate answer and should never shape normal output.

| | Before today | This morning (B238) | Now |
|---|---|---|---|
| Hard cap | 240 chars | 2000 chars | **4000 chars, runaway guard only** |
| How length is actually managed | truncate silently | truncate loudly | **stated in the prompt** |

**Is a long hypothesis expensive?** Not meaningfully. Sonnet 5 output is $10/MTok, so even a
full 4096-token response costs about **$0.04**. Across ~20 iterations that is under a dollar. The
context cost is similarly small: run memory shows full reasoning for kept improvements plus the
five most recent failures, so roughly 7 × 4000 chars ≈ 7k tokens against a 1M window. There is no
budget reason to want a short hypothesis, only a *density* reason — padding is worthless, but
length itself is fine.

---

## Q2. "Can't you just prompt the orchestrator to keep its response under a certain number of tokens/words instead of truncating or throwing an error?"

**Yes, and that is now what happens.** It is embarrassing that it did not before: nothing in the
prompt had *ever* told the model how long a hypothesis should be, or that an output ceiling
existed at all. It was being silently judged against a limit it was never given.

Added to the decision prompt:

> Aim for about **150 words**: long enough to name the specific buckets and confusion pairs
> driving the decision, because **YOU WILL BE SHOWN THIS TEXT AGAIN** on later iterations as the
> record of your own reasoning — a vague hypothesis is worthless to your future self. Do not pad
> it with restatement of the numbers above. Your ENTIRE response, including this field, must fit
> in **4096 output tokens**; if you exceed that, the JSON is cut off mid-object and your decision
> is DISCARDED.

Both numbers are substituted from the constants, so the prompt cannot drift from the real limits.
Knobs: `SLM_HYPOTHESIS_TARGET_WORDS` (150), `SLM_ITERATE_MAX_TOKENS` (4096),
`SLM_HYPOTHESIS_MAX_CHARS` (4000).

### What actually went wrong in the cancelled run

This is worth stating plainly because it was **a regression I introduced this morning**.

```
[cost] stage=iterate            tokens=5360->1536
Decision failed validation (ValueError: no parseable JSON object in LLM response:
  '{\n "intervention": "data_rebuild",\n "hypothesis": "Score 0.6688 is well below the 0.80 …
[cost] stage=iterate_json_reask tokens=5471->1536
Reask also failed … falling back
LLM call failed …; using test-agent suggestion: hyperparameter
```

Both responses are **exactly 1536 output tokens** — `_ITERATE_MAX_TOKENS`. The model did not
write bad JSON; it ran out of room and was cut off mid-string. Removing the 240-char cap removed
the only thing that had been keeping responses short, and I left the output ceiling untouched.
The previous run's iterate calls peaked at 1,154 output tokens and never hit the ceiling once.

The damage is in the last line: the orchestrator chose **`data_rebuild`**, and the fallback ran
**`hyperparameter`**. Its actual decision was thrown away and replaced with a different one.

(For the record, the run-memory block was not to blame — it made the prompt *smaller*, 5,360
input tokens against 7,836 for the same point in the old run.)

---

## Q3. "Fix the thing with the reask too."

**Done.** The reask had a specific flaw: it treated every failure as a formatting failure.

A length failure and a format failure need **opposite** corrections. The old reask always said
"return valid JSON" — so a model that overran wrote another over-long answer and failed
identically. That is precisely what the log shows: two attempts, same wall, same token count.

Two changes:

1. A `max_tokens` stop reason is now detected **before** parsing and raised as its own error —
   *"response was cut off after 4096 output tokens (stop_reason=max_tokens) … The decision was
   too long, NOT malformed."*
2. The reask branches on it:

> Your previous answer RAN OUT OF OUTPUT SPACE and was cut off mid-object. Send the same
> decision again but MUCH SHORTER: keep every required field, and compress 'hypothesis' to at
> most 75 words.

12 tests in `tests/test_output_budget.py`. BUGS B240.

---

## Q4. "Are you telling me the extraction failures are intentional? What does extraction failed exactly mean?"

**The mechanism is intentional. The occurrences are not.** Those are two different things and I
should have separated them.

### What it means

The eval prompt asks the student model to reply with one label and nothing else. The scorer then
has to turn whatever text comes back into exactly one of the 151 CLINC150 labels. It tries, in
order: an exact word-boundary match, then a substring match. If neither hits, it records the
sentinel `__EXTRACTION_FAILED__`, which counts as **wrong for every class**.

So `__EXTRACTION_FAILED__` means: *the model produced a string that is not any of the allowed
labels.* Your three examples:

| input | model said | correct label | why it failed |
|---|---|---|---|
| how much is my water and sewer | `water_and_sewer` | `bill_balance` | invented from the input text |
| stop, just stop | `stop` | `cancel` | not a label |
| in canadian dollars, what is $30 | `currency` | `exchange_rate` | right idea, wrong vocabulary |

`currency` is the instructive one: the model *understood* the intent, but scoring a near-synonym
as correct would need fuzzy semantic matching, which would inflate every number the pipeline
reports. Strict matching is deliberate.

The **sentinel** is intentional — without it, a formatting failure would be indistinguishable
from a wrong answer, and you could not tell "the model can't follow the output format" from "the
model doesn't know the answer". The **failures themselves** are just the model being wrong.

### Why you saw three out of three

Sampling bias in the logger, and it is deliberate:

```104:105:eval/harness.py
    failed = [i for i in range(n) if str(predictions[i]) == "__EXTRACTION_FAILED__"]
    chosen = (failed + [i for i in range(n) if i not in set(failed)])[:_PREDICTION_SAMPLE_N]
```

Failures are listed first, so if three or more exist anywhere in the eval, all three samples
shown are failures — regardless of the real rate. **That eval had 14 failures out of 800: 1.75%.**

### The rate is actually the success story

| eval | failures / 800 | |
|---|---|---|
| Qwen3.6-35B reference | 3 | 0.4% |
| Qwen3-0.6B **untrained** | 272 | **34%** |
| Qwen3-0.6B after one fine-tune | 14 | **1.75%** |

An untrained 0.6B model does not know the label vocabulary and fails a third of the time; one
round of fine-tuning takes that to under 2%. That is the system working.

---

## Q5. "Just feed all the gold labels to Qwen 3.6 so it doesn't make some label that's not in the vocabulary."

Two answers, because this lands on two different components.

**For the eval prompt: already done, and it is what fixed this class of bug before.** Every
classification prompt enumerates the full label set:

```6:9:eval/scorers/classification.py
CLASSIFY_PROMPT = (
    'Classify this message into exactly one of these labels: {labels}.\n'
    'Reply with only the label word — nothing else.\n\nMessage: {text}'
)
```

The docstring records that listing the labels is exactly what resolved an earlier 100%
`__EXTRACTION_FAILED__` collapse (B161). The remaining 1.75% is a 0.6B model failing to comply
with an instruction it was given, not an instruction it was never given.

Also note: the model doing the failing here is **Qwen3-0.6B, the student being trained** — not
Qwen3.6-35B. Qwen3.6 is the teacher/judge, and it fails extraction on 3 rows out of 800.

**For synthesis: it cannot happen by construction, which I had to check to be sure.** For
classification the teacher is never asked to choose a label at all. It is asked only to write an
utterance, and the label is copied from the real anchor row:

```554:559:data/curriculum.py
        return {
            "text": text,
            "label": anchor.get("label"),
            "_source": "synth",
            "_provenance": "synthetic_positive",
        }
```

So an out-of-vocabulary label is impossible on that path. (I started implementing your suggestion
before checking, then reverted it — it would have added a parameter that nothing could ever use.
The JSON-schema path that *does* let the teacher pick a label is generation-family only:
math/code/generation, which have no closed label set.)

The real synthesis risk is different and already handled: the teacher can write text that does
not actually belong to the assigned label. That is B221/B222, and the teacher verifier pass added
yesterday is what addresses it.

---

## Q6. "Explain what you're talking about — ModelSpec, est_params_b, confusion-pair targeting."

Fair — I used internal names as if they were common nouns.

### `ModelSpec`

The dataclass in `config/android_pool.py` describing **one deployable model variant**: which base
model, which quantization, its on-disk size, and its tier. "Variant" matters because the same base
model appears several times — `Qwen3-0.6B` at BF16, Q8_0, and Q4_K_M are three separate
`ModelSpec`s with different sizes and possibly different tiers.

### `est_params_b`

A helper on `ModelSpec` that **estimates the parameter count in billions** by dividing on-disk
weight size by the bytes-per-parameter of its quantization (Q4_K_M ≈ 0.55 GB per billion
parameters, Q8_0 ≈ 1.0, BF16 ≈ 2.0). It exists so all three variants of one model report the same
parameter count despite different file sizes.

**Why it broke the sizing.** My formula needs a parameter count to decide "smaller model → more
data". I wrote `getattr(model, "params") or getattr(model, "n_params")` — **neither attribute
exists**. It always got `None`, fell back to the neutral capacity factor 1.0, and the target
collapsed to the floor:

```
[sizing] curriculum target for Qwen/Qwen3-0.6B [Q4_K_M]: 5000 —
  no baseline yet, assuming neutral novelty 0.50; unknown params -> size factor 1.00
```

That is the *same* symptom as the `DATASET_SIZE_BY_TYPE` table I deleted for always clamping to
the floor. I replaced it with something that did the same thing for a different reason.

There was a second, independent cause in that one line: the **zero-shot baseline** is measured by
the first `evaluate`, which runs *after* the first `curate` — so novelty defaulted to the neutral
0.5, and because I only re-sized when the *model* changed, tier 0 never revisited it. Both inputs
to a two-input formula were missing.

There was also a trap in the fix: `est_params_b` is a **method**, not a property (unlike
`selector`), so reading it without calling it returns a truthy bound method. My first fix looked
correct and still silently returned `None`. I only caught it by printing the actual number.

With both fixed:

| when | target | why |
|---|---|---|
| first curate (no baseline yet) | **5,952** | 0.84B params → factor 1.19, neutral novelty |
| after baseline 0.3152 is measured | **7,052** | novelty 0.685 → the real target |

BUGS B241.

### Confusion-pair targeting

A "confusion pair" is an ordered pair *(what the label should have been, what the model said)*
with a count — e.g. `change_ai_name → change_user_name (7)` means seven test rows whose true
intent was `change_ai_name` were classified as `change_user_name`. The eval produces the top
pairs; the orchestrator sees them as failure evidence.

**Targeting** is surgical synthesis using them: take the most-confused pairs, and spend part of
the synthesis budget generating extra training rows for the *gold* class of each, to sharpen that
specific decision boundary.

**The problem I flagged.** `__EXTRACTION_FAILED__` shows up in that list as a "predicted" value,
producing pairs like `alarm → __EXTRACTION_FAILED__ (5)` — which was a top target in the previous
run. But that names **no boundary to sharpen**: the sentinel is not a class, it means "the output
wasn't a label at all". The remedy for out-of-vocabulary output is format adherence, which
ordinary in-class training already provides. Spending targeted budget there teaches nothing about
the confusion, and it pollutes the per-pair effectiveness tracking.

Such pairs are now skipped for targeting and logged, while remaining in the report for diagnosis:

```
[surgical] ignoring 2 extraction-failure pair(s) (no class boundary to target;
format adherence, not confusion)
```

BUGS B242.

---

## Status

- **831 passed, 2 failed** — the same two pre-existing, unrelated failures.
- Re-submitted as job **38180646**. It is queued behind other users on the `gpu-l40s-cse`
  association (contention, not an error).
- Bugs filed: **B240** (output ceiling / reask), **B241** (sizing inputs missing),
  **B242** (extraction sentinel as a synthesis target).

### Still open

1. **Nothing here has run end to end yet.** All three fixes are unit-tested and the sizing numbers
   were verified against the real `ModelSpec`, but the run is what proves them.
2. **`__EXTRACTION_FAILED__` still counts as a wrong answer for every class**, which is correct
   for scoring but means a format-compliance problem and a genuine accuracy problem are averaged
   into the same macro-F1. Worth separating if extraction failures stay above ~1%.
