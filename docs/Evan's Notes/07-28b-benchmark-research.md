# Task Suite — Surviving Benchmarks, Explained From Scratch

**Date:** 2026-07-28
**Source doc:** *Mobile AI Factory — Task Suite Design*
**Filter applied:** removed every task invalidated by being **too easy**, **too hard**, or **having
no public benchmark**. 18 tasks in → **6 tasks out**.
**Then:** every surviving benchmark explained assuming zero prior knowledge, with worked examples.

---

## 0. What I removed, and why

### Removed — no public benchmark exists (5)

| Task | What I found |
|---|---|
| **1.1 Notification triage** | No dataset, no leaderboard, no paper with a public eval set. Nearest proxy is the Enron corpus, which is email folders — not notification priority. |
| **2.2 RRULE extraction** | Zero. No dataset, no benchmark, no prior work. TempEval-3's `SET` type marks "every Tuesday" as recurring but emits no recurrence rule. |
| **3a.3 Interruptibility** | No public text benchmark. The published work (InterruptMe, ~60% accuracy) is phone-sensor data, not text. |
| **3b.1 What-to-whisper** | Nothing on any axis — no dataset, no metric, no baseline. |
| **3b.3 Field-domain QA** | Text-only version has only AgXQA, which is paywalled with an unstated row count. Everything else in the domain (PlantVillageVQA, AgroBench) is **image** input, which also breaks your text-only rule. |

### Removed — too easy (4)

The test: a model far below 1B already saturates it, so all three of your pilot sizes pass and the
size search learns nothing.

| Task | The number that kills it |
|---|---|
| **1.2 Intent classification** | SetFit hits few-shot SOTA on a **110M** sentence encoder; full SOTA on BANKING77 is ~94.8%. A 1–4B model is 10–36× oversized. |
| **2.1 PII redaction** | SOTA on the largest public de-identification set is **F1 97.85 from a BiLSTM-CRF well under 10M parameters**. Your 99% bar is *above* published SOTA and the task is *below* your size floor — it fails in both directions at once. |
| **3a.5 Sensitive-content gating** | Meta ships **Llama Guard 3-1B** as a product, and ShieldGemma ships at 2B. The task is solved and deployed at the bottom of your range. |
| **3a.6 Conversational repair** | SOTA is **F1 88.1** from a BERT-scale tagger. Also not data-scarce — Disfl-QA is public. |

### Removed — too hard (1)

| Task | The number that kills it |
|---|---|
| **2.4 Unified diff over code** | Qwen2.5-Coder-**32B** scores **8.0%** on Aider's polyglot benchmark. At 1–3B you would measure zero, repeatedly. *(The task survives in reframed form — see §2.2 — because the same edit-as-diff idea over **prose** is a different difficulty class entirely.)* |

### Removed — no text benchmark (1)

| Task | Why |
|---|---|
| **3a.1 When-to-speak** | Every published system is acoustic (voice activity projection). Switchboard and CANDOR are audio corpora. There is no text-only turn-taking benchmark to compare against. |

### Removed — benchmark exists but its size is unverifiable (1)

| Task | Why |
|---|---|
| **1.3 Smart reply** | EnronSR and MRS are public, but I could not establish a row count for either from primary sources. You cannot plan a pilot against a benchmark whose size you don't know. |

> **One flag before moving on.** The "no public benchmark" filter removes **RRULE**, which by my own
> analysis was the single strongest task in the suite — precisely *because* it has no benchmark (no
> "why not just use the base model" objection) and because `rrule` round-tripping gives you a free
> exact verifier and unlimited free training data. The filter and the reason it was good are the
> same fact. I've removed it as instructed; if you want it back, it's the one removal I'd argue with.

---

## 1. The six survivors

| # | Task | Primary benchmark | Rows | 1–4B evidence | Metric type |
|---|---|---|---|---|---|
| **A** | Intent → app action | **BFCL** + TOPv2/MTOP | 125k / 100k | **Yes — full 0.6B→3B curve** | Parser-computable |
| **B** | Text edit as diff (prose) | **CoEdIT** | 82,000 | **Yes — 770M beats 175B** | Parser-computable |
| **C** | Ambiguity detection | **AmbigQA** + CLAMBER | 14,042 / 3,000 | No (headroom open) | Parser-computable |
| **D** | Escalate-to-cloud router | **RouterBench** | 405,000 | No (headroom open) | Parser-computable |
| **E** | Email/dialogue summarization | **EmailSum**, SAMSum, DialogSum | 2,549 / 16,369 / 13,460 | Flan-T5-L (780M) | ⚠ ROUGE — weak |
| **F** | Health / medication QA | **MedQA** family | 12,723 / 194k / 674 | No (headroom open) | Mixed |

**Read this honestly:** only **A** and **B** have published numbers in the 1–4B band. **C, D, F**
have public evaluation sets with no small-model baselines, which means open headroom *and* that you
must produce those baselines yourself. **E** has a metric its own authors call weakly correlated
with human judgment — it survives your three filters but it will draw fire.

---

## 2. The benchmarks, explained from scratch

Everything below assumes you've never seen these before.

---

### 2.A — Function calling: **BFCL**, plus TOPv2 and MTOP

#### What the task actually is

You give a model (1) a list of functions it's allowed to call, written like an API spec, and
(2) something a user said. The model has to emit the correct function call.

```
FUNCTIONS AVAILABLE:
  send_message(recipient: string, body: string, app: enum[MESSAGES, WHATSAPP, SIGNAL])
  set_alarm(time: string, label: string, repeat: bool)
  get_weather(location: string, day: string)
  ... 147 more ...

USER SAYS: "text mom on whatsapp that I'll be 20 minutes late"

CORRECT OUTPUT:
  send_message(recipient="mom", body="I'll be 20 minutes late", app=WHATSAPP)
```

The thing base models get wrong isn't the meaning — it's that they **invent functions that don't
exist** (`sendWhatsApp(...)`) or **invent enum values** (`app="whatsapp_messenger"`). That's why
this is a *format-bound* task: the vocabulary is closed and the model doesn't know it.

#### What BFCL is

**BFCL = Berkeley Function Calling Leaderboard.** It's a public scoreboard, run by a Berkeley
group, that ranks models on exactly this ability. It has shipped four versions (v1→v4), each
adding harder categories.

The categories, in plain terms:

| Category | What it tests |
|---|---|
| **Non-live AST** | Hand-written test cases. The model's output is parsed and compared structurally. |
| **Live** | Real queries contributed by actual users — messier, more realistic. |
| **Executable** | The function is *actually run* and the return value is compared. Catches calls that parse fine but do the wrong thing. |
| **Multi-turn** | A conversation over several exchanges, where a later call depends on an earlier result. **This is where small models collapse.** |
| **Relevance detection** | Should the model call *any* function, or is the user just chatting? Punishes over-eager calling. |

#### How the "AST" metric works — ELI5

**AST = Abstract Syntax Tree.** It just means "parse the text into a structure and compare
structures instead of comparing strings."

Say the gold answer is:
```
send_message(recipient="mom", body="running late", app=WHATSAPP)
```

Parsing turns that into:
```
{ function: "send_message",
  args: { recipient: "mom", body: "running late", app: "WHATSAPP" } }
```

Now the model outputs, with different spacing and argument order:
```
send_message( app = WHATSAPP , recipient = "mom" , body = "running late" )
```
Parsed → **identical structure** → **correct**. String comparison would have called this wrong;
AST comparison correctly ignores cosmetic differences.

The checks, in order:
1. Is the function name in the allowed list? (No → wrong. This catches hallucinated functions.)
2. Is it the *right* function?
3. Are all required arguments present?
4. Does each argument value match, allowing sane type coercion (`"20"` vs `20`)?

Score = fraction of test cases where all four pass. One test case is 1 or 0 — no partial credit.

#### The numbers, and what they mean

From *TinyLLM* ([arXiv:2511.22138](https://arxiv.org/html/2511.22138)):

| Model | Params | BFCL overall | Multi-turn |
|---|---|---|---|
| xLAM-2-3b-fc-r | 3B | **65.74%** | 55.62% |
| Qwen3-4B | 4B | 62.04% | 16.88% |
| Qwen3-1.7B | 1.7B | 55.49% | 16.88% |
| xLAM-2-1b-fc-r | 1B | 53.97% | 8.38% |
| Qwen3-0.6B | 0.6B | 45.76% | — |
| TinyLlama-1.1B-32k-Inst | 1.1B | 19.73% | 0.00% |
| TinyAgent-1.1B | 1.1B | 19.70% | 0.00% |

For xLAM-2-3b the overall 65.74% breaks down as 88.22 non-live AST / 81.03 live / 55.62 multi-turn
— i.e. it's near-excellent on single calls and mediocre on conversations.

**Why this is the single best task in your suite:**

1. **There is a real accuracy gradient across your exact size range** — 45.76% at 0.6B rising to
   65.74% at 3B. Your size search has something genuine to find. Most benchmarks in this document
   are flat across 1–4B; this one is not.
2. **Fine-tuning demonstrably works here.** xLAM-2-3b (a fine-tuned 3B) beats Qwen3-4B (a bigger
   general model). That's your entire thesis in one row.
3. **Two comparisons are inconsistent, and that matters.** TinyLlama-1.1B and TinyAgent-1.1B score
   ~19.7% while xLAM-2-1b scores 53.97% — same size, 2.7× the score. The difference is training
   data, not capacity. That's the strongest argument in the literature that *data*, not size, is
   the binding constraint at 1B.

**Hammer** is a separate on-device family: **7B 83.92 · 4B 76.05 · 1.5B 73.04**
([arXiv:2410.04587](https://arxiv.org/pdf/2410.04587)). ⚠ **Do not put these in the same table as
the numbers above** — Hammer reports BFCL-**v3** and TinyLLM reports a later version. Different
test sets. Different scales. Comparing them would be an error a reviewer catches immediately.

#### The other two datasets

**TOPv2** — **125,000** utterance→parse pairs across 8 phone-assistant domains (alarm, event,
messaging, music, navigation, reminder, timer, weather). Older format: instead of a function call
it uses a nested bracket tree.

```
"remind me to call mom when I get home"
→ [IN:CREATE_REMINDER
     [SL:TODO [IN:CREATE_CALL [SL:CONTACT mom ] ] ]
     [SL:TRIGGER [IN:GET_LOCATION [SL:LOCATION home ] ] ] ]
```

Metric is **Exact Match**: the whole tree must be character-identical to gold. All or nothing.
SOTA is **87.52% EM**.

**MTOP** — same idea, **100,000** utterances, 6 languages, 11 domains. Multilingual SOTA **67.2% EM**.

**Training data you can just download:** **xlam-function-calling-60k** — **60,000** verified rows,
**21 domains, 3,673 distinct APIs**, >95% correct on a 600-row human audit
([HF](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)). It was generated
synthetically by APIGen, which is why §4 treats it as your template.

#### Hardware profile

Long input (a 150-function spec is thousands of tokens) → short structured output (one call).
**Prefill-bound.** This is your TTFT / prompt-length-scaling task.

---

### 2.B — Text editing as a diff: **CoEdIT**

#### What the task actually is

The user has some text and an instruction. The model must produce the edited version — but instead
of rewriting the whole thing, it emits a **patch** describing only what changed.

```
ORIGINAL:
  The meeting was attended by John and it was decided by the team
  that the deadline will be moved.

INSTRUCTION: "Make this more concise and use active voice."

WHOLE-REWRITE OUTPUT (what base models do):
  John attended the meeting and the team decided to move the deadline.

DIFF OUTPUT (what you want):
  @@ -1,2 +1,1 @@
  -The meeting was attended by John and it was decided by the team
  -that the deadline will be moved.
  +John attended the meeting and the team decided to move the deadline.
```

#### Why the diff format is the hard part

That `@@ -1,2 +1,1 @@` header is **line arithmetic**, and it's why this is format-bound:

- `-1,2` = "starting at line **1** of the original, **2** lines are being replaced"
- `+1,1` = "starting at line **1** of the new version, **1** line appears"
- `-` prefixed lines = removed. `+` prefixed = added. Space-prefixed = unchanged context.

A model has to **count lines** to write that header. Language models are famously bad at counting.
Get the arithmetic wrong by one and the patch is rejected wholesale — even if the actual edit was
perfect.

#### The metric is binary and free

**Does the patch apply cleanly?** You feed it to `patch` or `git apply --check`. It either succeeds
or it doesn't. No judge, no ROUGE, no human. Then, separately, you check whether the *result* is
the desired text.

**This gives you the two-column split your measurement notes ask for, for free:**

| Column | Question | How measured |
|---|---|---|
| **Format valid** | Did the patch apply? | `git apply --check` exit code |
| **Content correct** | Is the result the right text? | string comparison after applying |

#### Why 1–4B is the right band

**CoEdIT** ([EMNLP Findings 2023](https://aclanthology.org/2023.findings-emnlp.350.pdf)) is an
instruction-tuning dataset of **82,000** text-editing instruction pairs — "make this simpler,"
"make this more formal," "fix the grammar," "make this more neutral."

The headline result: **CoEdIT-L, at 770M parameters, outperforms GPT-3-Edit at 175B** on writing
assistance — **227× smaller, and better.** That is the cleanest published demonstration in this
entire document that a small fine-tuned specialist beats a giant generalist on a bounded task.

**Critical scoping note.** The **code** version of this task is far too hard — Qwen2.5-Coder-32B
gets **8.0%** on Aider's 225-exercise polyglot benchmark. Apply the diff *format* to *prose* edits
and the difficulty lands squarely in your range. Keep the mechanical format, drop the programming.

**Prior art to cite for the two-column split:** Aider's leaderboards report "percent completed
correctly" and "percent using correct edit format" as separate columns
([edit.html](https://aider.chat/docs/leaderboards/edit.html)). The best illustration of why:
**gemma-3-27b-it emits a 100.0% well-formed edit format while solving 4.9% of tasks.** Perfect
format, near-zero competence. If you reported one number you'd have no idea which was happening.

Related datasets: **IteraTeR** (multi-domain corpus of iterative revisions with edit-intention
labels; row count unverified), **EditEval**
([arXiv:2209.13331](https://ar5iv.labs.arxiv.org/html/2209.13331)), **XATU**
([arXiv:2309.11063](https://arxiv.org/pdf/2309.11063)).

#### Hardware profile

Long input (the original text) → medium output (the diff). **Both prefill and decode.** This is
your peak-memory / worst-case-footprint task.

---

### 2.C — Ambiguity detection: **AmbigQA** and **CLAMBER**

#### What the task actually is

A user says something that could mean two different things. The model must notice — *before*
acting — and say which part is unclear.

```
USER: "Play the new album"
      → ambiguous: WHICH artist? (missing slot: artist)

USER: "Cancel my 3pm"
      → ambiguous if there are two 3pm events (missing slot: which event)

USER: "Set a timer for 10"
      → ambiguous: 10 minutes or 10 seconds? (missing slot: unit)

USER: "Text Sarah I'm on my way"
      → NOT ambiguous (assuming one Sarah)
```

Output is a binary flag plus, optionally, which slot is underspecified.

#### Why this is worth doing on-device

It is the cheapest possible guard against a confident wrong action. A model that asks "which
Sarah?" costs one round trip; a model that texts the wrong Sarah costs trust.

#### **AmbigQA** — 14,042 questions

Built on top of NQ-open (real Google search questions). Researchers went through them and found
that a huge fraction are ambiguous without anyone noticing. For each ambiguous one they wrote out
the disambiguated versions.

```
ORIGINAL:  "When did the Giants win the Super Bowl?"
           ← ambiguous! NY Giants or SF Giants? (and they've won several times)

DISAMBIGUATED:
   Q: "When did the New York Giants win their first Super Bowl?"    A: 1987
   Q: "When did the New York Giants most recently win?"             A: 2012
```

**How it's scored:** you don't just get credit for saying "ambiguous." You must produce the *set*
of (disambiguated question, answer) pairs, and you're scored with F1 over that set — how many of
the real interpretations you found, versus how many you invented.

*(Full metric details: [EMNLP 2020](https://aclanthology.org/2020.emnlp-main.466/).)*

#### **CLAMBER** — 3,000 queries

Smaller and more directly usable. Deliberately **balanced** (roughly half ambiguous, half not), and
each query carries:

1. a **binary** label — ambiguous or not
2. a **9-way** category of *what kind* of ill-posedness:
   `{none, polysemy, co-reference, what, when, where, whom, ICL, NK}`

ELI5 on the interesting categories:
- **polysemy** — a word has multiple meanings ("play *Mercury*" — planet, band, or Freddie?)
- **co-reference** — "it"/"that one" has no clear antecedent
- **what/when/where/whom** — a specific required slot is simply missing
- **NK (not knowable)** — no amount of clarification helps; the answer doesn't exist

**This maps almost exactly onto your spec** ("binary + which-slot"). It's the one to build against.

#### Metric — ELI5

Binary accuracy, and macro-F1 over the 9 classes.

**Macro-F1 in plain terms:** compute F1 separately for each of the 9 categories, then average the
9 numbers *equally*. This is deliberately unforgiving — if `NK` only appears 40 times in the test
set and you get all 40 wrong, it drags the average down by a full ninth, even though it's 1.3% of
the data. Micro-F1 (pooling everything) would hide that. **Use macro** — it's the honest one for
imbalanced label sets, and it's why your eval-set config warns about needing ≥30–50 examples per
class.

#### Status

**No published 1–4B baseline.** Open headroom, but you must produce the baseline yourself.
Related recent work: Abg-CoQA ([AKBC 2021](https://www.akbc.ws/2021/assets/pdfs/SlDZ1o8FsJU.pdf)),
CondAmbigQA ([arXiv:2502.01523](https://arxiv.org/html/2502.01523v1)),
*Reasoning about Intent for Ambiguous Requests* ([arXiv:2511.10453](https://arxiv.org/pdf/2511.10453)).

#### Hardware profile

Short-to-medium input → 1 token. Cheapest task in the suite; good as your floor case.

---

### 2.D — Escalate-to-cloud routing: **RouterBench**

#### What the task actually is

You have a small model on the phone and a big model in the cloud. For each incoming query, a tiny
**router** decides: *can the local model handle this, or should I pay to send it up?*

```
"set a timer for 10 minutes"          → LOCAL   (trivial)
"summarize this email thread"          → LOCAL   (the 3B can do it)
"explain the tax implications of ..."  → CLOUD   (local will confidently make it up)
```

The router never answers the question. It only predicts **whether the local model would get it
right** — which is why your doc calls it counterfactual: the label depends on *your* model's
failure boundary, not on any fact about the world.

#### What RouterBench is, and why it's unusually good

**RouterBench** ([arXiv:2403.12031](https://arxiv.org/abs/2403.12031)) is **405,000+ inference
outcomes** across **64 tasks** and **11 different LLMs**, covering commonsense reasoning, QA,
conversation, math, coding, and RAG.

Here's the part that makes it special, and it's worth understanding:

> **For every query, they recorded what *all eleven* models answered, whether each was correct, and
> what each cost.**

That means you can **simulate any routing policy offline, for free, with zero API calls.** Want to
know how your router would have performed? Look up its decisions in the table. No inference, no
cost, no waiting. You can evaluate a thousand router variants in seconds.

For a project whose bottleneck is GPU-hours, that property is worth a great deal.

#### How routing is scored — ELI5

A router isn't scored on accuracy alone, because you can trivially hit 100% by sending everything
to the cloud — and trivially hit $0 by sending nothing. The metric has to capture the **trade-off**.

So you sweep the router's threshold from "never escalate" to "always escalate," and at each setting
plot a point: **(average cost, average quality)**. Connect the points and you get a curve.

```
quality
  ^
  |                    ╭────────  always cloud (best quality, worst cost)
  |              ╭─────╯
  |        ╭─────╯                 ← a GOOD router bulges up-and-left
  |    ╭───╯
  |  ╭─╯
  |╭─╯  ← always local (worst quality, best cost)
  +──────────────────────────────>  cost
```

A better router's curve sits **above and to the left** of a worse one — more quality per dollar at
every budget. The single-number summary is the **area under that curve**. The paper also compares
against a **cascade** baseline: try the cheap model first, and escalate only if its answer looks
low-confidence.

#### The honesty problem, and how to turn it into your best result

Your source doc lists this under "no real data exists — inherently counterfactual." **RouterBench
is 405,000 rows of exactly that counterfactual data.** A reviewer finds it in one search.

**Invert it.** Train your router on a **purely synthetic** corpus from the factory. Then evaluate
on RouterBench, which it never saw. If a 300M synthetic-only router matches routers fit to real
routing traces, you have **external validation of the entire data-generation claim** — and that's
evidence a task with no public benchmark can never give you. The removed tasks can't do this. This
one can.

Related: routing survey ([arXiv:2603.04445](https://arxiv.org/pdf/2603.04445)), causal routing
([arXiv:2505.16037](https://arxiv.org/pdf/2505.16037)), dueling feedback
([arXiv:2510.00841](https://arxiv.org/pdf/2510.00841)), LLMRank
([arXiv:2510.01234](https://arxiv.org/pdf/2510.01234)).

#### Hardware profile

Long input (the user's full query) → 1 token (the decision). **Pure prefill / TTFT.** This is the
cleanest instance of that profile in the suite, and it's latency-critical in a way the others
aren't — a router that takes 400ms to say "go local" has eaten the entire latency budget it was
supposed to protect.

---

### 2.E — Summarization: **EmailSum**, **SAMSum**, **DialogSum**

#### What the task actually is

Long multi-message thread in, short summary out.

```
INPUT (an 8-message email thread, ~1,200 words):
  Alice: Can we move Thursday's review to Friday?
  Bob:   Friday's bad for me, I'm out. Wednesday?
  Alice: Wednesday works. Carol?
  Carol: Wednesday afternoon only.
  ...

OUTPUT (<30 words):
  "The team rescheduled Thursday's review to Wednesday afternoon after Bob
   flagged a conflict with Friday."
```

#### The datasets

| Dataset | Size | What it is |
|---|---|---|
| **EmailSum** | **2,549** email threads (3–10 emails each) | Human-written **short** (<30 words) and **long** (<100 words) summaries. The one that matches your task. |
| **SAMSum** | **16,369** (14,731 / 818 / 819) | Messenger-style chat conversations, written by linguists to resemble real chats. |
| **DialogSum** | **13,460** (12,460 / 500 / 500) | Real-life spoken dialogue transcripts. |

#### ROUGE, explained properly — because you need to know why it's weak

**ROUGE just counts word overlap between your summary and a human reference.** That's it. There's
no understanding involved.

**ROUGE-1** counts single-word overlap. Worked example:

```
REFERENCE:  "the cat sat on the mat"
GENERATED:  "the cat sat on a mat"
```
Reference words: `the, cat, sat, on, the, mat` (6)
Generated words: `the, cat, sat, on, a, mat` (6)
Overlapping: `the, cat, sat, on, mat` → **5**

- Recall = 5/6 = 0.833 ("how much of the reference did I cover?")
- Precision = 5/6 = 0.833 ("how much of what I wrote was in the reference?")
- **F1 = 0.833** → reported as **83.3**

**ROUGE-2** does the same with adjacent word *pairs*:
```
REFERENCE pairs: "the cat", "cat sat", "sat on", "on the", "the mat"   (5)
GENERATED pairs: "the cat", "cat sat", "sat on", "on a",   "a mat"     (5)
Overlapping: 3  →  R = 3/5 = 0.60  →  reported as 60.0
```
ROUGE-2 is always much lower than ROUGE-1 because word order has to match too.

**ROUGE-L** uses the longest common subsequence — rewards getting the overall word order right
without requiring the words be adjacent.

#### The published numbers

- **EmailSum**: T5-base scores **36.88** (short) / **43.65** (long); Hierarchical T5 is best on long.
  *(These are ROUGE ×100; the paper reports several variants — check the table for which.)*
- **DialogSum**: **Flan-T5-Large (780M)** — R-1 **38.8** / R-2 **14.4** / R-L **30.9**
  *(secondary source)*. This is your only 1–4B-adjacent anchor.

#### ⚠ The problem you must acknowledge

**The EmailSum authors state outright that ROUGE and BERTScore are only weakly correlated with
human judgment on this task.** Look at the worked example above: "the cat sat on a mat" scored
83.3 for a one-word change. Now consider "the cat did **not** sit on the mat" — that would also
score highly while meaning the opposite. ROUGE cannot see negation, factual errors, or
hallucinated names.

This task passes your three filters (public benchmark ✓, right difficulty ✓, not saturated ✓) but
**fails your criterion 2 — "objective metric where possible."** Keep it if you want a
decode-heavy, long-context workload in the suite, but plan to supplement ROUGE with your local
judge and say so explicitly. Do not let ROUGE be the headline number.

#### Hardware profile

Long input (a full thread) → medium output. **Both bottlenecks; peak memory.** This is your
worst-case-footprint task.

---

### 2.F — Health and medication QA: the **MedQA** family

#### What the task actually is

Two quite different shapes live under this heading, and you should treat them separately.

**Shape 1 — multiple choice (objectively gradable):**

```
Q: A 55-year-old man on warfarin is prescribed a new antibiotic. Two weeks
   later his INR is 6.2. Which antibiotic was most likely prescribed?
   (A) Cephalexin  (B) Trimethoprim-sulfamethoxazole
   (C) Azithromycin  (D) Nitrofurantoin

MODEL OUTPUT: "B"          METRIC: accuracy — exact match on the letter.
```

**Shape 2 — free-text consumer questions (not objectively gradable):**

```
Q: "Can I take ibuprofen with my blood pressure medication?"
A: (a paragraph written by a medical expert)

METRIC: rubric-based LLM grading, or human review. No exact match possible.
```

#### The datasets

| Dataset | Rows | Format | Source |
|---|---|---|---|
| **MedQA** | **12,723** | 4–5 option MCQ | US Medical Licensing Exam (USMLE) questions |
| **MedMCQA** | **194,000** | 4-option MCQ | Indian medical entrance exams (AIIMS, NEET-PG) |
| **PubMedQA** | **1,000** expert + **211,000** auto-labeled | yes / no / maybe | Given a real PubMed abstract, answer the paper's own research question |
| **MedicationQA** | **674** | free text | **Real consumer questions submitted to MedlinePlus**, answered by experts |
| **HealthSearchQA** | **3,173** | free text | Real consumer health search queries |

**MedicationQA is the one that matches your task.** 674 rows is small, but they're *real people
asking real medication questions* — which is exactly your deployment story, and it's a fixed
held-out set you did not construct.

#### Where the field is

- **Med-PaLM 2**: **86.5%** on MedQA-USMLE — the first to clear the ~85% human-expert threshold.
- Ensemble systems: **96.8%** MedQA / **94.2%** MedMCQA on 500-question samples.
- Fine-tuned biomedical models: ~**78%** on PubMedQA.

*(All secondary, via the [Medical LLM Leaderboard 2026](https://awesomeagents.ai/leaderboards/medical-llm-leaderboard/).)*

**MedQA is saturating for frontier models** — strong general models pass without any medical
fine-tuning, which is why MedXpertQA was built to restore discrimination. **But it is nowhere near
saturated at 1–4B**, and no 1–4B numbers are published. Your headroom is real; just don't frame
MedQA as a hard benchmark in the abstract.

**Read before committing:** **HealthSLM-Bench** — *benchmarking small language models for mobile
and wearable healthcare* ([arXiv:2509.07260](https://arxiv.org/pdf/2509.07260)). It is the closest
published work to your exact framing.
Also: BRIDGE ([arXiv:2504.19467](https://arxiv.org/pdf/2504.19467)), Medmarks
([arXiv:2605.01417](https://arxiv.org/pdf/2605.01417)), OTC dosing under temporal uncertainty
([arXiv:2606.04262](https://arxiv.org/pdf/2606.04262)).

#### ⚠ Two risks, stated plainly

1. **This is the only task in the surviving suite where a wrong answer can hurt someone.** A
   fine-tuned 3B confidently giving drug-interaction advice is a different risk class from a
   mis-scheduled reminder. If you keep it, keep it multiple-choice (objectively gradable, no
   free-text generation shipped), and say in the paper that you are measuring knowledge, not
   deploying advice.
2. **The free-text half is graded by LLM rubric** (MedHELM does this). That's precisely the
   "factory grades its own homework" objection, on the highest-stakes task. Use MedicationQA's 674
   rows as a fixed set with human spot-checks, not as a training signal.

#### Hardware profile

MCQ: medium input → 1 token. Free-text: short input → **long output — decode-bound**, sustained
power, thermal. This is your KV-cache / battery task.

---

## 3. Does the surviving suite still meet your four criteria?

### Criterion 4 — hardware-profile spread ✅ **complete, 4/4**

| Output shape | Bottleneck | Which surviving task |
|---|---|---|
| Long input → 1 token | Prefill / TTFT | **D** Router (full query in, one decision out) |
| Long input → short structured | Balanced | **A** Function calling (API spec in, one call out) |
| Short input → long output | Decode / bandwidth | **F** MedicationQA free-text answers |
| Long input → medium output | Both / peak memory | **E** Summarization, **B** Text editing |

### Criterion 3 — size spread ✅ **preserved**

| Band | Tasks |
|---|---|
| 150–600M | **C** Ambiguity, **D** Router |
| 770M–3B | **B** Text editing (CoEdIT-L is 770M), **E** Summarization (Flan-T5-L is 780M) |
| 1–3B | **A** Function calling (measured gradient 0.6B → 3B) |
| 3B+ | **F** Medical QA |

### Criterion 2 — objective metric ⚠ **4 of 6 clean**

| Task | Metric | Objective? |
|---|---|---|
| **A** Function calling | AST match / exact match | ✅ Fully parser-computable |
| **B** Text editing | `git apply` + string compare | ✅ Fully parser-computable |
| **C** Ambiguity | Accuracy + macro-F1 | ✅ Fully parser-computable |
| **D** Router | Cost-quality curve area | ✅ Fully computable (offline replay) |
| **E** Summarization | ROUGE | ❌ Authors say it's weakly correlated with humans |
| **F** Medical QA | Accuracy (MCQ) / rubric (free text) | ⚠ Half and half |

### Criterion 1 — text-only, on-device justified ✅ **all six**

| Task | Why not the cloud |
|---|---|
| **A** Function calling | Latency — an assistant action must feel instant |
| **B** Text editing | Privacy — you're editing the user's own writing |
| **C** Ambiguity | Latency — it gates every other action |
| **D** Router | It *is* the cloud-avoidance mechanism; cloud-routing is self-defeating |
| **E** Summarization | Privacy — email and message content |
| **F** Medical QA | Privacy — health and medication data, by construction |

---

## 4. Cost of generating synthetic data

### 4.1 What your two completed runs actually cost

From `logs/runs/*/cost.json` (schema v2, pricing effective 2026-07-21):

| Run | Wall clock | Total | Anthropic calls | Input tok | Output tok |
|---|---|---|---|---|---|
| NER (37531245) | 44.8 h | **$13.43** | 231 | 3,584,126 | 177,361 |
| Math (37576194) | — | **$3.20** | 73 | 742,965 | 62,083 |

Per stage, NER:

| Stage | Calls | $ | Share |
|---|---|---|---|
| `iterate` | 141 | 8.4525 | 63.0% |
| `iterate_json_reask` | 86 | 4.9026 | 36.5% |
| `model_selection` | 1 | 0.0227 | |
| `acquire_dataset_discovery` (Exa) | 3 | 0.0210 | |
| `task_analysis` | 1 | 0.0153 | |
| `escalate` | 1 | 0.0136 | |
| `hardware_research` | 1 | 0.0060 | |
| `synth_preflight` (local) | 9 (**8 failed**) | 0.0000 | |

**Three findings:**

1. **Synthetic data cost $0.00 — because it never ran.** The only local-model events in either
   ledger are `synth_preflight`, which failed **8/9** in NER and **18/43** in math. There are zero
   `hard_negative_synthesis` events. Combined with `resample_existing` firing 31/31 in NER, this
   confirms **no synthetic data was generated in either completed run.** Fix the endpoint before
   building a suite that depends on it.
2. **`iterate` + `iterate_json_reask` = 99.4% of the bill.** Data generation is not your cost
   driver. Orchestrator decisions are.
3. **`cache_tokens = 0` — prompt caching is off.** `iterate` averages **15,391 input tokens/call**.
   Cached input is $0.30/Mtok vs $3.00. Enabling caching on the stable preamble should cut
   `iterate` from ~$8.45 toward **$1.50–2.50**. Cheapest win in the repo.

### 4.2 Marginal cost per synthetic row

**(a) Local vLLM — what the repo is designed to do.** `SYNTH_MODEL = Qwen/Qwen3.6-35B-A3B` served
by `scripts/serve_synth.slurm` ([`config/config.py:157`](../../config/config.py#L157)). MoE with ~3B
active params. 10,000 rows × ~150 output tokens ≈ 1.5M tokens ≈ **10–25 min of GPU time** ≈
**$0.25–0.85** at cloud-equivalent L40S rates; **free on Hyak.** The real cost is the reservation,
not the tokens — the synth server holds a GPU for the whole run regardless of volume.

**(b) Claude API.** Using the repo's own pricing table and the actual prompt shape (~250 in / 150 out):

| Model | $/row | 10k rows | 100k rows |
|---|---|---|---|
| claude-haiku-4-5 | $0.0010 | **$10** | $100 |
| claude-sonnet-5 | $0.0020 | **$20** | $200 |
| claude-sonnet-4-6 | $0.0030 | **$30** | $300 |
| claude-opus-5 | $0.0050 | **$50** | $500 |

For task **A** the prompt is much larger — a 150-function API spec plus a seed is ~2,000 input
tokens, with ~600-token outputs:

| Model | $/row (2k in / 600 out) | 10k rows |
|---|---|---|
| claude-haiku-4-5 | $0.005 | **$50** |
| claude-sonnet-4-6 | $0.015 | **$150** |
| claude-opus-5 | $0.025 | **$250** |

**Reducers:** prompt caching cuts the input term ~10× (the API spec is byte-identical across every
row — the ideal caching case); the Batch API halves the rest. Together ~$150 → ~$25 per 10k.

**External anchors:** Stanford Alpaca generated **52,000** samples for **<$500** ≈ **$0.0096/sample**
([Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)). APIGen produced the 60,000-row xlam
set with DeepSeek-V2-Chat + Mixtral-8x22B ([arXiv:2406.18518](https://arxiv.org/pdf/2406.18518)).

### 4.3 Whole-suite budget — 6 tasks

| Component | Estimate |
|---|---|
| Synthetic data, local | ~$0 on Hyak; **≤7 GPU-hours** total |
| Synthetic data, if API (cached + batched) | **$150–300** for all six |
| Orchestrator, 6 runs at the NER rate | **~$80** |
| Orchestrator with caching on | **~$20–35** |
| Pilot screening: 3 sizes × 6 tasks | **18 fine-tunes — the actual bottleneck** |

**Dollars are negligible. GPU-hours are not.** Your NER run had the evaluate node at **25.36 h vs
18.21 h for training** — evaluation is already the largest consumer, and "log accuracy at
intermediate data increments" multiplies it by the number of increments. **Do intermediate points
on a subsampled eval set; run the full set only at the final checkpoint.**

---

## 5. Making the pipeline handle these six tasks

### 5.1 There are exactly three output schemas today

[`data/loaders/web_acquire.py:1724`](../../data/loaders/web_acquire.py#L1724) branches on `task_type`:

| task_type | schema | validation |
|---|---|---|
| `classification` | `{"text", "label"}` | label ∈ allowed set |
| `NER` | `{"text", "entities":[{"text","type"}]}` | span is a substring of text |
| everything else | `{"text", "answer"}` | answer non-empty |

And the generator prompt in
[`data/curriculum.py:419`](../../data/curriculum.py#L419) is a hardcoded f-string; the only
orchestrator-controlled inputs are a one-sentence `pattern_hint` and `temperature`. Same on-rails
pattern you just fixed in `data_rebuild`.

### 5.2 How the six survivors map onto those schemas

| Fits | Tasks |
|---|---|
| `{text,label}` ✅ | **C** Ambiguity, **D** Router |
| `{text,answer}` ⚠ | **E** Summarization, **F** Medical QA — schema fits, but "answer must be correct and verifiable" is false; both need the local judge |
| **Nothing** ❌ | **A** Function calling, **B** Text editing |

The two that fit nothing are your two best tasks. They need a schema the system can't express, an
**executable** verifier (AST/arg match; `git apply --check`), and a generator prompt carrying a
closed vocabulary (the API spec, the diff grammar).

### 5.3 The fix: a `TaskContract` emitted at cold start

```jsonc
{
  "schema":           { /* JSON Schema for one row */ },
  "generator_prompt": "…{seed}…{target_label}…{difficulty}…{pattern_hint}…",
  "verifier":         {"kind": "registry", "name": "ast_call_match", "params": {...}},
  "metric":           {"kind": "exact_match" | "applies_cleanly" | "macro_f1" | "judge", ...}
}
```

- Versioned in `state`, and **re-generatable by `data_rebuild`** — so the contract stays flexible
  across iterations, exactly like the fix you just made. A rebuild that only reshuffles rows cannot
  repair a wrong schema.
- **Prefer a verifier *registry* over generated code.** Have the orchestrator *select and
  parameterize* from `{exact_match, json_schema, macro_f1, git_apply_check, ast_call_match,
  routing_curve, judge_rubric}` rather than write Python. That set covers all six surviving tasks,
  is ~200 lines, needs no sandbox, and can't execute a hallucinated `os.system`.
- **Log verifier accept-rate as a first-class metric.** It's the yield signal `data_rebuild`
  already tracks, and the natural home for schema-valid-rate reported separately from accuracy.

**Why the verifier matters most:** APIGen's 60k rows are >95% correct because every row passes
format check → **actual function execution** → semantic verification
([arXiv:2406.18518](https://arxiv.org/pdf/2406.18518)). Without an executable accept/reject test,
synthesis has no error signal and you train on the teacher's mistakes.

### 5.4 Split the score everywhere

`eval/scorers/` returns one number. Return two: **`format_valid`** and **`content_correct`**.
Aider has published these as separate columns for years, and gemma-3-27b-it's **100% format /
4.9% correct** shows exactly why one number is not enough. Without the split you cannot measure the
quantization effect below.

---

## 6. The one novel result available to you

**Nobody has published format-validity as a function of quantization bit-width with model, prompt,
and task held fixed.** The literature has each half separately:

- **Format failure is catastrophic and independent of task competence.** *When Correct Isn't Usable*
  ([arXiv:2605.02363](https://arxiv.org/html/2605.02363v1)) tested Llama-3.1-8B, Gemma-2-9B and
  Qwen-2.5-7B under a two-field JSON contract: **task accuracy 76–85% on GSM8K but JSON validity
  0%**, so output accuracy 0%. **Served at BF16 only — quantization was not a variable.**
- **The one 4-bit datapoint:** StructuredRAG
  ([arXiv:2408.11061](https://arxiv.org/html/2408.11061v1)) ran Llama 3 8B **at 4-bit** — mean JSON
  success **82.55%**, range **0–100%**. No BF16 control.
- ⚠ **Turn constrained decoding off when you measure this.** Grammar-constrained decoding
  (JSONSchemaBench, [arXiv:2501.10868](https://arxiv.org/html/2501.10868v3)) forces valid output and
  would mask the entire effect.

**Your semantic-half measurement is already done** (job 37818082, base Qwen3.5-4B, GSM8K, 800 examples):

| Variant | F1 | Δ vs bf16 | Size |
|---|---|---|---|
| bf16 | **0.8213** | — | 8888.1 MB |
| Q8_0 | **0.8100** | −0.0112 (−1.37%) | 4397.0 MB (49%) |
| Q4_K_M | **0.7913** | −0.0300 (−3.65%) | 2654.5 MB (30%) |

Monotonic in bit-width, and **37.5× the 0.0008 bf16→Q4 gap on fine-tuned NER** — same model, same
harness, same eval size, only the task type differs. **Quantization sensitivity is task-dependent**,
which is itself a reportable result: a suite spanning routing, structured output, and generation
will show a spread no single-task study can produce.

And that's all *without* a format contract. Tasks **A** and **B** are format-bound by construction.
Run the same sweep on them with the two-column scorer and you have the result.

**Also check deployability while you're there:** MobileAIBench
([arXiv:2406.10290](https://arxiv.org/pdf/2406.10290)) benchmarks on an iPhone 14 with everything
at 4-bit and finds **only models under 3B are deployable**. Consistent with your own measurement —
Qwen3.5-4B Q4_K_M is **2654.5 MB**, ~20% above what the pool's size arithmetic predicted, which is
why you moved it to tier 3. **State your reference device and its RAM budget explicitly**, or the
3B+ tier (task **F**) is a claim you can't cash. MobileAIBench also establishes **battery drain
rate** as a standard on-device metric — worth adding.

---

## Sources

**Function calling (A):**
[BFCL leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) ·
[BFCL paper](https://openreview.net/pdf?id=2GmDdhBdDk) ·
[BFCL repo](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/README.md) ·
[TinyLLM (1–4B numbers)](https://arxiv.org/html/2511.22138) ·
[Hammer](https://arxiv.org/pdf/2410.04587) · [xLAM](https://arxiv.org/pdf/2409.03215) ·
[APIGen](https://arxiv.org/pdf/2406.18518) ·
[xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) ·
[TOPv2 / RAF](https://ar5iv.labs.arxiv.org/html/2202.00901) ·
[MTOP](https://aclanthology.org/2021.eacl-main.257/) · [STOP](https://arxiv.org/pdf/2207.10643)

**Text editing (B):**
[CoEdIT](https://aclanthology.org/2023.findings-emnlp.350.pdf) ·
[CoEdIT blog](https://www.grammarly.com/blog/engineering/coedit-text-editing/) ·
[EditEval](https://ar5iv.labs.arxiv.org/html/2209.13331) · [XATU](https://arxiv.org/pdf/2309.11063) ·
[Aider code editing](https://aider.chat/docs/leaderboards/edit.html) ·
[Aider polyglot](https://aider.chat/docs/leaderboards/)

**Ambiguity (C):**
[AmbigQA](https://aclanthology.org/2020.emnlp-main.466/) ·
[Abg-CoQA](https://www.akbc.ws/2021/assets/pdfs/SlDZ1o8FsJU.pdf) ·
[CondAmbigQA](https://arxiv.org/html/2502.01523v1) ·
[Intent for ambiguous requests](https://arxiv.org/pdf/2511.10453)

**Routing (D):**
[RouterBench](https://arxiv.org/abs/2403.12031) ·
[RouterBench OpenReview](https://openreview.net/forum?id=IVXmV8Uxwh) ·
[Routing survey](https://arxiv.org/pdf/2603.04445) ·
[Causal routing](https://arxiv.org/pdf/2505.16037) ·
[Dueling feedback](https://arxiv.org/pdf/2510.00841) · [LLMRank](https://arxiv.org/pdf/2510.01234)

**Summarization (E):**
[EmailSum](https://aclanthology.org/2021.acl-long.537/) ·
[EmailSum arXiv](https://arxiv.org/pdf/2107.14691) ·
[Dialogue summarization sizes](https://pmc.ncbi.nlm.nih.gov/articles/PMC12192768/) ·
[Zero-shot conversational summarization](https://arxiv.org/pdf/2311.18041)

**Medical QA (F):**
[Med-PaLM / MultiMedQA](https://arxiv.org/pdf/2212.13138) ·
[Medical LLM leaderboard 2026](https://awesomeagents.ai/leaderboards/medical-llm-leaderboard/) ·
[MedQA/MedMCQA overview](https://www.emergentmind.com/topics/medqa-and-medmcqa) ·
[Consumer health QA survey](https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.12140) ·
[HealthSLM-Bench](https://arxiv.org/pdf/2509.07260) · [BRIDGE](https://arxiv.org/pdf/2504.19467) ·
[Medmarks](https://arxiv.org/pdf/2605.01417) · [OTC dosing](https://arxiv.org/pdf/2606.04262)

**Quantization & on-device:**
[When Correct Isn't Usable](https://arxiv.org/html/2605.02363v1) ·
[StructuredRAG](https://arxiv.org/html/2408.11061v1) ·
[JSONSchemaBench](https://arxiv.org/html/2501.10868v3) ·
[Quantization failure modes](https://arxiv.org/pdf/2604.19884) ·
[Extreme low-bit reasoning](https://arxiv.org/pdf/2606.02011) ·
[MobileAIBench](https://arxiv.org/pdf/2406.10290) ·
[MobileAIBench OpenReview](https://openreview.net/pdf?id=EEbRrNsiiD) ·
[PalmBench](https://arxiv.org/pdf/2410.05315)

**Synthetic data cost:**
[Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) ·
[Synthetic data cost guide 2026](https://blog.premai.io/how-to-generate-synthetic-training-data-for-llm-fine-tuning-2026-guide/)

**Evidence for the removals** (kept so the cuts are auditable):
[SetFit 110M](https://arxiv.org/pdf/2209.11055) · [FastFit](https://arxiv.org/pdf/2404.12365) ·
[BANKING77](https://huggingface.co/datasets/PolyAI/banking77) ·
[i2b2 de-id F1 97.85](https://arxiv.org/abs/1606.03475) ·
[TAB](https://aclanthology.org/2022.cl-4.19/) ·
[ai4privacy 300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k) ·
[Llama Guard 3-1B-INT4](https://arxiv.org/pdf/2411.17713) ·
[ShieldGemma](https://arxiv.org/pdf/2407.21772) ·
[Disfl-QA](https://arxiv.org/pdf/2106.04016) ·
[Disfluency SOTA 88.1](https://arxiv.org/pdf/1808.09092) ·
[TempEval-3](https://arxiv.org/abs/1206.5333) ·
[NLP-progress temporal](http://nlpprogress.com/english/temporal_processing.html) ·
[Multilingual VAP](https://arxiv.org/html/2403.06487v1) ·
[PlantVillageVQA](https://arxiv.org/abs/2508.17117) ·
[AgXQA](https://www.sciencedirect.com/science/article/abs/pii/S0168169924007403) ·
[EnronSR](https://sel.sise.bgu.ac.il/assets/pubs/enron-sr-icwsm-2024.pdf) ·
[Enron corpus](https://link.springer.com/chapter/10.1007/978-3-540-30115-8_22)
