#!/usr/bin/env python3
"""
A no-measurement baseline for the requirement filter in docs/SCENARIOS.md: ask a Claude chat that has only what the
SLM Factory knows without this harness (phone specs, model cards, runtime and backend) to filter each scenario's pool
and pick a configuration, then score its answer against the measured filter.

  python baseline.py prompt                  # writes baseline/baseline_prompt.md and baseline/scenario_ids.json
  python baseline.py naive                   # writes baseline/baseline_prompt_naive.md
  python baseline.py web                     # writes baseline/baseline_prompt_web.md (web search allowed)
  python baseline.py score RESPONSE.txt      # writes docs/BASELINE_COMPARISON.md
  python baseline.py score RESPONSE.txt --naive   # writes docs/BASELINE_COMPARISON_NAIVE.md
  python baseline.py score RESPONSE.txt --web     # writes docs/BASELINE_COMPARISON_WEB.md

The prompt uses neutral, shuffled scenario IDs (S01..S60), so the E/M/H tier is not given away. The naive prompt has
the same scenarios and IDs, asked the way a user would ask: phone names, the choices and plain-language wishes, with
no specs, sizes, limits or task details.
"""

from __future__ import annotations

import json
import random
import re
import statistics as st
import sys
from collections import Counter
from datetime import datetime

from req_data import DOCS, HERE, PARAMS_B, load
from requirements_md import LABEL, TIERS
from scenarios_md import BATTERY, COLUMNS, FW_LABEL, PHONE_ORDER, RAM_GIB, TASKS, num, request_j, scenarios

OUT_DIR = HERE / "baseline"
PROMPT = OUT_DIR / "baseline_prompt.md"
NAIVE_PROMPT = OUT_DIR / "baseline_prompt_naive.md"
ID_MAP = OUT_DIR / "scenario_ids.json"
REPORT = DOCS / "BASELINE_COMPARISON.md"
NAIVE_REPORT = DOCS / "BASELINE_COMPARISON_NAIVE.md"
WEB_PROMPT = OUT_DIR / "baseline_prompt_web.md"
WEB_REPORT = DOCS / "BASELINE_COMPARISON_WEB.md"
FLAGSHIP = "OnePlus 15"  # any 104-configuration phone: the pool's keys and file sizes are the same on all of them
FAMILY = {"smollm2-135m": "SmolLM2 (HuggingFace)", "gemma3-270m": "Gemma 3 (Google)", "smollm2-360m": "SmolLM2",
          "qwen3-0.6b": "Qwen3", "qwen3.5-0.8b": "Qwen3.5", "qwen3-1.7b": "Qwen3", "qwen3.5-2b": "Qwen3.5",
          "qwen3-4b-instruct-2507": "Qwen3 (Instruct-2507)", "qwen3.5-4b": "Qwen3.5"}
PHONE_SPECS = {
    "Galaxy S23": "Snapdragon 8 Gen 2 (SM8550): 1× Cortex-X3 3.36 GHz, 2× A715 + 2× A710 2.8 GHz, 3× A510 2.0 GHz; "
                  "Adreno 740 GPU; 8 GB LPDDR5X; Android 14",
    "OnePlus 15": "Snapdragon 8 Elite Gen 5 (SM8850): 2× Oryon prime 4.6 GHz, 6× Oryon performance 3.62 GHz; "
                  "Adreno 840 GPU; 16 GB LPDDR5X; Android 16",
    "OnePlus 8 Pro": "Snapdragon 865 (SM8250): 1× Cortex-A77 2.84 GHz, 3× A77 2.42 GHz, 4× A55 1.8 GHz; "
                     "Adreno 650 GPU; 12 GB LPDDR5; Android 13",
    "Redmi 9A": "MediaTek Helio G25 (MT6762): 8× Cortex-A53 up to 2.0 GHz; PowerVR GE8320 GPU (no usable OpenCL, "
                "so CPU only); 4 GB LPDDR4X; 32-bit Android 11 (armeabi-v7a builds)",
}


def pool_keys(data) -> list[tuple]:
    """Configuration IDs C001..C104 in a fixed order: model (smallest first), bits, runtime, backend."""
    rows = data[FLAGSHIP]
    order = list(LABEL)
    return sorted(rows, key=lambda k: (order.index(k[0]), rows[k]["bits"], COLUMNS.index((k[1], k[2]))))


def ids(data) -> dict:
    return {k: f"C{i:03d}" for i, k in enumerate(pool_keys(data), 1)}


def shuffled(chosen) -> list[tuple[str, dict]]:
    order = list(chosen)
    random.Random(20261028).shuffle(order)
    return [(f"S{i:02d}", s) for i, s in enumerate(order, 1)]


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

def write_prompt(web: bool = False):
    """The detailed prompt; with web=True, the same prompt with web search allowed."""
    data = load()
    _, chosen = scenarios(data)
    cid, rows = ids(data), data[FLAGSHIP]
    sources = ("You have no benchmark measurements from these phones. Search the web as much as you like: published "
               "benchmarks of these chips, runtimes and models, runtime documentation and issue trackers, model "
               "cards. You may use code for arithmetic. Commit to an answer for every configuration: this is a test "
               "of how well the choice can be made from published information, without measuring on the phone."
               if web else
               "You have no benchmark measurements from these phones. Use what you know about the phones' chips, the "
               "runtimes (llama.cpp and MNN) on Android CPUs and OpenCL GPUs, and the models. Do not search the web "
               "for benchmark numbers. You may use code for arithmetic. Commit to an answer for every configuration: "
               "this is a test of how well the choice can be made without measuring on the phone.")
    L = ["You are helping evaluate an \"SLM Factory\": a system that picks an on-device small language model "
         "configuration for an app on a specific Android phone. Below are four phones, a pool of 104 configurations "
         "(9 models × 3 precisions × 2 runtimes × 2 backends) and 60 scenarios. Each scenario deploys one of ten language "
         "tasks in a specific app setting on one phone, with a limit on four measured quantities. Your job, for every scenario, is to predict which "
         "configurations would meet all four limits on that phone, why the others would not, and which one to "
         "deploy.",
         "",
         sources,
         "",
         "## How a configuration is measured",
         "",
         "Each configuration was benchmarked on each phone with a 256-token prompt followed by a 256-token answer:",
         "",
         "- **Peak memory (MB):** the peak memory footprint of the inference process during that run, CPU memory "
         "plus GPU allocations (PSS). A scenario checks the measured peak plus the KV cache (F16 keys and values) "
         "of the tokens a request holds beyond those 512 (cached prefix + prefilled + answer); only the intent-"
         "routing (~1,060 tokens) and command-parsing (~930) tasks exceed 512.",
         "- **Prefill (tokens/s):** prompt-processing speed for the 256-token prompt.",
         "- **Decode (tokens/s):** generation speed over the 256-token answer (after the 256-token prompt).",
         "- **Energy (millijoules per token, prefill and decode separately):** whole-phone battery energy net of the "
         "phone's idle draw, screen off. A scenario's **energy per request** is prefilled tokens × prefill mJ/token "
         "+ answer tokens × decode mJ/token, in joules.",
         "",
         "Tools: llama.cpp `llama-bench` (CPU: no GPU layers; GPU: all layers on OpenCL) and MNN `llm_bench` (CPU or "
         "OpenCL). Default thread settings. Builds: Qwen3.5 models need newer builds (llama.cpp 9c2e0e4, MNN "
         "a8ec497); all other models use llama.cpp eadc418 and MNN 510ac8f.",
         "",
         "**Cannot run.** Some configurations cannot run on a phone at all: the model does not fit in memory, fails "
         "to load, crashes, or the GPU backend does not work on that phone's driver. The harness does not even try a "
         "configuration whose model file is larger than 55% of the phone's usable RAM (for MNN F16 on CPU, twice "
         "the file size, because MNN unpacks F16 weights to 32-bit on CPU). Others can still fail when tried.",
         "",
         "## Phones",
         "",
         "| Phone | Hardware | Usable RAM | Battery | Pool |",
         "|---|---|--:|--:|---|"]
    for p in PHONE_ORDER:
        mah, wh = BATTERY[p]
        pool = "CPU configurations only (52)" if p == "Redmi 9A" else "all 104"
        L.append(f"| {p} | {PHONE_SPECS[p]} | {RAM_GIB[p]:.2f} GiB | {mah:,} mAh ({wh} Wh) | {pool} |")
    L += ["", "## Configuration pool", "",
          "Parameters and file sizes are from the model cards and files. Precision names differ by runtime: "
          "llama.cpp Q4_0 / Q8_0 / F16 (GGUF), MNN Q4 / Q8 (4/8-bit weights) / F16.", "",
          "| ID | Model | Family | Parameters (B) | Precision | Bits | File size (MB) | Runtime | Backend |",
          "|---|---|---|--:|---|--:|--:|---|---|"]
    for k in pool_keys(data):
        r = rows[k]
        L.append(f"| {cid[k]} | {LABEL[k[0]]} | {FAMILY[k[0]]} | {PARAMS_B[k[0]]} | {r['quant']} | {r['bits']} | "
                 f"{r['size_gb'] * 1000:,.0f} | {FW_LABEL[k[1]]} | {k[2].upper()} |")
    L += ["", "## The ten tasks", "",
          "Every scenario deploys one of these tasks, each served by a fine-tuned model from the pool. Each task's "
          "prompt starts with fixed text (instructions, label lists, function schemas) that the app computes once "
          "and keeps in the KV cache, so a request only prefills its new tokens. Text longer than the task's input "
          "(an abstract, an essay) is sent sentence by sentence, as several requests.", "",
          "| Task | What the deployed model does | Request | Cached prefix (tokens) | Prefilled per request (tokens) "
          "| Answer (tokens) |", "|---|---|---|--:|--:|--:|",
          *[f"| {t.label} | {t.what} | {t.shape} | {t.cached:,} | {t.prompt:,} | {t.answer} |"
            for t in TASKS.values()],
          "", "## The filter and the pick", "",
          "For each scenario, on that scenario's phone and pool:", "",
          "1. Mark each configuration that would not run on that phone as **cannot run**.",
          "2. For each configuration that runs, check the four limits: **M** peak memory above the limit, **P** "
          "prefill below the limit, **D** decode below the limit, **E** energy per request above the limit. A "
          "configuration that misses any limit is **filtered**, with every limit it misses.",
          "3. The rest are **kept**. **Pick** the most capable kept configuration: the most parameters, then the "
          "most bits (F16 > 8-bit > 4-bit); if several runtimes or backends tie on model and precision, the one with "
          "the fastest decode, then the lowest energy. If nothing is kept, the pick is NONE.", "",
          "## Scenarios", "",
          "Limits: peak memory ≤ MB, prefill ≥ tokens/s, decode ≥ tokens/s, energy ≤ joules per request (with that "
          "task's prefilled and answer tokens).", "",
          "| Scenario | Phone | Task | Deployment | Prefilled / answer tokens | Peak memory ≤ (MB) | "
          "Prefill ≥ (tokens/s) | Decode ≥ (tokens/s) | Energy ≤ (J per request) |",
          "|---|---|---|---|--:|--:|--:|--:|--:|"]
    mapping = {}
    for sid, s in shuffled(chosen):
        uc, lim = s["uc"], s["limits"]
        mapping[sid] = s["id"]
        L.append(f"| {sid} | {s['phone']} | {uc.task.label} | {uc.name}: {uc.what} | {uc.prompt:,} / {uc.answer:,} | "
                 f"{lim['mem']:,.0f} | {num(lim['pre'])} | {lim['dec']:g} | {num(lim['energy'])} |")
    L += ["", "## Answer format", "",
          "Answer in three messages, so nothing is cut off: S01-S20 now; S21-S40 when I reply \"continue\"; S41-S60 "
          "after the next \"continue\". Before the first block you may write a short paragraph on your general "
          "assumptions (at most 200 words); after that, output only blocks in exactly this format, one per scenario:",
          "", "```",
          "SCENARIO S01",
          "PICK: C047",
          "KEPT: C001 C002 C005",
          "FILTERED: C003=P C004=MPE C006=D",
          "CANNOT_RUN: C099 C100",
          "END",
          "```", "",
          "Every configuration in the scenario's pool must appear exactly once across KEPT, FILTERED and "
          "CANNOT_RUN (104 IDs, or the 52 CPU IDs on the Redmi 9A). After `=` list every limit it misses, using the "
          "letters M, P, D, E. Write `PICK: NONE` if nothing is kept. Use an empty list (e.g. `CANNOT_RUN:`) when a "
          "group is empty."]
    OUT_DIR.mkdir(exist_ok=True)
    out = WEB_PROMPT if web else PROMPT
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    ID_MAP.write_text(json.dumps(mapping, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(out.read_text(encoding='utf-8')):,} characters) and {ID_MAP}")


# ---------------------------------------------------------------------------
# The naive prompt: the same scenarios, asked the way a user would ask
# ---------------------------------------------------------------------------

# Each deployment in scenarios_md.USE_CASES as a user would describe it: what they want, how quick and how often, in
# words. No limits, sizes or specs.
ASK = {
    "Voice-assistant intent router": "I want my phone's voice assistant to work out what I'm asking for and hand it "
    "to the right app, all on the phone. I use it maybe twenty times a day, and it has to answer within a second or "
    "two of me finishing talking.",
    "Banking app assistant": "I'm building my bank's app and want its help chat to work out what a customer is "
    "asking about (a transfer, their card, their balance) on the phone itself. Customers expect a reply within a "
    "couple of seconds, but it's only used once in a while.",
    "Offline assistant app": "I want a private assistant app that understands my typed requests without sending "
    "anything to the internet. I'd use it about thirty times a day and can wait a few seconds.",
    "Messaging app spam check": "I want my texting app to check every incoming text for spam before it notifies me, "
    "running quietly in the background. I get dozens of texts a day, so it can't eat my battery.",
    "Scam-check app": "I want an app where I paste a suspicious text and it tells me whether it's spam. I'd only use "
    "it a few times a day and can wait a few seconds.",
    "First-run inbox scan": "I want to check my whole text history, about 5,000 messages, for spam overnight while my "
    "phone charges. It has to finish by morning without using up much of the charge.",
    "Voice 'add to my calendar'": "I want to tell my voice assistant things like \"lunch with Sam Friday at 1\" and "
    "have it add the event to my calendar. It should confirm within a couple of seconds; I'd do this a few times a "
    "day.",
    "Calendar quick add": "I want to type events in plain English into my calendar app and have it create them. The "
    "event should show up within about three seconds.",
    "Smartwatch event dictation": "I want to dictate calendar events on my smartwatch and have my phone work them out "
    "in the background. A few seconds is fine, a few times a day.",
    "Planner app": "I want a planning app that turns what I type into calendar events. I plan my week in one go, "
    "about ten events, and can wait a few seconds for each.",
    "Voice-assistant actions": "I want my voice assistant to do things when I ask: send a text, set an alarm, check "
    "the weather. It should act within a second or two, a couple of dozen times a day.",
    "Smart-home app commands": "I want to type things like \"turn off the kitchen lights\" into my smart-home app and "
    "have it control the device within a few seconds. Maybe twenty commands a day.",
    "Smartwatch actions relay": "I want to speak requests into my watch and have my phone carry them out in the "
    "background. A few seconds is fine, about twenty times a day.",
    "Shortcuts app": "I want a shortcuts app where I type what I want and it runs the right action on my phone. I'd "
    "use it about twenty times a day and can wait a few seconds.",
    "Paper reader app": "I read about twenty medical research abstracts a day and want an app that highlights the "
    "drug and disease names as I read, keeping up with my reading.",
    "Reference-manager highlight tags": "When I highlight a sentence in my reference manager, I want it to tag the "
    "drugs and diseases in it within a few seconds. A couple of dozen times a day.",
    "Paper-alert abstract index": "Every night while my phone charges, I want it to go through the abstracts of the "
    "hundred or so papers I saved or got alerts for, and index the drugs and diseases so I can search them. It has "
    "to finish by morning.",
    "Keyboard tone check": "I want my keyboard to tell me the tone of a message before I send it. It has to be "
    "instant and can't slow my keyboard down; I send about a hundred messages a day.",
    "Comment tone check": "I'm building a social app and want it to warn people before they post a comment that "
    "sounds angry. The check should take about a second.",
    "Journaling app mood tags": "I want my journaling app to tag the emotions in my entry after I save it. I write "
    "one entry a day and can wait a few seconds.",
    "Keyboard mood insights": "I want my keyboard to go over what I typed each day, overnight while the phone "
    "charges, and give me a mood summary in the morning.",
    "English-learner keyboard": "I'm learning English and want my keyboard to suggest a grammar fix as soon as I "
    "finish each sentence. It can't slow down my typing; I type a couple of hundred sentences a day.",
    "Email 'fix grammar' button": "I want a \"fix grammar\" button in my email app that corrects a sentence within a "
    "few seconds of tapping it.",
    "English-learning feedback": "I want an English-learning app that corrects my practice sentences. I do about "
    "fifty a day and can wait a few seconds for each.",
    "Essay proofreading app": "I want an app that proofreads my English essays sentence by sentence; the corrections "
    "can trickle in while I read.",
    "English journal corrections": "I'm learning English and keep a daily journal. I want my phone to correct it "
    "overnight while it charges, so the corrections are ready in the morning.",
    "Search query understanding": "I want my search app to recognize the people, films and products in what I type, "
    "so the results are better. Search has to feel quick; I search dozens of times a day.",
    "Long-press smart actions": "When I long-press text on my screen, I want my phone to spot the names, places and "
    "products in it and offer actions. It runs in the background; a few seconds is fine.",
    "Read-later entity tags": "I save about five articles a day to read later and want an app to tag the people, "
    "places and products in them as I read.",
    "Photo and notes library index": "After I install it, I want an app to go through my 2,000 photo captions and "
    "notes overnight and tag the people, places and things in them.",
    "Dialer call summary": "I want my phone's dialer to summarize my short recorded calls, so I can read the summary "
    "right after hanging up. It should start within a second or two, and I read it as it appears.",
    "Chat catch-up": "When I open a conversation with unread messages, I want my messaging app to summarize what I "
    "missed, appearing as I read.",
    "Meeting-notes app": "I want a notes app that summarizes a short recorded conversation when I tap summarize; I "
    "read it as it appears.",
    "Overnight call summaries": "I want my phone to summarize the day's recorded calls overnight while it charges, "
    "ready in the morning.",
    "Voice reminders and weather": "I want to tell my voice assistant \"remind me to call mom tomorrow at 6\" or ask "
    "about the weather, and have it understood on the phone within a second or two.",
    "Reminders app quick add": "I want to type reminders in plain English into my reminders app and have them set up "
    "within a few seconds.",
    "Smartwatch command relay": "I want to say reminder and weather commands to my watch and have my phone work them "
    "out in the background.",
    "Reminders and weather app": "I want an app for reminders and weather that understands what I type, on the phone "
    "itself. I'd use it about twenty times a day.",
}
FAMILIES = ("SmolLM2 (Hugging Face), Gemma 3 (Google), Qwen3 and Qwen3.5 (Alibaba). All of them are fine-tuned for "
            "whatever the request is.")


def write_naive_prompt():
    data = load()
    _, chosen = scenarios(data)
    missing = {s["uc"].name for s in chosen} - set(ASK)
    assert not missing, f"no user wording for {missing}"
    cid, rows = ids(data), data[FLAGSHIP]
    L = ["I want to run a small language model on my Android phone for different things, without the cloud. Below "
         "are the options I have and 60 things I want to do, each on one of my phones. For each one, tell me which "
         "options would work well enough on that phone, which wouldn't and why, and which one I should use.",
         "",
         f"**Phones:** {', '.join(PHONE_ORDER)}.",
         "",
         f"**Models:** {FAMILIES}",
         "",
         "**Choices.** An option is a model plus three choices:",
         "",
         "- **Precision:** how the model's weights are stored. 4-bit (Q4) is smallest and fastest and loses a little "
         "quality; 8-bit (Q8) is in between; F16 is full quality and largest.",
         "- **Runtime:** the inference engine that runs the model on the phone, llama.cpp or MNN.",
         "- **Backend:** whether the runtime runs the model on the phone's CPU or its GPU.",
         "",
         "| Option | Model | Precision | Runtime | Backend |", "|---|---|---|---|---|"]
    for k in pool_keys(data):
        L.append(f"| {cid[k]} | {LABEL[k[0]]} | {rows[k]['quant']} | {FW_LABEL[k[1]]} | {k[2].upper()} |")
    L += ["", "**What \"works\" means.** An option works for a request if it fits all four:", "",
          "- **M (memory):** it doesn't use more memory than that kind of app can have without the phone closing it.",
          "- **P (prompt speed):** it reads my input fast enough to start answering in time.",
          "- **D (generation speed):** it writes the answer fast enough.",
          "- **E (battery):** it doesn't use too much battery for how often I use it.",
          "",
          "Some options can't run on a phone at all (they don't fit, fail to load or crash, or the GPU backend doesn't "
          "work there).",
          "",
          "**Which to use:** the most capable option that works: the biggest model, then the highest precision; if "
          "several runtimes or backends tie, the faster one, then the one that uses less battery.",
          "",
          "## What I want", ""]
    for sid, s in shuffled(chosen):
        L.append(f"- **{sid}** · {s['phone']}: {ASK[s['uc'].name]}")
    L += ["", "## How to answer", "",
          "Answer in three messages, so nothing is cut off: S01-S20 now, S21-S40 when I reply \"continue\", S41-S60 "
          "after the next \"continue\". Before the first block you may write a short paragraph on your assumptions "
          "(at most 200 words); after that, output only blocks in exactly this format, one per request:",
          "", "```",
          "SCENARIO S01",
          "PICK: C047",
          "KEPT: C001 C002 C005",
          "FILTERED: C003=P C004=MPE C006=D",
          "CANNOT_RUN: C099 C100",
          "END",
          "```", "",
          "Put every option C001-C104 in exactly one of KEPT (works), FILTERED (runs but doesn't work; after `=` list "
          "every reason, using the letters M, P, D, E) and CANNOT_RUN. Write `PICK: NONE` if nothing works. Use an "
          "empty list (e.g. `CANNOT_RUN:`) when a group is empty."]
    OUT_DIR.mkdir(exist_ok=True)
    NAIVE_PROMPT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {NAIVE_PROMPT} ({len(NAIVE_PROMPT.read_text(encoding='utf-8')):,} characters)")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def parse(text: str) -> dict:
    """{S01: {pick, kept: set, filtered: {cid: set(letters)}, cant: set}} from the chat's blocks."""
    out = {}
    for m in re.finditer(r"SCENARIO\s+(S\d+)(.*?)END", text, re.S):
        body, b = m.group(2), {"pick": None, "kept": set(), "filtered": {}, "cant": set()}
        for line in body.splitlines():
            line = line.strip().strip("`*")
            key, _, val = line.partition(":")
            key = key.strip().upper()
            if key == "PICK":
                p = re.search(r"C\d{3}", val)
                b["pick"] = p.group(0) if p else None
            elif key == "KEPT":
                b["kept"] = set(re.findall(r"C\d{3}", val))
            elif key == "CANNOT_RUN":
                b["cant"] = set(re.findall(r"C\d{3}", val))
            elif key == "FILTERED":
                b["filtered"] = {c: set(why.upper()) & set("MPDE") for c, why in
                                 re.findall(r"(C\d{3})\s*=\s*([MPDEmpde]+)", val)}
        out[m.group(1)] = b
    return out


def truth(s, cid) -> dict:
    """The measured verdicts for one scenario: {cid: (verdict, letters)} with verdict kept / filtered / cant."""
    t = {}
    for r, miss, _ in s["kept"] + s["failed"]:
        k = (r["model"], r["fw"], r["be"], r["quant"])
        letters = {m[0] for m in miss}
        t[cid[k]] = ("kept" if not miss else "filtered", letters)
    for r, _ in s["cant"]:
        t[cid[(r["model"], r["fw"], r["be"], r["quant"])]] = ("cant", set())
    return t


def score(path: str, variant: str = "detailed"):
    data = load()
    _, chosen = scenarios(data)
    cid = ids(data)
    back = {v: k for k, v in cid.items()}
    by_sid = dict(shuffled(chosen))
    answers = parse(open(path, encoding="utf-8").read())
    missing = [sid for sid in by_sid if sid not in answers]

    rows, field = [], {c: Counter() for c in "MPDE"}
    cant_conf = Counter()
    for sid, s in by_sid.items():
        if sid not in answers:
            continue
        a, t = answers[sid], truth(s, cid)
        rows_phone = data[s["phone"]]
        said = {c: "kept" for c in a["kept"]} | {c: "filtered" for c in a["filtered"]} | {c: "cant" for c in a["cant"]}
        unlisted = [c for c in t if c not in said]
        true_kept = {c for c, (v, _) in t.items() if v == "kept"}
        said_kept = {c for c in a["kept"] if c in t}
        inter = true_kept & said_kept
        for c, (v, letters) in t.items():
            cant_conf[(v == "cant", said.get(c) == "cant")] += 1
            if v != "cant" and said.get(c) in ("kept", "filtered"):  # per-limit verdicts where both say it runs
                guess = a["filtered"].get(c, set())
                for f in "MPDE":
                    if f == "E" and not any(m == "E" for m in letters) and request_j(rows_phone[back[c]],
                                                                                   s["uc"]) is None:
                        continue  # no energy reading: nothing to compare against
                    field[f][(f in letters, f in guess)] += 1
        pick, best = a["pick"], s["best"]
        best_id = cid[(best["model"], best["fw"], best["be"], best["quant"])]
        if pick and pick in t:
            pr = rows_phone[back[pick]]
            works = t[pick][0] == "kept"
            why = "cannot run" if t[pick][0] == "cant" else " ".join(sorted(t[pick][1])) if not works else ""
            cap = "same" if pick == best_id else (
                "same model and precision" if (pr["model"], pr["bits"]) == (best["model"], best["bits"]) else
                "less capable" if (pr["params_b"], pr["bits"]) < (best["params_b"], best["bits"]) else "more capable")
        else:
            works, why, cap, pr = False, "no pick" if not pick else "cannot run" if pick in back else "not in pool", "-", None
        rows.append(dict(sid=sid, id=s["id"], tier=s["tier"], phone=s["phone"], uc=s["uc"].name,
                         task=s["uc"].task.key,
                         n_true=len(true_kept), n_said=len(said_kept), inter=len(inter),
                         precision=len(inter) / len(said_kept) if said_kept else None,
                         recall=len(inter) / len(true_kept) if true_kept else None,
                         jaccard=len(inter) / len(true_kept | said_kept) if true_kept | said_kept else 1.0,
                         unlisted=len(unlisted), pick=pick, pick_name=_name(back.get(pick)) if pick else "NONE",
                         best_name=_name(back[best_id]), works=works, why=why, cap=cap))
    _report(rows, field, cant_conf, missing, path, variant)


def _name(k) -> str:
    if not k:
        return "?"
    m, fw, be, q = k
    return f"{LABEL[m]} {q} {FW_LABEL[fw]} {be.upper()}"


def _pct(n, d) -> str:
    return f"{n / d:.0%}" if d else "-"


def _report(rows, field, cant_conf, missing, path, variant="detailed"):
    report = {"detailed": REPORT, "naive": NAIVE_REPORT, "web": WEB_REPORT}[variant]
    given = {
        "detailed": "what the SLM Factory knows *without* this harness (phone specs, model cards, runtime and "
                    "backend, the numeric limits; no web search; prompt: `Benchmark-Harness/paper_table5/baseline/"
                    "baseline_prompt.md`",
        "web": "the same as the detailed baseline (phone specs, model cards, runtime and backend, the numeric "
               "limits) plus web search; prompt: `Benchmark-Harness/paper_table5/baseline/baseline_prompt_web.md`",
        "naive": "only what a user would say (the phone's name, the model families and choices, and a "
                 "plain-language wish with no limits, specs or sizes; prompt: `Benchmark-Harness/paper_table5/"
                 "baseline/baseline_prompt_naive.md`"}[variant]
    who = {"detailed": "without measurements", "web": "with web search", "naive": "asked like a user"}[variant]
    n = len(rows)
    works = sum(r["works"] for r in rows)
    same = sum(r["cap"] == "same" for r in rows)
    same_mq = sum(r["cap"] in ("same", "same model and precision") for r in rows)
    over = sum(r["works"] is False and r["cap"] == "more capable" for r in rows)
    L = [f"# Does measuring help? Claude {who} vs the harness", "",
         f"Generated {datetime.now():%Y-%m-%dT%H:%M} by `Benchmark-Harness/paper_table5/baseline.py` from "
         f"`{path}` and the measured filter in [SCENARIOS.md](SCENARIOS.md).", "",
         f"A Claude chat was given {given}) and asked to filter the pool and pick a configuration for the same 60 "
         "scenarios. Its answers are scored against the measured values and the scenarios' limits. With "
         "measurements the pick meets every limit by construction.", ""]
    if missing:
        L += [f"**Missing from the response:** {', '.join(missing)} (not scored).", ""]
    L += ["## Headline", "",
          f"| | Claude {who} | With the harness |", "|---|--:|--:|",
          f"| Pick actually meets all four limits on the phone | {works} of {n} ({_pct(works, n)}) | {n} of {n} |",
          f"| Pick is the same configuration as the measured best | {same} of {n} ({_pct(same, n)}) | {n} of {n} |",
          f"| Pick has the same model and precision as the measured best | {same_mq} of {n} ({_pct(same_mq, n)}) | "
          f"{n} of {n} |",
          f"| Pick is more capable than anything that actually fits (over-promised) | {over} of {n} | 0 |",
          f"| Pick fails with the right model and precision, on the wrong runtime or backend | "
          f"{sum(not r['works'] and r['cap'] == 'same model and precision' for r in rows)} of {n} | 0 |",
          f"| Pick cannot run on the phone at all | {sum(r['why'] == 'cannot run' for r in rows)} of {n} | 0 |"]
    prec = [r["precision"] for r in rows if r["precision"] is not None]
    rec = [r["recall"] for r in rows if r["recall"] is not None]
    jac = [r["jaccard"] for r in rows]
    L += [f"| Kept pool: share of predicted-kept configurations that really meet the limits (median precision) | "
          f"{st.median(prec):.0%} | 100% |" if prec else "",
          f"| Kept pool: share of really-passing configurations it kept (median recall) | {st.median(rec):.0%} | "
          f"100% |" if rec else "",
          f"| Kept pool overlap (median Jaccard) | {st.median(jac):.2f} | 1.00 |", ""]
    L += ["## By tier", "", "| Tier | Scenarios | Pick works | Same pick | Median precision | Median recall | "
          "Median Jaccard |", "|---|--:|--:|--:|--:|--:|--:|"]
    for t, _, _ in TIERS:
        rr = [r for r in rows if r["tier"] == t]
        if not rr:
            continue
        p = [r["precision"] for r in rr if r["precision"] is not None]
        q = [r["recall"] for r in rr if r["recall"] is not None]
        L.append(f"| {t} | {len(rr)} | {sum(r['works'] for r in rr)} | {sum(r['cap'] == 'same' for r in rr)} | "
                 f"{st.median(p):.0%} | {st.median(q):.0%} | {st.median(r['jaccard'] for r in rr):.2f} |"
                 if p and q else f"| {t} | {len(rr)} | {sum(r['works'] for r in rr)} | "
                 f"{sum(r['cap'] == 'same' for r in rr)} | - | - | {st.median(r['jaccard'] for r in rr):.2f} |")
    L += ["", "## By task", "", "| Task | Scenarios | Pick works | Same pick | Median Jaccard |",
          "|---|--:|--:|--:|--:|"]
    for t in TASKS:
        rr = [r for r in rows if r["task"] == t]
        if rr:
            L.append(f"| {t} | {len(rr)} | {sum(r['works'] for r in rr)} | {sum(r['cap'] == 'same' for r in rr)} | "
                     f"{st.median(r['jaccard'] for r in rr):.2f} |")
    L += ["", "## By phone", "", "| Phone | Scenarios | Pick works | Same pick | Median Jaccard |",
          "|---|--:|--:|--:|--:|"]
    for p in PHONE_ORDER:
        rr = [r for r in rows if r["phone"] == p]
        if rr:
            L.append(f"| {p} | {len(rr)} | {sum(r['works'] for r in rr)} | {sum(r['cap'] == 'same' for r in rr)} | "
                     f"{st.median(r['jaccard'] for r in rr):.2f} |")
    names = dict(M="Peak memory", P="Prefill", D="Decode", E="Energy")
    L += ["", "## Which limits it gets wrong", "",
          "For configurations that both the measurements and Claude say run: how often its call on each limit "
          "matches the measured one.", "",
          "| Limit | Agreement | Misses it predicted but did not happen | Misses that happened but it did not predict |",
          "|---|--:|--:|--:|"]
    for f, c in field.items():
        tot = sum(c.values())
        L.append(f"| {names[f]} | {_pct(c[(True, True)] + c[(False, False)], tot)} | {c[(False, True)]:,} | "
                 f"{c[(True, False)]:,} |")
    tp, fp, fn = cant_conf[(True, True)], cant_conf[(False, True)], cant_conf[(True, False)]
    L += ["", f"**Cannot run:** of {tp + fn:,} scenario-configurations that really cannot run, it flagged {tp:,}; it "
          f"also flagged {fp:,} that do run.", "",
          "## Every scenario", "",
          "| ID | Tier | Phone | Task | Deployment | Claude's pick | Works? | Measured best | Kept (measured / Claude / both) | "
          "Jaccard |", "|---|---|---|---|---|---|---|---|--:|--:|"]
    for r in sorted(rows, key=lambda r: r["id"]):
        ok = "yes" if r["works"] else f"**no** ({r['why']})"
        L.append(f"| [{r['id']}](SCENARIOS.md#{r['id'].lower()}) | {r['tier']} | {r['phone']} | {r['task']} | {r['uc']} | "
                 f"{r['pick_name']} | {ok} | {r['best_name']} | {r['n_true']} / {r['n_said']} / {r['inter']} | "
                 f"{r['jaccard']:.2f} |" + (f" {r['unlisted']} IDs unlisted" if r["unlisted"] else ""))
    L += ["", "Works? lists the limits the pick really misses (M memory, P prefill, D decode, E energy). Scenario "
          "IDs here are the SCENARIOS.md IDs; the prompt used shuffled IDs S01-S60 (`baseline/scenario_ids.json`)."]
    report.write_text("\n".join(x for x in L if x is not None) + "\n", encoding="utf-8")
    print(f"wrote {report}: pick works {works}/{n}, same pick {same}/{n}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "prompt":
        write_prompt()
    elif len(sys.argv) >= 2 and sys.argv[1] == "naive":
        write_naive_prompt()
    elif len(sys.argv) >= 2 and sys.argv[1] == "web":
        write_prompt(web=True)
    elif len(sys.argv) in (3, 4) and sys.argv[1] == "score" and sys.argv[3:] in ([], ["--naive"], ["--web"]):
        score(sys.argv[2], sys.argv[3][2:] if sys.argv[3:] else "detailed")
    else:
        sys.exit(__doc__)
