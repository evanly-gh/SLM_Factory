#!/usr/bin/env python3
"""
Write docs/SCENARIOS.md: a test of the requirement filter. Realistic app use cases set limits on the four measured
fields (peak memory, prefill tokens/s, decode tokens/s, energy per request); each configuration on the target phone is
checked against them with its measured values from docs/RESULTS.md (req_data.py), the ones that miss any limit are
filtered out, and the most capable survivor is picked. 20 easy, 20 medium and 20 hard scenarios.

  python scenarios_md.py
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from req_data import DOCS, HERE, load
from requirements_md import LABEL, TIERS

FIG_DIR = DOCS / "figures" / "scenarios"
CSV_OUT = HERE / "results" / "scenarios.csv"
PER_TIER = 20
PHONE_ORDER = ["Galaxy S23", "OnePlus 15", "OnePlus 8 Pro", "Redmi 9A"]
# Usable RAM (GiB, as reported in docs/RESULTS.md) and battery energy (rated mAh × 3.85 V nominal).
RAM_GIB = {"Galaxy S23": 6.91, "OnePlus 15": 14.76, "OnePlus 8 Pro": 11.24, "Redmi 9A": 3.72}
BATTERY = {"Galaxy S23": (3900, 15.0), "OnePlus 15": (7300, 28.1), "OnePlus 8 Pro": (4510, 17.4),
           "Redmi 9A": (5000, 19.3)}
COLUMNS = [("llama.cpp", "cpu"), ("llama.cpp", "gpu"), ("mnn", "cpu"), ("mnn", "gpu")]
FW_LABEL = {"llama.cpp": "llama.cpp", "mnn": "MNN"}
BITS_LABEL = {4: "4-bit", 8: "8-bit", 16: "F16"}


class Task:
    """One of the ten SLM Factory benchmark tasks: the benchmark it comes from, a neutral name and description (the
    baseline prompt uses only these, so the benchmark is not named), its request in tokens, and the most accurate
    student the Factory trained for it.

    A request is the task's real prompt (SLM Factory eval/scorers) with a median-length input. `cached` is the fixed
    part before the input (instructions, label list, tool schemas): the app computes it once and keeps its KV cache,
    so a request processes only `prompt` tokens (the input, any fixed text after it, the chat-template tail)."""

    def __init__(self, key, title, label, what, cached, prompt, answer, shape, best):
        self.key, self.title, self.label, self.what = key, title, label, what
        self.cached, self.prompt, self.answer, self.shape, self.best = cached, prompt, answer, shape, best
        self.best_key = BEST_STUDENT[key]

    def best_fits(self, kept) -> bool:
        """Whether the most accurate student (its model at its bit width, any runtime or backend) survived."""
        return any((r["model"], r["bits"]) == self.best_key for r, _, _ in kept)


# The most accurate student per task (SLM Factory RESULTS.md I.1) as (model, bits) in this pool. The Factory's
# Q4_K_M / Q8_0 GGUFs map to the pool's 4-bit and 8-bit builds of the same model.
BEST_STUDENT = {'clinc150': ('qwen3-4b-instruct-2507', 8),
                'sms_spam': ('qwen3-1.7b', 4),
                'calendar_json': ('qwen3.5-0.8b', 4),
                'xlam_bfcl': ('qwen3.5-4b', 4),
                'ner_bc5cdr': ('qwen3.5-0.8b', 4),
                'goemotions': ('qwen3-4b-instruct-2507', 8),
                'gec_bea19': ('qwen3-4b-instruct-2507', 8),
                'multiconer': ('qwen3-4b-instruct-2507', 8),
                'dialogsum': ('qwen3-4b-instruct-2507', 8),
                'topv2': ('qwen3-4b-instruct-2507', 8)}

# Token counts from the Qwen3 / Qwen3.5 tokenizers (they agree within 1-3%) on each task's real prompt and chat
# template, with an input of the task's median length (SLM Factory RESULTS.md IV.3) and a typical answer.
TASKS = {t.key: t for t in [
    Task("clinc150", "CLINC150", "Intent routing",
         "classify a user's request to an assistant into one of a large fixed set of intents (banking, travel, "
         "home, small talk, ...) or out-of-scope", 540, 515, 4,
         "the prompt lists every intent name before the request and again after it, so about 500 tokens follow the "
         "request and cannot be cached; the answer is one intent name",
         "0.9372 macro-F1, Qwen3-4B-Instruct-2507 Q8_0"),
    Task("sms_spam", "SMS Spam", "SMS spam check", "label a text message as spam or not", 67, 42, 2,
         "one text message; the answer is one label", "0.9609 minority-class F1, Qwen3-1.7B Q4_K_M"),
    Task("calendar_json", "Calendar NL→JSON", "Calendar event from a request",
         "turn a user's scheduling request (\"lunch with Sam Friday at 1\") into one create-event API call in JSON, "
         "with dates resolved against the current date", 370, 40, 85,
         "the API schema and rules are fixed; the request is the current date plus one sentence; the answer is one "
         "JSON call", "0.8112 argument match, Qwen3.5-0.8B Q4_K_M"),
    Task("xlam_bfcl", "Function calling (xLAM/BFCL)", "Tool calling",
         "turn a user's request into JSON calls to the app's declared functions (name and arguments)", 310, 35, 35,
         "the app's function schemas are fixed (about 300 tokens for three functions); the request is one sentence; "
         "the answer is usually one call", "0.8520 argument match, Qwen3.5-4B Q4_K_M"),
    Task("ner_bc5cdr", "Biomedical NER (BC5CDR)", "Drug and disease tagging",
         "find the chemical/drug and disease mentions in one sentence of biomedical text", 40, 32, 45,
         "one sentence; longer text is sent sentence by sentence; the answer is a JSON span list",
         "0.8301 span F1, Qwen3.5-0.8B Q4_K_M"),
    Task("goemotions", "GoEmotions", "Emotion tagging",
         "tag the emotions a short message expresses, from a fixed list of emotions (one or more)", 85, 57, 5,
         "one short message plus a fixed reply rule after it; longer text is sent sentence by sentence; the answer is "
         "one to three labels", "0.5979 Ekman macro-F1, Qwen3-4B-Instruct-2507 Q8_0"),
    Task("gec_bea19", "Grammar correction (BEA-2019)", "Grammar correction",
         "rewrite one English sentence with its grammatical errors fixed", 58, 24, 20,
         "one sentence; longer text is sent sentence by sentence; the answer is the corrected sentence",
         "0.5647 ERRANT F0.5, Qwen3-4B-Instruct-2507 Q8_0"),
    Task("multiconer", "Fine-grained NER (MultiCoNER II)", "Fine-grained entity tagging",
         "tag the named entities in a short query or sentence with about 30 fine-grained types (artist, software, "
         "food, medication, ...)", 153, 22, 35,
         "one query or short sentence; longer text is sent sentence by sentence; the answer is a JSON span list",
         "0.6532 micro-F1, Qwen3-4B-Instruct-2507 Q8_0"),
    Task("dialogsum", "DialogSum", "Conversation summary",
         "summarize a short conversation between two or more people in one to three sentences", 30, 175, 40,
         "one conversation of about 10 turns; the answer is a short summary, shown as it streams",
         "0.4817 ROUGE-L, Qwen3-4B-Instruct-2507 Q8_0"),
    Task("topv2", "TOPv2 (reminder, weather)", "Reminder and weather command parsing",
         "parse a reminder or weather command into a nested intent and slot tree", 876, 17, 40,
         "the intent and slot lists and the grammar are fixed (about 880 tokens); the command is a few words; the "
         "answer is a bracketed parse", "0.5990 exact match, Qwen3-4B-Instruct-2507 Q8_0"),
]}


# Each kind of deployment has the same memory budget and latency, so limits are not tuned per scenario. Memory is the
# peak an app can hold without the system killing it: MB (but never more than RAM_CAP of the phone's RAM), or a share
# of RAM for an app the user opened for this or an overnight job with nothing else running. Latency is first token
# within, whole answer within (seconds); an overnight job instead gets its share of the night (see UseCase).
KINDS = {
    "keyboard": (300, 0.3, 1.0, "keyboard (input method)",
                 "stays resident all day next to every app; a suggestion must appear within ~1 s"),
    "voice": (1536, 0.5, 2.0, "system voice assistant",
              "a privileged system app at foreground priority while invoked; the reply should come within ~2 s of "
              "the user finishing"),
    "in-app": (1024, 0.5, 3.0, "feature inside an app",
               "shares the app's memory; the user tapped or typed and waits up to ~3 s"),
    "service": (512, 1.0, 5.0, "background service",
                "runs while other apps are in the foreground; nobody is watching, a few seconds is fine"),
    "app": (0.4, 1.0, 5.0, "dedicated app",
            "the foreground app the user opened for this; a wait of up to ~5 s is accepted"),
    "overnight": (0.5, None, None, "overnight job",
                  "on the charger with nothing else running; all requests must finish within a 6-hour night"),
}
RAM_CAP = 0.25  # a fixed MB budget is capped at a quarter of the phone's RAM (it matters only on the 4 GB Redmi 9A)
OVERNIGHT_PCT = 25  # an overnight job may use up to a quarter of a full charge, so the phone is still charged by morning
NIGHT_S = 6 * 3600  # an overnight job's requests share a 6-hour night, at most a minute each
READ_TPS = 5  # a streamed answer the user reads as it arrives (silent reading ~4 words/s ≈ 5 tokens/s)


class UseCase:
    """A deployment of one task. Every request has the task's shape; text longer than what the task was trained on
    is sent as several task-sized requests, counted in per_day. The kind sets memory and latency. An answer that is
    only usable complete (a label, JSON, a parse, a corrected sentence) must be done in time, so decode >= answer
    tokens / (done - first token); a streamed answer the user reads needs decode >= READ_TPS. An overnight request
    gets NIGHT_S / per_day seconds (at most 60), half of it for the first token."""

    def __init__(self, task, kind, name, what, per_day, battery_pct, why, streamed=False):
        self.task, self.kind, self.name, self.what = TASKS[task], kind, name, what
        self.prompt, self.answer = self.task.prompt, self.task.answer
        self.mem, self.ttft_s, self.done_s = KINDS[kind][:3]
        if kind == "overnight":
            self.done_s = min(60.0, NIGHT_S / per_day)
            self.ttft_s, streamed = self.done_s / 2, False
        self.per_day, self.why, self.streamed = per_day, why, streamed
        self.battery_pct = OVERNIGHT_PCT if kind == "overnight" else battery_pct
        self.decode = READ_TPS if streamed else math.ceil(self.answer / (self.done_s - self.ttft_s))

    @property
    def ctx(self) -> int:
        """Tokens in the KV cache at the end of a request."""
        return self.task.cached + self.prompt + self.answer

    def limits(self, phone) -> dict:
        ram_mb = RAM_GIB[phone] * 1024
        mem = min(self.mem, round(RAM_CAP * ram_mb, -1)) if self.mem > 1 else round(self.mem * ram_mb, -2)
        joules = BATTERY[phone][1] * 3600 * self.battery_pct / 100 / self.per_day
        return dict(mem=mem, pre=self.prompt / self.ttft_s, dec=self.decode, energy=joules)

    def mem_txt(self):
        return f"{self.mem:,} MB" if self.mem > 1 else f"{self.mem:.0%} of RAM"

    def time_txt(self):
        if self.streamed:
            return f"first token ≤ {self.ttft_s:g} s, then ≥ {self.decode:g} tokens/s"
        return f"first token ≤ {self.ttft_s:.3g} s, answer done ≤ {self.done_s:.3g} s"


# KV-cache bytes per token for the layers whose cache grows with context (F16 K and V): 2 × layers × KV heads ×
# head dim × 2 bytes, from each model's config.json. Qwen3.5 caches only its full-attention layers (1 in 4; the
# linear-attention layers keep a fixed-size state) and Gemma 3 only its global layers beyond its 512-token window.
KV_BYTES = {"smollm2-135m": 2 * 30 * 3 * 64 * 2, "gemma3-270m": 2 * 3 * 1 * 256 * 2, "smollm2-360m": 2 * 32 * 5 * 64 * 2,
            "qwen3-0.6b": 2 * 28 * 8 * 128 * 2, "qwen3.5-0.8b": 2 * 6 * 2 * 256 * 2, "qwen3-1.7b": 2 * 28 * 8 * 128 * 2,
            "qwen3.5-2b": 2 * 6 * 2 * 256 * 2, "qwen3-4b-instruct-2507": 2 * 36 * 8 * 128 * 2,
            "qwen3.5-4b": 2 * 8 * 4 * 256 * 2}
MEASURED_CTX = 512  # peak memory was measured with a 256-token prompt and a 256-token answer


def peak_mb(r, uc) -> float:
    """Measured peak memory plus the KV cache of the tokens a request holds beyond the measured context."""
    return r["mem_mb"] + max(0, uc.ctx - MEASURED_CTX) * KV_BYTES[r["model"]] / 2 ** 20


# Three to five deployments per task, each something a phone user would actually use, with the task's input as it was
# trained on. per_day is requests a day (or a night); battery_pct is the share of a day's charge the feature may take.
# Every deployment assumes the model stays loaded; loading it is not counted.
USE_CASES = [
    # clinc150: the front door of an assistant
    UseCase("clinc150", "voice", "Voice-assistant intent router", "route each spoken request to the skill that "
            "handles it", 20, 1, "every voice request goes through it"),
    UseCase("clinc150", "in-app", "Banking app assistant", "route a typed question in a bank's assistant chat to the "
            "right flow; non-banking intents are answered as out-of-scope", 1, 0.5, "an occasional question"),
    UseCase("clinc150", "app", "Offline assistant app", "a private assistant app that routes typed requests on the "
            "device", 30, 1, "the user's main assistant"),
    # sms_spam
    UseCase("sms_spam", "service", "Messaging app spam check", "the default messaging app checks each incoming text "
            "before notifying", 50, 0.5, "dozens of texts a day; a filter must be nearly free"),
    UseCase("sms_spam", "app", "Scam-check app", "paste a suspicious message into a scam-checker app", 5, 0.5,
            "used only when a message looks suspicious"),
    UseCase("sms_spam", "overnight", "First-run inbox scan", "after install, check the whole SMS history (5,000 "
            "messages) overnight", 5000, None, "a one-time backlog"),
    # calendar_json
    UseCase("calendar_json", "voice", "Voice 'add to my calendar'", "create an event from a spoken request", 3, 0.5,
            "a few events a day"),
    UseCase("calendar_json", "in-app", "Calendar quick add", "type an event in plain language in the calendar app", 5,
            0.5, "a few events a day"),
    UseCase("calendar_json", "service", "Smartwatch event dictation", "events dictated on a paired watch, parsed on the "
            "phone", 5, 0.5, "the watch shows a spinner while the phone works"),
    UseCase("calendar_json", "app", "Planner app", "a planning assistant app that turns typed requests into events", 10,
            1, "the user plans the week in one sitting"),
    # xlam_bfcl
    UseCase("xlam_bfcl", "voice", "Voice-assistant actions", "turn a spoken request into a call to one of the "
            "assistant's phone actions (send a message, set an alarm, get the weather)", 20, 1,
            "a couple of dozen commands a day"),
    UseCase("xlam_bfcl", "in-app", "Smart-home app commands", "\"turn off the kitchen lights\" typed in a smart-home "
            "app, turned into a device call", 20, 0.5, "a few commands each morning and evening"),
    UseCase("xlam_bfcl", "service", "Smartwatch actions relay", "requests spoken to a paired watch, turned into calls "
            "on the phone", 20, 0.5, "the watch shows a spinner while the phone works"),
    UseCase("xlam_bfcl", "app", "Shortcuts app", "an on-device shortcuts app that turns a typed request into an "
            "action call", 20, 1, "used for quick actions through the day"),
    # ner_bc5cdr
    UseCase("ner_bc5cdr", "app", "Paper reader app", "highlight drug and disease names in abstracts, sentence by "
            "sentence, as the user reads", 200, 2, "20 abstracts of ~10 sentences a day"),
    UseCase("ner_bc5cdr", "in-app", "Reference-manager highlight tags", "tag the drugs and diseases in a sentence the "
            "user highlights in a reference-manager app", 20, 0.5, "a couple of dozen highlights a day"),
    UseCase("ner_bc5cdr", "overnight", "Paper-alert abstract index", "index the abstracts of the day's saved and "
            "alerted papers for search, sentence by sentence", 1000, None, "~100 abstracts of ~10 sentences"),
    # goemotions
    UseCase("goemotions", "keyboard", "Keyboard tone check", "show the tone of a message before it is sent", 100, 1,
            "checks every message the user writes"),
    UseCase("goemotions", "in-app", "Comment tone check", "warn before posting a comment that reads as angry, in a "
            "social app", 10, 0.5, "a check on the post button"),
    UseCase("goemotions", "app", "Journaling app mood tags", "tag the emotions in each sentence of a journal entry "
            "after it is saved", 20, 0.5, "one entry of ~20 sentences a day"),
    UseCase("goemotions", "overnight", "Keyboard mood insights", "an opt-in keyboard feature: tag the messages the "
            "user typed that day, for a mood summary in the morning", 100, None, "the day's typed messages"),
    # gec_bea19
    UseCase("gec_bea19", "keyboard", "English-learner keyboard", "a keyboard for English learners that suggests a "
            "correction when a sentence is finished", 200, 1, "checks every sentence the user types"),
    UseCase("gec_bea19", "in-app", "Email 'fix grammar' button", "correct a sentence in an email being written", 10,
            0.5, "used on the emails that matter"),
    UseCase("gec_bea19", "app", "English-learning feedback", "correct a learner's English practice sentence in a "
            "language-learning app", 50, 1, "a practice session of ~50 sentences a day"),
    UseCase("gec_bea19", "app", "Essay proofreading app", "proofread an English essay in a writing app, sentence by "
            "sentence, corrections filling a review list", 30, 1, "one essay of ~30 sentences a day"),
    UseCase("gec_bea19", "overnight", "English journal corrections", "correct the day's English journal entries of a "
            "learner, ready in the morning", 60, None, "a few entries of ~20 sentences"),
    # multiconer
    UseCase("multiconer", "in-app", "Search query understanding", "find the people, titles and products in a query "
            "typed into a search app", 50, 0.5, "dozens of searches a day"),
    UseCase("multiconer", "service", "Long-press smart actions", "a system feature: find the entities in text the "
            "user long-presses on screen, to offer actions", 20, 0.5, "a few dozen selections a day"),
    UseCase("multiconer", "app", "Read-later entity tags", "tag people, places and products in a saved article, "
            "sentence by sentence", 200, 1, "five articles of ~40 sentences a day"),
    UseCase("multiconer", "overnight", "Photo and notes library index", "after install, tag the entities in the "
            "existing photo captions and notes (2,000 items)", 2000, None, "a one-time backlog"),
    # dialogsum
    UseCase("dialogsum", "in-app", "Dialer call summary", "the phone's built-in dialer summarizes a short recorded call "
            "(from its transcript)", 3, 0.5, "the user opens the summary after the call and reads it as it streams",
            streamed=True),
    UseCase("dialogsum", "in-app", "Chat catch-up", "summarize the last ~10 unread messages of a one-to-one "
            "conversation when it is opened", 10, 0.5, "the user reads the summary as it streams", streamed=True),
    UseCase("dialogsum", "app", "Meeting-notes app", "summarize a short recorded conversation (from its transcript) "
            "in a notes app", 5, 0.5, "the user taps summarize and reads as it streams", streamed=True),
    UseCase("dialogsum", "overnight", "Overnight call summaries", "the built-in dialer summarizes the day's recorded "
            "calls overnight", 10, None, "read in the morning"),
    # topv2
    UseCase("topv2", "voice", "Voice reminders and weather", "parse a spoken reminder or weather command", 10, 0.5,
            "about ten commands a day"),
    UseCase("topv2", "in-app", "Reminders app quick add", "parse a typed reminder in the reminders app", 10, 0.5,
            "a few reminders a day"),
    UseCase("topv2", "service", "Smartwatch command relay", "parse reminder and weather commands relayed from a paired "
            "watch", 30, 1, "the watch shows a spinner while the phone works"),
    UseCase("topv2", "app", "Reminders and weather app", "an assistant app for reminders and weather that parses "
            "requests on the device", 20, 1, "the user's main way of setting reminders"),
]


def num(v) -> str:
    """3 significant figures, never scientific notation."""
    return f"{v:,.0f}" if v >= 100 else f"{v:.3g}"


def name(r) -> str:
    return f"{LABEL[r['model']]} {r['quant']} {FW_LABEL[r['fw']]} {r['be'].upper()}"


def request_j(r, uc) -> float | None:
    if r.get("e_pre") is None or r.get("e_dec") is None:
        return None
    return (uc.prompt * r["e_pre"] + uc.answer * r["e_dec"]) / 1e3


def capability(r):
    """Most capable first: more parameters, then more bits; ties go to the faster decode, then lower energy."""
    return (r["params_b"], r["bits"], r["dec"], -(r.get("req_j") or 0))


def evaluate(rows, uc, phone) -> dict:
    lim = uc.limits(phone)
    kept, failed, cant = [], [], []
    for r in rows.values():
        if not r["ok"]:
            cant.append((r, r["status"]))
            continue
        e = request_j(r, uc)
        miss = []
        mem = None if r["mem_mb"] is None else peak_mb(r, uc)
        if mem is None or mem > lim["mem"]:
            miss.append(("M", f"peak memory {num(mem)} MB > {num(lim['mem'])}", f"{num(mem)} MB"))
        if r["pre"] < lim["pre"]:
            miss.append(("P", f"prefill {num(r['pre'])} < {num(lim['pre'])} tokens/s", num(r["pre"])))
        if r["dec"] < lim["dec"]:
            miss.append(("D", f"decode {num(r['dec'])} < {num(lim['dec'])} tokens/s", num(r["dec"])))
        if e is None:
            miss.append(("E?", "energy not measured", None))
        elif e > lim["energy"]:
            miss.append(("E", f"energy {num(e)} J > {num(lim['energy'])} J", f"{num(e)} J"))
        (failed if miss else kept).append((r, miss, e))
    pool = len(rows)  # the whole pool, including configurations that cannot run on this phone
    share = len(kept) / pool
    tier = next(t for t, lo, hi in TIERS if lo <= share < hi)
    best = max(kept, key=lambda x: capability(x[0]))[0] if kept else None
    return dict(uc=uc, phone=phone, limits=lim, kept=kept, failed=failed, cant=cant, share=share, tier=tier,
                best=best, pool=pool)


def select(all_results) -> list[dict]:
    """PER_TIER scenarios per tier, each with at least one surviving configuration. Greedy for variety: always take
    the task, then the phone, then the use case, used least so far."""
    out, used = [], Counter()
    for tier, _, _ in TIERS:
        pool = [s for s in all_results if s["tier"] == tier and s["best"]]
        tasks, phones, cases = Counter(), Counter(), Counter()
        while len([s for s in out if s["tier"] == tier]) < PER_TIER and pool:
            s = min(pool, key=lambda s: (tasks[s["uc"].task.key], phones[s["phone"]],
                                         cases[s["uc"].name] + used[s["uc"].name],
                                         PHONE_ORDER.index(s["phone"]), USE_CASES.index(s["uc"])))
            pool.remove(s)
            tasks[s["uc"].task.key] += 1
            phones[s["phone"]] += 1
            cases[s["uc"].name] += 1
            out.append(s)
        used.update(cases)
    return out



def figure(s, fname):
    rows = {(r["model"], r["bits"], r["fw"], r["be"]): x for x in s["kept"] + s["failed"] for r in [x[0]]}
    cant = {(r["model"], r["bits"], r["fw"], r["be"]): why for r, why in s["cant"]}
    allr = [x[0] for x in s["kept"] + s["failed"]] + [r for r, _ in s["cant"]]
    models = sorted({r["model"] for r in allr}, key=lambda m: list(LABEL).index(m))
    ylabels = [(m, b) for m in models for b in (4, 8, 16) if any(r["model"] == m and r["bits"] == b for r in allr)]
    cols = [c for c in COLUMNS if any((r["fw"], r["be"]) == c for r in allr)]
    fig, ax = plt.subplots(figsize=(max(1.25 * len(cols) + 1.6, 4.4), 0.24 * len(ylabels) + 1.3))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    best = s["best"]
    for yi, (m, b) in enumerate(ylabels):
        for xi, (fw, be) in enumerate(cols):
            k = (m, b, fw, be)
            if k in rows:
                r, miss = rows[k][0], rows[k][1]
                if r is best:
                    fc, txt, tc, w = "#0ca30c", "★ pick", "#ffffff", "bold"
                elif not miss:
                    fc, txt, tc, w = "#d6ecd2", "✓", "#006300", "bold"
                else:
                    fc, txt, tc, w = "#f6e0d6", " ".join(m[0] for m in miss), "#3a2a24", "normal"
            elif k in cant:
                fc, txt, tc, w = "#ecebe6", "can't run", "#8a897f", "normal"
            else:
                continue
            ax.add_patch(plt.Rectangle((xi + 0.04, yi + 0.06), 0.92, 0.88, facecolor=fc, edgecolor="none"))
            ax.text(xi + 0.5, yi + 0.5, txt, ha="center", va="center", fontsize=7, color=tc, fontweight=w)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(ylabels), 0)
    ax.set_xticks([i + 0.5 for i in range(len(cols))],
                  [f"{FW_LABEL[fw]}\n{be.upper()}" for fw, be in cols], fontsize=7.5, color="#3d3d3a")
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(ylabels))],
                  [f"{LABEL[m]}  {BITS_LABEL[b]}" for m, b in ylabels], fontsize=7, color="#3d3d3a")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.suptitle(f"{s['id']} · {s['phone']} · {s['uc'].task.key}: {s['uc'].name}", fontsize=8.5, color="#1f1f1d", x=0.01, ha="left")
    fig.text(0.01, 0.004, "★ pick: chosen   ✓ kept   can't run: dropped first\nletters: filtered out by M memory, "
             "P prefill, D decode, E energy, E? no energy reading", fontsize=6.3, color="#5f5e58", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.045, 1, 0.98))
    fig.savefig(fname, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def fmt_lim(lim, uc):
    return (f"peak memory ≤ {num(lim['mem'])} MB · prefill ≥ {num(lim['pre'])} tokens/s · decode ≥ {lim['dec']:g} "
            f"tokens/s · energy ≤ {num(lim['energy'])} J per request")


def metrics(r, uc):
    e = request_j(r, uc)
    return (f"{num(peak_mb(r, uc))} MB, {num(r['pre'])} / {num(r['dec'])} tokens/s, "
            f"{num(e)} J" if e is not None else "")


def scenarios(data) -> tuple[list[dict], list[dict]]:
    """(every use case on every phone, the chosen 60 with their IDs E01..H20)."""
    results = [evaluate(data[p], uc, p) for p in PHONE_ORDER for uc in USE_CASES]
    chosen = select(results)
    assert {s["uc"].task.key for s in chosen} == set(TASKS), "every task must be used"
    for i, s in enumerate(chosen, 1):
        s["id"] = f"{s['tier'][0]}{sum(1 for t in chosen[:i] if t['tier'] == s['tier']):02d}"
    return results, chosen


def main():
    results, chosen = scenarios(load())
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for old in FIG_DIR.glob("*.png"):
        old.unlink()
    for s in chosen:
        figure(s, FIG_DIR / f"{s['id']}.png")

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "tier", "phone", "use_case", "mem_limit_mb", "prefill_min", "decode_min",
                    "energy_max_j", "model", "quant", "runtime", "backend", "verdict", "reasons", "peak_mb",
                    "prefill_tps", "decode_tps", "request_j"])
        for s in chosen:
            lim = s["limits"]
            head = [s["id"], s["tier"], s["phone"], s["uc"].name, round(lim["mem"]), round(lim["pre"], 1),
                    lim["dec"], round(lim["energy"], 2)]
            for r, miss, e in s["kept"] + s["failed"]:
                verdict = "pick" if r is s["best"] else "kept" if not miss else "filtered"
                w.writerow(head + [LABEL[r["model"]], r["quant"], FW_LABEL[r["fw"]], r["be"].upper(), verdict,
                                   "; ".join(m[1] for m in miss), round(peak_mb(r, s["uc"])), round(r["pre"], 2),
                                   round(r["dec"], 2), None if e is None else round(e, 3)])
            for r, why in s["cant"]:
                w.writerow(head + [LABEL[r["model"]], r["quant"], FW_LABEL[r["fw"]], r["be"].upper(),
                                   "cannot run", why, None, None, None, None])

    L = ["# Requirement-filter test: 60 scenarios from the ten SLM Factory tasks", "",
         f"Generated {datetime.now():%Y-%m-%dT%H:%M} by `Benchmark-Harness/paper_table5/scenarios_md.py` from the "
         "measured values in [RESULTS.md](RESULTS.md). Every configuration's verdict, with values, is in "
         "`Benchmark-Harness/paper_table5/results/scenarios.csv`.", "",
         "## The test", "",
         "A scenario is a use case on a target phone. The use case sets a limit on each of the four measured fields; "
         "the algorithm then works through the phone's pool (9 models × Q4/Q8/F16 × llama.cpp/MNN × CPU/GPU):", "",
         "1. **Drop configurations that cannot run on the phone** (too big for its RAM, fails to load, crashes, "
         "no GPU build).",
         "2. **Drop every configuration whose measured value misses any limit:** peak memory above the budget, "
         "prefill or decode below the minimum, energy per request above the budget. A configuration with no energy "
         "reading cannot be shown to meet an energy limit, so it is dropped too (only on the OnePlus 8 Pro, 5 "
         "llama.cpp GPU rows).",
         "3. **Pick the most capable survivor:** the largest model (parameters), then the highest precision "
         "(F16 > 8-bit > 4-bit); between runtimes and backends for that model and precision, the fastest decode, "
         "then the lowest energy.", "",
         "**Difficulty** uses the REQUIREMENTS.md cut-offs on the share of the phone's *whole* pool that survives, "
         "counting the configurations that cannot run on it as filtered out: easy ≥ 60%, medium 30-60%, hard < 30%. "
         "The pool is 104 configurations on the flagships and 52 on the CPU-only Redmi 9A. Because configurations "
         "that cannot run count against the share, a phone that runs few of them cannot have easy scenarios: the "
         "Galaxy S23 runs 82 of 104 and the OnePlus 8 Pro 79, so an easy scenario there would have to keep 77-80% "
         "of what runs.", "",
         "## The ten tasks", "",
         "Every scenario deploys one of the ten tasks the SLM Factory trains students for (SLM Factory "
         "RESULTS.md, Part I), with the task's own request: its real prompt (SLM Factory `eval/scorers`) and chat "
         "template around an input of the task's median length (RESULTS.md IV.3), counted with the Qwen3 and "
         "Qwen3.5 tokenizers (they agree within 1-3%), and a typical answer.", "",
         "**The fixed part of the prompt is cached.** Every task's prompt starts with fixed text (instructions, label "
         "lists, tool schemas) and puts the input after it. A deployed app computes that prefix once and keeps its "
         "KV cache, so a request only prefills the input, any fixed text after it and the chat-template tail. "
         "Intent routing is the exception that matters: its prompt repeats all 151 intent names after the "
         "request, so about 500 tokens are new on every request.", "",
         "**Inputs stay the size the model was trained on.** Text longer than that (an abstract, an essay, an "
         "article, a journal entry) is sent sentence by sentence, as several requests.", "",
         "| Task | Benchmark | What the deployed model does | Cached prefix (tokens) | Prefilled per request | "
         "Answer | Request | Factory's most accurate student (RESULTS.md I.1) |", "|---|---|---|--:|--:|--:|---|---|",
         *[f"| {t.key} | {t.title} | {t.what} | {t.cached:,} | {t.prompt:,} | {t.answer} | {t.shape} | {t.best} |"
           for t in TASKS.values()],
         "",
         "## How the limits are set", "",
         "Each task gets three to five deployments that a phone user would actually use. Every deployment of the same "
         "kind gets the same memory budget and latency, so the limits are not tuned per scenario; only the request "
         "count and the battery share differ. The limits come from the deployment, not from the measurements; the "
         "tier is whatever the measurements then give.", "",
         "| Kind | Peak memory ≤ | First token within | Whole answer within | Why |", "|---|--:|--:|--:|---|",
         *[f"| {label} | {f'{mem:,} MB' if mem > 1 else f'{mem:.0%} of RAM'} | "
           f"{f'{ttft:g} s' if ttft else 'half the window'} | {f'{done:g} s' if done else '6 h ÷ requests, ≤ 60 s'} | "
           f"{why} |" for mem, ttft, done, label, why in KINDS.values()],
         "",
         f"- **Peak memory**: a fixed MB budget is capped at {RAM_CAP:.0%} of the phone's RAM, which matters only on "
         "the 4 GB Redmi 9A. A configuration's peak is its measured peak (at a 512-token context) plus the KV cache "
         "of the tokens a request holds beyond 512: cached prefix + prefilled + answer, which exceeds 512 only for "
         "intent routing (~1,060) and reminder parsing (~930). The KV cache per token comes from each model's "
         "architecture (F16 keys and values; Qwen3.5 caches only its full-attention layers, 1 in 4), so it adds at "
         "most ~80 MB (Qwen3-4B).",
         "- **Prefill** ≥ tokens prefilled per request ÷ the first-token time.",
         "- **Decode**: an answer that is only usable complete (a label, JSON, a parse, a corrected sentence) must be "
         "done in time, so decode ≥ answer tokens ÷ (whole-answer time − first-token time). A summary the user reads "
         f"as it streams needs decode ≥ {READ_TPS} tokens/s (silent reading, ~4 words/s).",
         "- **Overnight**: all of a night's requests must finish in 6 hours, so each gets 6 h ÷ requests per night "
         "(at most a minute), half of it for the first token. 5,000 messages get 4.3 s each.",
         "- **Energy per request** = the deployment's battery budget ÷ its requests per day, using each phone's "
         "battery (" + ", ".join(f"{p} {mah:,} mAh ≈ {wh} Wh" for p, (mah, wh) in BATTERY.items()) + "). A "
         "configuration's request energy is prefilled tokens × its prefill mJ/token + answer tokens × its decode "
         f"mJ/token. An overnight job may use up to {OVERNIGHT_PCT}% of a full charge, so the phone still finishes "
         "charging by morning and does not heat up while it charges.", "",
         "| Task | Deployment | Kind | What it does | Requests | Energy budget | Prefill ≥ | Decode ≥ | Why |",
         "|---|---|---|---|--:|---|--:|--:|---|"]
    for uc in USE_CASES:
        L.append(f"| {uc.task.key} | {uc.name} | {KINDS[uc.kind][3]} | {uc.what} | {uc.per_day:,} a "
                 f"{'night' if uc.kind == 'overnight' else 'day'} | ≤ {uc.battery_pct}% of a charge | "
                 f"{num(uc.prompt / uc.ttft_s)} tokens/s | {uc.decode:g} tokens/s | {uc.why} |")
    L += ["", "**Assumptions and caveats.**", "",
          "- **Request size versus the measurement.** Values come from a 256-token prompt and a 256-token answer, "
          "run from an empty context. Here a request prefills 17-515 tokens and answers 2-85, on top of a cached "
          "prefix of up to ~880 tokens. Attention over the longer context makes prefill and decode somewhat "
          "slower. Very short prefills run below the measured tokens/s, most of all on the GPU, where a 256-token "
          "batch fills the hardware and a 20-token one does not, and each call has a fixed overhead the "
          "throughput numbers do not include. So interactive scenarios with short requests are optimistic, GPU "
          "ones most. Re-measuring at the task sizes (e.g. `llama-bench -p 32 -n 40 -d 880`) would remove this.",
          "- **Cached prefix.** llama.cpp can keep a prefix's KV cache and drop everything after it, or save and "
          "restore the state. For Qwen3.5, whose linear-attention layers cannot be cut back, the state has to be "
          "saved after the prefix and restored. We assume MNN can do the same with the pinned builds; if it cannot, "
          "its rows would have to prefill prefix + request. The prefix's one-time prefill is not counted.",
          "- **Resident model.** Every deployment assumes the model stays loaded. Loading it (seconds, and energy) "
          "is not counted, which matters most for keyboards and background services.",
          "- **Transcription.** The conversation summaries start from a transcript; speech recognition is not "
          "counted.",
          "- **Energy** is whole-phone energy net of idle, so it excludes the screen.", "",
          "## Which scenarios", "",
          f"All {len(USE_CASES)} use cases were run on all four phones ({len(results)} combinations). Tiers that "
          "came out:", "",
          "| Phone | Easy | Medium | Hard | Hard with no survivor |", "|---|--:|--:|--:|--:|"]
    for p in PHONE_ORDER:
        c = Counter(s["tier"] for s in results if s["phone"] == p)
        none = sum(1 for s in results if s["phone"] == p and not s["best"])
        L.append(f"| {p} | {c['Easy']} | {c['Medium']} | {c['Hard']} | {none} |")
    easy = Counter(s["phone"] for s in chosen if s["tier"] == "Easy")
    no_tier = sorted(f"{t} in {tier.lower()}" for t in TASKS for tier, _, _ in TIERS
                     if not any(s["uc"].task.key == t and s["tier"] == tier for s in chosen))
    L += ["", f"From these, {PER_TIER} per tier were chosen, spreading them across tasks, phones and deployments, "
          "and only combinations where at least one configuration survives (a scenario with no survivor has no "
          "answer to check). All ten tasks are used" + (
              f", but none appears as {', '.join(no_tier)}: none of its deployments lands in that tier on any phone"
              if no_tier
              else ", each in every tier") + ". Easy scenarios by phone: " +
          ", ".join(f"{p} {n}" for p, n in easy.most_common()) + ". Easy is rare away from the OnePlus 15, which "
          "runs 93 of 104 configurations and has 16 GB of RAM: elsewhere only overnight jobs keep 60% of the pool. "
          "The most any deployment keeps on each of the other phones: " + ", ".join(
              f"{p} {max(len(s['kept']) for s in results if s['phone'] == p)} of "
              f"{next(s['pool'] for s in results if s['phone'] == p)}" for p in PHONE_ORDER if p != "OnePlus 15") +
          ".", "",
          "## Results", "",
          "| ID | Task | Phone | Deployment | Limits (memory / prefill / decode / energy) | Kept | Pick (model, "
          "precision, runtime, backend) | Pick's values (memory, prefill / decode, energy) | Most accurate student "
          "fits? |",
          "|---|---|---|---|---|--:|---|---|---|"]
    for s in chosen:
        lim, b = s["limits"], s["best"]
        L.append(f"| [{s['id']}](#{s['id'].lower()}) | {s['uc'].task.key} | {s['phone']} | {s['uc'].name} | ≤ {num(lim['mem'])} MB / "
                 f"≥ {num(lim['pre'])} / ≥ {lim['dec']:g} tokens/s / ≤ {num(lim['energy'])} J | {len(s['kept'])} of "
                 f"{s['pool']} ({s['share']:.0%}) | **{LABEL[b['model']]} {b['quant']}** {FW_LABEL[b['fw']]} "
                 f"{b['be'].upper()} | {metrics(b, s['uc'])} | "
                 f"{'yes' if s['uc'].task.best_fits(s['kept']) else '**no**'} |")

    picks = Counter((FW_LABEL[s["best"]["fw"]], s["best"]["be"].upper()) for s in chosen)
    by_tier = {t: Counter(LABEL[s["best"]["model"]] for s in chosen if s["tier"] == t) for t, _, _ in TIERS}
    reasons = Counter()
    for s in chosen:
        for r, miss, e in s["failed"]:
            for m in miss:
                reasons[m[0]] += 1
    L += ["", "### What the picks show", "",
          f"- **The Factory's most accurate student for the task fits in "
          f"{sum(s['uc'].task.best_fits(s['kept']) for s in chosen)} of 60 scenarios** (its model at its bit "
          "width survives the filter, on any runtime and backend). In the rest, the most accurate model the "
          "Factory trained cannot meet the deployment's limits on that phone, and the pick is the most capable "
          "model that can.",
          "- **Runtime and backend of the pick:** " + ", ".join(f"{fw} {be} {n}" for (fw, be), n in
                                                                 picks.most_common()) + " of 60.",
          *[f"- **{t} picks:** " + ", ".join(f"{m} ×{n}" for m, n in c.most_common()) for t, c in by_tier.items()],
          "- **Why configurations were dropped** (a configuration can miss several limits): " +
          ", ".join(f"{dict(M='memory', P='prefill', D='decode', E='energy').get(c, 'no energy reading')} "
                    f"{n:,}" for c, n in reasons.most_common()) + ".", ""]
    L += _observations(chosen)

    L += ["## Scenarios", ""]
    for s in chosen:
        uc, lim, b = s["uc"], s["limits"], s["best"]
        L += [f"### {s['id']}", "",
              f"**{s['tier']} · {s['phone']} · {uc.task.key} · {uc.name}** ({KINDS[uc.kind][3]}): {uc.what}. "
              f"{uc.prompt:,} prefilled + {uc.answer:,} answer tokens per request ({uc.task.cached:,}-token prefix "
              f"cached); {uc.time_txt()}; {uc.per_day:,} requests ≤ {uc.battery_pct}% of a charge a "
              f"{'night' if uc.kind == 'overnight' else 'day'}.", "",
              f"Limits: {fmt_lim(lim, uc)}.", "",
              f"**Pick: {LABEL[b['model']]} {b['quant']}, {FW_LABEL[b['fw']]}, {b['be'].upper()}** "
              f"({metrics(b, uc)}). Kept {len(s['kept'])} of {s['pool']} configurations "
              f"({s['share']:.0%}); {len(s['failed'])} filtered out; {len(s['cant'])} cannot run on this phone. "
              f"The Factory's most accurate {uc.task.key} student ({uc.task.best}) "
              f"{'fits' if uc.task.best_fits(s['kept']) else '**does not fit**'} under these limits.", "",
              f"![{s['id']}](figures/scenarios/{s['id']}.png)", ""]
        kept = sorted((x for x in s["kept"]), key=lambda x: capability(x[0]), reverse=True)
        L += [f"<details><summary>Kept pool ({len(kept)}), most capable first</summary>", "",
              "| Configuration | Peak memory (MB) | Prefill (tokens/s) | Decode (tokens/s) | Energy (J per request) |",
              "|---|--:|--:|--:|--:|"]
        for r, _, e in kept:
            L.append(f"| {name(r)}{' **(pick)**' if r is b else ''} | {num(peak_mb(r, uc))} | {num(r['pre'])} | "
                     f"{num(r['dec'])} | {num(e)} |")
        groups = {}
        for r, miss, e in s["failed"]:
            for c, _, val in miss:
                groups.setdefault(c, []).append(f"{name(r)} ({val})" if val else name(r))
        L += ["", "</details>", "",
              f"<details><summary>Filtered out ({len(s['failed'])}) and cannot run ({len(s['cant'])})"
              "</summary>", ""]
        titles = {"M": f"Peak memory over {num(lim['mem'])} MB", "P": f"Prefill under {num(lim['pre'])} tokens/s",
                  "D": f"Decode under {lim['dec']:g} tokens/s", "E": f"Energy over {num(lim['energy'])} J per request",
                  "E?": "No energy reading"}
        for c in ("M", "P", "D", "E", "E?"):
            if c in groups:
                L.append(f"- **{titles[c]} ({len(groups[c])}):** " + ", ".join(groups[c]))
        if s["cant"]:
            L.append(f"- **Cannot run on this phone ({len(s['cant'])}):** " +
                     ", ".join(f"{name(r)} ({why.split(':')[0]})" for r, why in s["cant"]))
        L += ["", "</details>", ""]

    (DOCS / "SCENARIOS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {DOCS / 'SCENARIOS.md'}, {len(chosen)} figures in {FIG_DIR}, {CSV_OUT}")
    print(Counter(s["tier"] for s in chosen), Counter(s["phone"] for s in chosen))


def _observations(chosen) -> list[str]:
    """Same use case, different phones: does the pick change?"""
    by_uc = {}
    for s in chosen:
        by_uc.setdefault(s["uc"].name, []).append(s)
    multi = {u: ss for u, ss in by_uc.items() if len(ss) > 1}
    differ = [u for u, ss in multi.items() if len({name(s["best"]) for s in ss}) > 1]
    out = []
    if multi:
        out += [f"- **Same use case, different phone:** {len(multi)} use cases appear on more than one phone; the "
                f"pick differs between phones for {len(differ)} of them" +
                (" (" + "; ".join(f"{u}: " + ", ".join(f"{s['phone']} {name(s['best'])}" for s in multi[u])
                                  for u in differ[:4]) + ")" if differ else "") + ".", ""]
    return out


if __name__ == "__main__":
    main()
