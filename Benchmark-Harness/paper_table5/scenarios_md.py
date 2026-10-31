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


class UseCase:
    """A request shape (prompt and answer tokens) plus the limits an app developer would set for it. mem is MB, or a
    fraction of the phone's RAM for an app that is the only thing running (a dedicated assistant, an overnight job)."""

    def __init__(self, name, what, prompt, answer, ttft_s, decode, mem, per_day, battery_pct, why):
        self.name, self.what, self.prompt, self.answer = name, what, prompt, answer
        self.ttft_s, self.decode, self.mem, self.per_day, self.battery_pct, self.why = (
            ttft_s, decode, mem, per_day, battery_pct, why)

    def limits(self, phone) -> dict:
        mem = self.mem if self.mem > 1 else round(self.mem * RAM_GIB[phone] * 1024, -2)
        joules = BATTERY[phone][1] * 3600 * self.battery_pct / 100 / self.per_day
        return dict(mem=mem, pre=self.prompt / self.ttft_s, dec=self.decode, energy=joules)

    def mem_txt(self):
        return f"{self.mem:,} MB" if self.mem > 1 else f"{self.mem:.0%} of RAM"


USE_CASES = [
    UseCase("Keyboard next-phrase", "suggest the next words while typing", 64, 8, 0.3, 20, 300, 1000, 3,
            "an input method stays resident all day, so it must be small; suggestions are useless after ~0.3 s"),
    UseCase("Smart replies", "three reply chips under a received message", 128, 24, 0.5, 15, 512, 200, 2,
            "runs inside a messaging app; chips must appear before the user starts typing"),
    UseCase("Live message translation", "translate each chat message as it arrives", 64, 64, 0.5, 10, 1024, 200, 3,
            "part of a messaging app; the translation should keep up with the conversation"),
    UseCase("Game NPC dialogue", "a character answers the player", 200, 40, 1.0, 12, 768, 100, 2,
            "the game itself holds most of the memory; a reply within a second keeps the scene moving"),
    UseCase("Notification triage", "classify incoming notifications in the background", 128, 8, 5.0, 2, 512, 300, 1,
            "background services get killed above a few hundred MB; nobody waits on the result"),
    UseCase("Screen-reader descriptions", "describe a screen's content aloud", 400, 60, 2.0, 5, 1024, 100, 3,
            "output is spoken (~3.5 tokens/s), so decode only needs to stay ahead of the voice"),
    UseCase("Voice assistant", "spoken answer to a spoken question", 256, 64, 1.0, 5, 1536, 50, 2,
            "a pause over ~1 s feels broken in conversation; decode only needs to beat speech"),
    UseCase("Language tutor", "conversational practice with corrections", 300, 100, 2.0, 6, 1536, 40, 3,
            "read along at a learner's pace, slightly faster than reading speed"),
    UseCase("Journaling prompts", "a reflective question after a journal entry", 300, 150, 5.0, 5, 1536, 5, 1,
            "a few uses a day; the user is not in a hurry"),
    UseCase("Interactive chat", "a general chat assistant", 256, 256, 2.0, 10, 2048, 50, 3,
            "first token within 2 s and text faster than reading speed (~5-7 tokens/s) with margin"),
    UseCase("Email summary on open", "summarize an email when it is opened", 1000, 100, 4.0, 8, 2048, 30, 2,
            "the summary should be ready by the time the user has glanced at the email"),
    UseCase("Recipe / how-to helper", "step-by-step instructions", 150, 300, 5.0, 5, 2048, 10, 1,
            "read while cooking; reading speed is enough"),
    UseCase("Bedtime story", "a story read aloud by text-to-speech", 100, 500, 3.0, 4, 2048, 3, 1,
            "spoken output; must not drain the battery overnight on the nightstand"),
    UseCase("Notes Q&A (RAG)", "answer from retrieved personal notes", 1500, 200, 5.0, 10, 2560, 20, 2,
            "a long retrieved context must be read within ~5 s"),
    UseCase("Fluent premium chat", "a chat assistant that streams fast", 512, 256, 1.5, 20, 3072, 50, 3,
            "the flagship-app experience: fast first token and fast streaming"),
    UseCase("Coding helper", "explain or fix a code snippet", 1000, 150, 3.0, 15, 3072, 30, 3,
            "code context is long and answers are scanned quickly"),
    UseCase("Private chat app", "a dedicated on-device chat app the user opens", 256, 256, 2.0, 10, 0.4, 50, 3,
            "the user opened the AI app itself, so it can hold a large share of RAM; same speed as chat"),
    UseCase("Offline translator app", "a dedicated translation app for travel", 100, 100, 3.0, 5, 0.4, 50, 3,
            "the app in use; the translation is read, so reading speed is enough"),
    UseCase("Study tutor app", "explain homework questions", 300, 100, 5.0, 6, 0.4, 40, 3,
            "a dedicated app; students wait a few seconds for an explanation"),
    UseCase("Document Q&A app", "ask questions about an opened PDF", 1000, 200, 10.0, 8, 0.4, 20, 3,
            "a dedicated app; reading a long document takes a few seconds anyway"),
    UseCase("Essay feedback app", "comments on a pasted essay", 500, 300, 10.0, 5, 0.4, 20, 3,
            "a dedicated app; feedback is read at reading speed"),
    UseCase("Trip planner app", "plan a day from a list of places", 400, 400, 5.0, 5, 0.4, 10, 2,
            "a dedicated app, used a few times a day while travelling"),
    UseCase("Story-writing app", "co-write fiction with the user", 300, 600, 5.0, 6, 0.4, 20, 3,
            "a dedicated app; long answers read at a relaxed pace"),
    UseCase("Email drafting app", "draft an email from bullet points", 300, 200, 3.0, 8, 0.4, 20, 2,
            "a dedicated app; the draft is skimmed as it streams"),
    UseCase("Offline medical reference", "answer questions from a clinical reference", 300, 300, 5.0, 5, 0.4, 10, 2,
            "a dedicated app in clinics with poor connectivity; correctness over speed"),
    UseCase("Language practice app", "spoken-style conversation practice", 300, 150, 3.0, 6, 0.4, 30, 3,
            "a dedicated app; replies a little faster than a learner reads"),
    UseCase("Field-guide app", "explain a plant or bird from a description, offline", 200, 250, 5.0, 5, 0.4, 10, 2,
            "a dedicated app used outdoors without signal"),
    UseCase("Offline Q&A (travel)", "an assistant with no network", 200, 200, 10.0, 4, 0.4, 20, 3,
            "the only app in use; the user accepts a wait but needs to read the answer as it comes"),
    UseCase("Meeting-notes cleanup", "turn a transcript into notes after a meeting", 1500, 300, 30.0, 3, 0.4, 3, 2,
            "the user starts it and comes back; it must finish within a few minutes"),
    UseCase("Document translation", "translate a page while the user does something else", 500, 500, 30.0, 2, 0.4,
            10, 3, "batch work; finishing in a few minutes is fine"),
    UseCase("Voice-memo summary", "title and summarize a recorded voice memo", 800, 80, 20.0, 2, 0.4, 10, 1,
            "starts after the recording ends; the user checks back later"),
    UseCase("Morning digest", "a scheduled summary of overnight messages", 2000, 300, 120.0, 1, 0.4, 1, 1,
            "runs before the user wakes up; only the battery cost matters"),
    UseCase("Weekly journal summary", "summarize a week of entries", 3000, 400, 120.0, 1, 0.5, 1, 1,
            "rare, scheduled, nothing else running"),
    UseCase("Statement categorization", "categorize a month of bank transactions", 2500, 500, 120.0, 1, 0.4, 1, 1,
            "a scheduled monthly job; it only has to finish and not cost much battery"),
    UseCase("Overnight photo captioning", "caption the photo library while charging", 200, 40, 30.0, 1, 0.5, 300, 20,
            "on the charger, but heat and charging time still cap the energy"),
    UseCase("Overnight inbox sorting", "label every email while charging", 300, 10, 10.0, 1, 0.5, 500, 20,
            "on the charger; many short requests"),
    UseCase("Field data-entry helper", "fill forms from notes on a low-cost phone", 128, 64, 10.0, 2, 1024, 30, 3,
            "aimed at budget phones; a slow but steady answer is acceptable"),
    UseCase("SMS auto-reply", "draft a reply to a text message", 128, 32, 5.0, 3, 768, 50, 2,
            "a short reply; the user can wait a few seconds"),
    UseCase("Async FAQ answers", "answer a question, delivered like a text message", 128, 64, 30.0, 1, 1024, 20, 3,
            "the user is not watching; the answer arrives within a minute or two"),
    UseCase("Health-worker checklist", "guide an offline checklist on a budget phone", 256, 128, 20.0, 2, 1536, 20, 3,
            "offline, low-end hardware; correctness matters more than speed"),
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
        if r["mem_mb"] is None or r["mem_mb"] > lim["mem"]:
            miss.append(("M", f"peak memory {num(r['mem_mb'])} MB > {num(lim['mem'])}", f"{num(r['mem_mb'])} MB"))
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
    the phone, then the use case, used least so far."""
    out, used = [], Counter()
    for tier, _, _ in TIERS:
        pool = [s for s in all_results if s["tier"] == tier and s["best"]]
        phones, cases = Counter(), Counter()
        while len([s for s in out if s["tier"] == tier]) < PER_TIER and pool:
            s = min(pool, key=lambda s: (phones[s["phone"]], cases[s["uc"].name] + used[s["uc"].name],
                                         PHONE_ORDER.index(s["phone"]), USE_CASES.index(s["uc"])))
            pool.remove(s)
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
    fig.suptitle(f"{s['id']} · {s['phone']} · {s['uc'].name}", fontsize=8.5, color="#1f1f1d", x=0.01, ha="left")
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
    return (f"{num(r['mem_mb'])} MB, {num(r['pre'])} / {num(r['dec'])} tokens/s, "
            f"{num(e)} J" if e is not None else "")


def scenarios(data) -> tuple[list[dict], list[dict]]:
    """(every use case on every phone, the chosen 60 with their IDs E01..H20)."""
    results = [evaluate(data[p], uc, p) for p in PHONE_ORDER for uc in USE_CASES]
    chosen = select(results)
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
                                   "; ".join(m[1] for m in miss), round(r["mem_mb"]), round(r["pre"], 2),
                                   round(r["dec"], 2), None if e is None else round(e, 3)])
            for r, why in s["cant"]:
                w.writerow(head + [LABEL[r["model"]], r["quant"], FW_LABEL[r["fw"]], r["be"].upper(),
                                   "cannot run", why, None, None, None, None])

    L = ["# Requirement-filter test: 60 realistic scenarios", "",
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
         "## How the limits are set", "",
         "The limits come from the use case, not from the measurements; the tier is whatever the measurements "
         "then give.", "",
         "- **Prefill** = prompt tokens ÷ the longest acceptable wait for the first token.",
         "- **Decode**: silent reading is about 5-7 tokens/s, speech about 3.5 tokens/s; chat that feels fluent "
         "needs 10-20 tokens/s; background work only needs to finish.",
         "- **Peak memory**: what the app can hold without the system killing it. A keyboard stays resident all "
         "day (≤ 300 MB); background services and features inside another app get 0.5-1 GB; a foreground "
         "assistant 1.5-3 GB so it works on mainstream phones; an app that is the only thing running (a "
         "dedicated offline assistant, an overnight job) a share of the target phone's RAM.",
         "- **Energy per request** = a daily battery budget ÷ requests per day, using each phone's battery "
         "(" + ", ".join(f"{p} {mah:,} mAh ≈ {wh} Wh" for p, (mah, wh) in BATTERY.items()) + "). A configuration's "
         "request energy is prompt tokens × its prefill mJ/token + answer tokens × its decode mJ/token.", "",
         "| Use case | What it does | Request (prompt / answer tokens) | First token within | Decode ≥ | Peak memory ≤ | "
         "Energy budget | Why these limits |",
         "|---|---|--:|--:|--:|--:|---|---|"]
    for uc in USE_CASES:
        L.append(f"| {uc.name} | {uc.what} | {uc.prompt:,} / {uc.answer:,} | {uc.ttft_s:g} s | {uc.decode:g} "
                 f"tokens/s | {uc.mem_txt()} | {uc.per_day:,} request{'s' * (uc.per_day > 1)} ≤ {uc.battery_pct}% of battery a day | "
                 f"{uc.why} |")
    L += ["", "**Caveats.** Values come from the benchmark's 256-token prompt and 256-token answer. For longer prompts "
          "(RAG, coding, summaries) real prefill is somewhat slower and the KV cache adds memory, so those "
          "scenarios are slightly optimistic. Energy is whole-phone energy net of idle, so it excludes the "
          "screen.", "",
          "## Which scenarios", "",
          f"All {len(USE_CASES)} use cases were run on all four phones ({len(results)} combinations). Tiers that "
          "came out:", "",
          "| Phone | Easy | Medium | Hard | Hard with no survivor |", "|---|--:|--:|--:|--:|"]
    for p in PHONE_ORDER:
        c = Counter(s["tier"] for s in results if s["phone"] == p)
        none = sum(1 for s in results if s["phone"] == p and not s["best"])
        L.append(f"| {p} | {c['Easy']} | {c['Medium']} | {c['Hard']} | {none} |")
    L += ["", f"From these, {PER_TIER} per tier were chosen, spreading them across phones and use cases, and only "
          "combinations where at least one configuration survives (a scenario with no survivor has no answer to "
          "check). **All 20 easy scenarios are on the OnePlus 15.** It is the only phone where a realistic use case "
          "keeps 60% of the whole pool: it runs 93 of 104 configurations and has 16 GB of RAM. The most any use "
          "case keeps on the other phones: " + ", ".join(
              f"{p} {max(len(s['kept']) for s in results if s['phone'] == p)} of "
              f"{next(s['pool'] for s in results if s['phone'] == p)}" for p in PHONE_ORDER if p != "OnePlus 15") +
          ".", "",
          "## Results", "",
          "| ID | Phone | Use case | Limits (memory / prefill / decode / energy) | Kept | Pick (model, precision, "
          "runtime, backend) | Pick's values (memory, prefill / decode, energy) |",
          "|---|---|---|---|--:|---|---|"]
    for s in chosen:
        lim, b = s["limits"], s["best"]
        L.append(f"| [{s['id']}](#{s['id'].lower()}) | {s['phone']} | {s['uc'].name} | ≤ {num(lim['mem'])} MB / "
                 f"≥ {num(lim['pre'])} / ≥ {lim['dec']:g} tokens/s / ≤ {num(lim['energy'])} J | {len(s['kept'])} of "
                 f"{s['pool']} ({s['share']:.0%}) | **{LABEL[b['model']]} {b['quant']}** {FW_LABEL[b['fw']]} "
                 f"{b['be'].upper()} | {metrics(b, s['uc'])} |")

    picks = Counter((FW_LABEL[s["best"]["fw"]], s["best"]["be"].upper()) for s in chosen)
    by_tier = {t: Counter(LABEL[s["best"]["model"]] for s in chosen if s["tier"] == t) for t, _, _ in TIERS}
    reasons = Counter()
    for s in chosen:
        for r, miss, e in s["failed"]:
            for m in miss:
                reasons[m[0]] += 1
    L += ["", "### What the picks show", "",
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
              f"**{s['tier']} · {s['phone']} · {uc.name}**: {uc.what}; {uc.prompt:,}-token prompt, "
              f"{uc.answer:,}-token answer, first token within {uc.ttft_s:g} s, {uc.per_day:,} request{'s' * (uc.per_day > 1)} ≤ "
              f"{uc.battery_pct}% of battery a day.", "",
              f"Limits: {fmt_lim(lim, uc)}.", "",
              f"**Pick: {LABEL[b['model']]} {b['quant']}, {FW_LABEL[b['fw']]}, {b['be'].upper()}** "
              f"({metrics(b, uc)}). Kept {len(s['kept'])} of {s['pool']} configurations "
              f"({s['share']:.0%}); {len(s['failed'])} filtered out; {len(s['cant'])} cannot run on this phone.", "",
              f"![{s['id']}](figures/scenarios/{s['id']}.png)", ""]
        kept = sorted((x for x in s["kept"]), key=lambda x: capability(x[0]), reverse=True)
        L += [f"<details><summary>Kept pool ({len(kept)}), most capable first</summary>", "",
              "| Configuration | Peak memory (MB) | Prefill (tokens/s) | Decode (tokens/s) | Energy (J per request) |",
              "|---|--:|--:|--:|--:|"]
        for r, _, e in kept:
            L.append(f"| {name(r)}{' **(pick)**' if r is b else ''} | {num(r['mem_mb'])} | {num(r['pre'])} | "
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
