#!/usr/bin/env python3
"""
Write docs/REQUIREMENTS.md: realistic user requirements graded easy / medium / hard by how often a configuration chosen
WITHOUT the target phone's measurements meets them, across the four measured phones (req_data.py).

  python requirements_md.py
"""

from __future__ import annotations

import statistics as st
from datetime import datetime

from req_data import DOCS, PHONES, load

SNAP = ["Galaxy S23", "OnePlus 15", "OnePlus 8 Pro"]  # flagship-class phones: the main comparison
BUDGET = "Redmi 9A"
TIERS = [("Easy", 0.60, 1.01), ("Medium", 0.30, 0.60), ("Hard", 0.0, 0.30)]
LABEL = {"smollm2-135m": "SmolLM2-135M", "gemma3-270m": "Gemma3-270M", "smollm2-360m": "SmolLM2-360M",
         "qwen3-0.6b": "Qwen3-0.6B", "qwen3.5-0.8b": "Qwen3.5-0.8B", "qwen3-1.7b": "Qwen3-1.7B",
         "qwen3.5-2b": "Qwen3.5-2B", "qwen3-4b-instruct-2507": "Qwen3-4B", "qwen3.5-4b": "Qwen3.5-4B"}


class Req:
    """meta: what the Factory can check from the model card alone (size, parameters, bits); need: what only a
    measurement on the phone can tell (speed, energy, and whether the configuration runs at all)."""

    def __init__(self, rid, name, use, meta_txt, need_txt, meta, need, energy=False):
        self.rid, self.name, self.use, self.meta_txt, self.need_txt = rid, name, use, meta_txt, need_txt
        self.meta, self.need, self.energy = meta, need, energy

    def candidates(self, rows):
        c = [r for r in rows.values() if self.meta(r)]
        if self.energy:  # a configuration that ran but has no energy reading is unknown, not a miss
            if not any(r.get("req_j") for r in rows.values()):
                return None
            c = [r for r in c if not r["ok"] or r.get("req_j") is not None]
        return c

    def meets(self, r):
        return bool(r["ok"] and self.need(r))

    def rate(self, rows):
        c = self.candidates(rows)
        if not c:
            return None, 0
        return sum(self.meets(r) for r in c) / len(c), len(c)


REQS = [
    Req("E1", "Basic assistant", "a help bot where a slow but steady answer is fine",
        "any model", "decode ≥ 5 tokens/s", lambda r: True, lambda r: r["dec"] >= 5),
    Req("E2", "Small on-device helper", "smart replies, short rewrites",
        "model file ≤ 1 GB", "decode ≥ 10 tokens/s", lambda r: r["size_gb"] <= 1.0, lambda r: r["dec"] >= 10),
    Req("E3", "Capable model, patient user", "offline Q&A where answer quality matters more than speed",
        "≥ 1.5B parameters", "decode ≥ 5 tokens/s", lambda r: r["params_b"] >= 1.5, lambda r: r["dec"] >= 5),
    Req("M1", "Interactive chat", "a chat app: first token quickly, text faster than reading speed",
        "any model", "decode ≥ 10 tokens/s and first token ≤ 2 s after a 256-token prompt",
        lambda r: True, lambda r: r["dec"] >= 10 and r["ttft_s"] <= 2),
    Req("M2", "Mid-size model at reading speed", "assistant that needs more than a toy model",
        "≥ 0.6B parameters", "decode ≥ 10 tokens/s", lambda r: r["params_b"] >= 0.59, lambda r: r["dec"] >= 10),
    Req("M3", "Higher precision", "accuracy-sensitive extraction, no 4-bit quantization",
        "8-bit or 16-bit weights", "decode ≥ 15 tokens/s", lambda r: r["bits"] >= 8, lambda r: r["dec"] >= 15),
    Req("M4", "Snappy small model", "in-app assistant that must feel instant",
        "model file ≤ 1 GB", "decode ≥ 15 tokens/s and first token ≤ 1 s",
        lambda r: r["size_gb"] <= 1.0, lambda r: r["dec"] >= 15 and r["ttft_s"] <= 1),
    Req("M5", "Keyboard autocomplete", "suggestions while typing",
        "model file ≤ 0.5 GB", "first token ≤ 0.5 s and decode ≥ 30 tokens/s",
        lambda r: r["size_gb"] <= 0.5, lambda r: r["ttft_s"] <= 0.5 and r["dec"] >= 30),
    Req("H1", "Capable fluent chat", "a 1.5B+ assistant that streams faster than you read",
        "≥ 1.5B parameters", "decode ≥ 20 tokens/s", lambda r: r["params_b"] >= 1.5, lambda r: r["dec"] >= 20),
    Req("H2", "Capable interactive chat", "the M1 chat app with a 1.5B+ model",
        "≥ 1.5B parameters", "decode ≥ 10 tokens/s and first token ≤ 2 s",
        lambda r: r["params_b"] >= 1.5, lambda r: r["dec"] >= 10 and r["ttft_s"] <= 2),
    Req("H3", "Document summarization", "read a long note or email, then summarize",
        "any model", "prefill ≥ 300 tokens/s (a 2,000-token document in under 7 s) and decode ≥ 10 tokens/s",
        lambda r: True, lambda r: r["pre"] >= 300 and r["dec"] >= 10),
    Req("H4", "Battery-capped capable model", "always-available assistant that must not drain the battery",
        "≥ 1.5B parameters", "≤ 50 J per request (256-token prompt + 256-token answer, about 0.1% of a phone "
        "battery) and decode ≥ 10 tokens/s",
        lambda r: r["params_b"] >= 1.5, lambda r: r["req_j"] <= 50 and r["dec"] >= 10, energy=True),
]

BUDGET_REQS = [
    Req("B1", "Runs at all", "", "any model", "loads and completes the benchmark", lambda r: True, lambda r: True),
    Req("B2", "Tiny model, usable", "", "model file ≤ 0.5 GB", "decode ≥ 5 tokens/s",
        lambda r: r["size_gb"] <= 0.5, lambda r: r["dec"] >= 5),
    Req("B3", "Small model, usable", "", "model file ≤ 1 GB", "decode ≥ 5 tokens/s",
        lambda r: r["size_gb"] <= 1.0, lambda r: r["dec"] >= 5),
    Req("B4", "Mid-size model runs", "", "≥ 0.6B parameters", "loads and completes the benchmark",
        lambda r: r["params_b"] >= 0.59, lambda r: True),
    Req("B5", "Mid-size model, usable", "", "≥ 0.6B parameters", "decode ≥ 3 tokens/s",
        lambda r: r["params_b"] >= 0.59, lambda r: r["dec"] >= 3),
    Req("B6", "Any model, answer in a minute", "", "any model",
        "first token ≤ 10 s and decode ≥ 5 tokens/s (256 tokens in under a minute)",
        lambda r: True, lambda r: r["ttft_s"] <= 10 and r["dec"] >= 5),
]


def tier(v):
    return next(t for t, lo, hi in TIERS if lo <= v < hi)


def cfg(r):
    return f"{LABEL[r['model']]} {r['quant']} {'llama.cpp' if r['fw'] == 'llama.cpp' else 'MNN'} {r['be'].upper()}"


def pct(v):
    return "n/a" if v is None else f"{v:.0%}"


def bit_cost(rows):
    """Median 8-bit / 4-bit decode speed over the configurations that ran at both."""
    eight = {"Q4_0": "Q8_0", "Q4": "Q8"}
    v = [rows[(m, fw, be, eight[q])]["dec"] / r["dec"] for (m, fw, be, q), r in rows.items()
         if q in eight and r["ok"] and (m, fw, be, eight[q]) in rows and rows[(m, fw, be, eight[q])]["ok"]]
    return st.median(v) if v else float("nan")


def transfer(req, P):
    """The Factory has another phone's measurements: (a) deploy every configuration that met the requirement there;
    (b) deploy the single configuration with the most headroom there. Success = it meets the requirement here."""
    sets, picks = [], []
    for tgt in SNAP:
        for ref in SNAP:
            if ref == tgt or req.candidates(P[ref]) is None or req.candidates(P[tgt]) is None:
                continue
            known = lambda r: not req.energy or not r["ok"] or r.get("req_j") is not None  # energy unknown: skip
            ok_ref = [k for k, r in P[ref].items()
                      if req.meta(r) and known(r) and req.meets(r) and k in P[tgt] and known(P[tgt][k])]
            if not ok_ref:
                continue
            sets.append(sum(req.meets(P[tgt][k]) for k in ok_ref) / len(ok_ref))
            margin = lambda k: P[ref][k]["dec"]  # most decode headroom on the reference phone
            best = max(ok_ref, key=margin)
            picks.append(req.meets(P[tgt][best]))
    return (st.mean(sets) if sets else None), picks


def main():
    P = load()
    common = set(P["OnePlus 15"])  # the S23 also lists Qwen3.5-4B F16, which the other phones' tables omit
    P["Galaxy S23"] = {k: v for k, v in P["Galaxy S23"].items() if k in common}

    L = ["# Why each phone needs its own measurements: requirement difficulty", "",
         f"Generated {datetime.now().isoformat(timespec='minutes')} by `Benchmark-Harness/paper_table5/requirements_md.py` "
         "from the four measured phones ([RESULTS.md](RESULTS.md)).", "",
         "## The question", "",
         "A user asks the SLM Factory for a model that meets a requirement on *their* phone, for example \"a 1.5B+ "
         "model that streams at 20 tokens/s\". From the model card alone, the Factory knows each configuration's file "
         "size, parameter count and bit-width. It does not know, for this phone:", "",
         "- how fast prefill and decode are;",
         "- how much energy a request costs;",
         "- whether the configuration runs at all (it may not fit in memory, swap, or crash).", "",
         "**Difficulty** is the chance that a configuration passing the model-card filter also meets the measured part "
         "of the requirement on the target phone, i.e. the success rate of a choice made without that phone's "
         "measurements:", "",
         "| Tier | Chance of success without phone-specific measurements |", "|---|---|",
         "| Easy | 60–100%: the Factory alone usually gets it right |",
         "| Medium | 30–60%: a coin flip |",
         "| Hard | 0–30%: without measurements the Factory is usually wrong |", "",
         "A requirement's tier is the average over the three flagship-class phones; each phone's own rate and tier is "
         "shown too, because the same requirement changes tier from phone to phone. The budget Redmi 9A is graded "
         "separately (last section).", "",
         "**Phones:** " + "; ".join(f"{p} ({d})" for p, d in PHONES.items()) + ". Same pool (9 models × Q4/Q8/F16 × "
         "llama.cpp/MNN × CPU/GPU), same 256-token prompt and 256-token answer. Energy is missing on the OnePlus 8 "
         "Pro, so energy requirements are graded on the other phones.", ""]

    # ---- the catalog ----
    rates = {}
    for q in REQS:
        rates[q.rid] = {p: q.rate(P[p]) for p in SNAP}
    L += ["## Requirements by tier", "",
          "| ID | Requirement | Use case | Model-card part (Factory can check) | Measured part (needs the phone) | "
          + " | ".join(SNAP) + " | Average | Tier |",
          "|---|---|---|---|---|" + "--:|" * len(SNAP) + "--:|---|"]
    for q in REQS:
        vals = [v for v, _ in rates[q.rid].values() if v is not None]
        avg = st.mean(vals)
        cells = [f"{pct(v)} ({tier(v)[0]})" if v is not None else "n/a" for v, _ in rates[q.rid].values()]
        L.append(f"| {q.rid} | {q.name} | {q.use} | {q.meta_txt} | {q.need_txt} | " + " | ".join(cells)
                 + f" | {avg:.0%} | **{tier(avg)}** |")
    flips = [q.rid for q in REQS
             if len({tier(v) for v, _ in rates[q.rid].values() if v is not None}) > 1]
    L += ["", "Each cell: share of model-card-eligible configurations that meet the requirement on that phone, and "
          "that phone's tier (E/M/H).", "",
          f"**{len(flips)} of {len(REQS)} requirements change tier between phones** ({', '.join(flips)}). For example, "
          "H2 (a 1.5B+ model with interactive latency) is a coin flip on the OnePlus 15, hard on the Galaxy S23 and "
          "impossible on the OnePlus 8 Pro, even though all three are Snapdragon flagships.", ""]

    # ---- per-phone answer: which configs pass ----
    L += ["## What the measurements reveal: the passing configurations differ by phone", "",
          "For the hard requirements, the configurations that pass on each phone (fastest decode first):", ""]
    for q in [r for r in REQS if r.rid.startswith("H")]:
        L += [f"**{q.rid} {q.name}** ({q.meta_txt}; {q.need_txt})", ""]
        for p in SNAP:
            c = q.candidates(P[p])
            if c is None:
                L.append(f"- {p}: no energy data")
                continue
            ok = sorted((r for r in c if q.meets(r)), key=lambda r: -r["dec"])
            L.append(f"- {p}: " + (", ".join(cfg(r) for r in ok[:4]) + (f" (+{len(ok) - 4} more)" if len(ok) > 4 else "")
                                   if ok else "**none**"))
        L.append("")

    # ---- can another phone's data stand in? ----
    L += ["## Can another phone's measurements stand in?", "",
          "If the Factory measured one phone and reused those numbers for another, how often would its choice still "
          "work? For each pair of flagship phones: **all passing** = deploy any configuration that met the requirement "
          "on the measured phone; **best pick** = deploy the one with the most decode headroom there.", "",
          "| ID | Requirement | All passing: still meets it on the other phone | Best pick: still meets it |",
          "|---|---|--:|--:|"]
    for q in REQS:
        s, picks = transfer(q, P)
        L.append(f"| {q.rid} | {q.name} | {pct(s)} | "
                 + (f"{sum(picks)} of {len(picks)} phone pairs" if picks else "n/a") + " |")
    L += ["", "Reusing another flagship's measurements beats choosing blind. Its single fastest configuration "
          "usually transfers for easy and medium requirements, but not for hard ones (H1, H2, H4 above). And 24–34% "
          "of the configurations that pass a medium requirement on one phone fail it on another (33–100% for hard "
          "ones), so a Factory that needs a specific model, engine or bit-width still has to measure. Speed does not scale by a "
          "single factor between phones (next section).", ""]

    # ---- why: evidence ----
    def ratio(a, b, ph):
        rs = sorted(P[a][k][ph] / P[b][k][ph] for k in P[b] if P[a].get(k, {}).get("ok") and P[b][k]["ok"])
        return rs[0], st.median(rs), rs[-1]
    L += ["## Why the Factory can't predict this from the model card", ""]
    a = ratio("OnePlus 15", "Galaxy S23", "pre")
    b = ratio("OnePlus 15", "Galaxy S23", "dec")
    c = ratio("OnePlus 8 Pro", "Galaxy S23", "dec")
    L += [f"1. **Speed doesn't scale by one factor between phones.** The same configuration's prefill is "
          f"{a[0]:.2f}× to {a[2]:.0f}× faster on the OnePlus 15 than on the Galaxy S23 (median {a[1]:.1f}×), and "
          f"decode {b[0]:.2f}× to {b[2]:.1f}×. Against the S23, the OnePlus 8 Pro decodes at {c[0]:.2f}× to {c[2]:.1f}×. "
          "The engine, backend and weight type, not the phone's general speed, decide most of the outcome."]
    gpu = []
    for p in SNAP:
        pairs = [(P[p][k], P[p].get((k[0], k[1], "gpu", k[3]))) for k in P[p] if k[2] == "cpu"]
        pairs = [(x, g) for x, g in pairs if g and g["ok"] and x["ok"]]
        gpu.append(f"{p} {sum(g['pre'] > x['pre'] for x, g in pairs)} of {len(pairs)}")
    L += [f"2. **\"Use the GPU\" is phone-specific.** GPU prefill beats CPU prefill in: {'; '.join(gpu)} configurations. "
          "GPU decode is usually *slower* than CPU decode on all three. On the OnePlus 8 Pro, llama.cpp on the GPU "
          "is unusable (Qwen3-1.7B 8-bit: 0.88 tokens/s prefill)."]

    def best(p, meta, ph):
        cand = [r for r in P[p].values() if r["ok"] and meta(r)]
        return cfg(max(cand, key=lambda r: r[ph])) if cand else "none"
    L += ["3. **The best configuration for the same goal differs by phone:**", "",
          "   | Goal | " + " | ".join(P) + " |", "   |---|" + "---|" * len(P)]
    for goal, meta, ph in [("Fastest decode, ≥ 1.5B parameters", lambda r: r["params_b"] >= 1.5, "dec"),
                           ("Fastest prefill, Qwen3-0.6B", lambda r: r["model"] == "qwen3-0.6b", "pre"),
                           ("Fastest prefill, any model", lambda r: True, "pre")]:
        L.append(f"   | {goal} | " + " | ".join(best(p, meta, ph) for p in P) + " |")
    L.append("")
    runs = []
    for i, x in enumerate(SNAP):
        for y in SNAP[i + 1:]:
            ks = set(P[x]) & set(P[y])
            runs.append(f"{x} vs {y}: {sum(P[x][k]['ok'] != P[y][k]['ok'] for k in ks)} of {len(ks)}")
    L += [f"4. **Whether a configuration runs at all differs by phone** (it runs on one phone but not the other): "
          f"{'; '.join(runs)}. Memory limits, OpenCL driver support and crashes are device-specific: the head "
          "llama.cpp GPU build cannot even load on the OnePlus 8 Pro's driver.",
          "5. **The cost of more bits differs by phone.** Going from 4-bit to 8-bit keeps a median "
          + ", ".join(f"{bit_cost(P[x]):.2f}× on the {x}" for x in SNAP) + " of decode speed, and on the S23's "
          "llama.cpp GPU path 8-bit even decodes faster than 4-bit for some small models.", ""]

    # ---- budget phone ----
    L += [f"## The budget phone ({BUDGET})", "",
          "The flagship requirements are all **hard** on the Redmi 9A: almost nothing reaches 10 tokens/s. A budget "
          "phone needs its own requirements, and they show the same pattern:", "",
          "| ID | Requirement | Model-card part | Measured part | Redmi 9A | Tier |", "|---|---|---|---|--:|---|"]
    for q in BUDGET_REQS:
        v, n = q.rate(P[BUDGET])
        L.append(f"| {q.rid} | {q.name} | {q.meta_txt} | {q.need_txt} | {pct(v)} | **{tier(v)}** |")
    flag = [q.rid + " " + pct(q.rate(P[BUDGET])[0]) for q in REQS]
    L += ["", "Flagship requirements on the Redmi 9A: " + ", ".join(flag) + ". On this phone llama.cpp decodes "
          "1.7–3.5× slower than MNN for the same 4-bit model, so the engine choice alone decides whether a small model "
          "is usable, and which mid-size configurations even load depends on its 4 GB of RAM.", ""]

    # ---- conclusion ----
    L += ["## Takeaways", "",
          "- **Easy requirements** (loose speed targets) the Factory can meet without measuring: most configurations "
          "pass on a flagship phone.",
          "- **Medium requirements** (interactive latency, higher precision, snappy small models) are a coin flip "
          "without measurements, and the odds differ by 3–5× between flagship phones.",
          "- **Hard requirements** (a capable model that is fast, a long-prompt workload, a battery budget) are met by "
          "only a few configurations, and which ones depends on the phone's engine, GPU driver and memory. Without "
          "measuring that phone, the Factory is usually wrong, and even another flagship's measurements often pick a "
          "configuration that fails.",
          "- **One benchmark pass per phone** (the 104-configuration sweep in [BENCHMARKING_GUIDE.md](BENCHMARKING_GUIDE.md)) "
          "turns every hard requirement into a lookup.", ""]
    (DOCS / "REQUIREMENTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"-> {DOCS / 'REQUIREMENTS.md'}")


if __name__ == "__main__":
    main()
