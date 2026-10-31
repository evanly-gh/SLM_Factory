#!/usr/bin/env python3
"""
A no-measurement baseline for the requirement filter in docs/SCENARIOS.md: ask a Claude chat that has only what the
SLM Factory knows without this harness (phone specs, model cards, runtime and backend) to filter each scenario's pool
and pick a configuration, then score its answer against the measured filter.

  python baseline.py prompt                  # writes baseline/baseline_prompt.md and baseline/scenario_ids.json
  python baseline.py score RESPONSE.txt      # writes docs/BASELINE_COMPARISON.md

The prompt uses neutral, shuffled scenario IDs (S01..S60), so the E/M/H tier is not given away.
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
from scenarios_md import BATTERY, COLUMNS, FW_LABEL, PHONE_ORDER, RAM_GIB, num, request_j, scenarios

OUT_DIR = HERE / "baseline"
PROMPT = OUT_DIR / "baseline_prompt.md"
ID_MAP = OUT_DIR / "scenario_ids.json"
REPORT = DOCS / "BASELINE_COMPARISON.md"
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

def write_prompt():
    data = load()
    _, chosen = scenarios(data)
    cid, rows = ids(data), data[FLAGSHIP]
    L = ["You are helping evaluate an \"SLM Factory\": a system that picks an on-device small language model "
         "configuration for an app on a specific Android phone. Below are four phones, a pool of 104 configurations "
         "(9 models × 3 precisions × 2 runtimes × 2 backends) and 60 scenarios. Each scenario is an app use case on "
         "one phone, with a limit on four measured quantities. Your job, for every scenario, is to predict which "
         "configurations would meet all four limits on that phone, why the others would not, and which one to "
         "deploy.",
         "",
         "You have no benchmark measurements from these phones. Use what you know about the phones' chips, the "
         "runtimes (llama.cpp and MNN) on Android CPUs and OpenCL GPUs, and the models. Do not search the web for "
         "benchmark numbers. You may use code for arithmetic. Commit to an answer for every configuration: this is "
         "a test of how well the choice can be made without measuring on the phone.",
         "",
         "## How a configuration is measured",
         "",
         "Each configuration was benchmarked on each phone with a 256-token prompt followed by a 256-token answer:",
         "",
         "- **Peak memory (MB):** the peak memory footprint of the inference process during that run, CPU memory "
         "plus GPU allocations (PSS).",
         "- **Prefill (tokens/s):** prompt-processing speed for the 256-token prompt.",
         "- **Decode (tokens/s):** generation speed over the 256-token answer (after the 256-token prompt).",
         "- **Energy (millijoules per token, prefill and decode separately):** whole-phone battery energy net of the "
         "phone's idle draw, screen off. A scenario's **energy per request** is prompt tokens × prefill mJ/token + "
         "answer tokens × decode mJ/token, in joules.",
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
    L += ["", "## The filter and the pick", "",
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
          "scenario's prompt and answer lengths).", "",
          "| Scenario | Phone | Use case | Prompt / answer tokens | Peak memory ≤ (MB) | Prefill ≥ (tokens/s) | "
          "Decode ≥ (tokens/s) | Energy ≤ (J per request) |",
          "|---|---|---|--:|--:|--:|--:|--:|"]
    mapping = {}
    for sid, s in shuffled(chosen):
        uc, lim = s["uc"], s["limits"]
        mapping[sid] = s["id"]
        L.append(f"| {sid} | {s['phone']} | {uc.name}: {uc.what} | {uc.prompt:,} / {uc.answer:,} | "
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
    PROMPT.write_text("\n".join(L) + "\n", encoding="utf-8")
    ID_MAP.write_text(json.dumps(mapping, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {PROMPT} ({len(PROMPT.read_text(encoding='utf-8')):,} characters) and {ID_MAP}")


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


def score(path: str):
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
            works, why, cap, pr = False, "no pick" if not pick else "not in pool", "-", None
        rows.append(dict(sid=sid, id=s["id"], tier=s["tier"], phone=s["phone"], uc=s["uc"].name,
                         n_true=len(true_kept), n_said=len(said_kept), inter=len(inter),
                         precision=len(inter) / len(said_kept) if said_kept else None,
                         recall=len(inter) / len(true_kept) if true_kept else None,
                         jaccard=len(inter) / len(true_kept | said_kept) if true_kept | said_kept else 1.0,
                         unlisted=len(unlisted), pick=pick, pick_name=_name(back.get(pick)) if pick else "NONE",
                         best_name=_name(back[best_id]), works=works, why=why, cap=cap))
    _report(rows, field, cant_conf, missing, path)


def _name(k) -> str:
    if not k:
        return "?"
    m, fw, be, q = k
    return f"{LABEL[m]} {q} {FW_LABEL[fw]} {be.upper()}"


def _pct(n, d) -> str:
    return f"{n / d:.0%}" if d else "-"


def _report(rows, field, cant_conf, missing, path):
    n = len(rows)
    works = sum(r["works"] for r in rows)
    same = sum(r["cap"] == "same" for r in rows)
    same_mq = sum(r["cap"] in ("same", "same model and precision") for r in rows)
    over = sum(r["works"] is False and r["cap"] == "more capable" for r in rows)
    L = ["# Does measuring help? Claude without measurements vs the harness", "",
         f"Generated {datetime.now():%Y-%m-%dT%H:%M} by `Benchmark-Harness/paper_table5/baseline.py` from "
         f"`{path}` and the measured filter in [SCENARIOS.md](SCENARIOS.md).", "",
         "A Claude chat was given what the SLM Factory knows *without* this harness (phone specs, model cards, "
         "runtime and backend; prompt: `Benchmark-Harness/paper_table5/baseline/baseline_prompt.md`) and asked to "
         "filter the pool and pick a configuration for the same 60 scenarios. Its answers are scored against the "
         "measured values. With measurements the pick meets every limit by construction.", ""]
    if missing:
        L += [f"**Missing from the response:** {', '.join(missing)} (not scored).", ""]
    L += ["## Headline", "",
          "| | Claude without measurements | With the harness |", "|---|--:|--:|",
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
          "| ID | Tier | Phone | Use case | Claude's pick | Works? | Measured best | Kept (measured / Claude / both) | "
          "Jaccard |", "|---|---|---|---|---|---|---|--:|--:|"]
    for r in sorted(rows, key=lambda r: r["id"]):
        ok = "yes" if r["works"] else f"**no** ({r['why']})"
        L.append(f"| [{r['id']}](SCENARIOS.md#{r['id'].lower()}) | {r['tier']} | {r['phone']} | {r['uc']} | "
                 f"{r['pick_name']} | {ok} | {r['best_name']} | {r['n_true']} / {r['n_said']} / {r['inter']} | "
                 f"{r['jaccard']:.2f} |" + (f" {r['unlisted']} IDs unlisted" if r["unlisted"] else ""))
    L += ["", "Works? lists the limits the pick really misses (M memory, P prefill, D decode, E energy). Scenario "
          "IDs here are the SCENARIOS.md IDs; the prompt used shuffled IDs S01-S60 (`baseline/scenario_ids.json`)."]
    REPORT.write_text("\n".join(x for x in L if x is not None) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}: pick works {works}/{n}, same pick {same}/{n}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "prompt":
        write_prompt()
    elif len(sys.argv) == 3 and sys.argv[1] == "score":
        score(sys.argv[2])
    else:
        sys.exit(__doc__)
