"""How often does a blind pick of (model, precision, engine, backend) meet a user requirement, on each phone?

Inputs : results/all_results.md (OnePlus 15, OnePlus 8 Pro, Redmi 9A) and ~/Downloads/RESULTS.md (Galaxy S23, section 3).
Output : results/requirements_difficulty.md  (+ requirements_data.json with every per-config verdict)
Usage  : python make_requirements.py [path/to/S23/RESULTS.md]
"""
import itertools
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
S23_MD = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/RESULTS.md")

PHONES = ["OnePlus 15", "OnePlus 8 Pro", "Galaxy S23", "Redmi 9A"]
SHORT = {"OnePlus 15": "OP15", "OnePlus 8 Pro": "8 Pro", "Galaxy S23": "S23", "Redmi 9A": "Redmi"}
RAM_GIB = {"OnePlus 15": 14.76, "OnePlus 8 Pro": 11.24, "Galaxy S23": 6.91, "Redmi 9A": 3.72}

# model -> (display name, billions of parameters used as a capability proxy)
MODELS = {"SmolLM2-135M": 0.135, "Gemma3-270M": 0.27, "SmolLM2-360M": 0.36, "Qwen3-0.6B": 0.6, "Qwen3.5-0.8B": 0.8,
          "Qwen3-1.7B": 1.7, "Qwen3.5-2B": 2.0, "Qwen3-4B": 4.0, "Qwen3.5-4B": 4.0}
ALIASES = {"Gemma-3-270M": "Gemma3-270M", "Qwen3-4B-Instruct-2507": "Qwen3-4B"}
PREC = {"Q4_0": 4, "Q4": 4, "Q8_0": 8, "Q8": 8, "F16": 16}


def num(s):
    s = s.strip().replace(",", "")
    if s in ("", "—", "–", "-"):
        return None
    m = re.match(r"^([\d.]+(?:e[+-]?\d+)?)", s)
    return float(m.group(1)) if m else None


def key(model, fw, be, bits):
    return (model, "llama.cpp" if fw.lower().startswith("llama") else "MNN", be.upper(), bits)


def parse_ours():
    """phone -> {key: dict(prefill, decode, de_uj, size_gib, ok)} from all_results.md"""
    out = {p: {} for p in PHONES}
    phone = None
    for line in open(os.path.join(RES, "all_results.md")):
        if line.startswith("## "):
            phone = line[3:].strip()
            continue
        if phone in out and line.startswith("| ") and "---" not in line and not line.startswith("| Model"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            model = ALIASES.get(c[0], c[0])
            if model not in MODELS:
                continue
            ok = c[10].startswith("ok")
            out[phone][key(model, c[2], c[3], PREC[c[4]])] = dict(
                prefill=num(c[6]) if ok else None, decode=num(c[7]) if ok else None,
                de_uj=num(c[9]) * 1000 if ok and num(c[9]) else None, size_gib=num(c[5]), ok=ok)
    return out


def parse_s23():
    """Section 3 of the S23 RESULTS.md: model, backend, framework, quant, storage MB, peak MB, prefill, prefill uJ, decode, decode uJ."""
    out, model, in_sec = {}, None, False
    for line in open(S23_MD):
        if line.startswith("## "):
            in_sec = line.startswith("## 3.")
            continue
        if not in_sec or not line.startswith("| ") or "---" in line or line.startswith("| Model"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if c[0]:
            model = ALIASES.get(c[0], c[0])
        if model not in MODELS:
            continue
        ok = "FAIL" not in c[6]
        out[key(model, c[2], c[1], PREC[c[3]])] = dict(
            prefill=num(c[6]) if ok else None, decode=num(c[8]) if ok else None,
            de_uj=num(c[9]) if ok else None, size_gib=num(c[4]) * 1e6 / 2**30 if num(c[4]) else None, ok=ok)
    return out


# --------------------------------------------------------------------------------------- requirements
def has_energy(phone):
    return phone != "OnePlus 8 Pro"


REQS = [  # id, plain-language requirement, predicate(config, row, phone) -> True / False / None (cannot be judged)
    ("R1", "Readable streaming: decode >= 5 tokens/s", lambda k, r, p: r["ok"] and r["decode"] >= 5),
    ("R2", "Responsive start: a 256-token prompt starts answering within 2 s (prefill >= 128 t/s)",
     lambda k, r, p: r["ok"] and r["prefill"] >= 128),
    ("R3", "Light on storage: weights <= 1 GiB", lambda k, r, p: (r["size_gib"] or 99) <= 1.0),
    ("R4", "Comfortable chat: decode >= 10 tokens/s", lambda k, r, p: r["ok"] and r["decode"] >= 10),
    ("R5", "Interactive assistant: decode >= 10 t/s AND prefill >= 128 t/s",
     lambda k, r, p: r["ok"] and r["decode"] >= 10 and r["prefill"] >= 128),
    ("R6", "Frugal: decode energy <= 200 mJ/token",
     lambda k, r, p: None if not has_energy(p) else bool(r["ok"] and r["de_uj"] and r["de_uj"] <= 200_000)),
    ("R7", "Smooth streaming: decode >= 20 tokens/s", lambda k, r, p: r["ok"] and r["decode"] >= 20),
    ("R8", "Capable model (>= 1.5B parameters) at decode >= 10 tokens/s",
     lambda k, r, p: r["ok"] and MODELS[k[0]] >= 1.5 and r["decode"] >= 10),
    ("R9", "Near-instant start: prefill >= 512 t/s (256 tokens in <= 0.5 s)", lambda k, r, p: r["ok"] and r["prefill"] >= 512),
    ("R10", "Capable (>= 1.5B) AND interactive: decode >= 10 t/s AND prefill >= 128 t/s",
     lambda k, r, p: r["ok"] and MODELS[k[0]] >= 1.5 and r["decode"] >= 10 and r["prefill"] >= 128),
    ("R11", "Capable (>= 1.5B), interactive AND frugal: decode >= 10 t/s, prefill >= 128 t/s, decode energy <= 300 mJ/token",
     lambda k, r, p: None if not has_energy(p) else bool(r["ok"] and MODELS[k[0]] >= 1.5 and r["decode"] >= 10
                                                         and r["prefill"] >= 128 and r["de_uj"] and r["de_uj"] <= 300_000)),
    ("R13", "Just works: finishes and decodes at >= 2 tokens/s", lambda k, r, p: r["ok"] and r["decode"] >= 2),
    ("R14", "Fits a 4 GiB phone: weights <= 2 GiB", lambda k, r, p: (r["size_gib"] or 99) <= 2.0),
    ("R15", "Background summarizer: prefill >= 30 t/s (256-token prompt in about 9 s or less) AND decode >= 3 t/s",
     lambda k, r, p: r["ok"] and r["prefill"] >= 30 and r["decode"] >= 3),
    ("R12", "Capable (>= 1.5B) at decode >= 20 t/s AND weights <= 2.5 GiB",
     lambda k, r, p: r["ok"] and MODELS[k[0]] >= 1.5 and r["decode"] >= 20 and (r["size_gib"] or 99) <= 2.5),
]


def level(rate):
    return "Easy" if rate >= 0.60 else "Medium" if rate >= 0.30 else "Hard"


def main():
    ours, s23 = parse_ours(), parse_s23()
    data = {**ours, "Galaxy S23": s23}
    # candidate set = the pool configurations every GPU phone has (9 models x 3 precisions x 2 engines x 2 backends,
    # minus Qwen3.5-4B F16 which was never prepared for ours)
    cands = sorted(k for k in s23 if not (k[0] == "Qwen3.5-4B" and k[3] == 16))
    missing = {p: [k for k in cands if k not in data[p]] for p in PHONES}
    for p in PHONES:  # configs a phone cannot even try (Redmi has no OpenCL) count as not satisfied
        for k in missing[p]:
            data[p][k] = dict(prefill=None, decode=None, de_uj=None, size_gib=None, ok=False)
        for k in cands:  # weights size is a property of the model file, identical everywhere
            if data[p][k]["size_gib"] is None:
                data[p][k]["size_gib"] = s23[k]["size_gib"]
    rates, verdicts, rows = {}, {}, []
    for rid, text, pred in REQS:
        for p in PHONES:
            v = {}
            for k in cands:
                r = data[p][k]
                try:
                    v[k] = pred(k, r, p)
                except TypeError:
                    v[k] = False
            verdicts[(rid, p)] = v
            valid = [x for x in v.values() if x is not None]
            rates[(rid, p)] = (sum(bool(x) for x in valid) / len(valid)) if valid else None
    # cross-phone transfer: of the configs that pass on phone A, how many also pass on B (precision of A's profile on B)
    def transfer(rid):
        vals = []
        for a, b in itertools.permutations(PHONES, 2):
            va, vb = verdicts[(rid, a)], verdicts[(rid, b)]
            if any(x is None for x in va.values()) or any(x is None for x in vb.values()):
                continue
            pa = [k for k in cands if va[k]]
            if pa:
                vals.append(sum(bool(vb[k]) for k in pa) / len(pa))
        return sum(vals) / len(vals) if vals else None
    summary = []
    for rid, text, _ in REQS:
        rs = [rates[(rid, p)] for p in PHONES if rates[(rid, p)] is not None]
        pooled = sum(rs) / len(rs)
        summary.append((rid, text, pooled, transfer(rid)))
    json.dump({f"{rid}|{p}": {"rate": rates[(rid, p)], "pass": [list(k) for k, x in verdicts[(rid, p)].items() if x]}
               for (rid, p) in rates}, open(os.path.join(RES, "requirements_data.json"), "w"), indent=1)

    L = []
    L.append("# Why every phone needs its own profiling run: requirement difficulty without measurements\n")
    L.append("Generated by `results/graphs/make_requirements.py` from `results/all_results.md` (OnePlus 15, OnePlus 8 Pro, Redmi 9A) "
             "and the Galaxy S23 `RESULTS.md` (section 3). Per-configuration verdicts are in `results/requirements_data.json`.\n")
    L.append("## What we are trying to do\n")
    L.append("A phone app (or an agent) is asked to run a small language model under a **user-specified requirement**: how fast it must answer, "
             "how quickly it must start, how much storage or energy it may use, how capable the model must be. Something must then pick one "
             "configuration out of **104 choices per phone: 9 models x 3 precisions (4/8/16-bit) x 2 engines (llama.cpp, MNN) x 2 backends (CPU, GPU)**.\n")
    L.append("Without measurements the picker can only guess from rules of thumb (smaller is faster, 4-bit is smaller, a GPU is faster). "
             "Our four phones show that those rules fail in phone-specific ways:\n")
    L.append("- The same configuration can pass on one phone and fail on another: an 8 GB phone cannot load 8-bit 4B models that a 14.8 GiB phone runs.\n"
             "- GPU is not reliably faster: llama.cpp on the OnePlus 8 Pro's Adreno 650 prefills at 0.5-115 tokens/s, slower than its own CPU in all 16 measured cells; MNN GPU decode is slow on the "
             "Galaxy S23 and on the 8 Pro, but fast on the OnePlus 15; MNN GPU 16-bit crashes on every phone that has a GPU backend; the Redmi 9A has no OpenCL at all.\n"
             "- 'Smaller or lower-bit is faster' breaks: llama.cpp Q4_0 files keep a large Q6_K output matrix and the unoptimized OpenCL Q6_K kernel makes some Q4_0 models decode "
             "slower than Q8_0 on the GPU; on the Galaxy S23 llama.cpp GPU prefill is 13-53x slower for Q8_0 than for Q4_0 (on the OnePlus 15 the same gap is only 2-4x).\n"
             "- A profile does not transfer: the set of configurations that meet a requirement on one phone is only partly the set that meets it on another (last column below).\n")
    L.append("So we run the whole grid **once per phone** and keep the result as a lookup table. The model (or app) then answers 'which configuration meets this "
             "requirement on this phone?' from data instead of guessing. The tables below measure how much that profiling run is worth.\n")
    L.append("## How difficulty is defined\n")
    L.append("**Blind success rate** = the share of the 104 candidate configurations that satisfy the requirement on a given phone, i.e. the chance that picking "
             "one at random without any measurement works. A configuration that crashes, does not fit in memory, or cannot run on that phone counts as a failure "
             "(for the user it is one). The levels follow your cut-offs:\n")
    L.append("| Level | Blind success rate | Meaning |\n|---|---|---|\n"
             "| Easy | 60-100% | most configurations pass; a priori factors (model size, precision) are nearly enough |\n"
             "| Medium | 30-60% | about a coin-flip; you must know the phone class and the engine quirks |\n"
             "| Hard | 0-30% | few configurations pass; without this phone's measurements you almost never guess one |\n")
    L.append("The level of a requirement in the summary is the average of its four per-phone rates (energy requirements average the three phones that have energy). "
             "'Profile transfer' is the average chance that a configuration which passes on one phone also passes on another phone (averaged over all ordered phone pairs): "
             "low numbers mean one phone's profile cannot be reused on a different phone.\n")
    L.append("Candidate set and thresholds: the 104 configurations every GPU phone has (Qwen3.5-4B F16 was never prepared, so it is excluded everywhere). "
             "Decode and prefill thresholds are for a 256-token prompt and 256-token decode. 'Capable' means at least 1.5B parameters (Qwen3-1.7B, Qwen3.5-2B, Qwen3-4B, Qwen3.5-4B), "
             "used as a quality proxy because accuracy was not measured. Energy is whole-phone, net of idle (the OnePlus 8 Pro has no energy data, so energy requirements are not scored on it). "
             "Storage is the weight-file size.\n")
    L.append("## Summary: requirements ranked from easy to hard\n")
    L.append("| ID | Requirement | Blind success (avg of phones) | Level | Profile transfer |\n|---|---|--:|---|--:|")
    for rid, text, pooled, tr in sorted(summary, key=lambda t: -t[2]):
        L.append(f"| {rid} | {text} | {pooled*100:.0f}% | **{level(pooled)}** | {('%.0f%%' % (tr*100)) if tr is not None else '—'} |")
    L.append("")
    L.append("## Per-phone blind success rates\n")
    L.append("The same requirement is a different level on different phones. Cells show the percentage of the 104 configurations that pass; the letter is the level on that phone "
             "(E = easy, M = medium, H = hard).\n")
    L.append("| ID | Requirement | " + " | ".join(PHONES) + " |\n|---|---|" + "---:|" * len(PHONES))
    for rid, text, _ in REQS:
        cells = []
        for p in PHONES:
            r = rates[(rid, p)]
            cells.append("n/a (no energy data)" if r is None else f"{r*100:.0f}% ({level(r)[0]})")
        L.append(f"| {rid} | {text} | " + " | ".join(cells) + " |")
    L.append("")
    changes = [rid for rid, _, _ in REQS if len({level(rates[(rid, p)]) for p in PHONES if rates[(rid, p)] is not None}) > 1]
    L.append(f"**{len(changes)} of {len(REQS)} requirements change difficulty level depending on the phone** ({', '.join(changes)}).\n")
    # per-requirement what passes
    L.append("## Which configurations pass, by phone\n")
    L.append("For each requirement: how many of the 104 configurations pass on each phone, broken down by backend, engine and precision. "
             "This is the lookup a profiling run provides.\n")
    for rid, text, _ in REQS:
        L.append(f"**{rid}. {text}**\n")
        L.append("| Phone | Pass | Pass by backend (CPU / GPU) | Pass by engine (llama.cpp / MNN) | Pass by precision (4 / 8 / 16-bit) | Models with a passing config |\n|---|--:|---|---|---|---|")
        for p in PHONES:
            v = verdicts[(rid, p)]
            if rates[(rid, p)] is None:
                L.append(f"| {p} | n/a | n/a | n/a | n/a | no energy data |")
                continue
            ps = [k for k in cands if v[k]]
            by = lambda f, vals: " / ".join(str(sum(1 for k in ps if f(k) == x)) for x in vals)
            n_models = len({k[0] for k in ps})
            L.append(f"| {p} | {len(ps)} | {by(lambda k: k[2], ['CPU', 'GPU'])} | {by(lambda k: k[1], ['llama.cpp', 'MNN'])} | "
                     f"{by(lambda k: k[3], [4, 8, 16])} | {n_models} of 9 |")
        L.append("")
    L.append("## What the numbers say\n")
    easy = [s for s in summary if level(s[2]) == 'Easy']
    med = [s for s in summary if level(s[2]) == 'Medium']
    hard = [s for s in summary if level(s[2]) == 'Hard']
    L.append(f"- **Easy ({len(easy)}):** " + "; ".join(f"{s[0]} ({s[2]*100:.0f}%)" for s in easy) + ". A blind pick usually works, but only because the thresholds are low.\n"
             f"- **Medium ({len(med)}):** " + "; ".join(f"{s[0]} ({s[2]*100:.0f}%)" for s in med) + ". Whether the pick works depends on the phone.\n"
             f"- **Hard ({len(hard)}):** " + "; ".join(f"{s[0]} ({s[2]*100:.0f}%)" for s in hard) + ". Combined requirements (capable model AND fast AND frugal) leave only a handful of passing configurations per phone, "
             "and they are not the same handful on each phone.\n")
    L.append("- The cheap rules of thumb are not enough on their own: the passing set on every phone mixes engines, backends and precisions, so no single 'always use X' policy passes everywhere.\n")
    L.append("## Caveats\n")
    cpu_ks = [k for k in cands if k[2] == "CPU"]
    redmi_cpu = ", ".join(f"{rid} {100*sum(bool(verdicts[(rid, 'Redmi 9A')][k]) for k in cpu_ks)/len(cpu_ks):.0f}%"
                          for rid, _, _ in REQS if rates[(rid, 'Redmi 9A')] is not None)
    L.append("- The Redmi 9A has no OpenCL, so its 52 GPU configurations count as failures; that makes it look harder than a CPU-only comparison would. "
             f"Counting only its 52 CPU configurations the blind success rates are: {redmi_cpu}.\n"
             "- Accuracy (answer quality) was never measured, so 'capable' is a parameter-count proxy, not a quality result.\n"
             "- OnePlus 15 / 8 Pro / Redmi energy and the Galaxy S23 energy come from different instruments (Perfetto battery gauge for OnePlus 15 and S23, sysfs battery current for the Redmi), "
             "all whole-phone net of idle. The 8 Pro has none.\n"
             "- A small number of OnePlus 15 cells are suspected slow-governor runs (for example Qwen3-0.6B llama.cpp 8-bit CPU and Qwen3.5-4B llama.cpp Q4_0 CPU), and the S23 file lists its own cells to treat with caution; "
             "I used every value as reported.\n"
             "- Thresholds are my realistic choices; change them in `REQS` in `make_requirements.py` and rerun.\n")
    path = os.path.join(RES, "requirements_difficulty.md")
    open(path, "w").write("\n".join(L) + "\n")
    print("wrote", path)
    for rid, text, pooled, tr in sorted(summary, key=lambda t: -t[2]):
        print(f"{rid:4s} {pooled*100:5.1f}% {level(pooled):6s} transfer {tr if tr is None else round(tr*100)}  "
              + " ".join(f"{SHORT[p]}={'n/a' if rates[(rid,p)] is None else round(rates[(rid,p)]*100)}" for p in PHONES))


if __name__ == "__main__":
    main()
