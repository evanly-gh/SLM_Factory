"""Write results/all_results.md: one row per model x precision x engine x backend x phone with every metric we have.

Reads results/slm_pool_<phone>_{pinned,head}/configs/*.json.   Usage: python make_table.py
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)

PHONES = [  # key, label, SoC / RAM note, result-dir stems
    ("oneplus15", "OnePlus 15", "Snapdragon 8 Elite Gen 5 (SM8850), 14.8 GiB RAM", ["oneplus15_pinned", "oneplus15_head"]),
    ("oneplus8pro", "OnePlus 8 Pro", "Snapdragon 865 (SM8250), 11.2 GiB RAM (RAM expansion off)", ["oneplus8pro_pinned", "oneplus8pro_head"]),
    ("redmi9a", "Redmi 9A", "Helio G25 (MT6762), 8x Cortex-A53, 32-bit Android 11, 3.7 GiB RAM, CPU only", ["redmi9a_pinned", "redmi9a_head"]),
]
MODELS = [  # key, label, params note
    ("smollm2-135m", "SmolLM2-135M"), ("gemma3-270m", "Gemma3-270M"), ("smollm2-360m", "SmolLM2-360M"),
    ("qwen3-0.6b", "Qwen3-0.6B"), ("qwen3.5-0.8b", "Qwen3.5-0.8B"), ("qwen3-1.7b", "Qwen3-1.7B"),
    ("qwen3.5-2b", "Qwen3.5-2B"), ("qwen3-4b-instruct-2507", "Qwen3-4B-Instruct-2507"), ("qwen3.5-4b", "Qwen3.5-4B"),
]
MODEL_ORDER = {m[0]: i for i, m in enumerate(MODELS)}
MODEL_LABEL = dict(MODELS)
FW_ORDER = {"llama.cpp": 0, "mnn": 1}
BITS_ORDER = {4: 0, 8: 1, 16: 2}
BACKEND_ORDER = {"cpu": 0, "gpu": 1}


def status_text(j):
    if j.get("status") == "ok":
        return "ok"
    r = str(j.get("error") or j.get("reason") or "")
    if "exit 139" in r or "Segmentation" in r:
        return "crash (segfault, exit 139)"
    m = re.search(r"does not fit in RAM: ([\d.]+) GiB of weights", r)
    if m:
        return f"not run: does not fit RAM ({m.group(1)} GiB est. vs 55% limit)"
    m = re.search(r"does not fit in RAM: (\d+) MB .*swapped", r)
    if m:
        return f"stopped: swap guard ({m.group(1)} MB swapped, limit 512)"
    if "ggml_opencl" in r:
        return "crash (OpenCL model load/allocation, exit 1)"
    if "failed to load model" in r:
        return "crash (llama-bench could not load model, exit 1)"
    if "oom/killed" in r:
        return "killed (process vanished, exit 137)"
    return r[:60] or str(j.get("status"))


def fmt(v, sd=None, nd=2):
    if v is None:
        return "—"
    return f"{v:.{nd}f}" + (f" ± {sd:.{nd}f}" if sd else "")


def load(stems):
    rows = []
    for stem in stems:
        for f in glob.glob(os.path.join(RES, f"slm_pool_{stem}", "configs", "*.json")):
            j = json.load(open(f))
            c, mf = j["config"], j.get("model_file") or {}
            size = sum(mf["files"].values()) / 2**30 if mf.get("files") else None
            row = dict(model=c["model"], fw=c["framework"], bits=c["bits"], quant=c["quant"], backend=c["backend"],
                       build=stem.rsplit("_", 1)[1], size=size, status=status_text(j), pre=None, dec=None,
                       pre_sd=None, dec_sd=None, pe=None, de=None, id=c["id"], stem=stem, mem=None, mem_gpu=None, mem_note="")
            if j.get("status") == "ok":
                p, d = j["prefill"], j["decode"]
                row.update(pre=p.get("tps"), pre_sd=p.get("tps_std"), dec=d.get("tps"), dec_sd=d.get("tps_std"))
                if p.get("energy_valid") and p.get("uj_per_token"):
                    row["pe"] = p["uj_per_token"] / 1000
                if d.get("energy_valid") and d.get("uj_per_token"):
                    row["de"] = d["uj_per_token"] / 1000
                if (p.get("uj_per_token") and not p.get("energy_valid")):
                    row["status"] = "ok (energy marked invalid by harness: " + str(p.get("energy_reason")) + ")"
            rows.append(row)
    return rows


# Peak memory exists for the OnePlus 15 only (memory_usage.py, patched to read the system-wide GPUTotalUsed counter; see the setup notes).
# Exit 0 = clean run; exit 139 = the benchmark segfaulted after finishing its workload (MNN F16 on the GPU), the peak is still the workload's.
# One cell uses its repeat measurement: the first reading of Gemma3-270M MNN Q4 CPU was 442 MiB, the repeat 272 MiB with identical resident
# memory, i.e. PSS attribution noise on memory-mapped pages (results/_memory_repeat_oneplus15/).
MEM_REPEAT_OVERRIDE = {"gemma3-270m__Q4__mnn__cpu": "pinned"}


def attach_memory(rows):
    for r in rows:
        f = os.path.join(RES, f"slm_pool_{r['stem']}", "memory", f"{r['id']}.json")
        if r["id"] in MEM_REPEAT_OVERRIDE and r["stem"].endswith(MEM_REPEAT_OVERRIDE[r["id"]]):
            g = os.path.join(RES, "_memory_repeat_oneplus15", MEM_REPEAT_OVERRIDE[r["id"]], "memory", f"{r['id']}.json")
            if os.path.exists(g):
                f = g
                r["mem_note"] += " ‡"
        if not os.path.exists(f):
            continue
        m = json.load(open(f))
        if m.get("exit") in (0, 139) and m.get("peak_total_mib"):
            r["mem"] = m["peak_total_mib"]
            r["mem_gpu"] = m.get("peak_gpu_mib") or None
            if m["exit"] == 139:
                r["mem_note"] += " †"


def main():
    out = []
    out.append("# All results: every model, quantization, engine, backend and phone\n")
    out.append("Generated by `results/graphs/make_table.py` from `results/slm_pool_<phone>_{pinned,head}/configs/*.json`.\n")
    out.append("## Setup (same on every phone)\n")
    out.append("- 256-token prompt (prefill) and 256-token decode, 4 llama.cpp threads, 3+ recorded repetitions; "
               "values are mean ± standard deviation across repetitions (tokens per second).\n"
               "- **Build** `pinned` = the paper's llama.cpp `eadc418` / MNN `510ac8f` (Qwen3, SmolLM2, Gemma3). "
               "`head` = llama.cpp `9c2e0e4` / MNN `a8ec497` (Qwen3.5 only; its architecture needs a newer build).\n"
               "- **Precision** columns use each engine's own label: llama.cpp Q4_0 / Q8_0 / F16, MNN 4 / 8 / 16-bit (shown as Q4 / Q8 / F16).\n"
               "- **Size (GiB)** is the on-disk weight size of the model file(s) pushed to the phone. The RAM guard refuses a run when estimated resident weights exceed 55% of the phone's RAM "
               "(MNN F16 is estimated at about 2x its file size).\n"
               "- **Peak memory (MiB)** is the largest total memory of the inference process during one run of the same workload (256-token prompt, 256 generated tokens): "
               "PSS from `dumpsys meminfo` plus the GPU buffers that PSS does not show, sampled every 0.5 s by `memory_usage.py`; **GPU (MiB)** is the GPU part of it. "
               "**Measured on the OnePlus 15 only; — on the other phones means not measured.** On the OnePlus 15 `dumpsys meminfo` reports almost no GPU memory, so the GPU part is the rise of the "
               "system-wide `GPUTotalUsed` counter (`/proc/meminfo`) above its pre-run baseline (a patch to `memory_usage.py`). "
               "† = the benchmark segfaulted after completing the workload (MNN F16 on the GPU), so the peak is still the workload's. "
               "‡ = repeat measurement used: the first reading of Gemma3-270M MNN Q4 CPU was 442 MiB, the repeat 272 MiB with the same resident memory (PSS attribution noise on memory-mapped pages); "
               "a repeat of 24 cells otherwise agreed within 10% (median 0.4%). Peak memory can exceed the phone's RAM because PSS also counts memory-mapped weight pages that are copied into GPU memory; "
               "the drop in the phone's free memory is saved in the raw records.\n"
               "- **Energy (mJ/token)** is whole-device battery energy net of idle, measured through Perfetto (OnePlus 15) or sysfs battery current (Redmi 9A). "
               "— means not measured or not valid. **The OnePlus 8 Pro has no energy data at all** (Perfetto v25 on Android 13 cannot read its battery counters and sysfs battery files are not readable without root). The paper's energy is SoC-only, so it is not directly comparable.\n"
               "- **Status**: `ok` = measured. Anything else says why there is no number.\n")
    for key, label, note, stems in PHONES:
        rows = load(stems)
        if key == "oneplus8pro":  # head llama.cpp OpenCL build cannot even load here; add the cells that were never run
            for m in ("qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b"):
                for q, bits in (("Q4_0", 4), ("Q8_0", 8), ("F16", 16)):
                    if m == "qwen3.5-4b" and q == "F16":
                        continue
                    rows.append(dict(model=m, fw="llama.cpp", bits=bits, quant=q, backend="gpu", build="head", size=None,
                                     status="not run: head llama.cpp OpenCL build needs OpenCL symbols this phone's driver lacks",
                                     pre=None, dec=None, pre_sd=None, dec_sd=None, pe=None, de=None))
        rows = [r for r in rows if r["model"] in MODEL_ORDER]
        if key == "oneplus15":
            attach_memory(rows)
        if key != "oneplus8pro":  # 8 Pro never had energy; for the others flag measured cells that lack a valid reading
            for r in rows:
                if r["status"] == "ok" and r["pe"] is None and r["de"] is None:
                    r["status"] = "ok (no energy reading for this cell)"
        rows.sort(key=lambda r: (MODEL_ORDER[r["model"]], FW_ORDER[r["fw"]], BACKEND_ORDER[r["backend"]], BITS_ORDER[r["bits"]]))
        ok = sum(r["status"].startswith("ok") for r in rows)
        out.append(f"\n## {label}\n")
        out.append(f"{note}. {len(rows)} configurations, {ok} measured.\n")
        out.append("| Model | Build | Engine | Backend | Precision | Size (GiB) | Peak memory (MiB) | GPU (MiB) | Prefill (t/s) | Decode (t/s) | Prefill energy (mJ/tok) | Decode energy (mJ/tok) | Status |")
        out.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in rows:
            out.append("| " + " | ".join([
                MODEL_LABEL[r["model"]], r["build"], "llama.cpp" if r["fw"] == "llama.cpp" else "MNN", r["backend"].upper(), r["quant"],
                fmt(r["size"], nd=2) if r["size"] else "—",
                (f"{r['mem']:,.0f}{r['mem_note']}" if r.get("mem") else "—"), (f"{r['mem_gpu']:,.0f}" if r.get("mem_gpu") else "—"),
                fmt(r["pre"], r["pre_sd"], 2 if (r["pre"] or 99) < 10 else 1), fmt(r["dec"], r["dec_sd"], 2),
                fmt(r["pe"], nd=1), fmt(r["de"], nd=1), r["status"]]) + " |")
    path = os.path.join(RES, "all_results.md")
    open(path, "w").write("\n".join(out) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
