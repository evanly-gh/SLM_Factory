"""Per-phone configuration table for requirements_md.py: the Galaxy S23 from its result JSONs, the other phones from
docs/all_results.md (their raw results live elsewhere)."""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent.parent / "docs"
PARAMS_B = {"smollm2-135m": 0.135, "gemma3-270m": 0.268, "smollm2-360m": 0.362, "qwen3-0.6b": 0.596,
            "qwen3.5-0.8b": 0.773, "qwen3-1.7b": 1.72, "qwen3.5-2b": 1.9, "qwen3-4b-instruct-2507": 4.02,
            "qwen3.5-4b": 4.2}
NAME = {"SmolLM2-135M": "smollm2-135m", "Gemma3-270M": "gemma3-270m", "SmolLM2-360M": "smollm2-360m",
        "Qwen3-0.6B": "qwen3-0.6b", "Qwen3.5-0.8B": "qwen3.5-0.8b", "Qwen3-1.7B": "qwen3-1.7b",
        "Qwen3.5-2B": "qwen3.5-2b", "Qwen3-4B-Instruct-2507": "qwen3-4b-instruct-2507", "Qwen3.5-4B": "qwen3.5-4b"}
BITS = {"Q4_0": 4, "Q4": 4, "Q8_0": 8, "Q8": 8, "F16": 16}
PHONES = {"Galaxy S23": "Snapdragon 8 Gen 2, 8 GB", "OnePlus 15": "Snapdragon 8 Elite Gen 5, 16 GB",
          "OnePlus 8 Pro": "Snapdragon 865, 12 GB", "Redmi 9A": "Helio G25, 4 GB, CPU only"}


def num(cell: str):
    m = re.match(r"\s*([\d.]+)", cell.replace(",", ""))
    return float(m.group(1)) if m else None


def key(model, fw, be, quant):
    return (model, {"llama.cpp": "llama.cpp", "MNN": "mnn", "mnn": "mnn"}[fw], be.lower(), quant)


def load_all_results(path: Path = DOCS / "all_results.md") -> dict:
    phones, cur = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## ") and not line.startswith("## Setup"):
            cur = phones.setdefault(line[3:].strip(), {})
        elif cur is not None and line.startswith("| ") and not line.startswith("| Model") and not line.startswith("|---"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            model, build, fw, be, prec, size, pre, dec, ep, ed, status = c
            ok = status.startswith("ok")
            cur[key(NAME[model], fw, be, prec)] = dict(
                ok=ok, status=status, size_gb=(num(size) or 0) * 2**30 / 1e9 if num(size) else None,
                pre=num(pre) if ok else None, dec=num(dec) if ok else None,
                e_pre=num(ep) if ok else None, e_dec=num(ed) if ok else None, mem_mb=None)  # energy in mJ/token
    return phones


def load_s23() -> dict:
    out = {}
    for d in ("slm_pool_s23_pinned", "slm_pool_s23_head"):
        for f in (HERE / "results" / d / "configs").glob("*.json"):
            r = json.loads(f.read_text())
            c = r["config"]
            mf = HERE / "results" / d / "memory" / f.name
            m = json.loads(mf.read_text()) if mf.exists() else {}
            ok = r.get("status") == "ok"
            e = lambda ph: (r[ph]["uj_per_token"] / 1e3 if ok and r[ph].get("energy_valid") and r[ph].get("uj_per_token")
                            else None)
            out[(c["model"], c["framework"], c["backend"], c["quant"])] = dict(
                ok=ok, status="ok" if ok else (r.get("error") or "failed")[:60],
                size_gb=sum(r["model_file"]["files"].values()) / 1e9,
                pre=r["prefill"]["tps"] if ok else None, dec=r["decode"]["tps"] if ok else None,
                e_pre=e("prefill"), e_dec=e("decode"),
                mem_mb=m.get("peak_total_mib") * 1.048576 if m.get("exit") == 0 and m.get("peak_total_mib") else None)
    return out


def load() -> dict:
    """{phone: {(model, framework, backend, quant): row}}; rows carry params_b and bits too."""
    phones = {"Galaxy S23": load_s23(), **load_all_results()}
    sizes = {k: r["size_gb"] for p in phones.values() for k, r in p.items() if r.get("size_gb")}
    for p in phones.values():
        for k, r in p.items():
            r.update(model=k[0], fw=k[1], be=k[2], quant=k[3], params_b=PARAMS_B[k[0]], bits=BITS[k[3]])
            r["size_gb"] = r.get("size_gb") or sizes.get(k)
            r["ttft_s"] = 256 / r["pre"] if r.get("pre") else None
            r["req_j"] = 256 * (r["e_pre"] + r["e_dec"]) / 1e3 if r.get("e_pre") and r.get("e_dec") else None
    return phones


if __name__ == "__main__":
    for p, rows in load().items():
        print(p, len(rows), "configs,", sum(r["ok"] for r in rows.values()), "ok,",
              sum(1 for r in rows.values() if r.get("req_j")), "with request energy")
