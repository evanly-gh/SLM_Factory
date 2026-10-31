"""Per-phone configuration table for requirements_md.py, from the same merged data as docs/RESULTS.md
(results_all_md.py)."""
from __future__ import annotations

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


def load_phones() -> dict:
    """{phone label: {(model, framework, backend, quant): row}} from results_all_md.py's merged data (the same values
    as docs/RESULTS.md: raw result JSONs, the OnePlus 8 Pro / Redmi 9A reruns, the OnePlus 15 supplement)."""
    import results_all_md as ram
    out = {}
    for phone in ram.PHONES:
        d = ram.load_phone(phone)
        rows = {}
        for k in d["cands"]:
            m, be, fw, q = k
            r = d["pool"].get(k)
            ok = bool(r) and r.get("status") == "ok"
            e = lambda ph: ram.val(r, ph, "uj") / 1e3 if ok and ram.val(r, ph, "uj") else None
            rows[(m, fw, be, q)] = dict(
                ok=ok, status="ok" if ok else ((r or {}).get("error") or "not run")[:60],
                size_gb=(ram.storage(k) or 0) / 1e3 or None,
                pre=ram.val(r, "prefill", "tps") if ok else None, dec=ram.val(r, "decode", "tps") if ok else None,
                e_pre=e("prefill"), e_dec=e("decode"),
                mem_mb=ram.peak_mb(d["memory"].get(k)) if ok else None)
        out[phone["label"]] = rows
    return out


def load() -> dict:
    """{phone: {(model, framework, backend, quant): row}}; rows carry params_b and bits too."""
    phones = load_phones()
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
