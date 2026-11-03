#!/usr/bin/env python3
"""
End-to-end test of filter_pool.py's on-phone filter (FilterSession + the output contract) against the simulated phone
of fake_phone_test.py. Peak memory comes from a stub (the simulated phone has no memory sampler); everything else is
the real session: pool order, removal at the memory check without benchmarking, "does not fit", GPU paths that
cannot run, the prefill / decode / energy limits, and the per model x quantization choice.

  python tests/fake_filter_test.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))
sys.path.insert(0, str(HERE.parent))

import fake_phone_test as fake  # noqa: E402
import filter_pool as fp  # noqa: E402
import memory_usage  # noqa: E402
import run_paper_table5 as rpt  # noqa: E402

PEAK_MIB = {"tiny": 300, "mid": 900}  # stub peak memory per model


def fake_memory(adb, cfg, model_path, ref, threads, timeout_s):
    fake_memory.calls.append(cfg["id"])
    return {"exit": 0, "peak_total_mib": PEAK_MIB[cfg["model"]], "stderr_tail": None}


fake_memory.calls = []


def main():
    tmp = Path(tempfile.mkdtemp(prefix="pb_filter_"))
    models = tmp / "models"
    entries, sizes = [], {}
    # pool order is mid, tiny, huge: the opposite of alphabetical, so ordering is checked
    for name, size in (("mid", 123), ("tiny", 123), ("huge", 6 * 2**30)):
        for fw, quant, bits in (("llama.cpp", "Q4_0", 4), ("mnn", "Q4", 4), ("llama.cpp", "Q8_0", 8)):
            fname = f"{name}-{quant.lower()}" + (".gguf" if fw == "llama.cpp" else "-config.json")
            entries.append({"name": name, "framework": fw, "quant": quant, "bits": bits, "ref": "pinned",
                            "path": f"x/{fname}", "files": {fname: size}})
            sizes[fname] = size
    models.mkdir(parents=True)
    (models / "manifest.json").write_text(json.dumps({"models": entries}))
    fake.FakeAdb.phone = fake.FakePhone(sizes)
    rpt.Adb = fake.FakeAdb
    rpt.MODELS_DIR = models
    rpt.POLL_S = 0.05
    memory_usage.run_one = fake_memory

    pool = {"mid": ("org/mid", "pinned"), "tiny": ("org/tiny", "pinned"), "huge": ("org/huge", "pinned")}
    order = {n: i for i, n in enumerate(pool)}
    lim = {"max_peak_memory_mb": 500, "min_prefill_tps": 100, "min_decode_tps": 20, "max_energy_j": 50}
    out = tmp / "results" / "filter_fake_pinned"
    args = fp.session_args("FAKE123", "pinned", list(pool), ["Q4_0", "Q4", "Q8_0"], out, "auto", True,
                           ["--idle-seconds", "1", "--pre-idle-seconds", "0.5", "--gate-timeout", "60",
                            "--min-battery", "0"])
    no_gpu = {"llama.cpp"}  # as if the phone's llama.cpp OpenCL build found no device
    sess = fp.FilterSession(args, lim, order, no_gpu)
    planned = [c["id"] for c in sess.configs()]
    sess.run()

    results = {f.stem: json.loads(f.read_text()) for f in (out / "configs").glob("*.json")}
    # the pool asks for MNN 8-bit too: tiny's conversion "failed", mid's and huge's were never made (not measured)
    expected = fp.expected_matrix(pool, ["Q4_0", "Q8_0"], [4, 8])
    gpu = {"pinned": {"llama.cpp": "llama.cpp pinned OpenCL build finds no device", "mnn": None}}
    conversion = {"tiny": {"mnn": {"Q8": "conversion to MNN failed (prepare_models.py exit 1): boom"}}}
    phone = fp._phone_summary({"serial": "FAKE123", "manufacturer": "samsung", "model": "SM-S911U"},
                              {"label": "Fake phone", "id": "fake"})
    c = fp.build_contract(phone, [out], gpu, lim, {"prompt_tokens": 256, "answer_tokens": 256}, "decode", pool,
                          expected, fp.not_tried_reasons(pool, expected, gpu, conversion), conversion)
    fp.print_table(c)
    failures = []

    def check(cond, msg):
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            failures.append(msg)

    check([p.split("__")[0] for p in planned] == ["mid"] * 4 + ["tiny"] * 4 + ["huge"] * 4,
          f"configurations tried in pool order ({planned})")
    check([p.split("__")[1] for p in planned[:4]] == ["Q4_0", "Q4", "Q4", "Q8_0"], "4-bit before 8-bit within a model")
    check(not any("llama.cpp__gpu" in p for p in planned), "llama.cpp GPU not tried when its OpenCL path cannot run")
    for cid, r in sorted(results.items()):
        m = r["config"]["model"]
        if m == "mid":
            check(r["status"] == "failed" and r["error"].startswith(fp.OVER) and not r["invocations"],
                  f"{cid}: removed at the memory check, not benchmarked ({r.get('error')})")
        elif m == "huge":
            check(r["status"] == "failed" and "does not fit in RAM" in r["error"], f"{cid}: does not fit, not run")
        else:
            check(r["status"] == "ok" and len(r["prefill"]["reps"]) == 1, f"{cid}: benchmarked with 1 timed run")
    check(not any(x.startswith("huge") for x in fake_memory.calls), "no memory run for a model that cannot fit")
    check(c["complete"] is False and c["not_measured"] == ["mid__Q8__mnn__cpu", "mid__Q8__mnn__gpu",
                                                           "huge__Q8__mnn__cpu", "huge__Q8__mnn__gpu"],
          f"incomplete: never-made artifacts listed as not measured ({c['not_measured']})")
    check(len(c["entries"]) == 6 and all(len(e["tried"]) == 4 for e in c["entries"]),
          "full matrix: every model x bit-width with all 4 runtime/backend combinations")
    by = {(e["model"], e["bits"]): e for e in c["entries"]}
    check([(e["model"], e["bits"]) for e in c["entries"]] ==
          [("mid", 4), ("mid", 8), ("tiny", 4), ("tiny", 8), ("huge", 4), ("huge", 8)], "entries in pool order")
    t4 = by[("tiny", 4)]
    check(t4["kept"] and t4["best"]["runtime"] == "llama.cpp" and t4["best"]["backend"] == "cpu",
          f"tiny 4-bit: llama.cpp CPU chosen (fastest decode): {t4['best']}")
    check([(a["runtime"], a["backend"]) for a in t4["alternatives"]] == [("mnn", "cpu")],
          "tiny 4-bit: MNN CPU the only alternative (MNN GPU decodes 12 < 20 tokens/s)")
    mg = next(t for t in t4["tried"] if (t["runtime"], t["backend"]) == ("mnn", "gpu"))
    check(any("decode 12" in r for r in mg["reasons"]), f"MNN GPU removed for decode: {mg['reasons']}")
    lg = next(t for t in t4["tried"] if (t["runtime"], t["backend"]) == ("llama.cpp", "gpu"))
    check(lg["reasons"] and lg["reasons"][0].startswith("not tried:"), f"llama.cpp GPU listed as not tried: {lg['reasons']}")
    check(t4["best"]["energy_j_per_request"] is not None and t4["best"]["energy_j_per_request"] < 50,
          f"energy measured and within the limit ({t4['best']['energy_j_per_request']} J)")
    check(not by[("mid", 4)]["kept"] and all(any("peak memory 944 MB > 500 MB" in r for r in t["reasons"])
                                             for t in by[("mid", 4)]["tried"] if t["backend"] == "cpu"),
          "mid removed with its peak memory as the reason")
    check(not by[("huge", 4)]["kept"], "huge removed")
    t8 = {(t["runtime"], t["backend"]): t for t in by[("tiny", 8)]["tried"]}
    check(t8[("mnn", "cpu")]["reasons"] == ["not tried: conversion to MNN failed (prepare_models.py exit 1): boom"]
          and t8[("mnn", "gpu")]["reasons"][0].startswith("not tried: conversion"),
          "a failed conversion is listed as not tried, on both backends")
    check(c["filtered_pool"] == [{"model": "tiny", "hf_id": "org/tiny", "bits": 4, "quant": "Q4_0",
                                  "runtime": "llama.cpp", "format": "GGUF", "backend": "cpu"},
                                 {"model": "tiny", "hf_id": "org/tiny", "bits": 8, "quant": "Q8_0",
                                  "runtime": "llama.cpp", "format": "GGUF", "backend": "cpu"}],
          f"filtered pool {c['filtered_pool']}")

    # The contract matches the schema the rest of the pipeline validates against.
    try:
        import jsonschema
    except ImportError:
        print("SKIP schema validation (pip install jsonschema)")
    else:
        schema = json.loads((HERE / "filter_pool_contract.schema.json").read_text(encoding="utf-8"))
        v = jsonschema.Draft202012Validator(schema)
        errs = [f"{list(e.path)}: {e.message[:100]}" for e in v.iter_errors(json.loads(json.dumps(c)))]
        check(not errs, f"contract matches filter_pool_contract.schema.json {errs[:3]}")
        bad = json.loads(json.dumps(c))
        bad["filtered_pool"][0]["backend"] = "npu"
        bad["entries"][0]["tried"][0]["extra"] = 1
        check(len(list(v.iter_errors(bad))) == 2, "the schema rejects an unknown backend and an unknown field")

    # Without an energy limit nothing energy-only runs: no extension repetitions, no energy reasons.
    out2 = tmp / "results" / "filter_fake_noenergy_pinned"
    lim2 = {"max_peak_memory_mb": None, "min_prefill_tps": None, "min_decode_tps": 20, "max_energy_j": None}
    pool2 = {"tiny": ("org/tiny", "pinned")}
    args2 = fp.session_args("FAKE123", "pinned", list(pool2), ["Q4_0", "Q4"], out2, "auto", False,
                            ["--idle-seconds", "1", "--gate-timeout", "60", "--min-battery", "0"])
    fp.FilterSession(args2, lim2, {"tiny": 0}, set()).run()
    r2 = {f.stem: json.loads(f.read_text()) for f in (out2 / "configs").glob("*.json")}
    check(len(r2) == 4 and all(r["status"] == "ok" for r in r2.values()), f"no-energy session: 4 configurations ok")
    check(not any(k.startswith("energy_") for r in r2.values() for k in r["invocations"]),
          "no-energy session: no energy extension runs")
    exp2 = fp.expected_matrix(pool2, ["Q4_0"], [4])
    c2 = fp.build_contract(phone, [out2], {"pinned": {"llama.cpp": None, "mnn": None}}, lim2,
                           {"prompt_tokens": 256, "answer_tokens": 256}, "decode", pool2, exp2)
    e2 = c2["entries"][0]
    check(c2["complete"] is True and e2["kept"] and e2["best"]["runtime"] == "llama.cpp"
          and e2["best"]["backend"] == "cpu", f"no-energy: kept with llama.cpp CPU ({e2['best']})")
    check(not any("energy" in r for t in e2["tried"] for r in t["reasons"]), "no-energy: no energy reasons")
    check([(t["runtime"], t["backend"]) for t in e2["tried"] if not t["passes"]] == [("mnn", "gpu")],
          "no-energy: only MNN GPU (12 tokens/s) removed, for decode")
    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED'} (artifacts in {tmp})")
    if not failures:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
