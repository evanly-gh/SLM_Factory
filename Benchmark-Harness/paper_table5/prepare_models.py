#!/usr/bin/env python3
"""
Prepare benchmark models: download from Hugging Face, convert/quantize for llama.cpp (GGUF) and MNN, and
record every artifact in models/manifest.json (read by run_paper_table5.py).

  python prepare_models.py                                   # the paper's 4 models, paper quantizations
  python prepare_models.py --models llama3.2-1b --gguf Q4_0 --mnn 4
  python prepare_models.py --model smollm2-135m=HuggingFaceTB/SmolLM2-135M-Instruct --gguf Q4_K_M Q8_0 F16 --mnn 4 8 16

GGUF: the pinned llama.cpp's convert_hf_to_gguf.py writes F16 and Q8_0 directly; Q4_0 / Q4_K_M come from
llama-quantize over the F16 file. llama-quantize is looked up on this computer (--quantize-bin, PATH, the
SmolChat build); if there is none, the Android llama-quantize from build_binaries.py runs on the phone
(--serial) and the result is pulled back, so no host C++ compiler is needed.

MNN: Model-Conversion/convert_to_mnn.py, pointed at the pinned MNN checkout's llmexport.py with MNN's
default recipe (quant_block 64, no HQQ, lm_head at the body's width) unless --mnn-recipe pool. The MNNConvert
it calls must not be newer than the runtime: use SLM_MNN_CONVERT_BIN (built at 510ac8f) or `pip install
MNN==3.4.0` (28 commits after 510ac8f) into the exporter's Python ($SLM_MNN_PYTHON / .venv_mnn).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DEV_MODELS, MODELS_DIR, PAPER_MODELS, REFS, THIRD_PARTY, dev_bin_dir  # noqa: E402
from devenv import IS_WINDOWS, Adb  # noqa: E402

REPO_ROOT = HERE.parent.parent
CONVERT_TO_MNN = REPO_ROOT / "Model-Conversion" / "convert_to_mnn.py"
SMOLCHAT_QUANTIZE = [REPO_ROOT / "SmolChat-Android" / "llama.cpp" / "build" / "bin" / p
                     for p in ("llama-quantize", "llama-quantize.exe", "Release/llama-quantize.exe")]
GGUF_DIRECT = {"F16": "f16", "Q8_0": "q8_0", "BF16": "bf16"}
GGUF_QUANTIZED = {"Q4_0", "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q6_K"}
HF_IGNORE = ["*.gguf", "original/*", "*.pth", "consolidated*", "*.msgpack", "flax_model*", "tf_model*",
             "rust_model*", "onnx/*", "*.onnx"]
MANIFEST = MODELS_DIR / "manifest.json"


def run(cmd: list, env: dict | None = None, cwd: Path | None = None):
    print(f"[RUN] {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], env=env, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"command failed ({r.returncode})")


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"models": []}


def record(manifest: dict, entry: dict):
    manifest["models"] = [m for m in manifest["models"]
                          if not (m["name"] == entry["name"] and m["framework"] == entry["framework"]
                                  and m["quant"] == entry["quant"])] + [entry]
    manifest["models"].sort(key=lambda m: (m["name"], m["framework"], m["quant"]))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1))


def file_map(path: Path) -> dict:
    if path.is_file():
        return {path.name: path.stat().st_size}
    return {p.relative_to(path).as_posix(): p.stat().st_size for p in sorted(path.rglob("*")) if p.is_file()
            and not p.name.endswith(".validation.json")}


def download(hf_id: str) -> Path:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download
    dest = MODELS_DIR / "hf" / hf_id.split("/")[-1]
    print(f"[HF] {hf_id} -> {dest}")
    return Path(snapshot_download(repo_id=hf_id, local_dir=str(dest), ignore_patterns=HF_IGNORE))


def pinned_llama_src() -> Path:
    sha = REFS["pinned"]["llama.cpp"][:12]
    src = THIRD_PARTY / "src" / f"llama.cpp-{sha}"
    if not (src / "convert_hf_to_gguf.py").is_file():
        sys.exit(f"{src} missing: run `python build_binaries.py --fetch-only` first")
    return src


def find_host_quantize(explicit: str | None) -> Path | None:
    for cand in ([Path(explicit)] if explicit else []) + [Path(p) for p in filter(None, [shutil.which("llama-quantize")])] \
            + SMOLCHAT_QUANTIZE:
        if cand.is_file():
            return cand
    return None


def quantize_on_phone(serial: str | None, f16: Path, out: Path, qtype: str):
    adb = Adb(serial)
    qdir = dev_bin_dir("pinned", "llama_cpu")
    if adb.sh(f"test -x {qdir}/llama-quantize && echo ok").strip() != "ok":
        sys.exit(f"no host llama-quantize and none on the phone: run build_binaries.py --push first, or pass --quantize-bin")
    rf16, rout = f"{DEV_MODELS}/{f16.name}", f"{DEV_MODELS}/{out.name}"
    adb.sh(f"mkdir -p {DEV_MODELS}")
    if adb.sh(f"stat -c %s {rf16} 2>/dev/null").strip() != str(f16.stat().st_size):
        print(f"[PHONE] pushing {f16.name} for on-device quantization")
        adb.run(["push", str(f16), rf16], timeout=7200)
    print(f"[PHONE] llama-quantize {qtype} on the phone")
    log = adb.sh(f"cd {qdir} && ./llama-quantize {rf16} {rout} {qtype} 2>&1 | tail -3; rm -f {rf16}", timeout=7200)
    print(log.strip())
    r = adb.run(["pull", rout, str(out)], timeout=7200)
    if r.returncode != 0 or not out.is_file():
        sys.exit(f"on-device quantization failed: {log[-500:]} {r.stderr}")


def prepare_gguf(name: str, hf_id: str, hf_dir: Path, quant: str, args, manifest: dict):
    src = pinned_llama_src()
    gdir = MODELS_DIR / "gguf"
    gdir.mkdir(parents=True, exist_ok=True)
    out = gdir / f"{name}-{quant.lower()}.gguf"
    if out.is_file() and not args.force:
        print(f"[GGUF] {out.name} exists")
    elif quant in GGUF_DIRECT:
        run([sys.executable, src / "convert_hf_to_gguf.py", hf_dir, "--outtype", GGUF_DIRECT[quant], "--outfile", out])
    elif quant in GGUF_QUANTIZED:
        f16 = gdir / f"{name}-f16.gguf"
        if not f16.is_file():
            run([sys.executable, src / "convert_hf_to_gguf.py", hf_dir, "--outtype", "f16", "--outfile", f16])
        qbin = find_host_quantize(args.quantize_bin)
        if qbin:
            run([qbin, f16, out, quant])
        else:
            quantize_on_phone(args.serial, f16, out, quant)
    else:
        sys.exit(f"unsupported GGUF quant {quant}")
    bits = 16 if quant in ("F16", "BF16") else int(quant[1])
    record(manifest, {"name": name, "hf_id": hf_id, "framework": "llama.cpp", "quant": quant, "bits": bits,
                      "path": out.relative_to(MODELS_DIR).as_posix(), "files": file_map(out),
                      "converter": f"llama.cpp {REFS['pinned']['llama.cpp'][:12]} convert_hf_to_gguf"
                                   + ("" if quant in GGUF_DIRECT else " + llama-quantize")})
    print(f"[GGUF] {name} {quant}: {out.stat().st_size / 2**30:.2f} GiB")


def mnn_env(recipe: str) -> dict:
    env = dict(os.environ)
    src = THIRD_PARTY / "src" / f"mnn-{REFS['pinned']['mnn'][:12]}"
    if not (src / "transformers" / "llm" / "export" / "llmexport.py").is_file():
        sys.exit(f"{src} missing: run `python build_binaries.py --fetch-only` first")
    env["SLM_MNN_ROOT"] = str(src)
    env.pop("SLM_MNN_LLMEXPORT", None)
    if recipe == "paper-default":  # llmexport defaults: block 64, min/max (no HQQ), lm_head = body width
        env.update({"SLM_MNN_QUANT_BLOCK": "64", "SLM_MNN_HQQ": "0", "SLM_MNN_LM_QUANT_BIT": "4"})
    if not env.get("SLM_MNN_CONVERT_BIN"):
        py = Path(env.get("SLM_MNN_PYTHON") or (REPO_ROOT / ".venv_mnn" / ("Scripts/python.exe" if IS_WINDOWS
                                                                             else "bin/python")))
        if py.is_file():
            conv = py.parent / ("mnnconvert.exe" if IS_WINDOWS else "mnnconvert")
            ver = subprocess.run([str(py), "-c", "import importlib.metadata as m; print(m.version('MNN'))"],
                                 capture_output=True, text=True).stdout.strip()
            if conv.is_file():
                if ver != "3.4.0":
                    print(f"[WARN] {conv} is from MNN {ver or '?'}; the pinned runtime (510ac8f) matches MNN 3.4.0. "
                          f"`{py} -m pip install MNN==3.4.0` or set SLM_MNN_CONVERT_BIN to an MNNConvert built at 510ac8f.")
                env["SLM_MNN_CONVERT_BIN"] = str(conv)
                env["SLM_MNN_PYTHON"] = str(py)
    return env


def prepare_mnn(name: str, hf_id: str, hf_dir: Path, bits: int, args, manifest: dict):
    env = mnn_env(args.mnn_recipe)
    outdir = MODELS_DIR / "mnn" / args.mnn_recipe
    run([sys.executable, CONVERT_TO_MNN, "--model", hf_dir, "--output", outdir, "--quant", str(bits)], env=env)
    folder = outdir / f"{hf_dir.name.lower().replace('_', '-').replace(' ', '-')}-mnn-q{bits}"
    if not (folder / "config.json").is_file():
        sys.exit(f"expected MNN export at {folder}")
    dest = MODELS_DIR / "mnn" / f"{name}-mnn-{args.mnn_recipe}-q{bits}"
    if dest.resolve() != folder.resolve():
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(folder, dest)
    quant = "F16" if bits == 16 else f"Q{bits}"
    record(manifest, {"name": name, "hf_id": hf_id, "framework": "mnn", "quant": quant, "bits": bits,
                      "path": dest.relative_to(MODELS_DIR).as_posix(), "files": file_map(dest),
                      "recipe": args.mnn_recipe, "converter": f"llmexport.py @ MNN {REFS['pinned']['mnn'][:12]}, "
                      f"MNNConvert {env.get('SLM_MNN_CONVERT_BIN', 'default')}"})
    print(f"[MNN] {name} {quant}: {sum(file_map(dest).values()) / 2**30:.2f} GiB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=list(PAPER_MODELS), help="paper models (default: all 4)")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=HF_ID", help="extra model, repeatable")
    ap.add_argument("--gguf", nargs="*", default=["Q4_0", "Q4_K_M"],
                    help="GGUF quants (F16 Q8_0 Q4_0 Q4_K_M ...); the paper says only 'w4', so both 4-bit types")
    ap.add_argument("--mnn", nargs="*", type=int, default=[4], help="MNN quant_bit levels (4 8 16)")
    ap.add_argument("--mnn-recipe", choices=["paper-default", "pool"], default="paper-default")
    ap.add_argument("--quantize-bin", help="host llama-quantize to use")
    ap.add_argument("--serial", help="phone for on-device quantization when no host llama-quantize exists")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    todo = {n: PAPER_MODELS[n] for n in (args.models or ([] if args.model else PAPER_MODELS))}
    for spec in args.model:
        name, _, hf = spec.partition("=")
        if not hf:
            sys.exit(f"--model expects NAME=HF_ID, got {spec!r}")
        todo[name] = hf
    manifest = load_manifest()
    for name, hf_id in todo.items():
        hf_dir = download(hf_id)
        for q in args.gguf:
            prepare_gguf(name, hf_id, hf_dir, q.upper(), args, manifest)
        for b in args.mnn:
            prepare_mnn(name, hf_id, hf_dir, b, args, manifest)
    print(f"\nmanifest: {MANIFEST} ({len(manifest['models'])} artifacts)")


if __name__ == "__main__":
    main()
