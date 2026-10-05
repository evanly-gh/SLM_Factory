#!/usr/bin/env python3
"""
Build the paper's benchmark binaries for Android arm64 (Windows or macOS host), and optionally push them.

  python build_binaries.py                 # pinned versions: llama.cpp eadc418, MNN 510ac8f
  python build_binaries.py --ref head      # current upstream master of both (for newer architectures)
  python build_binaries.py --push [--serial S]

Work directory: ./third_party on macOS/Linux, %USERPROFILE%\\.pb on Windows (MAX_PATH), or $PB_WORK.

Targets (each into <work>/out/<ref>/<target>/, with build_info.json):
  llama_cpu   llama-bench + llama-quantize, -march=armv8.7a (llama.cpp's Android recipe), no OpenCL
  llama_gpu   llama-bench with GGML_OPENCL=ON + Adreno kernels, built against Khronos headers/ICD loader
              (at run time the phone's own /vendor/lib64/libOpenCL.so is used)
  mnn         llm_bench + MNN shared libs, MNN's own Android flags (build_64.sh + -DMNN_BUILD_LLM=true
              -DMNN_OPENCL=true, as in MNN's LLM docs), so CPU and OpenCL come from one build

Source edits (applied idempotently; each one checks its anchor text and fails loudly if it is missing):
  * llama-bench and llm_bench print one "PB_MARK ..." line per timed repetition with CLOCK_BOOTTIME
    begin/end (the same clock as /proc/uptime and Perfetto), so energy can be integrated over exactly
    the timed work - the role of the paper's PowerBench start()/stop().
  * MNN's Llm::is_stop() ignores EOS when PB_IGNORE_EOS is set, so decode always runs the full 256
    tokens (the paper's "end-of-sequence token replacement"); llm_bench divides by the requested count.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DEV_ROOT, REFS, REPOS, THIRD_PARTY, dev_bin_dir  # noqa: E402
from devenv import IS_WINDOWS, Adb, find_build_tool, find_ndk  # noqa: E402

SRC = THIRD_PARTY / "src"
BUILD = THIRD_PARTY / "build"
OUT = THIRD_PARTY / "out"
OPENCL_TAG = "v2024.10.24"
OPENCL_REPOS = {"OpenCL-Headers": "https://github.com/KhronosGroup/OpenCL-Headers.git",
                "OpenCL-ICD-Loader": "https://github.com/KhronosGroup/OpenCL-ICD-Loader.git"}
ANDROID_API = "28"
# llama.cpp's docs/android.md at eadc418; dotprod and i8mm on, fp16 vector arithmetic off
# (armv8.7a+fp16 turns it on, which matters for F16 models).
DEFAULT_MARCH = "armv8.7a"


def run(cmd: list, cwd: Path | None = None, env: dict | None = None, quiet: bool = False) -> str:
    print(f"[RUN] {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, text=True, encoding="utf-8", errors="replace",
                       stdout=subprocess.PIPE if quiet else None, stderr=subprocess.STDOUT if quiet else None)
    if r.returncode != 0:
        if quiet and r.stdout:
            print(r.stdout[-4000:])
        sys.exit(f"command failed ({r.returncode}): {' '.join(str(c) for c in cmd)}")
    return r.stdout or ""


def git(*args, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def fetch(name: str, url: str, ref: str) -> tuple[Path, str]:
    """Shallow checkout of `ref` (a SHA, tag or branch) into third_party/src/<name>-<sha12>."""
    staging = SRC / f"{name}-fetch"
    if not (staging / ".git").is_dir():
        staging.mkdir(parents=True, exist_ok=True)
        run(["git", "init", "-q"], cwd=staging)
        run(["git", "config", "core.longpaths", "true"], cwd=staging)
        run(["git", "remote", "add", "origin", url], cwd=staging)
    run(["git", "fetch", "-q", "--depth", "1", "origin", ref], cwd=staging)
    sha = git("rev-parse", "FETCH_HEAD", cwd=staging)
    dest = SRC / f"{name}-{sha[:12]}"
    if not (dest / ".git").exists():  # a worktree's .git is a file
        run(["git", "worktree", "add", "-f", "--detach", str(dest), sha], cwd=staging)
    return dest, sha


# ---------------------------------------------------------------------------
# Source edits
# ---------------------------------------------------------------------------

BOOTTIME_HELPER = """
#include <time.h>
// PB_MARK: CLOCK_BOOTTIME, the clock /proc/uptime and Perfetto use, so energy windows line up.
static unsigned long long pb_boottime_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_BOOTTIME, &ts);
    return (unsigned long long) ts.tv_sec * 1000000000ull + (unsigned long long) ts.tv_nsec;
}
"""


def edit(path: Path, anchor: str, replacement: str, marker: str):
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    if text.count(anchor) != 1:
        sys.exit(f"source edit failed: anchor found {text.count(anchor)}x in {path} (expected 1):\n{anchor}")
    path.write_text(text.replace(anchor, replacement), encoding="utf-8", newline="\n")
    print(f"[EDIT] {path.name}: {marker}")


def patch_llama(src: Path):
    f = src / "tools" / "llama-bench" / "llama-bench.cpp"
    edit(f, "// utils\n", "// utils\n" + BOOTTIME_HELPER, "pb_boottime_ns() {")
    edit(f, "        uint64_t t_start = get_time_ns();\n",
         "        unsigned long long pb_t0 = pb_boottime_ns();\n        uint64_t t_start = get_time_ns();\n",
         "pb_t0 = pb_boottime_ns()")
    edit(f, "        t.samples_ns.push_back(t_ns);\n",
         "        t.samples_ns.push_back(t_ns);\n"
         "        fprintf(stderr, \"PB_MARK llama rep=%d begin=%llu end=%llu n_prompt=%d n_gen=%d n_depth=%d\\n\",\n"
         "                i, pb_t0, pb_boottime_ns(), t.n_prompt, t.n_gen, t.n_depth);\n",
         "PB_MARK llama")


def patch_mnn(src: Path):
    bench = src / "transformers" / "llm" / "engine" / "tools" / "llm_bench.cpp"
    edit(bench, "static Llm* buildLLM(", BOOTTIME_HELPER + "\nstatic Llm* buildLLM(", "pb_boottime_ns() {")
    mark = ('fprintf(stderr, "PB_MARK mnn mode=%s rep=%d begin=%llu end=%llu prefill_us=%lld decode_us=%lld '
            'prompt=%d gen=%d\\n", {mode}, i, pb_t0, pb_boottime_ns(), (long long) context->prefill_us, '
            '(long long) context->decode_us, (int) {prompt}, (int) context->gen_seq_len);')
    edit(bench, "llm->response(tokens, nullptr, nullptr, decodeTokens);",
         "unsigned long long pb_t0 = pb_boottime_ns();\n"
         "                llm->response(tokens, nullptr, nullptr, decodeTokens);\n"
         "                " + mark.replace("{mode}", '"kv"').replace("{prompt}", "prompt_tokens"),
         'PB_MARK mnn mode=%s rep=%d begin=%llu end=%llu prefill_us=%lld decode_us=%lld prompt=%d gen=%d\\n", "kv"')
    edit(bench, "llm->response(tokens, nullptr, nullptr, 1);",
         "unsigned long long pb_t0 = pb_boottime_ns();\n"
         "                    llm->response(tokens, nullptr, nullptr, 1);\n"
         "                    " + mark.replace("{mode}", '"pp"').replace("{prompt}", "prompt_tokens"),
         '"pp", i, pb_t0')
    edit(bench, "llm->response(tokens1, nullptr, nullptr, decodeTokens);",
         "unsigned long long pb_t0 = pb_boottime_ns();\n"
         "                    llm->response(tokens1, nullptr, nullptr, decodeTokens);\n"
         "                    " + mark.replace("{mode}", '"tg"').replace("{prompt}", "1"),
         '"tg", i, pb_t0')
    llm = src / "transformers" / "llm" / "engine" / "src" / "llm.cpp"
    edit(llm, "    bool stop = mTokenizer->is_stop(token_id);\n",
         "    // PB_IGNORE_EOS: decode runs the full requested length (the paper's EOS replacement).\n"
         "    static const bool pb_ignore_eos = getenv(\"PB_IGNORE_EOS\") != nullptr;\n"
         "    bool stop = !pb_ignore_eos && mTokenizer->is_stop(token_id);\n",
         "PB_IGNORE_EOS")
    text = llm.read_text(encoding="utf-8")
    if "#include <cstdlib>" not in text:
        llm.write_text("#include <cstdlib>\n" + text, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# Builds
# ---------------------------------------------------------------------------

def android_cmake_args(ndk: Path, ninja: str) -> list[str]:
    return ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}",
            f"-DCMAKE_TOOLCHAIN_FILE={(ndk / 'build' / 'cmake' / 'android.toolchain.cmake').as_posix()}",
            "-DANDROID_ABI=arm64-v8a", f"-DANDROID_PLATFORM=android-{ANDROID_API}", "-DCMAKE_BUILD_TYPE=Release"]


def build_opencl_sdk(ndk: Path, cmake: str, ninja: str) -> tuple[Path, Path]:
    """Khronos headers + ICD loader cross-built for Android: link-time only (the phone supplies the driver)."""
    headers, _ = fetch("OpenCL-Headers", OPENCL_REPOS["OpenCL-Headers"], OPENCL_TAG)
    loader, _ = fetch("OpenCL-ICD-Loader", OPENCL_REPOS["OpenCL-ICD-Loader"], OPENCL_TAG)
    bdir = BUILD / "opencl-icd-loader"
    lib = bdir / "libOpenCL.so"
    if not lib.is_file():
        run([cmake, "-S", loader, "-B", bdir, *android_cmake_args(ndk, ninja),
             f"-DOPENCL_ICD_LOADER_HEADERS_DIR={headers.as_posix()}", "-DBUILD_TESTING=OFF"])
        run([cmake, "--build", bdir, "--target", "OpenCL"])
    return headers, lib


def build_llama(ref: str, ndk: Path, cmake: str, ninja: str, jobs: int, gpu: bool, march: str) -> dict:
    src, sha = fetch("llama.cpp", REPOS["llama.cpp"], REFS[ref]["llama.cpp"])
    patch_llama(src)
    target = "llama_gpu" if gpu else "llama_cpu"
    bdir = BUILD / f"{target}-{sha[:12]}-{march.replace('+', '_')}"
    flags = f"-march={march}"
    args = [*android_cmake_args(ndk, ninja), f"-DCMAKE_C_FLAGS={flags}", f"-DCMAKE_CXX_FLAGS={flags}",
            "-DGGML_OPENMP=OFF", "-DGGML_LLAMAFILE=OFF", "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_CURL=OFF",
            "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=OFF"]
    targets = ["llama-bench"] if gpu else ["llama-bench", "llama-quantize"]
    if gpu:
        headers, lib = build_opencl_sdk(ndk, cmake, ninja)
        args += ["-DGGML_OPENCL=ON", "-DGGML_OPENCL_USE_ADRENO_KERNELS=ON", "-DGGML_OPENCL_EMBED_KERNELS=ON",
                 f"-DOpenCL_INCLUDE_DIR={headers.as_posix()}", f"-DOpenCL_LIBRARY={lib.as_posix()}"]
    run([cmake, "-S", src, "-B", bdir, *args])
    run([cmake, "--build", bdir, "-j", str(jobs), "--target", *targets])
    out = OUT / ref / target
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for t in targets:
        shutil.copy2(bdir / "bin" / t, out / t)
    info = {"framework": "llama.cpp", "target": target, "ref": ref, "commit": sha, "cmake_args": args,
            "binaries": targets, "built": datetime.now().isoformat()}
    (out / "build_info.json").write_text(json.dumps(info, indent=1))
    return info


def build_mnn(ref: str, ndk: Path, cmake: str, ninja: str, jobs: int) -> dict:
    src, sha = fetch("mnn", REPOS["mnn"], REFS[ref]["mnn"])
    patch_mnn(src)
    bdir = BUILD / f"mnn-{sha[:12]}"
    # project/android/build_64.sh, then the LLM docs' Android line (-DMNN_BUILD_LLM=true -DMNN_OPENCL=true).
    # MNN_BUILD_LLM itself forces MNN_LOW_MEMORY and MNN_SUPPORT_TRANSFORMER_FUSE on.
    args = [*android_cmake_args(ndk, ninja), "-DANDROID_STL=c++_static", "-DMNN_USE_LOGCAT=false",
            "-DMNN_BUILD_BENCHMARK=ON", "-DMNN_USE_SSE=OFF", "-DMNN_BUILD_TEST=ON",
            "-DMNN_BUILD_FOR_ANDROID_COMMAND=true", "-DNATIVE_LIBRARY_OUTPUT=.", "-DNATIVE_INCLUDE_OUTPUT=.",
            "-DMNN_BUILD_LLM=true", "-DMNN_OPENCL=true", "-DMNN_ARM82=ON"]
    run([cmake, "-S", src, "-B", bdir, *args])
    run([cmake, "--build", bdir, "-j", str(jobs), "--target", "llm_bench"])
    out = OUT / ref / "mnn"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    shutil.copy2(bdir / "llm_bench", out / "llm_bench")
    libs = sorted({p.name: p for p in bdir.rglob("*.so")}.values(), key=lambda p: p.name)
    for so in libs:
        shutil.copy2(so, out / so.name)
    info = {"framework": "mnn", "target": "mnn", "ref": ref, "commit": sha, "cmake_args": args,
            "binaries": ["llm_bench"], "libs": [p.name for p in libs], "built": datetime.now().isoformat()}
    (out / "build_info.json").write_text(json.dumps(info, indent=1))
    return info


def push(ref: str, serial: str | None, targets: list[str]):
    adb = Adb(serial)
    for target in targets:
        local = OUT / ref / target
        if not local.is_dir():
            sys.exit(f"nothing built at {local}; build it first")
        remote = dev_bin_dir(ref, target)
        adb.sh(f"rm -rf {remote}; mkdir -p {remote}")
        for f in sorted(local.iterdir()):
            r = adb.run(["push", str(f), f"{remote}/{f.name}"], timeout=600)
            if r.returncode != 0:
                sys.exit(f"adb push {f} failed: {r.stderr.strip()}")
        adb.sh(f"chmod 755 {remote}/*")
        print(f"[PUSH] {local} -> {adb.serial}:{remote}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", choices=list(REFS), default="pinned")
    ap.add_argument("--targets", nargs="+", choices=["llama_cpu", "llama_gpu", "mnn"],
                    default=["llama_cpu", "llama_gpu", "mnn"])
    ap.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    ap.add_argument("--llama-march", default=DEFAULT_MARCH, help="llama.cpp -march (default: its Android docs)")
    ap.add_argument("--fetch-only", action="store_true", help="checkout + patch sources, no build")
    ap.add_argument("--push", action="store_true", help="push the built targets to the phone")
    ap.add_argument("--push-only", action="store_true", help="push previously built targets, no build")
    ap.add_argument("--serial")
    args = ap.parse_args()

    if not args.push_only:
        if args.fetch_only:
            if any(t.startswith("llama") for t in args.targets):
                patch_llama(fetch("llama.cpp", REPOS["llama.cpp"], REFS[args.ref]["llama.cpp"])[0])
            if "mnn" in args.targets:
                patch_mnn(fetch("mnn", REPOS["mnn"], REFS[args.ref]["mnn"])[0])
            return
        ndk, cmake, ninja = find_ndk(), find_build_tool("cmake"), find_build_tool("ninja")
        print(f"NDK {ndk}\ncmake {cmake}\nninja {ninja}\nhost {'Windows' if IS_WINDOWS else sys.platform}")
        for t in args.targets:
            info = (build_mnn(args.ref, ndk, cmake, ninja, args.jobs) if t == "mnn"
                    else build_llama(args.ref, ndk, cmake, ninja, args.jobs, gpu=(t == "llama_gpu"),
                                     march=args.llama_march))
            print(f"[DONE] {t}: {info['framework']} {info['commit'][:12]} -> {OUT / args.ref / t}")
    if args.push or args.push_only:
        push(args.ref, args.serial, args.targets)
        print(f"Binaries on the phone under {DEV_ROOT}/{args.ref}/")


if __name__ == "__main__":
    main()
