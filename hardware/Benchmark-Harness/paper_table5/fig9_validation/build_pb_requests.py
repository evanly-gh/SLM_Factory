#!/usr/bin/env python3
"""
Build pb_requests (pb_requests.cpp, this folder) for Android arm64 inside MNN's source tree, at the MNN commit the
factory exported and scored the Fig. 9 models with, and optionally push it to the phone.

  python fig9_validation/build_pb_requests.py                      # build (~15 min the first time)
  python fig9_validation/build_pb_requests.py --push --serial S    # build if needed, then push
  python fig9_validation/build_pb_requests.py --push-only --serial S

Same toolchain as build_binaries.py (Android NDK + CMake + Ninja from Android Studio) and the same MNN CMake flags
as its `mnn` target (build_64.sh + -DMNN_BUILD_LLM=true -DMNN_OPENCL=true), so CPU and OpenCL come from one build.

Why not the harness's own `--ref pinned` / `--ref head` MNN: the published models were exported by pymnn 3.6.1 /
MNN 47ccf6c6bb5b (recorded in each model's validation sidecar). The pinned runtime (510ac8f, January 2026) predates
that exporter, and `head` moves. Running the exact runtime version the factory scored with keeps the comparison to
the factory's own numbers clean. --mnn-ref head (or any SHA) is available if this commit does not load on a phone.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
from build_binaries import BUILD, OUT, android_cmake_args, fetch, run  # noqa: E402
from common import DEV_ROOT, REPOS, portable_record  # noqa: E402
from devenv import Adb, find_build_tool, find_ndk  # noqa: E402

MNN_SHA = "47ccf6c6bb5b6d357cd1f9b4370cbecb6188fd34"  # pymnn 3.6.1, the factory's exporter (validation sidecars)
FIG9_DEV = f"{DEV_ROOT}/fig9"
MARKER = "# PB_FIG9: pb_requests"


def dev_bin_dir(sha: str) -> str:
    return f"{FIG9_DEV}/mnn-{sha[:12]}"


def local_out(sha: str) -> Path:
    return OUT / "fig9" / f"mnn-{sha[:12]}"


def add_target(src: Path):
    """Copy pb_requests.cpp into MNN's LLM tools folder and register it next to llm_bench (idempotent)."""
    tools = src / "transformers" / "llm" / "engine" / "tools"
    shutil.copy2(HERE / "pb_requests.cpp", tools / "pb_requests.cpp")
    cml = tools / "CMakeLists.txt"
    text = cml.read_text(encoding="utf-8")
    if MARKER not in text:
        text += (f"\n{MARKER} (Benchmark-Harness/paper_table5/fig9_validation)\n"
                 "add_executable(pb_requests ${CMAKE_CURRENT_LIST_DIR}/pb_requests.cpp)\n"
                 "target_link_libraries(pb_requests ${LLM_DEPS})\n")
        cml.write_text(text, encoding="utf-8", newline="\n")
        print(f"[EDIT] {cml}: pb_requests target")


def build(ref: str, jobs: int) -> tuple[Path, str]:
    src, sha = fetch("mnn", REPOS["mnn"], ref)
    add_target(src)
    ndk, cmake, ninja = find_ndk(), find_build_tool("cmake"), find_build_tool("ninja")
    bdir = BUILD / f"mnn-fig9-{sha[:12]}"
    args = [*android_cmake_args(ndk, ninja), "-DANDROID_STL=c++_static", "-DMNN_USE_LOGCAT=false",
            "-DMNN_BUILD_BENCHMARK=ON", "-DMNN_USE_SSE=OFF", "-DMNN_BUILD_TEST=ON",
            "-DMNN_BUILD_FOR_ANDROID_COMMAND=true", "-DNATIVE_LIBRARY_OUTPUT=.", "-DNATIVE_INCLUDE_OUTPUT=.",
            "-DMNN_BUILD_LLM=true", "-DMNN_OPENCL=true", "-DMNN_ARM82=ON"]
    run([cmake, "-S", src, "-B", bdir, *args])
    run([cmake, "--build", bdir, "-j", str(jobs), "--target", "pb_requests"])
    out = local_out(sha)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    exe = next(bdir.rglob("pb_requests"), None)
    if exe is None:
        sys.exit(f"pb_requests not found under {bdir}")
    shutil.copy2(exe, out / "pb_requests")
    libs = sorted({p.name: p for p in bdir.rglob("*.so")}.values(), key=lambda p: p.name)
    for so in libs:
        shutil.copy2(so, out / so.name)
    info = {"tool": "pb_requests", "mnn_commit": sha, "cmake_args": args, "libs": [p.name for p in libs],
            "built": datetime.now().isoformat()}
    (out / "build_info.json").write_text(json.dumps(portable_record(info), indent=1))
    print(f"[DONE] pb_requests at MNN {sha[:12]} -> {out}")
    return out, sha


def push(out: Path, sha: str, serial: str | None):
    adb = Adb(serial)
    remote = dev_bin_dir(sha)
    adb.sh(f"rm -rf {remote}; mkdir -p {remote}/tmp")
    for f in sorted(out.iterdir()):
        r = adb.run(["push", str(f), f"{remote}/{f.name}"], timeout=600)
        if r.returncode != 0:
            sys.exit(f"adb push {f} failed: {r.stderr.strip()}")
    adb.sh(f"chmod 755 {remote}/pb_requests {remote}/*.so")
    print(f"[PUSH] {out} -> {adb.serial}:{remote}")


def resolve_sha(ref: str) -> str:
    if len(ref) == 40:
        return ref
    from build_binaries import SRC
    staging = SRC / "mnn-fetch"
    if (staging / ".git").is_dir():  # what fetch() last resolved (e.g. head)
        import subprocess
        return subprocess.run(["git", "rev-parse", "FETCH_HEAD"], cwd=staging, capture_output=True,
                              text=True).stdout.strip()
    sys.exit(f"--push-only with --mnn-ref {ref}: build first (no resolved commit)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mnn-ref", default=MNN_SHA, help="MNN commit/branch (default: the factory's exporter commit)")
    ap.add_argument("--jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK") or max(2, (os.cpu_count() or 4) - 1)))
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--push-only", action="store_true")
    ap.add_argument("--serial")
    a = ap.parse_args()
    if a.push_only:
        sha = resolve_sha(a.mnn_ref)
        out = local_out(sha)
        if not (out / "pb_requests").is_file():
            sys.exit(f"nothing built at {out}; run without --push-only first")
    else:
        out, sha = build(a.mnn_ref, a.jobs)
    if a.push or a.push_only:
        push(out, sha, a.serial)
        print(f"run_fig9.py uses it with --mnn-sha {sha} (the default when it is {MNN_SHA[:12]})")


if __name__ == "__main__":
    main()
