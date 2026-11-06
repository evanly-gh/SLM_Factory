"""
Host-side tool discovery (Windows and macOS/Linux) and a serial-aware adb wrapper.

adb lookup order: $ADB, PATH, $ANDROID_HOME / $ANDROID_SDK_ROOT, then the Android Studio default SDK
location for this OS. Device selection: --serial if given (or $ANDROID_SERIAL), otherwise the only
device in state "device"; "offline" / "unauthorized" entries are ignored when choosing.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = platform.system() == "Windows"
EXE = ".exe" if IS_WINDOWS else ""


def sdk_roots() -> list[Path]:
    roots = [os.environ.get(k) for k in ("ANDROID_HOME", "ANDROID_SDK_ROOT")]
    if IS_WINDOWS:
        roots.append(os.path.join(os.environ.get("LOCALAPPDATA", ""), "Android", "Sdk"))
    elif platform.system() == "Darwin":
        roots.append(str(Path.home() / "Library" / "Android" / "sdk"))
    else:
        roots.append(str(Path.home() / "Android" / "Sdk"))
    return [Path(r) for r in roots if r and Path(r).is_dir()]


def find_adb() -> str:
    env = os.environ.get("ADB")
    if env and Path(env).is_file():
        return env
    on_path = shutil.which("adb")
    if on_path:
        return on_path
    for root in sdk_roots():
        cand = root / "platform-tools" / f"adb{EXE}"
        if cand.is_file():
            return str(cand)
    sys.exit("adb not found: put platform-tools on PATH, or set ADB=<path to adb> or ANDROID_HOME=<sdk dir>")


def _version_key(p: Path):
    return [int(x) if x.isdigit() else x for x in re.split(r"[.\-]", p.name)]


def find_ndk() -> Path:
    for k in ("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT", "ANDROID_NDK", "NDK_ROOT"):
        v = os.environ.get(k)
        if v and (Path(v) / "build" / "cmake" / "android.toolchain.cmake").is_file():
            return Path(v)
    for root in sdk_roots():
        ndks = sorted((p for p in (root / "ndk").glob("*") if (p / "build/cmake/android.toolchain.cmake").is_file()),
                      key=_version_key)
        if ndks:
            return ndks[-1]
        legacy = root / "ndk-bundle"
        if (legacy / "build/cmake/android.toolchain.cmake").is_file():
            return legacy
    sys.exit("Android NDK not found: install it from Android Studio (SDK Manager > SDK Tools > NDK) "
             "or set ANDROID_NDK_HOME")


def find_build_tool(name: str) -> str:
    """cmake / ninja: PATH first, then the newest SDK cmake package (which ships both)."""
    on_path = shutil.which(name)
    if on_path:
        return on_path
    for root in sdk_roots():
        for d in sorted((root / "cmake").glob("*"), key=_version_key, reverse=True):
            cand = d / "bin" / f"{name}{EXE}"
            if cand.is_file():
                return str(cand)
    sys.exit(f"{name} not found: install CMake from Android Studio (SDK Manager > SDK Tools > CMake) or put it on PATH")


def list_devices(adb_bin: str) -> list[tuple[str, str]]:
    """[(serial, state)] from `adb devices`."""
    out = subprocess.run([adb_bin, "devices"], capture_output=True, text=True, timeout=30).stdout
    devs = []
    for line in out.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) == 2:
            devs.append((parts[0].strip(), parts[1].strip()))
    return devs


def pick_serial(adb_bin: str, requested: str | None) -> str:
    requested = requested or os.environ.get("ANDROID_SERIAL")
    devs = list_devices(adb_bin)
    ready = [s for s, st in devs if st == "device"]
    if requested:
        state = dict(devs).get(requested)
        if state != "device":
            sys.exit(f"device {requested!r} is {state or 'not connected'}; adb devices: {devs}")
        return requested
    if len(ready) == 1:
        return ready[0]
    if not ready:
        hint = ""
        if any(st == "unauthorized" for _, st in devs):
            hint = " (unlock the phone and accept the 'Allow USB debugging?' prompt)"
        sys.exit(f"no usable adb device{hint}; adb devices: {devs}")
    sys.exit(f"several devices connected, pass --serial one of: {ready}")


class Adb:
    """adb bound to one device. run(args, timeout) matches the interface bench_common expects."""

    def __init__(self, serial: str | None = None, adb_bin: str | None = None):
        self.bin = adb_bin or find_adb()
        self.serial = pick_serial(self.bin, serial)

    def run(self, args: list, timeout: int = 60) -> subprocess.CompletedProcess:
        cmd = [self.bin, "-s", self.serial, *args]
        try:
            return subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="TIMEOUT")

    def sh(self, cmd: str, timeout: int = 60) -> str:
        return self.run(["shell", cmd], timeout=timeout).stdout or ""
