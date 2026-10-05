"""
Shared settings for the Table 5 replication (arXiv 2607.05475): pinned framework versions, device
layout, the paper's models, device profiles, and the phone-side controls of the paper's protocol
(airplane mode, screen off, background isolation).
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Checkouts and builds. On Windows they live under a short path: MNN's KleidiAI dependency has file names
# that push a build tree inside the repo past the 260-character path limit.
THIRD_PARTY = Path(os.environ.get("PB_WORK") or (Path.home() / ".pb" if platform.system() == "Windows"
                                                  else HERE / "third_party"))
MODELS_DIR = HERE / "models"
RESULTS_DIR = HERE / "results"

# The paper's Table 3 versions. It prints MNN "51bac8f", which does not exist; 510ac8f (2026-01-27) does.
REPOS = {
    "llama.cpp": "https://github.com/ggml-org/llama.cpp.git",
    "mnn": "https://github.com/alibaba/MNN.git",
}
REFS = {
    "pinned": {"llama.cpp": "eadc4184caee5b5f68f31f19a2f65c6961748e46",
               "mnn": "510ac8f15d10ce44e8ab89a6fc972800b39a876e"},
    "head": {"llama.cpp": "master", "mnn": "master"},
}

DEV_ROOT = "/data/local/tmp/paper_bench"
DEV_MODELS = f"{DEV_ROOT}/models"


def dev_bin_dir(ref: str, target: str) -> str:
    """target: llama_cpu | llama_gpu | mnn"""
    return f"{DEV_ROOT}/{ref}/{target}"


# The paper's models (Section 3). Instruct variants, as in llm_bench's default model. The unsloth
# repos are ungated byte-identical copies of meta-llama's weights; pass --hf-id to use meta-llama.
PAPER_MODELS = {
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.2-1b": "unsloth/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "unsloth/Llama-3.2-3B-Instruct",
}

# Phase 2: the model pool (docs/superpowers/specs/2026-10-02-pool-sweep-design.md). min_ref "head": the
# architecture postdates the pinned versions (Qwen3.5, early 2026), so convert and run with --ref head.
POOL_MODELS = {
    "gemma3-270m": ("google/gemma-3-270m-it", "pinned"),
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M-Instruct", "pinned"),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M-Instruct", "pinned"),
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "pinned"),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "pinned"),
    "qwen3-4b-instruct-2507": ("Qwen/Qwen3-4B-Instruct-2507", "pinned"),
    "qwen3.5-0.8b": ("Qwen/Qwen3.5-0.8B", "head"),
    "qwen3.5-2b": ("Qwen/Qwen3.5-2B", "head"),
    "qwen3.5-4b": ("Qwen/Qwen3.5-4B", "head"),
}

DEVICES_FILE = HERE / "devices.json"


def load_profiles() -> list[dict]:
    return json.loads(DEVICES_FILE.read_text())["profiles"]


def match_profile(info: dict) -> dict:
    """First profile whose match rules all hold for the device (getprop values, case-insensitive
    regex); unknown phones get a generic profile instead of an error."""
    for prof in load_profiles():
        rules = prof.get("match", {})
        if rules and all(re.search(pat, str(info.get(key, "")), re.I) for key, pat in rules.items()):
            return prof
    return {"id": re.sub(r"[^a-z0-9]+", "_", f"{info.get('manufacturer', '')}_{info.get('model', '')}".lower()).strip("_"),
            "label": f"{info.get('manufacturer', '')} {info.get('model', '')}".strip(),
            "paper_column": None, "max_temp_c": 28.0, "generated": True}


def device_info(adb) -> dict:
    prop = lambda k: adb.sh(f"getprop {k}").strip()  # noqa: E731
    info = {
        "serial": adb.serial,
        "manufacturer": prop("ro.product.manufacturer"),
        "model": prop("ro.product.model"),
        "device": prop("ro.product.device"),
        "soc": prop("ro.soc.model") or prop("ro.board.platform"),
        "platform": prop("ro.board.platform"),
        "android": prop("ro.build.version.release"),
        "sdk": prop("ro.build.version.sdk"),
        "build": prop("ro.build.fingerprint"),
        "mem_total_kb": int((re.search(r"MemTotal:\s+(\d+)", adb.sh("cat /proc/meminfo")) or [0, 0])[1]),
    }
    clusters = {}
    for line in adb.sh("for p in /sys/devices/system/cpu/cpufreq/policy*; do "
                       "echo $(basename $p) $(cat $p/cpuinfo_max_freq) $(cat $p/related_cpus); done").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1].isdigit():
            clusters[parts[0]] = {"max_khz": int(parts[1]), "cpus": [int(c) for c in parts[2:] if c.isdigit()]}
    info["cpu_clusters"] = clusters
    info["ncpu"] = sum(len(c["cpus"]) for c in clusters.values()) or 8
    return info


# ---------------------------------------------------------------------------
# Phone-side protocol controls
# ---------------------------------------------------------------------------

def is_wireless_serial(serial: str) -> bool:
    """ip:port (adb tcpip / adb connect) or an mDNS name from Android 11+ Wireless debugging
    (adb-<serial>-<id>._adb-tls-connect._tcp)."""
    return ":" in serial or "._adb-tls-connect." in serial or serial.endswith("._tcp")


class DeviceControls:
    """Airplane mode, screen off, Do Not Disturb, background kill. Original state restored by restore().
    Over wireless adb, Wi-Fi is switched back on right after airplane mode (the command runs detached on
    the phone so it survives the adb connection dropping), then adb reconnects."""

    def __init__(self, adb, log=print):
        self.adb, self.log = adb, log
        self.wireless = is_wireless_serial(adb.serial)
        self.saved: dict = {}

    def _setting(self, ns, key):
        v = self.adb.sh(f"settings get {ns} {key}").strip()
        return None if v in ("", "null") else v

    def reconnect(self, timeout_s: int = 120):
        if not self.wireless:
            return
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if ":" in self.adb.serial:  # mDNS (Wireless debugging) devices reconnect on their own
                subprocess.run([self.adb.bin, "connect", self.adb.serial], capture_output=True, text=True, timeout=30)
            if self.adb.sh("echo ok", timeout=15).strip() == "ok":
                return
            time.sleep(3)
        raise RuntimeError(f"lost the wireless adb connection to {self.adb.serial}")

    def apply(self):
        self.saved = {"airplane": self._setting("global", "airplane_mode_on"),
                      "zen": self._setting("global", "zen_mode"),
                      "stay_on": self._setting("global", "stay_on_while_plugged_in")}
        self.adb.sh("am kill-all")
        self.adb.sh("cmd notification set_dnd priority")
        if self.wireless:
            self.adb.sh("setsid sh -c 'cmd connectivity airplane-mode enable; sleep 2; svc wifi enable' "
                        "</dev/null >/dev/null 2>&1 &")
            time.sleep(8)
            self.reconnect()
        else:
            self.adb.sh("cmd connectivity airplane-mode enable")
        self.screen_off()
        self.log("[DEVICE] airplane mode on" + (" (Wi-Fi kept for adb)" if self.wireless else "")
                 + ", Do Not Disturb on, background apps killed, screen off")

    def screen_off(self):
        self.adb.sh("input keyevent 223")

    def restore(self):
        try:
            self.adb.sh("input keyevent 224")
            if self.saved.get("airplane") != "1":
                self.adb.sh("cmd connectivity airplane-mode disable")
                if self.wireless:
                    time.sleep(5)
                    self.reconnect()
            if self.saved.get("zen") in (None, "0"):
                self.adb.sh("cmd notification set_dnd off")
            self.log("[DEVICE] settings restored")
        except Exception as e:  # noqa: BLE001 - restoring must never mask the original error
            self.log(f"[DEVICE] WARNING: could not restore settings: {e}")


def boottime_s(adb) -> float:
    """Phone CLOCK_BOOTTIME (same clock as /proc/uptime, Perfetto and the PB_MARK lines)."""
    return float(adb.sh("cat /proc/uptime").split()[0])
