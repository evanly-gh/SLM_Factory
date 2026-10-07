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

class ScreenKeeper:
    """Keeps the phone awake with the screen on at minimum brightness, restoring the settings afterwards.
    Needed over wireless adb: with the screen off nothing holds a wake lock (a USB connection does), so
    Android suspends the whole system every few seconds and freezes the benchmark mid-run."""

    KEYS = (("system", "screen_off_timeout"), ("system", "screen_brightness_mode"), ("system", "screen_brightness"))

    def __init__(self, adb):
        self.adb = adb
        self.saved: dict = {}

    def apply(self):
        for ns, key in self.KEYS:
            v = self.adb.sh(f"settings get {ns} {key}").strip()
            self.saved[(ns, key)] = None if v in ("", "null") else v
        self.adb.sh("settings put system screen_off_timeout 2147483647")
        self.adb.sh("settings put system screen_brightness_mode 0")
        self.adb.sh("settings put system screen_brightness 1")
        self.wake()

    def wake(self):
        self.adb.sh("input keyevent 224")  # KEYCODE_WAKEUP; no-op when already awake
        self.adb.sh("wm dismiss-keyguard")

    def restore(self):
        for (ns, key), v in self.saved.items():
            if v is not None:
                self.adb.sh(f"settings put {ns} {key} {v}")


class WakeHolder:
    """Keeps the CPU awake with the screen OFF over wireless adb: a partial wake lock held by tools/PbWake.java,
    run with app_process as the shell user (no app install). Replaces ScreenKeeper's screen-on workaround."""

    JAR = f"{DEV_ROOT}/pbwake.jar"
    STOP = f"{DEV_ROOT}/pbwake.stop"
    LOG = f"{DEV_ROOT}/pbwake.log"

    def __init__(self, adb):
        self.adb = adb

    def apply(self):
        from build_binaries import build_pbwake
        jar = build_pbwake()
        self.adb.sh(f"mkdir -p {DEV_ROOT}; rm -f {self.STOP} {self.LOG}; pkill -f 'PbWak[e]'")
        r = self.adb.run(["push", str(jar), self.JAR], timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"pushing the wake-lock helper failed: {r.stderr.strip()}")
        self.adb.sh(f"CLASSPATH={self.JAR} setsid app_process / PbWake {self.STOP} </dev/null >{self.LOG} 2>&1 &")
        for _ in range(20):
            time.sleep(0.5)
            if "PB_WAKELOCK held" in self.adb.sh(f"cat {self.LOG}"):
                break
        else:
            raise RuntimeError(f"the wake-lock helper did not start: {self.adb.sh(f'cat {self.LOG}')[-500:]}")
        if "pb:benchmark" not in self.adb.sh("dumpsys power | grep -i 'pb:benchmark'"):
            raise RuntimeError("the wake-lock helper runs but dumpsys power lists no pb:benchmark wake lock")

    def held(self) -> bool:
        return "pb:benchmark" in self.adb.sh("dumpsys power | grep -i 'pb:benchmark'")

    def restore(self):
        self.adb.sh(f"touch {self.STOP}")
        time.sleep(2)
        self.adb.sh("pkill -f 'PbWak[e]'")


def is_wireless_serial(serial: str) -> bool:
    """ip:port (adb tcpip / adb connect) or an mDNS name from Android 11+ Wireless debugging
    (adb-<serial>-<id>._adb-tls-connect._tcp)."""
    return ":" in serial or "._adb-tls-connect." in serial or serial.endswith("._tcp")


class DeviceControls:
    """Airplane mode, screen off, Do Not Disturb, background kill. Original state restored by restore().
    Over wireless adb: airplane mode only if Wi-Fi stays on in it (Android remembers Wi-Fi turned back on in airplane
    mode: secure wifi_apm_state = 1), since dropping Wi-Fi ends Wireless debugging; else mobile data, Bluetooth and
    location are switched off one by one. Screen off as in the paper; over wireless adb a partial wake lock
    (WakeHolder) keeps the CPU running, falling back to screen on at minimum brightness (ScreenKeeper)."""

    def __init__(self, adb, log=print, screen: str = "auto"):
        self.adb, self.log = adb, log
        self.wireless = is_wireless_serial(adb.serial)
        self.screen_on = screen == "on"
        self.keeper = ScreenKeeper(adb) if self.screen_on else None
        self.waker = WakeHolder(adb) if self.wireless and not self.screen_on else None
        self.saved: dict = {}
        self.radio_mode = None

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
        if not self.wireless:
            self.adb.sh("cmd connectivity airplane-mode enable")
            self.radio_mode = "airplane mode on"
        elif self.saved["airplane"] == "1":
            self.radio_mode = "airplane mode already on (Wi-Fi kept on in it)"
        elif self._setting("secure", "wifi_apm_state") == "1":
            # Wi-Fi stays on in airplane mode on this phone; detached so a brief drop cannot kill the command
            self.adb.sh("setsid sh -c 'cmd connectivity airplane-mode enable' </dev/null >/dev/null 2>&1 &")
            time.sleep(6)
            self.reconnect(60)
            self.radio_mode = "airplane mode on (Wi-Fi stays on in it)"
        else:
            # Airplane mode would drop Wi-Fi, and Android turns Wireless debugging off with it (its port changes when
            # re-enabled), so the session could not reconnect. Switch off the other radios one by one instead.
            self.saved["radios"] = {"mobile_data": self._setting("global", "mobile_data"),
                                    "bluetooth_on": self._setting("global", "bluetooth_on"),
                                    "location": self.adb.sh("cmd location is-location-enabled").strip()}
            self.adb.sh("svc data disable; svc bluetooth disable; cmd location set-location-enabled false")
            self.radio_mode = ("mobile data, Bluetooth and location off (Wi-Fi kept for adb; airplane mode would end "
                               "Wireless debugging: turn Wi-Fi back on once while in airplane mode to change that)")
        if self.waker:
            try:
                self.waker.apply()
            except Exception as e:  # noqa: BLE001 - fall back to the screen-on workaround
                self.log(f"[DEVICE] WARNING: wake lock failed ({e}); keeping the screen on at minimum brightness")
                self.waker, self.screen_on, self.keeper = None, True, ScreenKeeper(self.adb)
        if self.keeper:
            self.keeper.apply()
        self.ensure_screen()
        screen = ("on at minimum brightness (wireless adb: keeps the phone from suspending)" if self.screen_on else
                  "off, CPU kept awake by a partial wake lock (wireless adb)" if self.waker else "off")
        self.log(f"[DEVICE] {self.radio_mode}, Do Not Disturb on, background apps killed, screen {screen}")

    def ensure_screen(self):
        if self.keeper:
            self.keeper.wake()
        else:
            self.adb.sh("input keyevent 223")  # KEYCODE_SLEEP

    def screen_off(self):
        self.ensure_screen()

    def restore(self):
        try:
            if self.waker:
                self.waker.restore()
            self.adb.sh("input keyevent 224")
            if self.keeper:
                self.keeper.restore()
            radios = self.saved.get("radios")
            if radios is not None:
                if radios["mobile_data"] != "0":
                    self.adb.sh("svc data enable")
                if radios["bluetooth_on"] not in (None, "0"):
                    self.adb.sh("svc bluetooth enable")
                if radios["location"] == "true":
                    self.adb.sh("cmd location set-location-enabled true")
            elif self.saved.get("airplane") != "1":
                self.adb.sh("cmd connectivity airplane-mode disable")
                if self.wireless:
                    time.sleep(5)
                    self.reconnect(60)
            if self.saved.get("zen") in (None, "0"):
                self.adb.sh("cmd notification set_dnd off")
            self.log("[DEVICE] settings restored")
        except Exception as e:  # noqa: BLE001 - restoring must never mask the original error
            self.log(f"[DEVICE] WARNING: could not restore settings: {e}")


def boottime_s(adb) -> float:
    """Phone CLOCK_BOOTTIME (same clock as /proc/uptime, Perfetto and the PB_MARK lines)."""
    return float(adb.sh("cat /proc/uptime").split()[0])
