# agent/hardware_research.py
"""
Autonomous hardware research.

Given a free-text device/chip description from the user (e.g. "moto g stylus 5g 2023
with 6 GB of ram and 256 gb of ROM"), the orchestrator grounds itself with a web search
(Exa) and then uses Claude Sonnet to resolve the chipset / RAM / storage / NPU and emit a
concrete HardwareConstraints object that gates model selection in android_pool.filter_pool.

Returns (HardwareConstraints, info_dict). info_dict is saved as device_research.json.
"""
import json
import re

from android_pool import HardwareConstraints

RESEARCH_MODEL = "claude-sonnet-4-6"
# Reference chips for which the Android pool has tok/s benchmarks; the agent maps the
# real device chipset to the nearest of these.
REFERENCE_CHIPS = ["snapdragon_660", "snapdragon_778g", "snapdragon_8gen3"]

_PROMPT = """You are the hardware-analysis stage of an autonomous on-device fine-tuning agent. \
A user wants to run a small quantized language model on their phone. Given the user's device \
description and some web snippets about the device, resolve its real specs and the constraints \
that bound a deployable INT4 model.

Reply with ONLY a JSON object (no prose, no code fences) with keys:
- "device": resolved device name
- "chipset": the SoC (e.g. "Qualcomm Snapdragon 6 Gen 1")
- "ram_gb": integer total RAM
- "storage_gb": integer total storage
- "npu": short note on the NPU/AI accelerator, or null
- "usable_ram_mb": integer — RAM available to the model after OS/other apps (reserve ~2 GB on a phone)
- "storage_budget_mb": integer — a sensible on-disk budget for the model file (do NOT use the whole ROM; \
keep it small for efficient on-device use, typically 1000-3000 MB)
- "latency_ttft_ms": integer target time-to-first-token for interactive use (e.g. 2000)
- "power_watts": float sustained-inference power budget (e.g. 5.0)
- "reference_chip": the closest of {chips} to this device's SoC by performance tier
- "rationale": one sentence

User device description:
\"\"\"{description}\"\"\"

Web snippets:
{snippets}
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"Hardware research did not return JSON: {text[:200]!r}")
        return json.loads(m.group())


def _exa_snippets(description: str, log) -> str:
    try:
        from config import EXA_API_KEY
        from exa_py import Exa
        exa = Exa(api_key=EXA_API_KEY)
        r = exa.search_and_contents(
            f"{description} smartphone chipset SoC RAM storage NPU full specifications",
            num_results=3, type="auto", text={"max_characters": 600},
        )
        return "\n".join(f"- {x.title[:80]}: {(x.text or '')[:300]}" for x in r.results)
    except Exception as e:
        log(f"      [hw] Exa lookup failed ({e}); proceeding from the description alone.")
        return "(no web snippets available)"


def research_device(description: str, anthropic_client=None, log=print):
    """Resolve a device description into HardwareConstraints via Exa + Claude. Best-effort."""
    if anthropic_client is None:
        import anthropic
        from config import ANTHROPIC_API_KEY
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    snippets = _exa_snippets(description, log)
    try:
        resp = anthropic_client.messages.create(
            model=RESEARCH_MODEL, max_tokens=700,
            messages=[{"role": "user", "content": _PROMPT.format(
                chips=REFERENCE_CHIPS, description=description, snippets=snippets)}],
        )
        info = _extract_json(resp.content[0].text)
    except Exception as e:
        log(f"      [hw] research failed ({e}); using conservative defaults.")
        info = {"device": description, "usable_ram_mb": 3000, "storage_budget_mb": 1500,
                "latency_ttft_ms": 2000, "power_watts": 5.0,
                "reference_chip": "snapdragon_778g", "rationale": "fallback defaults"}

    ref = info.get("reference_chip")
    if ref not in REFERENCE_CHIPS:
        ref = "snapdragon_778g"
    hw = HardwareConstraints(
        storage_mb=int(info.get("storage_budget_mb", 1500)),
        memory_mb=int(info.get("usable_ram_mb", 3000)),
        latency_ttft_ms=int(info.get("latency_ttft_ms", 2000)),
        power_watts=float(info.get("power_watts", 5.0)),
        target_chip=ref,
    )
    log(f"      [hw] {info.get('device','?')} | {info.get('chipset','?')} | "
        f"RAM {info.get('ram_gb','?')}GB -> usable {hw.memory_mb}MB | "
        f"model budget {hw.storage_mb}MB | ref_chip={hw.target_chip}")
    log(f"      [hw] rationale: {info.get('rationale','')}")
    return hw, info
