"""ToolBench loader (2026-08-24) — the corpus and test split of arXiv:2512.15943.

WHAT THIS IS
    ToolBench is the ToolLLM / OpenBMB benchmark (Qin et al., arXiv:2307.16789): ~16,000 real
    RapidAPI endpoints, and multi-step solution paths in which a model interleaves a free-text
    ``Thought`` with an ``Action`` / ``Action Input`` function call and terminates by calling the
    pseudo-function ``Finish``. It is NOT SambaNova's identically-named benchmark
    (arXiv:2305.16504), which is eight hand-built execution tasks.

    ``Small Language Models for Efficient Agentic Tool Calling`` (arXiv:2512.15943) fine-tunes on
    it and reports ToolEval pass rate. Its stated training size — 187,542 examples — is exactly the
    row count of ToolBench's own ``toolllama_G123_dfs_train.json``, so that file is what this
    loader trains from.

TRAIN: A 2 GB FILE READ 51 MB AT A TIME
    ``toolllama_G123_dfs_train.json`` is a single 2,002,419,658-byte pretty-printed JSON array.
    ``json.load`` on it costs the whole 2 GB and several GB of resident memory to produce 187,542
    rows, of which a cold-start curriculum takes 5,000 — measured at 390 complete objects per 4 MiB,
    so ~51 MiB is enough. It is served over HTTPS with byte-range support (verified: HTTP 206), so
    this reads a growing PREFIX and decodes objects out of it incrementally.

    The prefix is cached on disk and EXTENDED rather than refetched, which is what makes rung 1 of
    the mining ladder cheap here: ``_reread_known_sources`` calls ``load`` again with a larger
    ``max_train`` every rebuild (see ``agent/nodes/curate.py``), and a growing head slice is
    precisely what it needs to see. Re-reading 6,000 rows after 5,000 costs the 11 MB difference,
    not 2 GB — and B303 is the open bug for the task whose loader could NOT do this.

TRAIN: ONE COMPLETE PATH PER ROW
    The ``_dfs`` file is already exploded by step: an ``id`` of ``"Step 9: <query>"`` means this row
    is that query's trajectory truncated after 9 assistant turns, and the same query appears
    several times at different depths (measured on the eval split: 762 rows over 300 distinct
    queries). Consecutive depths share nearly all of their text.

    This loader keeps only rows whose FINAL assistant turn is ``Finish`` with
    ``return_type == "give_answer"``, i.e. the complete, successful paths. Three things fall out of
    that one rule:

      * the target is a whole solution path, which is what the eval asks the model to produce, so
        train and serve agree about the unit of work;
      * near-duplicate step prefixes disappear without a similarity filter — which matters because
        a trigram filter CANNOT be used here (step N and step N+1 differ by one action, so any
        threshold that removes them also removes genuinely distinct trajectories);
      * paths that ended in ``give_up_and_restart`` are excluded, since a curriculum built from
        them teaches giving up.

EVAL: THE SIX TOOLEVAL SUBSETS
    ToolEval's test set is six named subsets (G1/G2/G3 x instruction/category/tool), and the
    paper's headline number is their query-count-weighted mean. Reproducing that needs each query's
    ``api_list``, which ToolBench itself distributes only inside 1.7 GB of ``instruction/G*_query.json``
    — its own ``test_query_ids`` files carry ids and nothing else.

    StableToolBench (Guo et al., arXiv:2403.07714) republishes the same six subsets with the
    ``api_list`` inline, one small file each, and additionally filters them for SOLVABILITY by
    majority vote of three frontier models. That filter is not a deviation for the sake of
    convenience — it is load-bearing for this metric. Pass rate asks "did the model solve it"; a
    query that is not solvable by anyone caps the achievable score below 1.0 and reports a model
    failure for a benchmark defect. 765 queries survive it (163/153/158/106/124/61).

    See ``docs/Evan's Notes/08-24-toolbench-tooleval-harness.md`` for the subset-size discrepancy
    between the paper (200 per G1/G2 subset), ToolBench's release (100), and this (the solvable
    subset) — the aggregation is weighted by ACTUAL subset size, so it is correct for whichever
    set it is given.

WHAT IS NOT REPRODUCED, AND WHY
    ToolEval scores a model's own INTERACTIVE rollout: it calls RapidAPI, feeds each observation
    back, and lets DFSDT backtrack. There is no API server in this pipeline, RapidAPI's 2023
    endpoints have largely decayed (the reason StableToolBench exists), and the paper itself states
    its evaluator "assessed solution paths without requiring live API execution". So the model here
    emits a COMPLETE path in one generation and is judged on it. ``eval/scorers/toolbench.py``
    documents what that changes about the metric; it is the single most important caveat on this
    task and it is stated there rather than here because it is a scoring property, not a data one.
"""
from __future__ import annotations

import ast
import json
import os
import re
import tarfile
from collections.abc import Iterable, Iterator
from pathlib import Path

from data.loaders.toolbench_prompt import (
    build_system_prompt,
    build_user_prompt,
    standardize,
)

# The community mirror of ToolBench's preprocessed G1+G2+G3 DFSDT training file. The canonical
# `ToolBench/ToolBench` repo 404s for unauthenticated and token-authenticated reads alike (checked
# 2026-08-24); this mirror is ungated and its train split is 187,542 rows — the paper's number.
TRAIN_REPO = "Yhyu13/ToolBench_toolllama_G123_dfs"
TRAIN_FILE = "toolllama_G123_dfs_train.json"

# StableToolBench's republished ToolEval test subsets, with `api_list` inline.
STB_BASE = (
    "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/master/solvable_queries"
    "/test_instruction"
)
# The RapidAPI tool environment, for the one field the test files omit: each tool's prose
# description, which ToolBench's system prompt lists under "You have access of the following
# tools:". 11.7 MB, cached once.
TOOLENV_REPO = "stabletoolbench/ToolEnv2404"
TOOLENV_FILE = "toolenv2404_filtered.tar.gz"

# ToolEval's six subsets, in the order the paper's Table 2 lists them.
SUBSETS = (
    "G1_instruction",
    "G1_category",
    "G1_tool",
    "G2_instruction",
    "G2_category",
    "G3_instruction",
)

# Bytes per range request, and the observed density used to size the first one. 390 objects per
# 4 MiB measured 2026-08-24, rounded down for headroom.
_RANGE_CHUNK_BYTES = 8 * 1024 * 1024
_BYTES_PER_ROW_ESTIMATE = 12_000
_PREFIX_CACHE_ENV = "SLM_TOOLBENCH_PREFIX_CACHE"

# Characters per token assumed when bounding a row against the task's context, and the reason it is
# a character bound at all: the loader runs before model selection, so no tokenizer is available, and
# the two candidate tokenizers disagree anyway.
#
# WHY A BOUND IS NEEDED. `training/lora_trainer._validate_training_sequence_lengths` and the eval
# generator both REFUSE a row that does not fit rather than truncating it — correctly, since
# truncating a ToolBench prompt removes the tail of the API list, i.e. the names the model is meant
# to choose between. But a refusal aborts the whole run, so a row that cannot fit has to be dropped
# at load time. Measured 2026-08-24 across the full eval split with both candidate tokenizers, the
# longest eval prompt is 10,195 tokens; SmolLM2-360M's position limit is 8,192, so NO context setting
# fits every row and dropping is unavoidable rather than a tuning choice.
#
# WHY 2.5. Among rows near the limit — the only ones this bound decides — the measured ratio is
# 2.59-4.02 characters per token on both tokenizers, so 2.5 has margin at the point it matters. The
# global minimum over all rows is 2.14, but that comes from short rows whose length is irrelevant
# here. At 2.5 this drops 5 of 765 eval rows (0.65%) and 14 of 5,000 train rows (0.28%). If a row
# ever did slip through, both consumers raise loudly rather than scoring a truncated prompt, so the
# failure mode of a bad estimate is a visible abort and not a silently wrong number.
_CHARS_PER_TOKEN = 2.5

_ACTION = re.compile(r"\nAction:\s*(.*?)\nAction Input:\s*(.*)\Z", re.DOTALL)
_API_BLOCK = re.compile(
    r"Specifically, you have access to the following APIs:\s*(\[.*)\Z", re.DOTALL
)
_STEP_ID = re.compile(r"^Step (\d+):\s*", re.DOTALL)


# --------------------------------------------------------------------------
# Reading a JSON array without reading all of it
# --------------------------------------------------------------------------


def iter_json_array_objects(text: str, limit: int | None = None) -> Iterator[dict]:
    """Yield objects from `text`, a JSON array that may be TRUNCATED mid-element.

    A pure function so the streaming path is testable without a 2 GB download: the interesting
    cases are a prefix that stops inside an object, one that stops inside a string containing
    ``]``, and one that is complete. Decoding stops at the first element that does not parse, which
    is the truncation point — everything before it is intact by construction, because
    ``raw_decode`` only succeeds on a complete value.
    """
    decoder = json.JSONDecoder()
    index = text.find("[")
    if index < 0:
        return
    index += 1
    count = 0
    length = len(text)
    while limit is None or count < limit:
        while index < length and text[index] in " \t\r\n,":
            index += 1
        if index >= length or text[index] == "]":
            return
        try:
            value, index = decoder.raw_decode(text, index)
        except ValueError:
            return
        if isinstance(value, dict):
            yield value
            count += 1


def _prefix_cache_path() -> Path:
    """Where the growing byte prefix of the train file lives."""
    configured = os.environ.get(_PREFIX_CACHE_ENV)
    if configured:
        root = Path(configured).expanduser()
    else:
        hf_home = os.environ.get("HF_HOME")
        root = Path(hf_home).expanduser() / "toolbench-prefix" if hf_home else (
            Path.cwd() / "artifacts" / "toolbench-prefix"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root / TRAIN_FILE


def _extend_prefix(path: Path, want_bytes: int, log=print) -> str:
    """Ensure `path` holds at least `want_bytes` of the train file, then return its text.

    Extends with a ranged GET from the current length rather than refetching, so a second call for
    a deeper slice costs only the difference. A server that ignores `Range` (HTTP 200 rather than
    206) is handled by rewriting the file from the response, which is correct but expensive — it is
    logged so the cost is visible rather than mysterious.
    """
    import requests
    from huggingface_hub import hf_hub_url

    have = path.stat().st_size if path.exists() else 0
    if have >= want_bytes:
        return path.read_text(encoding="utf-8", errors="ignore")

    url = hf_hub_url(TRAIN_REPO, TRAIN_FILE, repo_type="dataset")
    headers = {"Range": f"bytes={have}-{want_bytes - 1}"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(url, headers=headers, timeout=300, stream=True)
    response.raise_for_status()
    if response.status_code == 200 and have:
        log(f"      [toolbench] range request was ignored; rewriting the {want_bytes} byte prefix")
        have = 0
    mode = "ab" if have else "wb"
    with path.open(mode) as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            handle.write(chunk)
    log(
        f"      [toolbench] train prefix {have} -> {path.stat().st_size} bytes "
        f"(of 2,002,419,658; full download avoided)"
    )
    return path.read_text(encoding="utf-8", errors="ignore")


def load_complete_paths(
    max_train: int, log=print,
) -> tuple[list[dict], dict[str, int], int]:
    """`max_train` complete solution paths, growing the byte prefix until it has that many.

    THE POINT OF THE LOOP
        The complete-path rule keeps roughly a quarter of raw rows (measured: 274 of 1,200, the rest
        being 777 incomplete step prefixes and 138 abandoned DFSDT branches). So reading exactly
        `max_train` RAW rows returns about a quarter of what was asked for — and a short return is
        precisely how ``_reread_known_sources`` detects that a split has run out. A fixed multiplier
        has the same failure one step further along: guess low and mining retires the free rung of
        the ladder on its first call, which is the open bug B303 records for `calendar_json`.

        So the target is the KEPT count, and the next prefix size is derived from the keep rate this
        corpus actually just exhibited rather than from a constant that could drift away from it.

    Returns `(rows, drop_reasons, raw_seen)`.
    """
    path = _prefix_cache_path()
    want_bytes = max(_RANGE_CHUNK_BYTES, int(max_train * _BYTES_PER_ROW_ESTIMATE / 0.22))
    rows: list[dict] = []
    dropped: dict[str, int] = {}
    raw_seen = 0
    while True:
        text = _extend_prefix(path, want_bytes, log=log)
        raw = list(iter_json_array_objects(text))
        raw_seen = len(raw)
        rows, dropped = convert_train_rows(raw)
        if len(rows) >= max_train:
            break
        if len(text) < want_bytes:
            # The whole 2 GB has been read; there is genuinely nothing left.
            log(f"      [toolbench] the train split is exhausted at {len(rows)} complete path(s)")
            break
        keep_rate = max(len(rows) / raw_seen, 0.05) if raw_seen else 0.05
        needed_raw = int((max_train - len(rows)) / keep_rate * 1.2) + 64
        want_bytes += max(_RANGE_CHUNK_BYTES, needed_raw * _BYTES_PER_ROW_ESTIMATE)
        log(
            f"      [toolbench] {len(rows)} complete path(s) of {raw_seen} raw row(s) "
            f"({len(rows) / raw_seen:.1%} kept); growing the prefix to {want_bytes} bytes"
        )
    return rows, dropped, raw_seen


# --------------------------------------------------------------------------
# Train rows
# --------------------------------------------------------------------------


def parse_action(turn: str) -> tuple[str, str] | None:
    """`(action_name, action_input_text)` from one assistant turn, or None if it has no call."""
    match = _ACTION.search(str(turn or ""))
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def _declared_functions(system_message: str) -> list[dict] | None:
    """The function dicts a ToolBench system prompt declares.

    Read with ``ast.literal_eval`` because the prompt appends ``str(functions)`` — a Python repr
    with single quotes, not JSON (see ``data/loaders/toolbench_prompt.py``). Parsed cleanly on
    762/762 rows of the eval split when this was checked.
    """
    match = _API_BLOCK.search(system_message)
    if not match:
        return None
    try:
        functions = ast.literal_eval(match.group(1).strip())
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    if not isinstance(functions, list) or not functions:
        return None
    if not all(isinstance(f, dict) and isinstance(f.get("name"), str) for f in functions):
        return None
    return functions


def compact_tools(functions: Iterable[dict]) -> list[dict]:
    """Function schemas with prose stripped: `{name, parameters:{properties, required, optional}}`.

    The descriptions are already in the row's ``text`` — they are most of the system prompt — so
    carrying them a second time on ``tools`` would only cost. What ``tools`` is FOR is the three
    programmatic consumers: the scorer's undeclared-API check, ``verify_toolbench_row``'s schema
    check, and ``_row_context_block``, which renders the row's non-answer fields into the teacher's
    verification prompt and truncates at 6,000 characters. A full ToolBench schema list is ~5,000
    characters on its own and would routinely be clipped there; this is ~1,000 and is not.
    """
    out: list[dict] = []
    for function in functions or []:
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        parameters = function.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        properties = parameters.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        out.append({
            "name": function["name"],
            "parameters": {
                "properties": {
                    name: str((spec or {}).get("type", "string"))
                    if isinstance(spec, dict) else "string"
                    for name, spec in properties.items()
                },
                "required": [str(v) for v in (parameters.get("required") or [])],
                "optional": [str(v) for v in (parameters.get("optional") or [])],
            },
        })
    return out


def _bare_query(user_turn: str) -> str:
    """The query itself, with ToolBench's `\\n...\\nBegin!\\n` wrapper removed.

    The judge is asked "does this answer solve this query", so it must receive the query and not
    the framing — a trailing ``Begin!`` in the judged text is an instruction to the judge.
    """
    text = str(user_turn or "").strip()
    if text.endswith("Begin!"):
        text = text[: -len("Begin!")].rstrip()
    return text.strip()


def convert_train_rows(raw_rows: Iterable[dict]) -> tuple[list[dict], dict[str, int]]:
    """Shape raw ToolBench conversations into `{text, query, answer, tools}` rows.

    Returns the rows and a count of WHY rows were dropped, because on this corpus the drops are
    large and structural (roughly two thirds of rows are incomplete step prefixes) and a loader
    that reported only a final total would look like it was silently losing data.
    """
    rows: list[dict] = []
    dropped: dict[str, int] = {}

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    for raw in raw_rows:
        if not isinstance(raw, dict):
            drop("not an object")
            continue
        conversations = raw.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 3:
            drop("no conversation")
            continue
        system = next(
            (c.get("value") for c in conversations
             if isinstance(c, dict) and c.get("from") == "system"), None,
        )
        user = next(
            (c.get("value") for c in conversations
             if isinstance(c, dict) and c.get("from") == "user"), None,
        )
        assistant_turns = [
            str(c.get("value") or "") for c in conversations
            if isinstance(c, dict) and c.get("from") == "assistant"
        ]
        if not system or not user or not assistant_turns:
            drop("missing system / user / assistant turn")
            continue

        final = parse_action(assistant_turns[-1])
        if final is None:
            drop("final turn is not an Action")
            continue
        name, action_input = final
        if name != "Finish":
            # An incomplete step prefix: this trajectory continues in another row.
            drop("path does not end in Finish")
            continue
        try:
            parsed_input = json.loads(action_input) if action_input else {}
        except ValueError:
            drop("Finish input is not valid JSON")
            continue
        if not isinstance(parsed_input, dict):
            drop("Finish input is not an object")
            continue
        if parsed_input.get("return_type") != "give_answer":
            drop("path gave up rather than answering")
            continue
        if not str(parsed_input.get("final_answer") or "").strip():
            drop("give_answer with an empty final_answer")
            continue

        functions = _declared_functions(str(system))
        if functions is None:
            drop("system prompt declares no readable APIs")
            continue
        declared = {f["name"] for f in functions}
        actions = [parse_action(turn) for turn in assistant_turns]
        if any(action is None for action in actions):
            drop("an intermediate turn is not an Action")
            continue
        if any(action[0] not in declared for action in actions):
            # The ToolBench analogue of BFCL's `simple_363`: gold calls a function the prompt never
            # offered. Measured at 33 of 762 eval rows (4.3%). The scorer rejects an undeclared
            # call, so such a row is unwinnable by construction and would silently cap the ceiling
            # below 1.0 — the same reason `data/loaders/xlam_bfcl.py` drops its equivalents.
            drop("gold calls an undeclared API")
            continue

        query = _bare_query(str(user))
        if not query:
            drop("empty query")
            continue
        rows.append({
            "text": str(system) + str(user),
            "query": query,
            "answer": "\n".join(turn.strip("\n") for turn in assistant_turns),
            "tools": compact_tools(functions),
            "label": "toolbench",
        })
    return rows, dropped


# --------------------------------------------------------------------------
# Eval rows
# --------------------------------------------------------------------------


def load_tool_descriptions(log=print) -> dict[str, str]:
    """`{standardized tool name -> prose description}` from the RapidAPI environment archive.

    The test files give an ``api_list`` but no tool-level description, and ToolBench's system
    prompt lists one per tool. Without them every eval prompt would carry ``None`` where a training
    prompt carries real prose — a systematic train/serve difference in the largest block of the
    prompt. Verified against a training row: `greyhound_racing_uk` resolves to exactly the string
    that row's system prompt contains.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(TOOLENV_REPO, TOOLENV_FILE, repo_type="dataset")
    descriptions: dict[str, str] = {}
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            try:
                tool = json.load(handle)
            except (ValueError, UnicodeDecodeError):
                continue
            name = standardize(str(tool.get("tool_name") or ""))
            if name and name not in descriptions:
                descriptions[name] = str(tool.get("tool_description") or "")
    log(f"      [toolbench] {len(descriptions)} tool description(s) from {TOOLENV_FILE}")
    return descriptions


def convert_eval_rows(
    subset: str, raw_rows: Iterable[dict], tool_descriptions: dict[str, str],
) -> list[dict]:
    """Shape one ToolEval subset into rows carrying a rebuilt ToolBench prompt.

    Eval rows have NO ``answer``. That is not an omission: ToolEval's pass rate is reference-free —
    it asks a judge whether the model's own final answer addresses the query — and the test queries
    ship without a gold solution. The reference paths ToolBench distributes exist only to compute
    WIN rate, which the paper does not report.
    """
    rows: list[dict] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        query = str(raw.get("query") or "").strip()
        api_list = raw.get("api_list")
        if not query or not isinstance(api_list, list) or not api_list:
            continue
        system = build_system_prompt(api_list, tool_descriptions)
        functions = _declared_functions(system)
        if not functions:
            continue
        rows.append({
            "text": system + build_user_prompt(query),
            "query": query,
            "answer": "",
            "tools": compact_tools(functions),
            "_subset": subset,
            "_query_id": str(raw.get("query_id") or ""),
        })
    return rows


def _read_subset(subset: str) -> list[dict]:
    import requests

    response = requests.get(f"{STB_BASE}/{subset}.json", timeout=120)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []


def load_toolbench_eval(max_test: int = 1000, log=print) -> list[dict]:
    """The six ToolEval subsets, interleaved round-robin so a truncation stays representative.

    Round-robin matters for the reported metric and not only for tidiness: the overall pass rate is
    weighted by subset size, so concatenating and truncating would silently drop G3 entirely and
    report an average over five subsets under a six-subset name.
    """
    descriptions = load_tool_descriptions(log=log)
    per_subset: list[list[dict]] = []
    for subset in SUBSETS:
        try:
            raw = _read_subset(subset)
        except Exception as error:  # noqa: BLE001 — one missing subset must not kill the run
            log(f"      [toolbench] subset {subset} unavailable: "
                f"{type(error).__name__}: {error}")
            continue
        rows = convert_eval_rows(subset, raw, descriptions)
        log(f"      [toolbench] {subset}: {len(rows)} usable of {len(raw)} solvable queries")
        if rows:
            per_subset.append(rows)

    out: list[dict] = []
    index = 0
    while len(out) < max_test and any(index < len(rows) for rows in per_subset):
        for rows in per_subset:
            if index < len(rows) and len(out) < max_test:
                out.append(rows[index])
        index += 1
    return out


# --------------------------------------------------------------------------
# The entry point the spec names
# --------------------------------------------------------------------------


def context_char_budgets() -> tuple[int, int]:
    """`(eval prompt budget, train row budget)` in characters, from the task's own spec.

    The eval budget is the PROMPT budget — the context minus the output reserve, since the model has
    to generate a whole path into the same window. The train budget is the full context, because a
    training row is prompt and target together.

    Read from the spec lazily so this module does not import the registry at import time (the spec
    imports this loader), and so an `SLM_MAX_SEQ_LENGTH` override is picked up: that env var takes
    precedence over the spec inside `task_max_seq_length`, which is exactly how the first toolbench
    run died — the shared launcher body defaults it to 4096 and clamped this task's measured 8192.
    """
    from training.slm_helpers import task_max_seq_length

    from tasks import get_task

    spec = get_task("toolbench")
    context = int(task_max_seq_length("toolbench"))
    prompt_tokens = max(context - int(spec.max_new_tokens), 256)
    return (
        int(prompt_tokens * _CHARS_PER_TOKEN),
        int(context * _CHARS_PER_TOKEN),
    )


def load_toolbench(
    max_train: int = 5000, max_test: int = 1000, log=print,
) -> tuple[list[dict], list[dict]]:
    """Return `(train, test)` for the `toolbench` task.

    `max_train` is a count of COMPLETE PATHS, so asking for 5,000 returns 5,000 whenever the corpus
    has them — see `load_complete_paths` for why that distinction is what keeps the mining ladder
    working on this task.
    """
    from data.loaders.dataset_integrity import remove_normalized_train_overlap

    eval_budget, train_budget = context_char_budgets()
    train, dropped, raw_seen = load_complete_paths(max_train, log=log)
    if dropped:
        detail = ", ".join(f"{reason}: {count}" for reason, count in sorted(dropped.items()))
        log(f"      [toolbench] kept {len(train)} complete path(s) of {raw_seen} raw row(s) "
            f"({detail})")

    fitted = [
        row for row in train
        if len(row["text"]) + len(row["answer"]) <= train_budget
    ]
    if len(fitted) < len(train):
        log(f"      [toolbench] dropped {len(train) - len(fitted)} train row(s) longer than the "
            f"{train_budget}-char context bound — the trainer refuses to truncate a target rather "
            f"than silently learning a clipped path")
    train = fitted[:max_train]

    test = [
        row for row in load_toolbench_eval(max_test, log=log)
        if len(row["text"]) <= eval_budget
    ]
    if not test:
        raise RuntimeError(
            "ToolBench produced zero eval rows — refusing to proceed with an empty held-out set. "
            "Check network access to raw.githubusercontent.com for StableToolBench's "
            "solvable_queries and to the Hub for the tool environment archive."
        )
    train, removed = remove_normalized_train_overlap(train, test)
    if removed:
        log(f"      [toolbench] eval firewall removed {removed} train row(s) matching a test query")
    log(
        f"      [toolbench] {len(train)} train / {len(test)} eval row(s) "
        f"(context bounds: {train_budget} train chars, {eval_budget} eval prompt chars)"
    )
    return train, test
