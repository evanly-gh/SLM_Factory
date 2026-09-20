"""Freeze the on-device SFT suite's non-Hub corpora into checksummed `data/local/` bundles.

WHY THESE TWO ARE NOT IN `scripts/download_datasets.py`
    That script's catalog is `load_dataset`-shaped: an id, a config, and a split name. Two of the
    suite's corpora are not that shape and cannot be made into it without inventing a loader
    mechanism for a catalog of one entry each:

        bea19       a tar.gz off a university web server, containing eight M2 annotation files
        multiconer  raw `.conll` files in a Hub repo, read as text rather than as a dataset

    `dialogsum` IS in that catalog, because the original release's JSONL fits the existing `json`
    loader path. Same reasoning as `scripts/vendor_sms_spam.py`, which exists for the same kind of
    mismatch.

WHY VENDOR AT ALL, WHEN BOTH LOADERS CAN FETCH
    Because a compute node reaching out to `cl.cam.ac.uk` or a Hub CDN in the middle of a
    seven-day run is a failure mode with no upside. This repo already learned it the expensive way:
    `tner/bc5cdr` is script-based and DEAD under datasets 4.3.0 and `spyysalo/bc5cdr` was removed
    from the Hub outright, which is why `data/local/bc5cdr` exists. A frozen, checksummed copy also
    means two runs weeks apart provably trained on the same rows.

THE SHAPE ON DISK
    The loaders read their SOURCE format from the bundle directory, not a normalized JSONL:
    `data/local/bea19/m2/*.m2` and `data/local/multiconer/EN-English/*.conll`. That is deliberate —
    the loaders' parsers are the tested code path, and a bundle in a second format would mean a
    second parser and two ways to disagree about a row. `checksums.sha256` is written beside them.

USAGE
    python scripts/vendor_suite_datasets.py                 # both
    python scripts/vendor_suite_datasets.py --only bea19
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_ROOT = os.path.join(PROJ, "data", "local")
SCHEMA_VERSION = 2


def _log(*parts) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_integrity(bundle: str, manifest: dict) -> None:
    """Write `manifest.json` and a checksum sidecar over every file in the bundle.

    The sidecar format matches `dataset_integrity.write_checksum_sidecar`: 64 hex characters, two
    spaces, then the bundle-relative path.
    """
    files = sorted(
        os.path.relpath(os.path.join(root, name), bundle)
        for root, _dirs, names in os.walk(bundle)
        for name in names
        if name not in ("manifest.json", "checksums.sha256")
    )
    manifest["integrity"] = {
        "algorithm": "sha256",
        "checksum_file": "checksums.sha256",
        "files": {name: _sha256(os.path.join(bundle, name)) for name in files},
    }
    with open(os.path.join(bundle, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    lines = [f"{_sha256(os.path.join(bundle, name))}  {name}" for name in files]
    lines.append(f"{_sha256(os.path.join(bundle, 'manifest.json'))}  manifest.json")
    with open(os.path.join(bundle, "checksums.sha256"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    _log(f"  wrote manifest.json and checksums.sha256 over {len(files)} file(s)")


def _fresh(bundle: str) -> str:
    """An empty bundle directory, replacing any previous one atomically enough to be safe.

    The old directory is moved aside and deleted only after the new one is in place, so an
    interrupted run cannot leave a half-populated bundle that a loader would happily read.
    """
    staging = bundle + ".new"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    return staging


def _promote(staging: str, bundle: str) -> None:
    previous = bundle + ".old"
    shutil.rmtree(previous, ignore_errors=True)
    if os.path.exists(bundle):
        os.rename(bundle, previous)
    os.rename(staging, bundle)
    shutil.rmtree(previous, ignore_errors=True)


def vendor_bea19() -> None:
    """W&I+LOCNESS: download the release tarball and keep the eight per-level M2 files.

    Per-LEVEL files, not the combined `ABC.train` / `ABCN.dev`. The CEFR level is this task's
    episode axis and it is recoverable only from the file names — and the per-level files
    reassemble the combined ones exactly (10,493 + 13,032 + 10,783 = 34,308 on train;
    1,037 + 1,290 + 1,069 + 988 = 4,384 on dev), so nothing is lost by keeping only them.
    """
    from data.loaders.gec_bea19 import BEA19_URL, EXPECTED_DEV, EXPECTED_TRAIN, M2_FILES, parse_m2

    bundle = os.path.join(LOCAL_ROOT, "bea19")
    staging = _fresh(bundle)
    _log(f"bea19: fetching {BEA19_URL}")
    with tempfile.TemporaryDirectory(prefix="bea19-vendor-") as tmp:
        archive = os.path.join(tmp, "wi_locness.tar.gz")
        urllib.request.urlretrieve(BEA19_URL, archive)  # noqa: S310 - pinned https URL
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        extracted = os.path.join(tmp, "wi+locness")
        os.makedirs(os.path.join(staging, "m2"))
        counts: dict[str, int] = {}
        for level, paths in sorted(M2_FILES.items()):
            for path in paths:
                if not path:
                    continue
                source = os.path.join(extracted, path)
                if not os.path.exists(source):
                    raise RuntimeError(f"bea19: the release is missing {path}")
                shutil.copy2(source, os.path.join(staging, path))
                with open(source, encoding="utf-8") as handle:
                    counts[path] = len(parse_m2(handle.read()))
        # Licence files travel with the data. This corpus is research/educational only, and a
        # bundle that dropped its licence would make that invisible to the next reader.
        for name in ("licence.wi.txt", "license.locness.txt", "readme.txt"):
            source = os.path.join(extracted, name)
            if os.path.exists(source):
                shutil.copy2(source, os.path.join(staging, name))

    train = sum(n for path, n in counts.items() if ".train." in path)
    dev = sum(n for path, n in counts.items() if ".dev." in path)
    _log(f"  bea19: train={train} dev={dev} across {len(counts)} M2 file(s)")
    if (train, dev) != (EXPECTED_TRAIN, EXPECTED_DEV):
        raise RuntimeError(
            f"bea19: got train={train} dev={dev}, expected {EXPECTED_TRAIN}/{EXPECTED_DEV}. "
            "Refusing to freeze an incomplete bundle."
        )
    _write_integrity(staging, {
        "schema_version": SCHEMA_VERSION,
        "name": "bea19",
        "task": "gec_bea19",
        "source_url": BEA19_URL,
        "format": "m2",
        "counts": {"train": train, "dev": dev, "by_file": counts},
        "licence": "research/educational only (W&I and LOCNESS licences included in the bundle)",
        "notes": (
            "Per-CEFR-level M2 files, which is where the A/B/C/N labels come from. The BEA-2019 "
            "TEST split is blind (CodaLab only) and is deliberately not vendored: it cannot be "
            "scored locally, so dev is what this task reports."
        ),
    })
    _promote(staging, bundle)
    _log(f"  bea19: {bundle}")


def vendor_multiconer() -> None:
    """MultiCoNER II English: keep the three raw `.conll` files.

    English only. The other 11 languages are the cross-lingual episode axis and are ~500 MB; this
    suite's model pool is scored on English on-device workloads, so vendoring them would be paying
    disk for a dimension nothing here measures.
    """
    from huggingface_hub import hf_hub_download

    from data.loaders.multiconer import (
        CONLL_FILES,
        ENTITY_TYPES,
        MULTICONER_ID,
        parse_conll,
        to_rows,
    )

    bundle = os.path.join(LOCAL_ROOT, "multiconer")
    staging = _fresh(bundle)
    os.makedirs(os.path.join(staging, "EN-English"))
    counts: dict[str, int] = {}
    types: set[str] = set()
    for split, path in sorted(CONLL_FILES.items()):
        _log(f"multiconer: fetching {path}")
        cached = hf_hub_download(MULTICONER_ID, path, repo_type="dataset")
        shutil.copy2(cached, os.path.join(staging, path))
        with open(cached, encoding="utf-8") as handle:
            rows = to_rows(parse_conll(handle.read()))
        counts[split] = len(rows)
        types.update(entity["type"] for row in rows for entity in row["entities"])
        _log(f"  {split}: {len(rows)} sentence(s)")

    missing = sorted(set(ENTITY_TYPES) - types)
    if missing:
        raise RuntimeError(
            f"multiconer: {len(missing)} of the 33 declared types never appear in the vendored "
            f"files ({missing[:5]}...). Either the release changed or COARSE_GROUPS is wrong; "
            "refusing to freeze a bundle the scorer cannot score against."
        )
    _write_integrity(staging, {
        "schema_version": SCHEMA_VERSION,
        "name": "multiconer",
        "task": "multiconer",
        "hf_id": MULTICONER_ID,
        "url": f"https://huggingface.co/datasets/{MULTICONER_ID}",
        "format": "conll",
        "counts": counts,
        "entity_types": sorted(types),
        "licence": "cc-by-4.0",
        "notes": (
            "English only; the other 11 languages are the cross-lingual episode axis and are not "
            "vendored. The test split carries gold labels but NO clean/corrupted marker — every "
            "test sentence header is `# id <uuid>` with none of the `domain=en` attribute train "
            "and dev have — so the robustness gap is not derivable from this release."
        ),
    })
    _promote(staging, bundle)
    _log(f"  multiconer: {bundle}")


VENDORS = {"bea19": vendor_bea19, "multiconer": vendor_multiconer}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--only", nargs="*", choices=sorted(VENDORS), default=sorted(VENDORS),
        help="which bundles to build; default is all of them",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HOME", "/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.makedirs(LOCAL_ROOT, exist_ok=True)
    for name in args.only:
        VENDORS[name]()
    _log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
