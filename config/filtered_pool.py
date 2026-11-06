# config/filtered_pool.py
"""
The phone-measured filtered pool: which deployment variants a run may use, and which runtime each
one is built for.

WHERE IT COMES FROM
    `filter_pool.py` (the on-phone half of the factory) benchmarks every pool model × bit-width on
    llama.cpp and MNN, CPU and GPU, against the operator's memory / prefill / decode / energy
    limits, and writes a contract-v1 JSON. Its `filtered_pool` lists the kept model × bit-width
    rows, each with the runtime and backend that won. Everything else in the file (`entries`,
    `summary`, ...) explains removals and is not read here.

ON BY DEFAULT
    The contract is read from `config/filtered_pool.json` whenever that file exists — copy or
    symlink filter_pool's `results/filter_<profile>_filtered_pool.json` there. `SLM_FILTERED_POOL`
    points at a different file (another phone, a Slurm script) or, set to `off`, disables it. With
    no contract the run uses the full ANDROID_POOL and the run-wide `SLM_QUANT_BACKEND`, exactly as
    before this existed.

WHAT IT CHANGES
    * `config.android_pool.filter_pool` drops every variant the contract did not keep. That is the
      one function cold start, escalation and the downward probe all draw candidates from, so a
      removed variant can never be selected, and a tier whose variants were all removed is simply
      empty — escalation already steps past empty tiers to the nearest non-empty one.
    * `training.quant_backend.backend_for_model` returns the contract's runtime for a kept variant,
      so each variant is quantized and scored on the runtime that won on the phone. One run can
      therefore mix runtimes across tiers; their artifact caches live in separate trees.

    The pipeline's quant recipe is unchanged: a contract `bits: 4` row maps onto the pool's
    `@Q4_K_M` variant even though filter_pool benchmarked llama.cpp at Q4_0. Only the runtime
    moves. The contract's cpu/gpu `backend` is recorded for the report; scoring runs on the cluster.

Deliberately free of `config.config` imports: `training/quant_backend.py` reads this from CUDA
workers that hold no API key.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from dataclasses import dataclass, field

CONTRACT_VERSION = 1
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "filtered_pool.json")
ENV_VAR = "SLM_FILTERED_POOL"
_DISABLED = {"off", "0", "none", "false", "no"}

# Contract runtime names → training/quant_backend.py names.
_RUNTIMES = {"llama.cpp": "llama_cpp", "mnn": "mnn"}
_BACKENDS = ("cpu", "gpu")
# Contract bit-width → ANDROID_POOL quant field (None = the BF16 base variant).
_BITS_TO_QUANT = {16: None, 8: "Q8_0", 4: "Q4_K_M"}


@dataclass(frozen=True)
class FilteredPool:
    path: str
    sha256: str
    complete: bool | None
    not_measured: tuple[str, ...]
    phone: dict
    # selector → {"runtime", "backend", "format", "contract_quant", "model"}
    deployments: dict = field(default_factory=dict)

    def allows(self, model) -> bool:
        return model.selector in self.deployments

    def deployment_for(self, model_id: str, quant: str | None) -> dict | None:
        return self.deployments.get(f"{model_id}@{quant or 'bf16'}")

    def runtime_for(self, model_id: str, quant: str | None) -> str | None:
        deployment = self.deployment_for(model_id, quant)
        return deployment["runtime"] if deployment else None

    def describe(self) -> str:
        phone = self.phone.get("label") or self.phone.get("model") or self.phone.get("serial") or "?"
        return f"{self.path} (phone {phone}, {len(self.deployments)} kept variants)"


def contract_path() -> str | None:
    """The contract this process uses: `SLM_FILTERED_POOL`, else the default file if it exists."""
    value = os.environ.get(ENV_VAR, "").strip()
    if value.lower() in _DISABLED:
        return None
    if value:
        return os.path.abspath(value)
    return DEFAULT_PATH if os.path.isfile(DEFAULT_PATH) else None


def _match_pool_model(row: dict, pool) -> str:
    """The ANDROID_POOL model_id a contract row names, by hf_id, else by the `model` slug."""
    model_ids = {spec.model_id for spec in pool}
    hf_id = row.get("hf_id")
    if hf_id in model_ids:
        return hf_id
    slug = str(row.get("model") or "").strip().lower()
    by_slug = [mid for mid in model_ids if mid.rsplit("/", 1)[-1].lower() == slug]
    if len(by_slug) == 1:
        return by_slug[0]
    raise ValueError(
        f"filtered pool row {row.get('model')!r} (hf_id={hf_id!r}) matches no ANDROID_POOL model; "
        f"known: {sorted(model_ids)}"
    )


@functools.lru_cache(maxsize=None)
def load(path: str) -> FilteredPool:
    """Parse and validate a contract-v1 file and map its kept rows onto ANDROID_POOL selectors."""
    from config.android_pool import ANDROID_POOL

    with open(path, "rb") as fh:
        raw = fh.read()
    data = json.loads(raw)
    if data.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(
            f"{path}: contract_version {data.get('contract_version')!r} is not supported "
            f"(expected {CONTRACT_VERSION})"
        )
    rows = data.get("filtered_pool")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: `filtered_pool` must be a list")

    selectors = {spec.selector for spec in ANDROID_POOL}
    deployments: dict[str, dict] = {}
    for row in rows:
        runtime = _RUNTIMES.get(row.get("runtime"))
        if runtime is None:
            raise ValueError(f"{path}: unknown runtime {row.get('runtime')!r} in {row}")
        if row.get("backend") not in _BACKENDS:
            raise ValueError(f"{path}: unknown backend {row.get('backend')!r} in {row}")
        if row.get("bits") not in _BITS_TO_QUANT:
            raise ValueError(f"{path}: unsupported bits {row.get('bits')!r} in {row}")
        model_id = _match_pool_model(row, ANDROID_POOL)
        selector = f"{model_id}@{_BITS_TO_QUANT[row['bits']] or 'bf16'}"
        if selector not in selectors:
            raise ValueError(f"{path}: {selector} is not an ANDROID_POOL variant")
        if selector in deployments:
            raise ValueError(f"{path}: {selector} appears twice in filtered_pool")
        deployments[selector] = {
            "model": row.get("model"),
            "runtime": runtime,
            "backend": row["backend"],
            "format": row.get("format"),
            "contract_quant": row.get("quant"),
        }

    pool = FilteredPool(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        complete=data.get("complete"),
        not_measured=tuple(data.get("not_measured") or ()),
        phone=data.get("phone") or {},
        deployments=deployments,
    )
    if pool.complete is not True:
        message = (
            f"[filtered_pool] WARNING: {path} has complete={pool.complete!r} — the pool may not be "
            f"final. Using filtered_pool as given; {len(pool.not_measured)} config(s) not measured"
            + (f": {', '.join(pool.not_measured)}" if pool.not_measured else "")
        )
        print(message)
    return pool


def active() -> FilteredPool | None:
    """The contract in force for this process, or None when the run uses the full pool."""
    path = contract_path()
    return load(path) if path else None
