"""Shared utilities: configuration, repository-relative paths, logging, seeding and run manifests."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = os.environ.get("L2R_CONFIG", "configs/config.yaml")


def resolve(path: str | os.PathLike) -> Path:
    """A path relative to the repository root (absolute paths are returned unchanged)."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


@lru_cache(maxsize=None)
def _load(path: str) -> dict:
    with open(resolve(path)) as f:
        return yaml.safe_load(f)


def load_config(path: str | None = None) -> dict[str, Any]:
    """The YAML configuration (`configs/config.yaml`, or the file named by $L2R_CONFIG)."""
    return _load(str(path or CONFIG))


def _dir(kind: str, *parts: str) -> Path:
    return resolve(load_config()["paths"][kind]).joinpath(*parts)


def data(*parts: str) -> Path:
    """A path under the data folder (datasets and benchmarks; see the README for its layout)."""
    return _dir("data", *parts)


def work(*parts: str) -> Path:
    """A path under the work folder (everything the pipeline generates). Parent folders are created."""
    p = _dir("work", *parts); p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ckpt(*parts: str) -> Path:
    """A path under the checkpoint folder (the tree, the trained heads, downloaded encoder weights)."""
    p = _dir("checkpoints", *parts); p.parent.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "numpy"):
        mod = sys.modules.get(name)
        if mod is not None:
            out[name] = getattr(mod, "__version__", "?")
    return out


def setup_run(step: str, cfg: dict | None = None) -> tuple[Path, logging.Logger]:
    """Create `<work>/runs/<step>/`, seed the RNGs, write a manifest and return (run_dir, logger)."""
    cfg = cfg or load_config()
    run_dir = work("runs", step, "log.txt").parent
    set_seed(cfg.get("seed", 0))
    log = logging.getLogger(step)
    log.setLevel(logging.INFO)
    log.propagate = False
    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "log.txt")):
            h.setFormatter(fmt)
            log.addHandler(h)
    manifest = {"step": step, "time_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv, "versions": _versions(), "config": cfg}
    save_json(manifest, run_dir / "manifest.json")
    log.info("=== %s ===", step)
    return run_dir, log


def save_json(obj: Any, path: str | os.PathLike, indent: int | None = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path: str | os.PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    """Tolerant JSON-lines reader: a half-written last line (an interrupted job) is skipped."""
    out = []
    p = Path(path)
    for line in (p.read_text().splitlines() if p.exists() else []):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


class CheckError(AssertionError):
    """A validity check of a pipeline step failed."""


def check(condition: bool, message: str, log: logging.Logger | None = None) -> None:
    """Assert a validity condition of a step and log the outcome."""
    if condition:
        if log:
            log.info("  [check ok] %s", message)
        return
    if log:
        log.error("  [check FAILED] %s", message)
    raise CheckError(message)


def audio(path: str | os.PathLike) -> Path:
    """An audio file named in a manifest: paths are stored relative to the data folder."""
    p = Path(path)
    return p if p.is_absolute() else data(str(p))
