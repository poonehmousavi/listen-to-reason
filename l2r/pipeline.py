"""One command per part of the pipeline: run its stages in order and skip the ones whose output exists.

    python -m l2r.dataset      python -m l2r.tree      python -m l2r.router
    python -m l2r.retrieve     python -m l2r.reason    python -m l2r.adapt

Every part accepts --list (show the stages and whether they are done), --from STAGE, --only STAGE and --force
(re-run stages whose output exists). Each stage is a module of its own with the same name in the stage table, so
one stage can also be run alone with its own options (`python -m l2r.<part>.<module> --help`).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Step:
    name: str
    cmd: list[str]                                  # module and arguments, run as `python -m <cmd>`
    outputs: list[Path] = field(default_factory=list)   # the stage is done when every output exists ...
    done: Callable[[], bool] | None = None          # ... or when this says so
    before: Callable[[], None] | None = None        # run before the command (e.g. remove a stale cache)

    def is_done(self) -> bool:
        return self.done() if self.done else bool(self.outputs) and all(p.exists() for p in self.outputs)


def parser(part: str, doc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=f"python -m l2r.{part}", description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show the stages and their state, run nothing")
    ap.add_argument("--from", dest="start", default="", metavar="STAGE", help="start at this stage (skip the earlier ones)")
    ap.add_argument("--only", default="", metavar="STAGE", help="run this stage alone")
    ap.add_argument("--force", action="store_true", help="re-run stages whose output exists")
    return ap


def run(steps: list[Step], a: argparse.Namespace) -> None:
    names = [s.name for s in steps]
    for x in (a.start, a.only):
        assert not x or x in names, f"unknown stage {x!r}; stages: {', '.join(names)}"
    if a.list:
        for s in steps:
            print(f"{'done' if s.is_done() else '    '}  {s.name:18s} python -m {' '.join(s.cmd)}")
        return
    todo = [s for s in steps if (not a.only or s.name == a.only) and (not a.start or names.index(s.name) >= names.index(a.start))]
    for s in todo:
        if s.is_done() and not a.force:
            print(f"[{s.name}] done, skipped", flush=True)
            continue
        if s.before:
            s.before()
        print(f"[{s.name}] python -m {' '.join(s.cmd)}", flush=True); t = time.time()
        r = subprocess.run([sys.executable, "-m", *s.cmd])
        if r.returncode:
            sys.exit(f"[{s.name}] failed (exit {r.returncode}); fix and re-run with --from {s.name}")
        print(f"[{s.name}] finished in {(time.time() - t) / 60:.1f} min", flush=True)
