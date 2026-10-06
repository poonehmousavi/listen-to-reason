"""Retrieval: transcribe, route every chunk into the tree, detect the event timeline, write the reader's context.

    python -m l2r.retrieve                              # MMAU, MMAR, SAKURA
    python -m l2r.retrieve --benchmarks mmaupro
    python -m l2r.retrieve --benchmarks birds --domains birds_k5

One stage per benchmark (`python -m l2r.retrieve.context --benchmark <name> --run`), which runs the steps whose cache is
missing: transcripts (<work>/asr/), the question classifier (<work>/question/), the traversal (<work>/nodes/), the event
timeline (<work>/events/); the context goes to <work>/contexts/<benchmark>.json.
"""
from __future__ import annotations

from l2r.pipeline import Step, parser, run
from l2r.retrieve.context import context_path

if __name__ == "__main__":
    ap = parser("retrieve", __doc__); ap.add_argument("--benchmarks", default="mmau,mmar,sakura")
    ap.add_argument("--domains", default="", help="new-domain routers to add to the traversal, e.g. birds_k5")
    ap.add_argument("--tree-only", action="store_true", help="also write the context without the transcript (<benchmark>_tree_only.json)")
    a = ap.parse_args(); opt = (["--domains", a.domains] if a.domains else []) + (["--tree-only"] if a.tree_only else [])
    run([Step(b, ["l2r.retrieve.context", "--benchmark", b, "--run"] + opt, [context_path(b)] + ([context_path(b, "tree_only")] if a.tree_only else []))
         for b in a.benchmarks.split(",")], a)
