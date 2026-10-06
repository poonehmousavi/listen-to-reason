"""The traversal trace of a clip: every served node with where in the clip it was heard and how confident the router was.

    python -m l2r.retrieve.trace --benchmark mmau --clip MMAU/audio/<id>.wav

The reader does not see the trace. It is for a person: it shows where each node of the context came from, so a
wrong answer can be located in a node.

    sound animal: dog (Bark)             whole clip          p 0.93
    speech emotion: happy                0-3 s, 6-10.5 s     p 0.71
"""
from __future__ import annotations

import argparse
import collections

from l2r.common import load_json
from l2r.retrieve.context import clause_head, node_text
from l2r.retrieve.serve import nodes_path


def spans(rows: list[dict]) -> list[tuple[float, float]]:
    """Merge the (overlapping) chunks a node was served on into time spans."""
    out: list[list[float]] = []
    for t0, t1 in sorted((r["t0"], r["t1"]) for r in rows):
        if out and t0 <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], t1)
        else:
            out.append([t0, t1])
    return [(a, b) for a, b in out]


def trace(rows: list[dict]) -> list[dict]:
    """rows of one clip (from the nodes file) -> [{attr, node, where, p}], in the order the nodes were reached."""
    seen = collections.OrderedDict()
    for r in rows:
        for n in r["nodes"]:
            seen.setdefault((n["attr"], n["value"], n["leaf"]), []).append((r, n))
    out = []
    for (attr, _, _), hits in seen.items():
        chunks = [r for r, _ in hits if r["grid"] != "clip"]
        where = ", ".join(f"{a:g}-{b:g} s" for a, b in spans(chunks)) if chunks and len(chunks) == len(hits) else "whole clip"
        ps = [n["p"] for _, n in hits if "p" in n]
        out.append({"attr": attr, "node": node_text(hits[0][1]), "where": where, "p": max(ps) if ps else None})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", required=True); ap.add_argument("--clip", required=True, help="audio path as in the benchmark items")
    a = ap.parse_args()
    clips = load_json(nodes_path(a.benchmark))["clips"]
    assert a.clip in clips, f"{a.clip} is not a clip of {a.benchmark}"
    for t in trace(clips[a.clip]):
        print(f"{clause_head(t['attr']) + ': ' + t['node']:<60s} {t['where']:<28s} " + (f"p {t['p']:.2f}" if t["p"] is not None else ""))


if __name__ == "__main__":
    main()
