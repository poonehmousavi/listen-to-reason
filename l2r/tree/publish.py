"""Publish the tree: label table of a set -> the tree files in `<checkpoints>/tree/`.

    python -m l2r.tree.publish --raw       # the tree before the LLM cleaning step        -> <checkpoints>/tree_raw/
    python -m l2r.tree.publish             # the final tree: the apply table of `l2r.tree.llm_clean`
                                           # (<checkpoints>/tree_raw/llm_groups.json) applied -> <checkpoints>/tree/

Only clips whose annotation is complete enter. The set's `index.jsonl` is rewritten with, per row, the labels that
enter the tree (`labels`: canonical leaf and node path next to the raw `fired` records); the router is trained on it.
The step checks that the tree and the table agree node by node, and writes MANIFEST.json.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json

from l2r.common import ckpt, read_jsonl, resolve, save_json, setup_run
from l2r.dataset import index as I, schema as S
from l2r.tree import build as T


def publish(name: str, groups: str | None = None, model: bool = True, log=None, out: str = "tree") -> dict:
    d = I.sdir(name)
    done = {r["done"] for r in read_jsonl(d / f"fired_{T.LALM}.jsonl") if "done" in r}
    by = collections.defaultdict(list)
    for p in sorted(d.glob("fired_*.jsonl")):
        for r in read_jsonl(p):
            if "done" not in r:
                by[r["row"]].append({k: r[k] for k in I.FIELDS if k != "row"})
    all_rows = I.rows(name)
    table = [{**r, "fired": by.get(r["row"], [])} for r in all_rows if r["clip"] in done]
    assert len({r["row"] for r in table}) == len(table), "row ids collide"
    write = lambda: (d / "index.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in table))
    write()
    schema = S.load_schema(); res = T.build(name, schema, model=model, log=log, llm_groups=groups, out=out)
    node_clips = collections.defaultdict(set)                   # every row carries its clean labels: canonical leaf + node path
    for r in table:
        r["labels"] = [l for l in T.clean_labels(r["fired"], res) if l["attr"] not in res.get("pruned", set())]
        for l in r["labels"]:
            node_clips[l["node"]].add(r["clip"])
            base = "/".join(l["node"].split("/")[:3])
            for g in l.get("generic", []):                      # a generic leaf above the served one counts the clip too
                node_clips[f"{base}/{g}"].add(r["clip"])
            if l["leaf"]:
                node_clips[base].add(r["clip"])
    write()
    tree = res["tree"]
    for reg, A in tree.items():                                 # the tree and the table agree, node by node
        for at, P in A.items():
            for v, q in P.items():
                assert len(node_clips[f"{reg}/{at}/{v}"]) == q["clips"], (reg, at, v, len(node_clips[f"{reg}/{at}/{v}"]), q["clips"])
                for l, ql in q["leaves"].items():
                    assert len(node_clips[f"{reg}/{at}/{v}/{l}"]) == ql["clips"], (reg, at, v, l, len(node_clips[f"{reg}/{at}/{v}/{l}"]), ql["clips"])
    man = {"built": datetime.datetime.now().isoformat(timespec="seconds"), "set": name,
           "clips_annotated": len(done), "clips_total": len({r["clip"] for r in all_rows}), "rows": len(table),
           "sources": dict(collections.Counter(r["source"] for r in table if r["grid"] == "clip")), "llm_groups": bool(groups),
           "attributes": sum(len(R) for R in tree.values()), "classes": sum(len(A) for R in tree.values() for A in R.values()),
           "leaves": sum(len(v["leaves"]) for R in tree.values() for A in R.values() for v in A.values()),
           "refused": dict(collections.Counter(r["why"] for r in res["refused"])), "vote": {k: v for k, v in res["stat"].items() if not k.startswith("refused")}}
    save_json(man, ckpt(out, "MANIFEST.json"), indent=1)
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default="pool"); ap.add_argument("--raw", action="store_true", help="the tree before the LLM cleaning step")
    ap.add_argument("--groups", default="", help="apply table (default: <checkpoints>/tree_raw/llm_groups.json)")
    ap.add_argument("--light", action="store_true", help="skip the sentence-embedding synonym merge")
    a = ap.parse_args(); _, log = setup_run("tree_publish")
    if a.raw:
        man = publish(a.set, None, model=not a.light, log=log, out="tree_raw")
    else:
        groups = resolve(a.groups) if a.groups else ckpt("tree_raw", "llm_groups.json")
        assert groups.exists(), f"{groups} is missing: run `python -m l2r.tree.publish --raw` and `python -m l2r.tree.llm_clean` first"
        man = publish(a.set, str(groups), model=not a.light, log=log)
    print(json.dumps(man, indent=1))
