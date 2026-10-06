"""The served tree: every node the trained router can serve, with the encoder that reads each attribute.

The annotator tree (`<checkpoints>/tree/tree.json`) lists every admitted leaf; a leaf becomes a node of a head only
with enough training clips. This module reads the trained heads and prints the tree as the paper reports it
(attributes, classes and leaves per region):

    python -m l2r.tree.summary [--json <work>/reports/served_tree.json]
    python -m l2r.tree.summary --annotator-tree          # the annotator tree instead: every admitted leaf

Sources: the region routers (`<checkpoints>/routers/`), the transcript language classifier, the identity
attributes of environmental sound (`<checkpoints>/sound/`) and the music-property heads.
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np

from l2r.common import ckpt, save_json
from l2r.dataset.schema import GATE

ROUTERS = ("sound", "music_clap", "music_muq", "speech_whisper", "speech_emotion", "speech_gender", "music_labelled")   # a later router owns a shared attribute
REPLACED = {"sound.event", "sound.activity"}        # annotator attributes replaced by the AudioSet identity attributes
SOUND_ENCODERS = ("clap", "beats")


def served_tree(routers=ROUTERS) -> dict:
    """{attribute: {"encoder": name, "classes": {class: [leaves]}}}"""
    import torch
    attrs = {}
    for n in routers:
        ck = torch.load(ckpt("routers", f"{n}.pt"), weights_only=False, map_location="cpu")
        for a, sp in ck["space"].items():
            if a in REPLACED or a == GATE:
                continue
            leaves = {}
            for v, l in ck["classes"].get(a, []):
                if l:
                    leaves.setdefault(v, []).append(l)
            attrs[a] = {"encoder": ck["encoder"], "classes": {v: sorted(leaves.get(v, [])) for v in sp["values"]}}
    lang = pickle.load(open(ckpt("language", "text_head.pkl"), "rb")); by = {}
    for leaf in lang["classes"]:
        by.setdefault(lang["leaf2val"].get(leaf, "other"), []).append(leaf)
    attrs["speech.language"] = {"encoder": "transcript text classifier", "classes": {v: sorted(ls) for v, ls in by.items()}}
    nodes = json.load(open(ckpt("sound", "nodes.json")))
    res = {t: json.load(open(ckpt("sound", f"result_{t}.json"))) for t in SOUND_ENCODERS}
    for a, pairs in nodes.items():                     # each identity attribute is read by the encoder with the higher held-out mAP
        own = max(res, key=lambda t: np.nanmean([res[t]["test"].get(a, np.nan), res[t]["fsd_eval"].get(a, np.nan)]))
        d = {}
        for v, l in pairs:
            d.setdefault(v, [])
            if l:
                d[v].append(l)
        attrs[a] = {"encoder": own, "classes": {v: sorted(ls) for v, ls in d.items()}}
    for field, head in torch.load(ckpt("music_properties", "heads.pt"), weights_only=False, map_location="cpu").items():
        a = field.replace("music_", "music.", 1)
        if a not in attrs:
            attrs[a] = {"encoder": "muq_mulan", "classes": {x: [] for x in head["values"]}}
    return dict(sorted(attrs.items()))


def annotator_tree() -> dict:
    """The same table for the annotator tree (`<checkpoints>/tree/tree.json`): every admitted leaf, before the heads' clip threshold."""
    tree = json.load(open(ckpt("tree", "tree.json")))
    return {f"{r}.{a}": {"encoder": "", "classes": {v: sorted(q["leaves"]) for v, q in V.items()}} for r, A in sorted(tree.items()) for a, V in sorted(A.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--json", default="", help="also write the table to this file")
    ap.add_argument("--annotator-tree", action="store_true", help="count the annotator tree (every admitted leaf) instead of the served heads")
    a = ap.parse_args(); attrs = annotator_tree() if a.annotator_tree else served_tree(); tot = [0, 0, 0]
    print(f"{'attribute':34s} {'classes':>7s} {'leaves':>6s}  encoder")
    for region in ("speech", "music", "sound") + (("cross",) if a.annotator_tree else ()):
        A = {k: v for k, v in attrs.items() if k.startswith(region + ".")}
        nc = sum(len(v["classes"]) for v in A.values()); nl = sum(len(l) for v in A.values() for l in v["classes"].values())
        for k, v in A.items():
            print(f"{k:34s} {len(v['classes']):7d} {sum(len(l) for l in v['classes'].values()):6d}  {v['encoder']}")
        print(f"{'  ' + region + ': ' + str(len(A)) + ' attributes':34s} {nc:7d} {nl:6d}\n"); tot = [tot[0] + len(A), tot[1] + nc, tot[2] + nl]
    print(f"total: {tot[0]} attributes, {tot[1]} classes, {tot[2]} leaves")
    if a.json:
        save_json(attrs, a.json, indent=1)


if __name__ == "__main__":
    main()
