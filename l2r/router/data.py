"""Router training data: a set's label table -> (chunk, attribute) pairs.

One chunk carries labels for several attributes (a 3 s chunk is speech and sound; inside speech it has a gender,
an emotion and a style), and a multiple-choice attribute carries several classes at once. The training unit is
therefore the (chunk, attribute) pair: `pairs.jsonl` holds one record per pair with its classes and, where the
tree admits them, its leaves; `space.json` holds the label space (attribute -> region, grid, scope, classes).

An attribute decided once per clip (`scope: constant`) is labelled on the clip row, which has no audio of its
own, so its label is copied to every chunk of that clip on the attribute's grid. Clips are split into train and
held-out by a hash of the clip id, so all chunks of one recording fall on one side.

    python -m l2r.router.data --set pool
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json

from l2r.common import check, ckpt, read_jsonl, save_json, setup_run
from l2r.dataset import index as I, schema as S, segment as G

PAIRS = "pairs.jsonl"
SPACE = "space.json"


def space(schema: dict) -> dict[str, dict]:
    """attribute -> {region, grid, scope, multi, values}: the closed part of the tree a head is trained on."""
    out = {}
    for region in schema["regions"]:
        for name, a in S.fields(schema, region):
            out[f"{region}.{name}"] = {"region": region, "grid": G.GRID_OF[region], "scope": a.get("scope", "clip"),
                                       "multi": bool(a.get("multi")), "values": [v.lower() for v in a["values"]]}
    return out


def canon_leaves(tree: dict) -> dict[tuple[str, str], dict[str, str]]:
    """(attribute, class) -> {surface form: leaf of the tree}. A raw leaf absent here was not admitted to the
    tree, and its pair trains the class only."""
    out = collections.defaultdict(dict)
    for region, attrs in tree.items():
        for a, classes in attrs.items():
            for value, node in classes.items():
                for leaf, q in node.get("leaves", {}).items():
                    for form in [leaf, *q.get("forms", [])]:
                        out[(f"{region}.{a}", value)][form] = leaf
    return dict(out)


def split_of(clip: str, hold: float = 0.3) -> str:
    """Held out by clip: every chunk of one recording lands on one side."""
    return "test" if int(hashlib.md5(clip.encode()).hexdigest()[:8], 16) % 1000 < hold * 1000 else "train"


def load_tree() -> dict | None:
    p = ckpt("tree", "tree.json")
    return json.loads(p.read_text()) if p.exists() else None


def pairs(name: str, schema: dict, tree: dict | None) -> tuple[list[dict], collections.Counter]:
    """The label table of a set -> one record per (chunk, attribute) that carries a label.

    Each row's labels are first merged over the annotators by the tree's vote rule. Leaves are mapped to the
    tree's leaves; an attribute the tree does not hold is not trained."""
    from l2r.tree.build import vote
    table = read_jsonl(I.sdir(name) / "index.jsonl") or I.assemble(name)
    canon = canon_leaves(tree) if tree else {}
    tree_attrs = {f"{r}.{a}" for r, attrs in tree.items() for a in attrs} if tree else None
    sp = space(schema)
    stat = collections.Counter()

    rows, voted = {}, {}
    for r in table:
        rows[r["row"]] = r
        labs, _ = vote(r["fired"])
        if tree_attrs is not None:
            n0 = len(labs)
            labs = [l for l in labs if l["attr"] in tree_attrs]
            stat["label: attribute not in the tree"] += n0 - len(labs)
        voted[r["row"]] = labs
        stat["rows"] += 1

    by_clip_grid = collections.defaultdict(list)
    for r in table:
        if r["grid"] != "clip":
            by_clip_grid[(r["clip"], r["grid"])].append(r["row"])

    acc = collections.defaultdict(dict)                        # (row, attribute) -> {class: (leaf, where the label came from)}
    for row, labs in voted.items():
        r = rows[row]
        for l in labs:
            attr = l["attr"]
            s = sp.get(attr)
            if s is None:
                stat["label: attribute not in the schema"] += 1
                continue
            if l["value"] not in s["values"]:
                stat[f"label: class not in the schema ({attr})"] += 1
                continue
            leaf = canon.get((attr, l["value"]), {}).get(l["leaf"], "") if l["leaf"] else ""
            if l["leaf"] and not leaf:
                stat["leaf dropped: not a leaf of the tree"] += 1
            # A label on a chunk of the attribute's own grid is evidence about that chunk and always wins. A label
            # on the clip row is copied to the clip's chunks only for an attribute decided once per clip.
            if r["grid"] == s["grid"]:
                targets, src = [row], "own"
            elif r["grid"] == "clip" and s["scope"] != "local":
                targets, src = by_clip_grid[(r["clip"], s["grid"])], "clip"
                if not targets:
                    stat[f"dropped: clip label, no {s['grid']} chunk on the clip ({attr})"] += 1
            elif r["grid"] == "clip":
                targets, src = [], ""
                stat[f"not used: clip-level label of a per-chunk attribute ({attr})"] += 1
            else:
                targets, src = [], ""
                stat[f"not used: label on a {r['grid']} chunk, the attribute lives on {s['grid']} ({attr})"] += 1
            for t in targets:
                cur = acc[(t, attr)]
                if l["value"] in cur and not (leaf and not cur[l["value"]][0]) and not (src == "own" and cur[l["value"]][1] == "clip"):
                    continue                                   # keep the copy that has a leaf / that is the chunk's own
                cur[l["value"]] = (leaf, src)
            if targets:
                stat["label: used"] += 1

    out = []
    for (row, attr), vl in sorted(acc.items()):
        r = rows[row]
        s = sp[attr]
        # a single-choice attribute keeps one class: the chunk's own evidence first, then a copy with a leaf
        vals = sorted(vl) if s["multi"] else [max(vl, key=lambda v: (vl[v][1] == "own", bool(vl[v][0]), v))]
        out.append({"row": row, "clip": r["clip"], "source": r["source"], "grid": r["grid"], "t0": r["t0"], "t1": r["t1"],
                    "attr": attr, "values": vals, "leaves": {v: vl[v][0] for v in vals if vl[v][0]},
                    "from": {v: vl[v][1] for v in vals}, "split": split_of(r["clip"])})
        stat["pairs"] += 1
    return out, stat


def write(name: str, pr: list[dict], sp: dict[str, dict]):
    """pairs.jsonl and space.json of a set (the space keeps, per attribute, the train count of every class)."""
    d = I.sdir(name)
    tmp = d / (PAIRS + ".tmp")
    tmp.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pr))
    tmp.replace(d / PAIRS)
    out = {}
    for a, s in sp.items():
        mine = [p for p in pr if p["attr"] == a]
        cnt = collections.Counter(v for p in mine if p["split"] == "train" for v in p["values"])
        out[a] = {**s, "counts": {"pairs": len(mine), "clips": len({p["clip"] for p in mine}), "train": dict(cnt)}}
    save_json({"attributes": out}, d / SPACE, indent=1)


def read(name: str) -> tuple[list[dict], dict[str, dict]]:
    d = I.sdir(name)
    return read_jsonl(d / PAIRS), json.loads((d / SPACE).read_text())["attributes"]


def build(name: str, log) -> list[dict]:
    schema = S.load_schema()
    sp = space(schema)
    pr, stat = pairs(name, schema, load_tree())
    write(name, pr, sp)
    log.info("%s: %d pairs over %d rows", name, len(pr), stat["rows"])
    for k, v in sorted(stat.items(), key=lambda kv: -kv[1]):
        if k not in ("rows", "pairs", "label: used"):
            log.info("  %-70s %d", k, v)
    by = collections.defaultdict(list)
    for p in pr:
        by[p["attr"]].append(p)
    for a in sorted(by, key=lambda a: -len(by[a])):
        clips = {p["clip"] for p in by[a]}
        log.info("  %-26s %6d pairs  %5d clips  %2d classes  %3d leaves", a, len(by[a]), len(clips),
                 len({v for p in by[a] for v in p["values"]}), len({l for p in by[a] for l in p["leaves"].values()}))
    check(len(pr) == len({(p["row"], p["attr"]) for p in pr}), "one record per (chunk, attribute)", log)
    check(all(set(p["leaves"]) <= set(p["values"]) for p in pr), "every leaf sits under a class of its own pair", log)
    check(bool(pr), "at least one attribute carries a pair", log)
    return pr


def selftest():
    import shutil
    schema = S.load_schema()
    sp = space(schema)
    assert sp["speech.gender"]["grid"] == "g3" and sp["music.genre"]["grid"] == "g10"
    assert sp["speech.gender"]["scope"] == "constant" and sp["speech.emotion"]["scope"] == "local"
    assert sp["sound.event"]["multi"] and not sp["speech.emotion"]["multi"]
    tree = {"speech": {"emotion": {"angry": {"leaves": {"sarcastic": {"forms": ["sarky"]}}}}, "gender": {}}, "sound": {"event": {}}, "music": {"genre": {}}}
    assert canon_leaves(tree)[("speech.emotion", "angry")] == {"sarcastic": "sarcastic", "sarky": "sarcastic"}
    # one clip, two 3 s chunks: a per-clip attribute reaches both, a per-chunk one only its own chunk, and a
    # music attribute reaches neither because the clip has no 10 s chunk
    name = "_selftest_pairs"
    shutil.rmtree(I.sdir(name), ignore_errors=True)
    I.write_rows(name, [{"row": "c:clip", "clip": "c", "audio_path": "x/a.wav", "source": "s", "grid": "clip", "t0": 0, "t1": 6},
                        {"row": "c:g3:0", "clip": "c", "audio_path": "x/a.wav", "source": "s", "grid": "g3", "t0": 0, "t1": 3},
                        {"row": "c:g3:3", "clip": "c", "audio_path": "x/a.wav", "source": "s", "grid": "g3", "t0": 3, "t1": 6}])
    F = I.fired
    for tool, recs in (("wavlm_sv", [F("c:clip", "speech.gender", "male voice", "wavlm_sv", 0.9), F("c:g3:3", "speech.gender", "female voice", "wavlm_sv", 0.8)]),
                       ("emotion2vec", [F("c:g3:3", "speech.emotion", "angry", "emotion2vec", 0.9)]),
                       ("panns", [F("c:g3:0", "sound.event", "animal", "panns", 0.8), F("c:g3:0", "sound.event", "machine or vehicle", "panns", 0.7),
                                  F("c:g3:0", "sound.event", "zipper", "panns", 0.7)]),
                       ("qwen3_omni", [F("c:clip", "music.genre", "rock", "qwen3_omni", leaf="garage rock"),
                                       F("c:g3:3", "speech.emotion", "angry", "qwen3_omni", leaf="sarky")])):
        s = I.Shard(name, tool); s.add("c", recs); s.close()
    I.assemble(name)
    pr, stat = pairs(name, schema, tree)
    got = {(p["row"], p["attr"]): p for p in pr}
    assert got[("c:g3:0", "speech.gender")]["values"] == ["male voice"] and got[("c:g3:0", "speech.gender")]["from"] == {"male voice": "clip"}
    assert got[("c:g3:3", "speech.gender")]["values"] == ["female voice"] and got[("c:g3:3", "speech.gender")]["from"] == {"female voice": "own"}
    assert ("c:clip", "speech.gender") not in got
    assert got[("c:g3:3", "speech.emotion")]["leaves"] == {"angry": "sarcastic"} and ("c:g3:0", "speech.emotion") not in got
    assert got[("c:g3:0", "sound.event")]["values"] == ["animal", "machine or vehicle"]
    assert stat["label: class not in the schema (sound.event)"] == 1
    assert not any(a == "music.genre" for _, a in got) and stat["dropped: clip label, no g10 chunk on the clip (music.genre)"] == 1
    assert len({p["split"] for p in pr}) == 1
    write(name, pr, sp)
    pr2, sp2 = read(name)
    assert pr2 == pr and sp2["speech.gender"]["values"] == sp["speech.gender"]["values"]
    shutil.rmtree(I.sdir(name))
    print("router data selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default="pool")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        _, log = setup_run(f"router_data_{a.set}")
        build(a.set, log)
