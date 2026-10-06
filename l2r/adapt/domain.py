"""Adding a domain: the label set of a new attribute, and a routing probe of its trained head.

A new domain (bird species, marine-mammal species) is one new attribute under the sound region. Its head is
trained on k labelled clips per species plus a `none` class drawn from chunks of the tree-generation pool, so the
head learns when a clip does not belong to the domain. The tree, the other heads and the reader are untouched.

    python -m l2r.adapt.domain build --domain birds --k 5          # -> <work>/sets/birds_k5/{rows.jsonl, pairs.jsonl, space.json}
    python -m l2r.router.features --encoder birdnet --sets birds_k5
    python -m l2r.router.train --name birds_k5 --sets birds_k5 --regions sound --encoder birdnet \
        --layers 1 --lr 1e-3 --balanced 0.5 --min-clips 5 --val-frac 0 --epochs 40 --steps 200
    python -m l2r.adapt.domain probe --domain birds --k 5          # species accuracy of the head, no reader

Marine mammals use `--encoder clap`; `--min-clips` is min(k, 5). At inference the head serves its best species
unless p(none) >= 0.995.

The k clips of a species are the first k pool clips, in the order of `l2r.adapt.data.candidates`, that are not
within 0.95 CLAP cosine similarity of any evaluation clip (containment rule). The test rows of the label set are
the evaluation items with their four options, used only by the probe.
"""
from __future__ import annotations

import argparse
import collections
import json
import random

import numpy as np

from l2r.adapt import data as D
from l2r.common import audio, check, load_config, save_json, setup_run
from l2r.dataset import index as I, segment as G

ATTR = {"birds": "sound.bird_species", "marine": "sound.marine_species"}
CONTAIN_TAU = 0.95
N_NONE = 2000


def set_name(domain: str, k: int) -> str:
    return f"{domain}_k{k}"


# ----------------------------------------------------------------------------- the k labelled clips
def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def _clap_window(clap, path, span):
    """The 10 s CLAP window of a pool clip: the given span, grown around itself when the file is shorter."""
    import librosa
    y = librosa.load(str(audio(path)), sr=clap.SR, mono=True)[0]
    a, b = max(0, int(span[0] * clap.SR)), min(len(y), int(span[1] * clap.SR))
    if b > a:
        seg = y[a:b]
        if len(seg) >= clap.WIN:
            off = (len(seg) - clap.WIN) // 2
            return seg[off:off + clap.WIN]
        lo = max(0, a - (clap.WIN - len(seg)) // 2)
        hi = min(len(y), lo + clap.WIN)
        return y[max(0, hi - clap.WIN):hi]
    return y if len(y) <= clap.WIN else y[(len(y) - clap.WIN) // 2:][:clap.WIN]


def _clap_embed(clap, chunk):
    """CLAP vectors of [(path, span or None)]."""
    if not any(s for _, s in chunk):
        return _unit(clap.embed([str(audio(p)) for p, _ in chunk]))
    wins = [_clap_window(clap, p, s) for p, s in chunk]
    wins = [w if len(w) else np.zeros(4800, dtype=np.float32) for w in wins]
    e = [np.asarray(clap.m.get_audio_embedding_from_data(x=wins[j:j + clap.bs], use_tensor=False)) for j in range(0, len(wins), clap.bs)]
    return _unit(_unit(np.vstack(e)))


def few_shot(domain: str, k: int, cfg, log, dom: dict | None = None) -> dict[str, list[tuple[str, tuple | None]]]:
    """{species label: the k labelled clips (path, window)} under the containment rule."""
    from l2r.encoders import build_encoder
    dom = dom or D.load_domain(domain)
    clap = build_encoder("clap", cfg)
    ev = _unit(clap.embed([str(audio(it["audio"])) for it in dom["items"]]))
    out, skipped, short = {}, 0, []
    for lab, sel in D.candidates(domain, dom).items():
        keep, i = [], 0
        while len(keep) < k and i < len(sel):
            chunk = sel[i:i + max(k, 8)]; i += len(chunk)
            mx = (_clap_embed(clap, chunk) @ ev.T).max(axis=1)
            for c, m in zip(chunk, mx):
                if len(keep) >= k:
                    break
                if m >= CONTAIN_TAU:
                    skipped += 1
                    continue
                keep.append(c)
        if len(keep) < k:
            short.append(lab)
        out[lab] = keep
    log.info("k=%d per species: %d candidates skipped as within %.2f of an evaluation clip, %d species short of k %s",
             k, skipped, CONTAIN_TAU, len(short), short[:8])
    check(all(out.values()), "every species has at least one labelled clip", log)
    return out


# ----------------------------------------------------------------------------- the label set
def build(domain: str, k: int, cfg, log, n_none: int = N_NONE, seed: int = 0) -> dict:
    dom = D.load_domain(domain); attr = ATTR[domain]; name = set_name(domain, k)
    name_of = {lab: v["name"] for lab, v in dom["pool"].items()}
    values = [name_of[lab] for lab in dom["species"]] if all(s in name_of for s in dom["species"]) else sorted(name_of.values())
    rows, pairs = [], []

    def add(path, span, split, value, src, extra=None):
        c = G.clip_id(path); t0, t1 = (float(span[0]), float(span[1])) if span else (0.0, D.WINDOW_S)
        row = {"clip": c, "audio_path": str(path), "source": f"{domain}_{src}", "row": f"{c}:g10:{t0:g}", "grid": "g10", "t0": round(t0, 2), "t1": round(t1, 2)}
        rows.append(row)
        pairs.append({"row": row["row"], "clip": c, "source": row["source"], "grid": "g10", "t0": row["t0"], "t1": row["t1"], "attr": attr,
                      "values": [value], "leaves": {}, "from": {value: "own"}, "split": split, **(extra or {})})

    few = few_shot(domain, k, cfg, log, dom)
    for lab, clips in few.items():                               # train: the k labelled clips of every species
        for path, span in clips:
            add(path, span, "train", name_of[lab], "pool")
    n_train = len(rows)
    for it in dom["items"]:                                      # test: the evaluation items with their options
        add(it["audio"], None, "test", it["sub_category"], "eval",
            {"options": list(it["choices"].values()), "gold_name": it["choices"][it["gold"]], "item_id": it["id"]})
    base = [r for r in I.rows("pool") if r["grid"] == "g3"]      # `none`: random chunks of the tree-generation pool
    check(len(base) >= n_none, f"the pool set holds at least {n_none} 3 s chunks (run the dataset stage first)", log)
    random.Random(seed).shuffle(base)
    for i, r in enumerate(base[:n_none]):
        rows.append({q: r[q] for q in ("clip", "audio_path", "source", "row", "grid", "t0", "t1")})
        pairs.append({"row": r["row"], "clip": r["clip"], "source": r["source"], "grid": r["grid"], "t0": r["t0"], "t1": r["t1"], "attr": attr,
                      "values": ["none"], "leaves": {}, "from": {"none": "own"}, "split": "train" if i % 10 else "test"})
    out = I.sdir(name)
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    counts = collections.Counter(v for p in pairs if p["split"] == "train" for v in p["values"])
    save_json({"attributes": {attr: {"region": "sound", "values": values + ["none"], "multi": False, "scope": "constant", "grid": "g10",
                                     "leaf": "closed", "counts": {"train": dict(counts)}}}}, out / "space.json", indent=1)
    log.info("%s: %d labelled clips over %d species (k=%d), %d test items, %d `none` chunks -> %s", name, n_train, len(values), k, len(dom["items"]), n_none, out)
    return {"set": name, "train_clips": n_train, "species": len(values), "test": len(dom["items"])}


# ----------------------------------------------------------------------------- routing probe
def probe(domain: str, k: int, router: str | None = None) -> dict:
    """Species accuracy of the trained head on the evaluation items, without a reader: top-1 over all species,
    top-1 among the item's four options, abstention rates, and the nearest-centroid rule on the same k clips."""
    import torch
    from l2r.common import read_jsonl, work
    from l2r.router import features as F, model as M
    name = set_name(domain, k); attr = ATTR[domain]
    m, _ = M.load(router or name)
    idx, X = F.load(m.encoder, name)
    pairs = [p for p in read_jsonl(I.sdir(name) / "pairs.jsonl") if p["attr"] == attr]
    feats = lambda ps: X[[idx[p["row"]] for p in ps]]
    vals = m.meta[attr]["values"]; vi = {v: i for i, v in enumerate(vals)}; none = vi["none"]
    ev = [p for p in pairs if p["split"] == "test" and p["values"][0] != "none"]
    ng = [p for p in pairs if p["split"] == "test" and p["values"][0] == "none"]
    tr = [p for p in pairs if p["split"] == "train" and p["values"][0] != "none"]

    def logits(ps):
        with torch.no_grad():
            return m.class_logits(attr, m.embed(torch.from_numpy(feats(ps)))).float().numpy() if ps else np.zeros((0, len(vals)))
    L, Ln = logits(ev), logits(ng)
    species = [i for i in range(len(vals)) if i != none]
    gold = np.array([vi[p["values"][0]] for p in ev])
    top = np.array([species[int(np.argmax(l[species]))] for l in L])
    four = []
    for p, l in zip(ev, L):
        o = [vi[x] for x in p["options"] if x in vi]
        four.append(bool(o) and o[int(np.argmax(l[o]))] == vi[p["values"][0]])
    cent = collections.defaultdict(list)                         # reference: nearest centroid of the k clips, no training
    for p, x in zip(tr, feats(tr)):
        cent[p["values"][0]].append(x)
    cent = {s: np.mean(v, 0) / (np.linalg.norm(np.mean(v, 0)) + 1e-8) for s, v in cent.items()}
    names = list(cent); S = _unit(feats(ev)) @ np.stack([cent[s] for s in names]).T
    c_all = [names[int(np.argmax(s))] == p["values"][0] for p, s in zip(ev, S)]
    c_four = [max((s[names.index(o)], o) for o in p["options"] if o in cent)[1] == p["values"][0] if any(o in cent for o in p["options"]) else False
              for p, s in zip(ev, S)]
    per = collections.defaultdict(list)
    for p, t, g in zip(ev, top, gold):
        per[p["values"][0]].append(int(t == g))
    res = {"router": router or name, "encoder": m.encoder, "items": len(ev), "species": len(species),
           "all_way": float((top == gold).mean()), "four_way": float(np.mean(four)),
           "abstain_eval": float((L.argmax(1) == none).mean()), "abstain_none": float((Ln.argmax(1) == none).mean()) if len(ng) else float("nan"),
           "centroid_all_way": float(np.mean(c_all)), "centroid_four_way": float(np.mean(c_four)),
           "per_species": {s: float(np.mean(v)) for s, v in sorted(per.items(), key=lambda t: -np.mean(t[1]))}}
    save_json(res, work("results", f"probe_{router or name}.json"))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build", "probe"]); ap.add_argument("--domain", choices=D.DOMAINS, required=True)
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--router", default=None, help="probe: router name (default <domain>_k<k>)")
    a = ap.parse_args()
    if a.stage == "build":
        cfg = load_config(); _, log = setup_run(f"adapt_domain_{set_name(a.domain, a.k)}", cfg)
        print(json.dumps(build(a.domain, a.k, cfg, log)))
    else:
        r = probe(a.domain, a.k, a.router)
        print(json.dumps({q: v for q, v in r.items() if q != "per_species"}, indent=1))


if __name__ == "__main__":
    main()
