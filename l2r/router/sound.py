"""Environmental-sound identity heads: which animal, vehicle, machine, tool, ... is heard.

The sound tree (`checkpoints/tree/sound_tree.json`) has one identity attribute per kind of source; under it classes
(`dog`) and leaves (`Bark`), all placed from the AudioSet ontology. One model per frozen encoder (CLAP, BEATs) reads
the clip embedding with a shared layer and predicts three levels per attribute,

    p(leaf | x) = p(attribute | x) * p(class | attribute, x) * p(leaf | class, x),

each a sigmoid trained with a masked binary cross-entropy: a class output is trained only on clips where its
attribute is present, a leaf output only where its class is present, and a label that stops at a class leaves the
leaves below it unknown rather than negative. Targets are human AudioSet labels mapped onto tree nodes, plus the
predictions of a VGGSound classifier (a teacher used at training only) as soft targets. The CLAP heads also train on
FSD50K development clips (leaf level) and ESC-50 clips (class level). Each attribute is served by the encoder with
the higher held-out mean average precision.

    python -m l2r.router.sound teacher             # VGGSound classifier probabilities on the training clips
    python -m l2r.router.sound targets             # tree nodes + target matrices
    python -m l2r.router.sound extra               # FSD50K dev + ESC-50 clips for the CLAP heads
    python -m l2r.router.sound train --encoder clap
    python -m l2r.router.sound train --encoder beats

Needs `python -m l2r.router.audioset items` and `beats` first. Outputs: checkpoints/sound/{nodes.json,
heads_<encoder>.pt, result_<encoder>.json}.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from l2r import containment
from l2r.common import audio, check, ckpt, data, load_config, resolve, setup_run
from l2r.router import audioset as AS

DIM = {"clap": 512, "beats": 768}
ENCODERS = ("clap", "beats")                # order breaks ties when choosing the serving encoder of an attribute
SKIP_ATTR = {"music.instrument"}            # placed by the tree builder, but routed by the music heads
SOUND_CLASSES = "configs/sound_classes.yaml"
EXTRA_TAU = 0.95                            # containment of the FSD50K dev / ESC-50 clips against the benchmarks
TAUS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)       # candidate serving thresholds
VGG_CLASSES = 308


def out(name: str) -> Path:
    return ckpt("sound", name)


# ------------------------------------------------------------------------------------- the tree's nodes
def node_index(tree: dict | None = None):
    """-> (attribute -> [(class, leaf or "")], (source, label name) -> [(attribute, class, leaf)]).
    The first lists the output nodes of each attribute: every class, then its leaves. The second maps a label of an
    external vocabulary (`audioset`, `vggsound`) to the tree nodes it stands for."""
    tree = tree or json.load(open(ckpt("tree", "sound_tree.json")))
    attrs, name2 = {}, defaultdict(list)
    for at, classes in tree.items():
        if at in SKIP_ATTR:
            continue
        nodes = []
        for v, leaves in classes.items():
            nodes.append((v, ""))
            names = {x["leaf"] for x in leaves}
            canon = {l["leaf"] for l in leaves if not (l.get("same_as") and l["same_as"] in names)}
            nodes += [(v, l) for l in sorted(canon)]
            for l in leaves:
                name2[(l["src"], l["leaf"])].append((at, v, l["same_as"] if l.get("same_as") in canon else l["leaf"]))
        attrs[at] = nodes
    return attrs, name2


def load_nodes() -> dict[str, list]:
    return json.load(open(out("nodes.json")))


# ------------------------------------------------------------------------------------- model
class Heads(nn.Module):
    """A shared layer on the frozen clip embedding and one linear head per attribute (one output per node), plus the
    `kind` head: one output per attribute, "a source of this kind is present"."""

    def __init__(self, dim: int, sizes: dict[str, int]):
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1024), nn.GELU(), nn.Dropout(0.2))
        self.heads = nn.ModuleDict({k.replace(".", "__"): nn.Linear(1024, n) for k, n in sizes.items()})

    def forward(self, x):
        h = self.body(x)
        return {k.replace("__", "."): m(h) for k, m in self.heads.items()}


def build(encoder: str, nodes: dict) -> Heads:
    return Heads(DIM[encoder], {**{k: len(v) for k, v in nodes.items()}, "kind": len(nodes)})


def joint(p: np.ndarray, p_kind: np.ndarray, nodes: list) -> np.ndarray:
    """Sigmoid outputs of one attribute [N, nodes] -> joint probabilities: p(class) = p(attribute) p(class | attribute),
    p(leaf) = p(class) p(leaf | class)."""
    q = p.copy()
    parent = {v: j for j, (v, l) in enumerate(nodes) if not l}
    for j, (v, l) in enumerate(nodes):
        if not l:
            q[:, j] = p_kind * p[:, j]
    for j, (v, l) in enumerate(nodes):
        if l:
            q[:, j] = q[:, parent[v]] * p[:, j]
    return q


def load(encoder: str, dev="cpu") -> tuple[Heads, dict, dict]:
    """-> (model, attribute -> serving threshold, attribute -> nodes). `model(x)[attribute]` are node logits and
    `model(x)["kind"][:, i]` the logit of the i-th attribute (in the order of the nodes dict)."""
    ck = torch.load(out(f"heads_{encoder}.pt"), map_location="cpu", weights_only=False)
    m = build(encoder, ck["nodes"])
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), ck["tau"], ck["nodes"]


def serving_encoder(nodes: dict | None = None) -> dict[str, str]:
    """attribute -> the encoder whose heads serve it: the higher mean of the two held-out mean average precisions."""
    nodes = nodes or load_nodes()
    R = {e: json.load(open(out(f"result_{e}.json"))) for e in ENCODERS}
    return {at: max(ENCODERS, key=lambda e: np.nanmean([R[e]["test"].get(at, np.nan), R[e]["fsd_eval"].get(at, np.nan)])) for at in nodes}


# ------------------------------------------------------------------------------------- teacher (training only)
def vggsound_classifier(dev=None):
    """The SpeechBrain ECAPA classifier trained on VGGSound -> (model, head(embedding) -> logits, class names).
    The classification layers are applied by hand from the checkpoint's tensors."""
    from speechbrain.inference.classifiers import EncoderClassifier
    spec = load_config()["encoders"].get("vggsound") or {"source": "Ubenwa/sb-ecapa-vggsound", "dir": "pretrained/vggsound_ecapa"}
    d = ckpt(spec["dir"])
    m = EncoderClassifier.from_hparams(source=spec["source"], savedir=str(d), run_opts={"device": dev or ("cuda" if torch.cuda.is_available() else "cpu")})
    ck = {k: v.to(m.device) for k, v in torch.load(d / "classifier.ckpt", map_location="cpu").items() if v.dim()}

    def bn(x, p):
        return (x - ck[p + ".running_mean"]) / torch.sqrt(ck[p + ".running_var"] + 1e-5) * ck[p + ".weight"] + ck[p + ".bias"]

    def head(e):
        x = bn(F.leaky_relu(e), "norm.norm")
        x = x @ ck["DNN.block_0.linear.w.weight"].T + ck["DNN.block_0.linear.w.bias"]
        x = bn(F.leaky_relu(x), "DNN.block_0.norm.norm")
        return x @ ck["out.w.weight"].T + ck["out.w.bias"]
    names = [m.hparams.label_encoder.decode_ndim(torch.tensor([i]))[0] for i in range(VGG_CLASSES)]
    return m, head, names


def teacher(a, log):
    """Class probabilities of the VGGSound classifier on the centre 10 s of 40,000 training clips and every held-out clip."""
    import librosa
    from torch.utils.data import DataLoader, Dataset
    it = AS.load_items()["items"]
    rng = np.random.default_rng(0)
    tr = [i for i, x in enumerate(it) if x["set"] == "train"]
    pick = sorted(rng.choice(tr, min(a.n_train, len(tr)), replace=False).tolist() + [i for i, x in enumerate(it) if x["set"] != "train"])
    log.info("teacher over %d clips (%d train + every held-out clip)", len(pick), min(a.n_train, len(tr)))
    n = 16000 * AS.CLIP_S

    class Clips(Dataset):
        def __len__(self):
            return len(pick)

        def __getitem__(self, j):
            p = it[pick[j]]["path"]
            try:
                assert AS.readable(p)
                y, _ = librosa.load(str(audio(p)), sr=16000, mono=True, duration=60)
                ok = 1
            except Exception:
                y, ok = np.zeros(n, np.float32), 0
            return AS._centre(y, n), ok
    m, head, names = vggsound_classifier()
    P, OK = [], []
    with torch.no_grad():
        for b, (y, ok) in enumerate(DataLoader(Clips(), batch_size=32, num_workers=a.workers)):
            P.append(F.softmax(head(m.encode_batch(y.to(m.device))[:, 0]), -1).cpu().numpy().astype(np.float16))
            OK.append(ok.numpy())
            if b % 200 == 0:
                log.info("%d / %d", (b + 1) * 32, len(pick))
    ok = np.concatenate(OK).astype(bool)
    np.savez(AS.out("vggsound_teacher.npz"), idx=np.array(pick)[ok], p=np.concatenate(P)[ok], names=np.array(names))
    check(ok.mean() > 0.95, f"readable clips {ok.mean():.3f} > 0.95", log)


# ------------------------------------------------------------------------------------- targets
def targets(a, log):
    """Target matrices over the tree's nodes for every item: human AudioSet labels (1.0; a leaf also marks its class)
    and the teacher's probabilities summed onto the nodes they map to (soft)."""
    attrs, name2 = node_index()
    items = AS.load_items()
    names, it = items["names"], items["items"]
    nidx = {at: {n: j for j, n in enumerate(nodes)} for at, nodes in attrs.items()}
    Y = {at: np.zeros((len(it), len(nodes)), np.float16) for at, nodes in attrs.items()}

    def put(i, at, v, leaf, p):
        y = Y[at][i]
        jv = nidx[at][(v, "")]
        y[jv] = max(y[jv], p)
        if leaf and (v, leaf) in nidx[at]:
            jl = nidx[at][(v, leaf)]
            y[jl] = max(y[jl], p)
    for i, x in enumerate(it):
        for c in x["clip"]:
            for at, v, leaf in name2.get(("audioset", names[c]), []):
                put(i, at, v, leaf, 1.0)
    D = np.load(AS.out("vggsound_teacher.npz"))
    P = D["p"].astype(np.float32)
    vm = [name2.get(("vggsound", str(n)), []) for n in D["names"]]
    n_soft = 0
    for r, i in enumerate(D["idx"]):
        acc = defaultdict(float)
        for j in np.nonzero(P[r] >= 0.02)[0]:
            for at, v, leaf in vm[j]:
                acc[(at, v, leaf)] += float(P[r, j])
        for (at, v, leaf), p in acc.items():
            put(i, at, v, leaf, min(p, 1.0))
            n_soft += 1
    np.savez_compressed(AS.out("sound_targets.npz"), **{at.replace(".", "__"): y for at, y in Y.items()})
    json.dump({at: [list(n) for n in nodes] for at, nodes in attrs.items()}, open(out("nodes.json"), "w"))
    for at, y in Y.items():
        log.info("%-24s nodes %4d | clips with a positive %6d | mean positives %.2f", at, y.shape[1], int((y.max(1) >= 0.5).sum()), float((y >= 0.5).sum(1).mean()))
    log.info("teacher soft entries: %d over %d clips", n_soft, len(D["idx"]))


def unknown_masks(Y: dict, nodes: dict) -> dict:
    """M[attribute] = 1 where a node is supervised. When a clip's only positive under a positive class is the leaf
    named like the class itself (the label stopped at the class), the other leaves of that class are masked."""
    M = {}
    for at, y in Y.items():
        nd = nodes[at]
        m = np.ones_like(y, dtype=np.float16)
        for j, (v, l) in enumerate(nd):
            if l:
                continue
            other = [i for i, (vv, ll) in enumerate(nd) if ll and vv == v and ll.lower() != v.lower()]
            if not other:
                continue
            sel = (y[:, j] >= 0.5) & (y[:, other].max(1) < 0.5)
            m[np.ix_(np.nonzero(sel)[0], other)] = 0
        M[at] = m
    return M


def source_classes():
    """-> (class name of the hand-written class list for an AudioSet id, ESC-50 category -> AudioSet class name)."""
    import yaml
    spec = yaml.safe_load(open(resolve(SOUND_CLASSES)))
    onto = json.load(open(data("AudioSet", "ontology.json")))
    by_name = {o["name"]: o["id"] for o in onto}
    parents = defaultdict(set)
    for o in onto:
        for c in o["child_ids"]:
            parents[c].add(o["id"])
    root = {by_name[n]: c["name"] for cl in spec["kinds"].values() for c in cl for n in c["audioset"]}

    def up(i, seen):
        if i not in seen:
            seen.add(i)
            for p in parents[i]:
                up(p, seen)
        return seen

    def classes_of(audioset_name):
        return sorted({root[x] for x in up(by_name[audioset_name], set()) if x in root})
    return classes_of, spec["esc50_to_audioset"]


def extra_labels(nodes: dict, keep=None) -> tuple[list[str], dict[str, np.ndarray]]:
    """Labelled FSD50K development and ESC-50 clips -> (paths, attribute -> target matrix). FSD50K clips are labelled
    to the leaf through their AudioSet ids, ESC-50 clips to the class through the hand-written class list.
    `keep(path)` optionally filters the clips."""
    _, name2 = node_index()
    nidx = {at: {tuple(n): j for j, n in enumerate(nd)} for at, nd in nodes.items()}
    onto = {o["id"]: o["name"] for o in json.load(open(data("AudioSet", "ontology.json")))}
    of_class = defaultdict(list)                                    # class name -> [(attribute, class)]
    for at, nd in nodes.items():
        for v, l in nd:
            if not l:
                of_class[v].append((at, v))
    paths, Y = [], {at: [] for at in nodes}

    def add(path, hits):
        """hits = [(attribute, class, leaf or None)]"""
        if not hits or (keep is not None and not keep(path)):
            return
        y = {at: np.zeros(len(nd), np.float16) for at, nd in nodes.items()}
        for at, v, leaf in hits:
            y[at][nidx[at][(v, "")]] = 1
            if leaf and (v, leaf) in nidx[at]:
                y[at][nidx[at][(v, leaf)]] = 1
        paths.append(path)
        for at in nodes:
            Y[at].append(y[at])
    for r in csv.DictReader(open(data("FSD50K", "FSD50K.ground_truth", "dev.csv"))):
        hits = [h for mid in r["mids"].split(",") for h in name2.get(("audioset", onto.get(mid, "")), [])]
        add(f"FSD50K/FSD50K.dev_audio/{r['fname']}.wav", hits)
    classes_of, esc_map = source_classes()
    for r in csv.DictReader(open(data("ESC-50", "meta", "esc50.csv"))):
        if r["category"] in esc_map:
            add(f"ESC-50/audio/{r['filename']}", [(at, v, None) for c in classes_of(esc_map[r["category"]]) for at, v in of_class.get(c, [])])
    return paths, {at: np.stack(v) for at, v in Y.items()}


def extra(a, log):
    """More human-labelled clips for the CLAP heads (FSD50K development, ESC-50). Clips within 0.95 of a benchmark
    clip are dropped."""
    from l2r.encoders import build_encoder
    nodes = load_nodes()
    paths, Y = extra_labels(nodes, keep=AS.readable)
    clap = build_encoder("clap")
    X = AS.embed_clap(clap, paths, log)
    keep = containment.nearest(X, containment.benchmark_embeddings(clap, log)) < EXTRA_TAU
    Y = {at: y[keep] for at, y in Y.items()}
    M = unknown_masks(Y, nodes)
    np.savez_compressed(AS.out("sound_extra.npz"), X=X[keep], **{"Y_" + at.replace(".", "__"): y for at, y in Y.items()},
                        **{"M_" + at.replace(".", "__"): m for at, m in M.items()})
    log.info("extra: %d labelled FSD50K dev / ESC-50 clips, %d kept after the containment cut", len(paths), int(keep.sum()))
    check(int(keep.sum()) > 10000, "more than 10,000 extra clips", log)


# ------------------------------------------------------------------------------------- training
def features(encoder: str) -> np.ndarray:
    return np.asarray(np.load(AS.out("clap_clip.npy" if encoder == "clap" else "beats_clip.npy"), mmap_mode="r"), np.float32)


def train(a, log):
    from sklearn.metrics import average_precision_score
    nodes = load_nodes()
    Z = np.load(AS.out("sound_targets.npz"))
    Y = {k.replace("__", "."): Z[k].astype(np.float32) for k in Z.files}
    sets = np.array([x["set"] for x in AS.load_items()["items"]])
    M = unknown_masks(Y, nodes)
    X = features(a.encoder)
    tr = np.nonzero(sets == "train")[0]
    va = np.random.default_rng(0).choice(tr, len(tr) // 20, replace=False)
    tr = np.setdiff1d(tr, va)
    Xtr = X[tr]
    Ytr = {k: v[tr] for k, v in Y.items()}
    Mtr = {k: v[tr].astype(np.float32) for k, v in M.items()}
    is_class = {k: np.array([j for j, (v, l) in enumerate(nd) if not l]) for k, nd in nodes.items()}
    parent = {k: np.array([next(i for i, (vv, ll) in enumerate(nd) if not ll and vv == v) if l else -1 for v, l in nd]) for k, nd in nodes.items()}
    if a.encoder == "clap":
        E = np.load(AS.out("sound_extra.npz"))
        Xtr = np.concatenate([Xtr, E["X"]])
        Ytr = {k: np.concatenate([Ytr[k], E["Y_" + k.replace(".", "__")].astype(np.float32)]) for k in Ytr}
        Mtr = {k: np.concatenate([Mtr[k], E["M_" + k.replace(".", "__")].astype(np.float32)]) for k in Mtr}
        log.info("clap: + %d FSD50K dev / ESC-50 clips", len(E["X"]))
    for k in Ytr:                                   # a class is trained where its attribute is present, a leaf where its class is
        y, mk = Ytr[k], Mtr[k]
        present = y[:, is_class[k]].max(1) >= 0.5
        for j, (v, l) in enumerate(nodes[k]):
            mk[:, j] *= present if not l else (y[:, parent[k][j]] >= 0.5)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = build(a.encoder, nodes).to(dev)
    kinds = list(nodes)
    Xt = torch.tensor(Xtr, device=dev)
    Yt = {k: torch.tensor(v, device=dev) for k, v in Ytr.items()}
    Mt = {k: torch.tensor(v, device=dev) for k, v in Mtr.items()}
    pw = {k: torch.tensor(np.clip((len(v) - (v >= 0.5).sum(0)) / np.maximum((v >= 0.5).sum(0), 1), 1, 30) ** 0.5, device=dev).float() for k, v in Ytr.items()}
    Xall = torch.tensor(X, device=dev)

    def predict(ix):
        m.eval()
        o = defaultdict(list)
        with torch.no_grad():
            for s in range(0, len(ix), 4096):
                for k, v in m(Xall[ix[s:s + 4096]]).items():
                    o[k].append(torch.sigmoid(v).cpu().numpy())
        m.train()
        o = {k: np.concatenate(v) for k, v in o.items()}
        return {k: joint(o[k], o["kind"][:, i], nodes[k]) for i, k in enumerate(kinds)}

    def mean_ap(P, ix):
        r = {}
        for k, p in P.items():
            y = Y[k][ix] >= 0.5
            keep = y.sum(0) >= 5
            r[k] = float(average_precision_score(y[:, keep], p[:, keep], average="macro")) if keep.any() else float("nan")
        return r
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    best, state, best_epoch = -1, None, 0
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xt), device=dev)
        for s in range(0, len(Xt), 512):
            b = perm[s:s + 512]
            o = m(Xt[b] + (0.02 * torch.randn_like(Xt[b]) if a.encoder == "clap" else 0))
            loss = sum((F.binary_cross_entropy_with_logits(o[k], Yt[k][b], pos_weight=pw[k], reduction="none") * Mt[k][b]).sum() / Mt[k][b].sum().clamp(min=1)
                       for k in nodes) / len(nodes)
            present = torch.stack([(Yt[k][b][:, is_class[k]].max(1).values >= 0.5).float() for k in kinds], 1)
            loss = loss + F.binary_cross_entropy_with_logits(o["kind"], present)
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = float(np.nanmean(list(mean_ap(predict(va), va).values())))
        if state is None or v > best:
            best, state, best_epoch = v, {k: t.detach().clone() for k, t in m.state_dict().items()}, ep
        log.info("%s epoch %d loss %.4f validation mean AP %.4f", a.encoder, ep, float(loss), v)
    m.load_state_dict(state)
    Pv = predict(va)
    tau = {}                                        # per attribute: the threshold with the best class-level F1 on validation
    for k, nd in nodes.items():
        y = Y[k][va][:, is_class[k]] >= 0.5
        p = Pv[k][:, is_class[k]]
        f1 = [2 * ((p >= t) & y).sum() / max((p >= t).sum() + y.sum(), 1) for t in TAUS]
        tau[k] = TAUS[int(np.argmax(f1))]
    res = {"encoder": a.encoder, "epoch": best_epoch, "val": round(best, 4), "tau": tau}
    for name in ("test", "fsd_eval"):
        ix = np.nonzero(sets == name)[0]
        r = mean_ap(predict(ix), ix)
        res[name] = {"mean_mAP": round(float(np.nanmean(list(r.values()))), 4), **{k: round(v, 3) for k, v in r.items()}}
    log.info("RESULT %s", json.dumps(res))
    json.dump(res, open(out(f"result_{a.encoder}.json"), "w"), indent=1)
    torch.save({"state": m.cpu().state_dict(), "nodes": nodes, "tau": tau, "encoder": a.encoder}, out(f"heads_{a.encoder}.pt"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["teacher", "targets", "extra", "train"])
    ap.add_argument("--encoder", choices=list(DIM), default="clap")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-train", type=int, default=40000, help="teacher: number of training clips it labels")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    _, log = setup_run(f"sound_{a.stage}")
    {"teacher": teacher, "targets": targets, "extra": extra, "train": train}[a.stage](a, log)


if __name__ == "__main__":
    main()
