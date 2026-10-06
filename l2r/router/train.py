"""Train one router: a small body on a frozen encoder's cached embeddings, with one head per attribute.

All attributes of a router are trained at once. A chunk is labelled for only some attributes, so the loss is
masked: an attribute contributes on the chunks where it is labelled and nowhere else (a missing label is never a
negative). Per labelled chunk the loss is the cross-entropy over classes (binary cross-entropy for a
multiple-choice attribute) plus, when the label names a leaf, the cross-entropy over the nodes inside that class.

    python -m l2r.router.train --name sound              # one of the routers of the `routers:` config section
    python -m l2r.router.train --name birds_k5 --encoder birdnet --sets birds_k5 --regions sound \
        --layers 1 --lr 1e-3 --balanced 0.5 --min-clips 5 --epochs 40 --steps 200 --val-frac 0     # a new-domain head

Training reads `<work>/sets/<set>/{pairs.jsonl, space.json}` (router/data.py) and the feature stores
(router/features.py). `--sets "pool:music.mood+music.vocals,musiccaps"` restricts a set to some attributes.
"""
from __future__ import annotations

import argparse
import collections
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from l2r.common import check, load_config, set_seed, setup_run, work
from l2r.dataset import index as I
from l2r.router import data as D, features, model as M

DEFAULTS = {"regions": [], "attributes": [], "sets": "pool", "layers": 2, "lr": 3e-4, "balanced": 0.3, "min_clips": 5,
            "epochs": 40, "steps": 300, "label_smooth": 0.0, "val_frac": 0.1}
BATCH = 64
ATTR_POW = 0.5          # a step trains the attributes of one grid; it is drawn with probability ~ (train pairs) ** ATTR_POW
PRECISION = 0.75        # a served leaf must be right at least this often on held-out chunks


# ------------------------------------------------------------------------------------- data
def parse_sets(spec: str) -> tuple[list[str], dict[int, set]]:
    """"pool:music.mood+music.vocals,musiccaps" -> (["pool", "musiccaps"], {0: {"music.mood", "music.vocals"}})."""
    names, allow = [], {}
    for k, t in enumerate(spec.split(",")):
        name, _, at = t.partition(":")
        names.append(name)
        if at:
            allow[k] = set(at.split("+"))
    return names, allow


class Data:
    """The pairs of one or more sets with their cached features, indexed for batching."""

    def __init__(self, sets: list[str], encoder: str, log=None, regions: set | None = None, attrs: set | None = None,
                 val_frac: float = 0.0, allow: dict[int, set] | None = None):
        idx, X = {}, None
        for name in sets:
            i2, X2 = features.load(encoder, name)
            assert len(i2), f"no {encoder} features for set {name}: run `python -m l2r.router.features --encoder {encoder} --sets {name}`"
            n = 0 if X is None else len(X)
            idx.update({r: n + i for r, i in i2.items()})
            X = X2 if X is None else np.concatenate([X, X2])
        self.idx, self.X = idx, X
        self.pairs, self.space, self.where = [], {}, {}
        for k, name in enumerate(sets):
            for r in I.rows(name):
                if r["grid"] in features.GRIDS:
                    self.where[r["row"]] = k                       # the set a chunk belongs to
            prs, sp = D.read(name)
            self.space.update({a: v for a, v in sp.items() if (not regions or v["region"] in regions) and (not attrs or a in attrs)})
            keep = (allow or {}).get(k)
            got = [p for p in prs if p["row"] in self.where and p["attr"] in self.space and (keep is None or p["attr"] in keep)]
            if log:
                log.info("set %s: %d pairs", name, len(got))
            self.pairs += got
        if val_frac > 0:                                           # a slice of the train clips the router never fits: epoch selection
            for p in self.pairs:
                if p["split"] == "train" and int(hashlib.md5((p["clip"] + "|val").encode()).hexdigest()[:8], 16) % 1000 < val_frac * 1000:
                    p["split"] = "val"
        self.by_attr = collections.defaultdict(lambda: {"train": [], "val": [], "test": []})
        self.by_row = collections.defaultdict(list)
        for p in self.pairs:
            self.by_attr[p["attr"]][p["split"]].append(p)
            self.by_row[p["row"]].append(p)

    def counts(self, attr: str, split="train") -> dict[str, int]:
        c = collections.Counter(v for p in self.by_attr[attr][split] for v in p["values"])
        return {v: c[v] for v in self.space[attr]["values"] if c[v]}

    def x(self, rows: list[str], dev) -> torch.Tensor:
        miss = [r for r in rows if r not in self.idx]
        assert not miss, f"{len(miss)} chunks have no cached feature, e.g. {miss[:3]}"
        return torch.from_numpy(self.X[[self.idx[r] for r in rows]]).to(dev)


def node_space(data: Data, attrs: list[str], min_clips: int) -> dict[str, list]:
    """The nodes of each head: every class seen in training, and every leaf with at least `min_clips` train clips."""
    out = {}
    for a in attrs:
        cnt = data.counts(a)
        clips = collections.defaultdict(set)
        for p in data.by_attr[a]["train"]:
            for v, lf in (p.get("leaves") or {}).items():
                if lf and v in cnt:
                    clips[(v, lf)].add(p["clip"])
        nodes = []
        for v in data.space[a]["values"]:
            if v in cnt:
                nodes += [(v, "")] + sorted(k for k, c in clips.items() if k[0] == v and len(c) >= min_clips)
        out[a] = nodes
    return out


def leaf_targets(model: M.Router, a: str, p: dict) -> list[int]:
    """The node a pair's leaf trains. A leaf too rare to be a node trains its class's own node, so the head learns
    when not to be specific."""
    out = []
    for v, lf in (p.get("leaves") or {}).items():
        if lf and (v, "") in model.cix[a]:
            out.append(model.cix[a].get((v, lf), model.cix[a][(v, "")]))
    return out


def batch_labels(data: Data, model: M.Router, rows: list[str], grid: str, split: str, dev) -> dict:
    """Every labelled attribute of these chunks: {attr: {ix [n], y [n, classes] multi-hot, lt_ix [m], lt_c [m]}};
    `lt` are the (chunk, node) leaf targets."""
    pos = {r: i for i, r in enumerate(rows)}
    out = {}
    for a in model.attrs:
        if model.meta[a]["grid"] != grid:
            continue
        vi = {v: i for i, v in enumerate(model.meta[a]["values"])}
        ix, y, li, lc = [], [], [], []
        for r in rows:
            for p in data.by_row[r]:
                if p["attr"] != a or p["split"] != split:
                    continue
                hit = [vi[v] for v in p["values"] if v in vi]
                if not hit:
                    continue
                row = np.zeros(len(vi), np.float32)
                row[hit] = 1.0
                ix.append(pos[r]); y.append(row)
                for c in leaf_targets(model, a, p):
                    li.append(pos[r]); lc.append(c)
        if ix:
            out[a] = {"ix": torch.tensor(ix, device=dev), "y": torch.from_numpy(np.stack(y)).to(dev),
                      "lt_ix": torch.tensor(li, dtype=torch.long, device=dev), "lt_c": torch.tensor(lc, dtype=torch.long, device=dev)}
    return out


# ------------------------------------------------------------------------------------- loss
def losses(model: M.Router, z, batch: dict, smooth: float = 0.0):
    """The masked multi-attribute loss (Eq. 2): an attribute with no labelled chunk in the batch contributes nothing."""
    total = torch.zeros((), device=z.device)
    log = {}
    for a, b in batch.items():
        s = model.node_logits(a, z)
        V = model.class_score(a, s[b["ix"]])
        if model.meta[a]["multi"]:
            lv = F.binary_cross_entropy_with_logits(V, b["y"])
        else:
            lv = F.cross_entropy(V, b["y"].argmax(1), label_smoothing=smooth)
        loss = lv
        log[f"{a}/class"] = float(lv.detach())
        if len(b["lt_ix"]):
            um = model.under(a)[model.class_of(a)[b["lt_c"]]]            # the nodes inside the labelled class
            lp = (s[b["lt_ix"]] + um).log_softmax(1)
            inside = (um == 0).float()
            tgt = F.one_hot(b["lt_c"], lp.shape[1]).float() * (1 - smooth) + smooth * inside / inside.sum(1, keepdim=True)
            ll = -(tgt * lp * inside).sum(1).mean()
            loss = loss + ll
            log[f"{a}/leaf"] = float(ll.detach())
        total = total + loss
    return total, log


# ------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(data: Data, model: M.Router, dev, split: str = "test", attrs: list[str] | None = None) -> tuple[dict, dict]:
    """-> (per attribute class metrics, per attribute leaf records). `first` marks chunks of the first set."""
    model.eval()
    res, leaf = {}, {}
    for a in attrs or model.attrs:
        m = model.meta[a]
        vi = {v: i for i, v in enumerate(m["values"])}
        prs = [p for p in data.by_attr[a][split] if any(v in vi for v in p["values"])]
        if not prs:
            continue
        maj = collections.Counter(v for p in data.by_attr[a]["train"] for v in p["values"]).most_common(1)[0][0]
        rec, lr = [], []
        for s0 in range(0, len(prs), 128):
            b = prs[s0:s0 + 128]
            s = model.node_logits(a, model.embed(data.x([p["row"] for p in b], dev)))
            V = model.class_score(a, s)
            pv = V.argmax(1)
            sg = torch.sigmoid(V)
            for j, p in enumerate(b):
                gold = {vi[v] for v in p["values"] if v in vi}
                first = data.where[p["row"]] == 0
                rec.append({"ok": int(pv[j]) in gold, "gold": min(gold), "maj": maj in p["values"], "first": first, "clip": p["clip"]})
                for v, lf in (p.get("leaves") or {}).items():
                    if not lf or v not in vi:
                        continue
                    c = model.cix[a].get((v, lf), model.cix[a][(v, "")])
                    ids = [k for k, (vv, _) in enumerate(m["classes"]) if vv == v]
                    if len(ids) < 2:
                        continue
                    q = s[j, ids].softmax(0)
                    k = int(q.argmax())
                    class_ok = int(pv[j]) == vi[v] or (m["multi"] and float(sg[j, vi[v]]) >= 0.5)
                    lr.append({"named_gold": bool(m["classes"][c][1]), "named_pred": bool(m["classes"][ids[k]][1]), "right": ids[k] == c,
                               "class_ok": bool(class_ok), "conf": float(q[k]), "first": first})

        def summ(rs):
            if not rs:
                return None
            by = collections.defaultdict(list)
            for r in rs:
                by[r["gold"]].append(r["ok"])
            return {"n": len(rs), "acc": float(np.mean([r["ok"] for r in rs])), "balanced": float(np.mean([np.mean(v) for v in by.values()])),
                    "majority": float(np.mean([r["maj"] for r in rs])), "clips": len({r["clip"] for r in rs})}
        res[a] = {"all": summ(rec), "first": summ([r for r in rec if r["first"]]), "other": summ([r for r in rec if not r["first"]]),
                  "values": len(vi), "classes": len(m["classes"])}
        leaf[a] = lr
    model.train()
    return res, leaf


def leaf_summary(lr: list[dict]) -> dict | None:
    """Leaf accuracy on the chunks whose labelled leaf is a node: given the labelled class, and end to end."""
    ng = [r for r in lr if r["named_gold"]]
    if not ng:
        return None
    return {"n": len(ng), "given_class": float(np.mean([r["right"] for r in ng])), "e2e": float(np.mean([r["right"] and r["class_ok"] for r in ng]))}


def fit_tau(lr: list[dict], precision: float) -> tuple[float, float, float]:
    """The smallest threshold at which a served leaf is right at least `precision` of the time, end to end.
    -> (threshold, precision, share of the chunks with a labelled leaf that are served right). 1.01 = never serve."""
    best = (1.01, 0.0, 0.0)
    ng = max(sum(r["named_gold"] for r in lr), 1)
    for t in np.arange(0.95, 0.19, -0.05):
        s = [r for r in lr if r["named_pred"] and r["class_ok"] and r["conf"] >= t]
        if len(s) >= 10:
            pr = float(np.mean([r["right"] for r in s]))
            if pr >= precision:
                best = (float(round(t, 2)), pr, sum(r["right"] for r in s) / ng)
    return best


# ------------------------------------------------------------------------------------- training
def settings(name: str, cfg: dict, args) -> dict:
    """The router's settings: the `routers.<name>` config entry, overridden by command-line arguments."""
    s = {**DEFAULTS, **((cfg.get("routers") or {}).get(name) or {})}
    for k in ("encoder", "sets", "regions", "attributes", "layers", "lr", "balanced", "min_clips", "epochs", "steps", "label_smooth", "val_frac"):
        v = getattr(args, k)
        if v is not None:
            s[k] = v.split(",") if k in ("regions", "attributes") else v
    assert s.get("encoder"), f"router {name!r} is not in the config: give --encoder and --sets"
    return s


def train(name: str, s: dict, log, seed: int = 0):
    set_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sets, allow = parse_sets(s["sets"])
    data = Data(sets, s["encoder"], log, regions=set(s["regions"]) or None, attrs=set(s["attributes"]) or None, val_frac=s["val_frac"], allow=allow)
    missing = [p for p in data.pairs if p["row"] not in data.idx]
    check(not missing, f"every labelled chunk has a cached {s['encoder']} feature ({len(missing)} missing)", log)
    live = [a for a in sorted(data.by_attr) if len({p["clip"] for p in data.by_attr[a]["train"]}) >= 2 and len(data.counts(a)) >= 2]
    check(bool(live), "at least one attribute has two classes and two train clips", log)
    nodes = node_space(data, live, s["min_clips"])
    model = M.Router({a: data.space[a] for a in live}, nodes, s["encoder"], int(data.X.shape[1]), layers=s["layers"]).to(dev)
    log.info("router %s: %.2fM parameters (body %.2fM), %d heads on frozen %s", name, sum(p.numel() for p in model.parameters()) / 1e6,
             sum(p.numel() for p in model.enc.parameters()) / 1e6, len(model.attrs), s["encoder"])
    for a in live:
        log.info("  %-28s %2d classes, %3d nodes (%3d leaves) -> head width %3d | train pairs %6d", a, len(model.meta[a]["values"]), len(nodes[a]),
                 sum(bool(c[1]) for c in nodes[a]), model.hdim[a], len(data.by_attr[a]["train"]))

    # sampling pools: per attribute, the train pairs grouped by their terminal node (leaf node, else the class's own node)
    pool = {}
    for a in live:
        by = collections.defaultdict(list)
        for p in data.by_attr[a]["train"]:
            lt = leaf_targets(model, a, p)
            v = next((v for v in p["values"] if v in model.meta[a]["values"]), None)
            if v is not None:
                by[lt[0] if lt else model.cix[a][(v, "")]].append(p)
        pool[a] = (list(by.values()), [p for ps in by.values() for p in ps])
    aw = np.array([len(pool[a][1]) for a in live], float) ** ATTR_POW
    aw = aw / aw.sum()
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=s["lr"], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=s["lr"], total_steps=s["epochs"] * s["steps"])
    best = (-1.0, -1, None)
    n_bal = int(BATCH * s["balanced"])                               # this share of a batch is drawn class-balanced, the rest follows the data

    def val_score():
        res, lf = evaluate(data, model, dev, "val", live)
        sc = [r["all"]["balanced"] for r in res.values() if r["all"]] + [q["e2e"] for q in (leaf_summary(v) for v in lf.values()) if q]
        return float(np.mean(sc)) if sc else 0.0

    for ep in range(s["epochs"]):
        model.train()
        agg, n = collections.defaultdict(float), collections.Counter()
        for _ in range(s["steps"]):
            a = live[int(np.random.choice(len(live), p=aw))]
            groups, flat = pool[a]
            picked = [groups[i] for i in np.random.randint(0, len(groups), n_bal)]
            prs = [g[np.random.randint(len(g))] for g in picked]
            prs += [flat[i] for i in np.random.randint(0, len(flat), BATCH - n_bal)]
            grid = model.meta[a]["grid"]
            rows = sorted({p["row"] for p in prs})
            z = model.embed(data.x(rows, dev))
            loss, lg = losses(model, z, batch_labels(data, model, rows, grid, "train", dev), s["label_smooth"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            for k, v in lg.items():
                agg[k] += v; n[k] += 1
        if s["val_frac"] > 0 and (ep % 4 == 3 or ep == s["epochs"] - 1):
            vs = val_score()
            if vs > best[0]:
                best = (vs, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
            log.info("epoch %2d  val score %.3f (best %.3f at %d) | %s", ep, vs, best[0], best[1],
                     "  ".join(f"{k.split('.')[1]} {agg[k] / max(n[k], 1):.2f}" for k in sorted(agg))[:400])
    if best[2] is not None:
        model.load_state_dict(best[2])
        log.info("keeping epoch %d (val %.3f)", best[1], best[0])

    res, lf = evaluate(data, model, dev, "test", live)
    tau = {}
    for a in live:
        if lf.get(a):
            tau[a] = fit_tau(lf[a], PRECISION)
            model.tau[a] = tau[a][0]
    report = _report(name, s, model, nodes, res, lf, tau, best[1])
    out = work("results", f"router_{name}.md")
    out.write_text(report)
    print(report)
    space = {a: {k: data.space[a][k] for k in ("region", "grid", "scope", "multi", "values")} for a in live}
    M.save(model, name, space, {"settings": s, "seed": seed, "epoch": best[1],
                                     "result": {"class": res, "leaf": {a: leaf_summary(v) for a, v in lf.items()}}})
    log.info("wrote %s and %s", M.path(name), out)
    return model


def _report(name, s, model, nodes, res, lf, tau, epoch) -> str:
    f3 = lambda q, k: f"{q[k]:.3f}" if q else "-"                                           # noqa: E731
    sets = parse_sets(s["sets"])[0]
    T = [f"# Router `{name}`: frozen `{s['encoder']}`, sets `{s['sets']}`\n",
         f"- {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, {s['epochs']} x {s['steps']} steps, epoch {epoch} kept\n",
         f"Held-out chunks; `{sets[0]}` and the other sets are reported separately.\n",
         f"| attribute | classes | nodes | {sets[0]}: chunks | accuracy | **balanced** | majority | other sets: chunks | accuracy | balanced |", "|---|---|---|---|---|---|---|---|---|---|"]
    for a in sorted(res, key=lambda a: -((res[a]["first"] or res[a]["all"])["balanced"])):
        g, c = res[a]["first"], res[a]["other"]
        T.append(f"| {a} | {res[a]['values']} | {res[a]['classes']} | {g['n'] if g else 0} | {f3(g, 'acc')} | **{f3(g, 'balanced')}** | {f3(g, 'majority')} | "
                 f"{c['n'] if c else 0} | {f3(c, 'acc')} | {f3(c, 'balanced')} |")
    T += ["\n## Leaves (chunks whose labelled leaf is a node)\n",
          "| attribute | leaves | held-out chunks | leaf accuracy given the class | **end to end** | serving threshold | precision | coverage |", "|---|---|---|---|---|---|---|---|"]
    for a in sorted(res):
        q = leaf_summary(lf.get(a) or [])
        if q:
            T.append(f"| {a} | {sum(bool(c[1]) for c in nodes[a])} | {q['n']} | {q['given_class']:.3f} | **{q['e2e']:.3f}** | "
                     f"{tau[a][0]:.2f} | {tau[a][1]:.3f} | {tau[a][2]:.3f} |")
    return "\n".join(T) + "\n"


def selftest():
    """The model and the loss on synthetic labels: a leaf implies its class, a missing label gives no gradient."""
    torch.manual_seed(0)
    space = {"speech.emotion": {"values": ["angry", "happy", "sad"], "multi": False, "grid": "g3", "region": "speech"},
             "sound.event": {"values": ["animal", "tool"], "multi": True, "grid": "g3", "region": "sound"}}
    nodes = {"speech.emotion": [("angry", ""), ("angry", "shouting"), ("happy", ""), ("happy", "amused"), ("happy", "excited"), ("sad", "")],
             "sound.event": [("animal", ""), ("animal", "dog barking"), ("animal", "cat meowing"), ("tool", "")]}
    m = M.Router(space, nodes, "fake", 16)
    z = m.embed(torch.randn(4, 16))
    s = m.node_logits("speech.emotion", z)
    V = m.class_score("speech.emotion", s)
    assert V.shape == (4, 3) and torch.allclose(V[:, 1], torch.logsumexp(s[:, 2:5], 1), atol=1e-4)
    assert torch.allclose(V[:, 2], s[:, 5], atol=1e-4)                                # a class with no leaf is its own node
    batch = {"speech.emotion": {"ix": torch.tensor([0, 1]), "y": torch.tensor([[1., 0, 0], [0, 1., 0]]), "lt_ix": torch.tensor([1]), "lt_c": torch.tensor([3])}}
    loss, lg = losses(m, z, batch)
    loss.backward()
    assert "speech.emotion/leaf" in lg and m.head[M.key("sound.event")][1].weight.grad is None      # no label -> no gradient
    assert parse_sets("pool:music.mood+music.vocals,musiccaps") == (["pool", "musiccaps"], {0: {"music.mood", "music.vocals"}})
    recs = [{"named_gold": True, "named_pred": True, "right": i % 5 != 0, "class_ok": True, "conf": 0.9, "first": True} for i in range(40)]
    assert fit_tau(recs, 0.75)[0] == 0.2 and fit_tau(recs, 0.9)[0] == 1.01
    print("router train selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="", help="a router of the `routers:` config section, or a new name with --encoder / --sets")
    ap.add_argument("--encoder"); ap.add_argument("--sets"); ap.add_argument("--regions"); ap.add_argument("--attributes")
    ap.add_argument("--layers", type=int, help="depth of the body"); ap.add_argument("--lr", type=float)
    ap.add_argument("--balanced", type=float, help="share of a batch drawn class-balanced"); ap.add_argument("--min-clips", dest="min_clips", type=int)
    ap.add_argument("--epochs", type=int); ap.add_argument("--steps", type=int); ap.add_argument("--label-smooth", dest="label_smooth", type=float)
    ap.add_argument("--val-frac", dest="val_frac", type=float); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        assert a.name, "--name"
        cfg = load_config()
        _, log = setup_run(f"router_train_{a.name}", cfg)
        train(a.name, settings(a.name, cfg, a), log, a.seed)
