"""Event-timeline head: PANNs' classification layer, fine-tuned on frame-level AudioSet strong labels.

The head is one linear layer over PANNs' frame features (2048-d, 31 frames per 10 s). It starts as a copy of PANNs'
own 527-way classification layer and is trained with a binary cross-entropy on the strong labels (exact onsets and
offsets, on the classes the strong labels cover) plus a binary cross-entropy to PANNs' original frame scores, so
classes the strong labels rarely cover are kept. PANNs itself stays frozen.

    python -m l2r.router.event_head          # needs `python -m l2r.router.audioset strong`; -> checkpoints/events/head.pt
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from l2r.common import ckpt, setup_run
from l2r.router import audioset as AS

N_CLASSES = 527


def load(dev="cpu") -> nn.Linear:
    """The trained head: frame features [.., 2048] -> 527 class logits."""
    head = nn.Linear(2048, N_CLASSES)
    head.load_state_dict(torch.load(ckpt("events", "head.pt"), map_location="cpu")["state"])
    return head.to(dev).eval()


def _runs(col):
    out, t = [], 0
    while t < len(col):
        if col[t]:
            s = t
            while t < len(col) and col[t]:
                t += 1
            out.append((s, t))
        else:
            t += 1
    return out


def event_f1(head, orig, batch, test, Y, has_strong, n=3000) -> dict:
    """Held-out agreement with the strong labels (per second and class), for the tuned head and for PANNs' own layer."""
    from l2r import events as EV
    from l2r.encoders import panns_labels
    labels = panns_labels()
    anc = EV.load_ancestors(labels)
    P = dict(EV.DEFAULTS, smooth=1)

    def seg(p):
        return EV.segment(p, labels, AS.PANNS_FRAME_S, P["tau_on"], P["tau_off"], P["min_dur"], P["gap"], int(P["k"]), 1, ancestors=anc)

    def f1(a, b):
        cells = lambda E: {(int(t), e.label) for e in E for t in np.arange(np.floor(e.t0), np.ceil(e.t1)) if t < AS.CLIP_S}  # noqa: E731
        A, B = cells(a), cells(b)
        return 2 * len(A & B) / max(len(A) + len(B), 1)

    def gold(i):
        return [EV.Event(labels[c], s * AS.PANNS_FRAME_S, e * AS.PANNS_FRAME_S, 1.0)
                for c in np.nonzero(Y[i].any(0))[0] if has_strong[c] for s, e in _runs(Y[i][:, c])]
    R = {"tuned": [], "panns": []}
    with torch.no_grad():
        for j in range(0, min(len(test), n), 256):
            ix = np.sort(test)[j:j + 256]
            x, _ = batch(ix)
            pt, po = torch.sigmoid(head(x)).cpu().numpy(), torch.sigmoid(orig(x)).cpu().numpy()
            for k, i in enumerate(ix):
                g = gold(i)
                if g:
                    R["tuned"].append(f1(seg(pt[k]), g))
                    R["panns"].append(f1(seg(po[k]), g))
    return {"held_out_event_f1": float(np.mean(R["tuned"])), "held_out_event_f1_panns": float(np.mean(R["panns"])), "n": len(R["tuned"])}


def train(a, log):
    from l2r.encoders import load_panns
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    panns, _ = load_panns("cpu")
    X = np.load(AS.out("panns_fc1.npy"), mmap_mode="r")
    Y = np.unpackbits(np.load(AS.out("strong_y.npy")), axis=2)[:, :, :N_CLASSES]
    sets = json.load(open(AS.out("strong_sets.json")))
    has = torch.zeros(N_CLASSES, dtype=torch.bool)
    has[np.load(AS.out("has_strong.npy"))] = True
    has_np = has.numpy().copy()
    has = has.to(dev)
    tr = np.array([i for i, s in enumerate(sets) if s == "train"])
    te = np.array([i for i, s in enumerate(sets) if s == "test"])
    rng = np.random.default_rng(0)
    va = rng.choice(tr, len(tr) // 20, replace=False)
    tr = np.setdiff1d(tr, va)
    orig = nn.Linear(2048, N_CLASSES)
    orig.load_state_dict(panns.fc_audioset.state_dict())
    orig.to(dev).eval()
    head = nn.Linear(2048, N_CLASSES)
    head.load_state_dict(panns.fc_audioset.state_dict())
    head.to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.0)

    def batch(ix):
        return torch.from_numpy(np.asarray(X[ix], np.float32)).to(dev), torch.from_numpy(Y[ix].astype(np.float32)).to(dev)

    def loss_on(ix):
        x, y = batch(ix)
        z = head(x)
        with torch.no_grad():
            t = torch.sigmoid(orig(x))
        return F.binary_cross_entropy_with_logits(z[..., has], y[..., has]) + a.keep * F.binary_cross_entropy_with_logits(z, t)
    best, state, best_epoch = 9e9, None, 0
    for ep in range(a.epochs):
        perm = rng.permutation(tr)
        for j in range(0, len(perm), 256):
            loss = loss_on(np.sort(perm[j:j + 256]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = float(np.mean([loss_on(np.sort(va)[j:j + 512]).item() for j in range(0, len(va), 512)]))
        log.info("epoch %d validation loss %.5f", ep, v)
        if v < best:
            best, state, best_epoch = v, {k: t.detach().clone() for k, t in head.state_dict().items()}, ep
    head.load_state_dict(state)
    torch.save({"state": {k: t.cpu() for k, t in state.items()}, "epoch": best_epoch}, ckpt("events", "head.pt"))
    res = {"epoch": best_epoch, **event_f1(head, orig, batch, te, Y, has_np)}
    log.info("RESULT %s", res)
    json.dump(res, open(ckpt("events", "result.json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--keep", type=float, default=1.0, help="weight of the term that keeps the head close to PANNs' original frame scores")
    a = ap.parse_args()
    _, log = setup_run("event_head")
    train(a, log)
