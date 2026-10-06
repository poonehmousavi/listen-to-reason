"""Music-property heads: period, vocal style, form, rhythm feel and dynamics of a music clip.

The audio LLM annotator fills a closed form about the music of every pool clip (the form's answers, one JSON line per
clip: {"audio": path, "new": {"music_period": ..., ...}}, in `<work>/sets/pool/music_form*.jsonl`). Each field
becomes one attribute, read by a two-layer MLP on the frozen MuQ-MuLan clip embedding and trained with a
class-weighted cross-entropy. The annotator never runs at inference. A class needs at least 10 training clips;
the answers "unclear", "unsure" and "not music" are never classes.

    python -m l2r.router.music_properties embed     # MuQ-MuLan embeddings of the pool's music clips
    python -m l2r.router.music_properties train     # -> checkpoints/music_properties/heads.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from l2r.common import audio, check, ckpt, read_jsonl, setup_run, work

ENCODER = "muq_mulan"
ATTR = {"music_period": "music.period", "music_vocal_style": "music.vocal_style", "music_form": "music.form",
        "music_rhythm_feel": "music.rhythm_feel", "music_dynamics": "music.dynamics"}       # form field -> attribute
NEVER = {"unclear", "unsure", "not music"}
MIN_CLIPS = 10
EPOCHS = 60


def form_answers() -> dict[str, dict]:
    """audio path -> the form's answers, for the pool clips the annotator heard as music."""
    out = {}
    for f in sorted(work("sets", "pool", "x").parent.glob("music_form*.jsonl")):
        for r in read_jsonl(f):
            new = r.get("new") or {}
            if new and new.get("music_period") not in (None, "not music"):
                out[r["audio"]] = new
    return out


def embed_paths(paths: list[str], log=None, encoder=None) -> np.ndarray:
    """MuQ-MuLan embeddings (centre 10 s of each file)."""
    if encoder is None:
        from l2r.encoders import build_encoder
        encoder = build_encoder(ENCODER)
    out = []
    for i in range(0, len(paths), 32):
        out.append(np.asarray(encoder.embed([str(audio(p)) for p in paths[i:i + 32]]), np.float32))
        if log and i % 640 == 0:
            log.info("embed %d / %d", i, len(paths))
    return np.concatenate(out)


def embed(log):
    paths = sorted(form_answers())
    np.savez(work("music_properties", "pool.npz"), paths=np.array(paths), x=embed_paths(paths, log))
    log.info("pool: %d music clips embedded", len(paths))


def head(dim: int, n_classes: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, n_classes))


def train(log):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    answers = form_answers()
    E = np.load(work("music_properties", "pool.npz"))
    paths = [str(p) for p in E["paths"]]
    X = E["x"] / np.linalg.norm(E["x"], axis=1, keepdims=True)
    held = np.array([int(hashlib.md5(p.encode()).hexdigest(), 16) % 5 == 0 for p in paths])       # one clip in five is held out
    heads, res = {}, {}
    for field, attr in ATTR.items():
        y = [answers[p].get(field) if p in answers else None for p in paths]
        ok = np.array([v is not None and v not in NEVER for v in y])
        count = Counter(v for v, o, h in zip(y, ok, held) if o and not h)
        classes = [v for v, c in count.items() if c >= MIN_CLIPS]
        if len(classes) < 2:
            log.info("%s: fewer than two classes with %d training clips, skipped", attr, MIN_CLIPS)
            continue
        ci = {v: i for i, v in enumerate(classes)}
        sel = np.array([o and y[i] in ci for i, o in enumerate(ok)])
        Y = np.array([ci.get(v, -1) for v in y])
        tr, te = sel & ~held, sel & held
        m = head(X.shape[1], len(classes))
        w = torch.tensor([1.0 / count[v] for v in classes])
        w = w / w.mean()
        opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-3)
        Xt, Yt = torch.tensor(X[tr]), torch.tensor(Y[tr])
        for _ in range(EPOCHS):
            m.train()
            perm = torch.randperm(len(Xt))
            for j in range(0, len(perm), 64):
                b = perm[j:j + 64]
                loss = F.cross_entropy(m(Xt[b]), Yt[b], weight=w)
                opt.zero_grad()
                loss.backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            pred = m(torch.tensor(X[te])).argmax(1).numpy()
        probe = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], Y[tr])
        res[attr] = {"classes": classes, "train": int(tr.sum()), "held_out": int(te.sum()),
                     "balanced_accuracy": round(balanced_accuracy_score(Y[te], pred), 3),
                     "linear_probe": round(balanced_accuracy_score(Y[te], probe.predict(X[te])), 3), "chance": round(1 / len(classes), 3)}
        log.info("%s %s", attr, res[attr])
        heads[attr] = {"state": m.state_dict(), "values": classes, "dim": X.shape[1]}
    torch.save(heads, ckpt("music_properties", "heads.pt"))
    json.dump(res, open(ckpt("music_properties", "result.json"), "w"), indent=1)
    check(any(r["balanced_accuracy"] > r["chance"] + 0.1 for r in res.values()), "at least one attribute beats chance by 10 points on held-out clips", log)


def load() -> dict[str, tuple[nn.Sequential, list[str]]]:
    """attribute -> (head, classes). A head reads an L2-normalised MuQ-MuLan clip embedding."""
    H = torch.load(ckpt("music_properties", "heads.pt"), map_location="cpu", weights_only=False)
    out = {}
    for attr, h in H.items():
        m = head(h["dim"], len(h["values"]))
        m.load_state_dict(h["state"])
        out[attr] = (m.eval(), h["values"])
    return out


@torch.no_grad()
def predict(heads: dict, x: np.ndarray, tau: float = 0.5) -> list[tuple[str, str, float]]:
    """(attribute, class, p) of every head whose best class has probability >= tau, for one clip embedding."""
    x = torch.tensor(x / np.linalg.norm(x))[None].float()
    out = []
    for attr, (m, classes) in heads.items():
        p = m(x).softmax(1)[0].numpy()
        if p.max() >= tau:
            out.append((attr, classes[int(p.argmax())], float(p.max())))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["embed", "train"])
    a = ap.parse_args()
    _, log = setup_run(f"music_properties_{a.stage}")
    embed(log) if a.stage == "embed" else train(log)
