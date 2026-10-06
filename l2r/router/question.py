"""Question classifier: does the answer to this question name an environmental sound, and of which kind?

At inference the sound-identity heads read the chunks whose region gate heard environmental sound. When no chunk
did, this classifier decides from the question alone whether the answer names a sound; if so, the heads read the
whole clip. It never removes nodes.

Training data are question-answer records of AF-Think (text only; their audio is training audio, never a
benchmark). The label of a record is the set of sound-identity attributes whose class or leaf names occur in its
ANSWER; the classifier (one logistic regression per attribute on a MiniLM sentence embedding) predicts that set
from the QUESTION.

    python -m l2r.router.question data       # -> <work>/question/data.json
    python -m l2r.router.question train      # -> checkpoints/question/model.pkl
    python -m l2r.router.question predict --benchmarks mmau,mmar,sakura     # -> <work>/question/<benchmark>.json

AF-Think's record files are read from `data/AF-Think/<subset>/<split>.json` when present there, otherwise downloaded
from the Hugging Face hub (`nvidia/AF-Think`, JSON files only).
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np

from l2r.common import check, ckpt, data, load_config, save_json, setup_run, work

AF_THINK = "nvidia/AF-Think"
SKIP_SPLITS = ("TUT_Urban",)                # acoustic-scene recordings held out of all training
STOP = {"sound", "sounds", "noise", "other", "generic", "specific", "surface", "type", "effect", "background"}
PER_SPLIT = 4000
MIN_POSITIVES = 20
KEEP_P = 0.3                                # predictions below this are not stored


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def vocabulary(nodes: dict) -> dict[str, set]:
    """node name -> {(attribute, class)} for every class and leaf of the sound tree; generic single words are dropped."""
    V = {}
    for at, nd in nodes.items():
        for v, l in nd:
            for name in (v, l):
                n = norm(name.split("(")[0])
                if n and n not in STOP and len(n) >= 3:
                    V.setdefault(n, set()).add((at, v))
    return V


def af_think_dir() -> Path:
    """The folder holding AF-Think's record files (<subset>/<split>.json)."""
    local = data("AF-Think")
    if local.exists() and any(local.glob("*/*.json")):
        return local
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(AF_THINK, repo_type="dataset", allow_patterns=["*.json"]))


def af_think_splits(base: Path):
    """(split name, file) for every record file of AF-Think."""
    for d in sorted(p.name for p in base.iterdir() if p.is_dir()):
        for f in sorted(p.name for p in (base / d).iterdir()):
            if f.endswith(".json") and f != "counts_report.json":
                yield f"{d}/{f[:-5]}", base / d / f


def answer_of(rec: dict) -> str:
    """The record's answer: its <CONCLUSION>, else the last sentence of the response."""
    gpt = "\n".join(t.get("value", "") for t in rec.get("conversations", ()) if t.get("from") == "gpt")
    m = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", gpt, re.S)
    if m:
        return m.group(1)
    sents = re.split(r"(?<=[.!?])\s+", gpt.strip())
    return sents[-1] if sents else ""


def question_of(rec: dict) -> str:
    h = next((t.get("value", "") for t in rec.get("conversations", ()) if t.get("from") == "human"), "")
    return re.split(r"Output the answer with", h.replace("<sound>", ""))[0].strip()


def build_data(log, per_split: int = PER_SPLIT):
    from l2r.router import sound
    V = vocabulary(sound.load_nodes())
    pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, V), key=len, reverse=True)) + r")\b")
    out, per = [], Counter()
    for split, p in af_think_splits(af_think_dir()):
        if any(s in split for s in SKIP_SPLITS):
            continue
        recs = json.loads(Path(p).read_text())
        recs = recs if isinstance(recs, list) else []
        for i in np.random.default_rng(0).permutation(len(recs))[:per_split]:
            q, ans = question_of(recs[i]), norm(answer_of(recs[i]))
            if not q or not ans:
                continue
            hits = {av for w in pat.findall(ans) for av in V[w]}
            out.append({"split": split, "q": q, "attrs": sorted({x[0] for x in hits})})
            per[split] += 1
    save_json(out, work("question", "data.json"), indent=None)
    n = sum(bool(r["attrs"]) for r in out)
    log.info("records %d over %d splits; %d (%.1f%%) name a sound in the answer; per attribute %s", len(out), len(per), n, 100 * n / len(out),
             dict(Counter(x for r in out for x in r["attrs"]).most_common()))
    check(n > 5000, "more than 5,000 questions whose answer names a sound", log)


_EMBEDDER = None


def embed(questions: list[str]) -> np.ndarray:
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(load_config()["models"]["text_embedder"])
    return _EMBEDDER.encode(list(questions), batch_size=256, show_progress_bar=False, normalize_embeddings=True)


def train(log):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from l2r.router import sound
    D = json.load(open(work("question", "data.json")))
    attrs = list(sound.load_nodes())
    X = embed([d["q"] for d in D])
    Y = np.array([[at in d["attrs"] for at in attrs] for d in D], dtype=int)
    splits = sorted({d["split"] for d in D})
    held = set(np.random.default_rng(0).choice(splits, max(1, len(splits) // 5), replace=False))     # held out by AF-Think split
    te = np.array([d["split"] in held for d in D])
    tr = ~te
    clf, rows = {}, []
    for j, at in enumerate(attrs):
        if Y[tr, j].sum() < MIN_POSITIVES:
            continue
        clf[at] = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced").fit(X[tr], Y[tr, j])
        if Y[te, j].sum() >= 5:
            p = clf[at].predict_proba(X[te])[:, 1]
            rows.append({"attribute": at, "held_out_positives": int(Y[te, j].sum()), "average_precision": round(float(average_precision_score(Y[te, j], p)), 3),
                         "base_rate": round(float(Y[te, j].mean()), 3)})
    pickle.dump({"clf": clf, "attrs": attrs}, open(ckpt("question", "model.pkl"), "wb"))
    save_json({"held_out_splits": sorted(held), "attributes": rows}, ckpt("question", "result.json"))
    for r in rows:
        log.info("%s", r)


_MODEL = None


def predict(questions: list[str], keep: float = KEEP_P) -> list[dict[str, float]]:
    """Per question: {sound attribute: probability that the answer names a sound of this kind}, probabilities >= `keep`
    only, rounded to three decimals. The sound fallback opens when any value is >= 0.5."""
    global _MODEL
    if _MODEL is None:
        _MODEL = pickle.load(open(ckpt("question", "model.pkl"), "rb"))
    X = embed(questions)
    P = {at: m.predict_proba(X)[:, 1] for at, m in _MODEL["clf"].items()}
    return [{at: round(float(P[at][i]), 3) for at in P if P[at][i] >= keep} for i in range(len(questions))]


def predict_benchmarks(names: list[str], log) -> dict:
    """<work>/question/<benchmark>.json = {item id: {attribute: p}} for every item of the benchmark."""
    from l2r import benchmarks
    out = {}
    for name in names:
        items = benchmarks.load(name)
        P = predict([it["question"] for it in items])
        out[name] = {str(it.get("id", i)): p for i, (it, p) in enumerate(zip(items, P))}
        save_json(out[name], work("question", f"{name}.json"), indent=None)
        log.info("%s: %d of %d questions point to a sound (p >= 0.5)", name, sum(any(v >= 0.5 for v in p.values()) for p in P), len(items))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["data", "train", "predict"])
    ap.add_argument("--benchmarks", default="mmau,mmar,sakura")
    a = ap.parse_args()
    _, log = setup_run(f"question_{a.stage}")
    if a.stage == "data":
        build_data(log)
    elif a.stage == "train":
        train(log)
    else:
        predict_benchmarks(a.benchmarks.split(","), log)
