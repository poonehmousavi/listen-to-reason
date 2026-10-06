"""The language head: `speech.language` predicted from the transcript's text.

When speech is transcribed, the language of the transcript is in its characters: TF-IDF over character 1-2-grams
and logistic regression, trained on the FLEURS reference sentences (text only) of the languages the tree holds as
leaves, so the head can only name a node that exists.

    python -m l2r.router.language        ->  <checkpoints>/language/text_head.pkl

Needs `<data>/FLEURS/meta.json`: {code: {"language": name, "clips": [{"text": reference sentence, ...}]}}.
"""
from __future__ import annotations

import json
import pickle
import re

import numpy as np

from l2r.common import check, ckpt, data, setup_run

FLEURS_META = "FLEURS/meta.json"
ALIAS = {"mandarin chinese": "chinese", "cantonese chinese": "cantonese", "norwegian bokmal": "norwegian", "spanish (latin america)": "spanish"}


def path():
    return ckpt("language", "text_head.pkl")


def clean(text) -> str:
    """Transcript text as the head reads it: timestamps removed, lower case."""
    return re.sub(r"\[[0-9:.]+\]", "", str(text)).strip().lower()


def sentences(leaf2val: dict[str, str]) -> tuple[list[str], list[str], set]:
    """FLEURS reference sentences of the languages that are leaves of the tree -> (texts, leaves, languages left out)."""
    X, Y, skipped = [], [], set()
    for d in json.loads(data(FLEURS_META).read_text()).values():
        name = d["language"].lower()
        name = ALIAS.get(name, name)
        if name not in leaf2val:
            name = next((lf for lf in leaf2val if lf in name.split()), None)
        if name is None:
            skipped.add(d["language"])
            continue
        for c in d["clips"]:
            if c.get("text"):
                X.append(clean(c["text"]))
                Y.append(name)
    return X, Y, skipped


def train(log) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    tree = json.loads(ckpt("tree", "tree.json").read_text())["speech"]["language"]
    leaf2val = {lf: v for v, node in tree.items() for lf in (node.get("leaves") or {})}
    X, Y, skipped = sentences(leaf2val)
    log.info("%d sentences over %d of the tree's %d language leaves; %d FLEURS languages are not leaves", len(X), len(set(Y)), len(leaf2val), len(skipped))
    ix = np.random.RandomState(0).permutation(len(X))
    held = set(ix[: len(X) // 7].tolist())
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=40000)
    clf = LogisticRegression(max_iter=300, C=30).fit(vec.fit_transform([X[i] for i in ix if i not in held]), [Y[i] for i in ix if i not in held])
    acc = float(np.mean(clf.predict(vec.transform([X[i] for i in held])) == np.array([Y[i] for i in held])))
    check(acc > 0.85, f"held-out sentences: accuracy {acc:.3f} (n={len(held)})", log)
    head = {"vec": vec, "clf": clf, "leaf2val": leaf2val, "heldout": acc, "classes": sorted(set(Y))}
    pickle.dump(head, open(path(), "wb"))
    log.info("wrote %s", path())
    return head


def load() -> dict:
    return pickle.load(open(path(), "rb"))


def predict(head: dict, transcript: str) -> tuple[str, str, float]:
    """-> (class, leaf, probability) of the most likely language of a transcript."""
    p = head["clf"].predict_proba(head["vec"].transform([clean(transcript)]))[0]
    j = int(p.argmax())
    leaf = head["clf"].classes_[j]
    return head["leaf2val"][leaf], leaf, float(p[j])


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    _, log = setup_run("language_head")
    train(log)
