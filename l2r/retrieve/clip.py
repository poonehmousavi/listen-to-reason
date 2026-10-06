"""One audio file -> its tree nodes, the reader's context and, given a question with options, the reader's answer.

    python -m l2r.retrieve.clip --audio clip.wav
    python -m l2r.retrieve.clip --audio clip.wav --question "Who calls the dog?" --options "a man" "a woman" "a child"
    python -m l2r.retrieve.clip --audio clip.wav --question ... --options ... --answer     # the reader (models.reader) answers

In Python:

    from l2r.retrieve import clip
    r = clip.describe("clip.wav", question="Who calls the dog?", options=["a man", "a woman", "a child"])
    r["context"]       # the text the reader receives
    r["trace"]         # every served node with its chunk span and probability
    clip.answer(r)     # {"answer": "b", "option": "a woman", ...}

The chunks, embeddings, transcript and nodes are cached under <work>/ by a name derived from the file.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

from l2r import benchmarks as B
from l2r.common import load_config, setup_run
from l2r.retrieve import asr, context as C, serve, timeline, trace as TR
from l2r.router import question as Q

DEFAULT_QUESTION = "What is heard in the recording?"


def item(audio: Path, question: str, options) -> dict:
    st = audio.stat(); name = "clip_" + hashlib.md5(f"{audio}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:12]
    opts = {B.LETTERS[i]: o for i, o in enumerate(options)}
    return B._item(name, str(audio), "clip", question or DEFAULT_QUESTION, opts, None, lambda v: v.lower())


def describe(audio, question: str = "", options=(), domains: list[str] | None = None, cfg=None, log=None) -> dict:
    """Route one clip into the tree: its nodes per chunk, the trace and the reader's context."""
    cfg = cfg or load_config(); log = log or logging.getLogger("l2r")
    p = Path(audio).resolve(); it = item(p, question, options); name = it["id"]; items = [it]
    transcripts = asr.transcribe([str(p)], name, cfg, log)
    scores = {name: Q.predict([it["question"]])[0]} if question and cfg["serve"].get("sound_heads", True) else {}
    nodes = serve.Retriever(cfg, log, domains).run(name, items, transcripts, scores)
    detected = timeline.detect(name, items, cfg, log)
    ctx = C.build(items, nodes, transcripts, timeline.lines(items, detected), scores, cfg)[0]["ctx"]["ours"]
    rows = nodes["clips"].get(str(p), [])
    return {"item": it, "context": ctx, "transcript": transcripts.get(str(p), ""), "nodes": rows, "trace": TR.trace(rows)}


def answer(r: dict, model: str | None = None, cfg=None) -> dict:
    """The frozen reader answers the item's question from the context (letter log-probs, with the paper's weak blind prior)."""
    from l2r.reason import reader as R
    cfg = cfg or load_config(); model = model or cfg["models"]["reader"]
    assert r["item"]["choices"], "describe() needs a question and its options for an answer"
    it = {**r["item"], "ctx": {"ours": r["context"], "blind": ""}}
    res = R.answer(R.Reader(model, model.split("/")[-1].lower()), [it], "ours", [], 1, cfg["reader"]["beta"])[0]
    return {"answer": res["ours_poe"], "option": it["choices"][res["ours_poe"]], "without_context": res["blind"], "model": model}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audio", required=True); ap.add_argument("--question", default="")
    ap.add_argument("--options", nargs="*", default=[], help="answer options, in order (a), (b), ...")
    ap.add_argument("--domains", default="", help="new-domain routers to add, e.g. birds_k5")
    ap.add_argument("--answer", action="store_true", help="run the reader (models.reader) on the question")
    a = ap.parse_args(); cfg = load_config(); _, log = setup_run("clip", cfg)
    r = describe(a.audio, a.question, a.options, [d for d in a.domains.split(",") if d], cfg, log)
    print("\n".join(f"{TR.clause_head(t['attr']) + ': ' + t['node']:<60s} {t['where']:<28s} " + (f"p {t['p']:.2f}" if t["p"] is not None else "") for t in r["trace"]))
    print("\n" + r["context"])
    if a.answer:
        out = answer(r, cfg=cfg); print(f"\nanswer: ({out['answer']}) {out['option']}    [without the context: ({out['without_context']})]")


if __name__ == "__main__":
    main()
