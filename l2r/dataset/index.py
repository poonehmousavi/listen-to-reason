"""The label table of a set of clips: which annotator fired which label on which chunk.

A *set* is a folder `<work>/sets/<name>/`:
  rows.jsonl             the chunk grid: {row, clip, audio_path, source, grid, t0, t1, gold?}
  fired_<tool>.jsonl     one shard per annotator, so one can be re-run alone:
                         {row, attr, value, leaf, score, tool, span} and {"done": clip} markers (resume is per clip)
  index.jsonl            the assembled table, rows with their fired labels; the tree is always re-derived from it
Every (label, annotator) pair is kept; which label wins is decided when the tree is built.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

from l2r.common import read_jsonl, work

FIELDS = ("row", "attr", "value", "leaf", "score", "tool", "span")


def sdir(name: str) -> Path:
    """The folder of a set."""
    d = work("sets", name, "x").parent; d.mkdir(parents=True, exist_ok=True)
    return d


def write_rows(name: str, rows: list[dict]):
    (sdir(name) / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def rows(name: str) -> list[dict]:
    return read_jsonl(sdir(name) / "rows.jsonl")


def fired(row: str, attr: str, value: str, tool: str, score=None, leaf: str = "", span=None) -> dict:
    assert attr.count(".") == 1 and value, (attr, value)
    return {"row": row, "attr": attr, "value": str(value).lower(), "leaf": leaf or "",
            "score": None if score is None else round(float(score), 4), "tool": tool,
            "span": None if span is None else [round(float(span[0]), 2), round(float(span[1]), 2)]}


class Shard:
    """Append-only shard of one annotator, resumable per clip."""

    def __init__(self, name: str, tool: str):
        self.p = sdir(name) / f"fired_{tool}.jsonl"; self.tool = tool
        recs = read_jsonl(self.p); self.done = {r["done"] for r in recs if "done" in r}
        # records of a clip whose `done` marker never landed (an interrupted job) are dropped, so the clip is redone once
        keep = [r for r in recs if "done" in r or r["row"].split(":", 1)[0] in self.done]
        if len(keep) != len(recs) or (self.p.exists() and self.p.stat().st_size and not self.p.read_text().endswith("\n")):
            tmp = self.p.with_suffix(".jsonl.tmp"); tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep)); tmp.replace(self.p)
        self.repaired = len(recs) - len(keep)
        self.fh = self.p.open("a")

    def add(self, clip: str, recs: list[dict]):
        for r in recs:
            assert r["tool"] == self.tool and set(r) == set(FIELDS), r
            self.fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.fh.write(json.dumps({"done": clip}) + "\n"); self.fh.flush(); self.done.add(clip)

    def close(self):
        self.fh.close()


def assemble(name: str) -> list[dict]:
    """rows + every annotator's shard -> index.jsonl."""
    d = sdir(name); by = collections.defaultdict(list)
    for p in sorted(d.glob("fired_*.jsonl")):
        for r in read_jsonl(p):
            if "done" not in r:
                by[r["row"]].append({k: r[k] for k in FIELDS if k != "row"})
    table = [{**r, "fired": by.get(r["row"], [])} for r in rows(name)]
    (d / "index.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in table))
    return table


def selftest():
    import shutil
    name = "_selftest"; shutil.rmtree(sdir(name))
    write_rows(name, [{"row": "c:clip", "clip": "c", "audio_path": "x/a.wav", "source": "t", "grid": "clip", "t0": 0, "t1": 6},
                      {"row": "c:g3:0", "clip": "c", "audio_path": "x/a.wav", "source": "t", "grid": "g3", "t0": 0, "t1": 3}])
    s = Shard(name, "emotion2vec"); s.add("c", [fired("c:g3:0", "speech.emotion", "Happy", "emotion2vec", 0.91)]); s.close()
    s = Shard(name, "qwen3_omni"); assert "c" not in s.done
    s.add("c", [fired("c:g3:0", "speech.emotion", "happy", "qwen3_omni", leaf="amused"),
                fired("c:g3:0", "music.instrument", "acoustic guitar", "qwen3_omni", span=(0, 3))]); s.close()
    assert "c" in Shard(name, "emotion2vec").done
    with (sdir(name) / "fired_qwen3_omni.jsonl").open("a") as fh:             # an interrupted job: clip d half-written
        fh.write(json.dumps(fired("d:clip", "speech.gender", "male voice", "qwen3_omni")) + "\n" + '{"row": "d:g3:0", "attr": "spee')
    s = Shard(name, "qwen3_omni"); assert s.repaired == 1 and "d" not in s.done and "c" in s.done
    s.add("d", [fired("d:clip", "speech.gender", "male voice", "qwen3_omni")]); s.close()
    assert len([r for r in read_jsonl(sdir(name) / "fired_qwen3_omni.jsonl") if r.get("row", "").startswith("d:")]) == 1
    assert [len(r["fired"]) for r in assemble(name)] == [0, 3]
    shutil.rmtree(sdir(name))
