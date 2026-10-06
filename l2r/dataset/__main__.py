"""Dataset generation: chunk the pool, label every chunk with the expert classifiers and the audio LLM, assemble.

    python -m l2r.dataset                    # every stage in order; a finished stage is skipped, an interrupted one resumes
    python -m l2r.dataset --from annotate

Stages (each is also `python -m l2r.dataset.build <stage>`; outputs under <work>/sets/pool/ unless noted):
  gender-centroids   speaker-embedding centroids of the gender expert (IEMOCAP)        -> <checkpoints>/experts/gender_centroids.npz
  chunk              the chunk grid and the chunk wavs of the pool                       -> rows.jsonl, <work>/chunks/
  experts            PANNs, WavLM-SV, emotion2vec, Whisper language id, librosa          -> fired_<expert>.jsonl
  annotate           the audio LLM fills the annotation template on every chunk form     -> fired_qwen3_omni.jsonl   (GPU-days)
  music-properties   the music-property form on every clip                              -> music_form.jsonl
  assemble           the label table the tree is built from                              -> index.jsonl
"""
from __future__ import annotations

from l2r.common import ckpt, read_jsonl
from l2r.dataset import index as I
from l2r.dataset.annotator import MUSIC_FILE, TOOL
from l2r.dataset.experts import GENDER_CENTROIDS, ORDER
from l2r.pipeline import Step, parser, run


def complete(name: str, tool: str) -> bool:
    """Every clip of the set carries the tool's `done` marker."""
    rows, shard = I.sdir(name) / "rows.jsonl", I.sdir(name) / f"fired_{tool}.jsonl"
    if not rows.exists() or not shard.exists():
        return False
    return {r["clip"] for r in I.rows(name)} <= {r["done"] for r in read_jsonl(shard) if "done" in r}


def music_done(name: str) -> bool:
    rows, out = I.sdir(name) / "rows.jsonl", I.sdir(name) / MUSIC_FILE
    return rows.exists() and out.exists() and {r["audio"] for r in read_jsonl(out)} >= {r["audio_path"] for r in I.rows(name) if r["grid"] == "clip"}


def steps(name: str) -> list[Step]:
    b = ["l2r.dataset.build"]; s = ["--set", name]
    return [Step("gender-centroids", ["l2r.dataset.experts", "--gender-centroids"], [ckpt(*GENDER_CENTROIDS)]),
            Step("chunk", b + ["chunk"] + s, [I.sdir(name) / "rows.jsonl"]),
            Step("experts", b + ["experts"] + s, done=lambda: all(complete(name, t) for t in ORDER)),
            Step("annotate", b + ["annotate"] + s, done=lambda: complete(name, TOOL)),
            Step("music-properties", b + ["music-properties"] + s, done=lambda: music_done(name)),
            Step("assemble", b + ["assemble"] + s, [I.sdir(name) / "index.jsonl"])]


if __name__ == "__main__":
    ap = parser("dataset", __doc__); ap.add_argument("--set", default="pool", help="name of the clip set (its manifest is assets/<set>.jsonl)")
    a = ap.parse_args(); run(steps(a.set), a)
