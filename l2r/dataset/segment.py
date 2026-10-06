"""Chunking: the fixed grids every label is indexed on.

Each clip gets one `clip` row and two grids of rows: `g3` (3 s chunks, for speech, sound and the region
gate) and `g10` (10 s chunks, for music). Every grid row is written once as a 16 kHz mono wav, which is what
the encoders read.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from l2r.common import audio, work

MAX_S = 60.0                     # only the first minute of a clip is chunked
SR = 16000
GRID_OF = {"speech": "g3", "sound": "g3", "cross": "g3", "music": "g10"}


def clip_id(path: str) -> str:
    return hashlib.md5(str(path).encode()).hexdigest()[:16]


def grid(dur: float, chunk: float, cap: float = MAX_S) -> list[tuple[float, float]]:
    """Non-overlapping windows. A tail shorter than half a chunk is dropped; a clip shorter than one chunk is one row."""
    dur = min(float(dur), cap)
    if dur < chunk:
        return [(0.0, round(dur, 2))] if dur >= 0.5 else []
    out, t = [], 0.0
    while t + chunk <= dur + 1e-6:
        out.append((round(t, 2), round(t + chunk, 2))); t += chunk
    if dur - t >= chunk / 2:
        out.append((round(t, 2), round(dur, 2)))
    return out


def chunk_secs(schema: dict) -> dict[str, float]:
    g = {}
    for r, R in schema["regions"].items():
        c = float(R["chunk_sec"]); k = GRID_OF[r]
        assert g.setdefault(k, c) == c, f"regions sharing grid {k} disagree on chunk_sec"
    return g


def rows_for(path: str, dur: float, schema: dict, source: str = "", extra: dict | None = None, hop: float = 0.0) -> list[dict]:
    """The rows of one clip. `hop` > 0 adds a second 3 s grid shifted by `hop` seconds (overlapping chunks, used at inference)."""
    cid = clip_id(path); base = {"clip": cid, "audio_path": str(path), "source": source, **(extra or {})}
    rows = [{**base, "row": f"{cid}:clip", "grid": "clip", "t0": 0.0, "t1": round(min(dur, MAX_S), 2)}]
    secs = chunk_secs(schema)
    for g, c in sorted(secs.items()):
        rows += [{**base, "row": f"{cid}:{g}:{a:g}", "grid": g, "t0": a, "t1": b} for a, b in grid(dur, c)]
    if hop:
        for a, b in grid(max(min(dur, MAX_S) - hop, 0.0), secs["g3"]):
            rows.append({**base, "row": f"{cid}:g3:{a + hop:g}", "grid": "g3", "t0": round(a + hop, 2), "t1": round(b + hop, 2)})
    return rows


def project(span: tuple[float, float], rows: list[dict], min_ov: float = 0.5) -> list[dict]:
    """Grid rows a time span belongs to: overlap >= min_ov seconds, or >= half of a span shorter than that."""
    a, b = span; need = min(min_ov, (b - a) / 2) if b > a else 0.0
    return [r for r in rows if r["grid"] != "clip" and min(b, r["t1"]) - max(a, r["t0"]) >= max(need, 1e-6)]


def wav_path(row: dict) -> Path:
    return work("chunks", row["clip"], f"{row['grid']}_{row['t0']:g}.wav")


def materialise(path: str, rows: list[dict]) -> float:
    """Write every grid row of one clip as a wav (idempotent); returns the clip's duration as loaded."""
    import librosa
    import soundfile as sf
    y, _ = librosa.load(str(audio(path)), sr=SR, mono=True, duration=MAX_S)
    for r in rows:
        if r["grid"] == "clip":
            continue
        w = wav_path(r)
        if not w.exists():
            sf.write(str(w), y[int(r["t0"] * SR): int(r["t1"] * SR)], SR)
    return len(y) / SR


def chunk_wav(row: dict) -> Path:
    """The wav of one grid row, cut on demand."""
    p = wav_path(row)
    if not p.exists():
        materialise(row["audio_path"], [row])
    return p


def selftest(schema: dict):
    assert grid(10.0, 3.0) == [(0, 3), (3, 6), (6, 9)], grid(10.0, 3.0)
    assert grid(11.0, 3.0)[-1] == (9.0, 11.0) and grid(2.0, 3.0) == [(0.0, 2.0)] and grid(0.3, 3.0) == []
    assert grid(200.0, 10.0)[-1] == (50.0, 60.0) and len(grid(30.0, 10.0)) == 3
    assert chunk_secs(schema) == {"g3": 3.0, "g10": 10.0}
    rows = rows_for("x/a.wav", 10.0, schema, "t")
    assert [r["grid"] for r in rows] == ["clip", "g10", "g3", "g3", "g3"] and len({r["row"] for r in rows}) == 5
    assert [r["t0"] for r in rows_for("x/a.wav", 10.0, schema, hop=1.5) if r["grid"] == "g3"] == [0.0, 3.0, 6.0, 1.5, 4.5, 7.5]
    assert [r["t0"] for r in project((2.4, 4.0), rows)] == [0.0, 0.0, 3.0]
    assert [r["t0"] for r in project((2.9, 3.2), rows) if r["grid"] == "g3"] == [3.0]
    assert [r["t0"] for r in project((2.85, 3.15), rows) if r["grid"] == "g3"] == [0.0, 3.0]
