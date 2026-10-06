"""Containment: training audio must stay away from the benchmarks.

Some training corpora and the benchmarks draw from the same sources. A training clip whose CLAP embedding is
within a cosine threshold of any benchmark clip is therefore left out (0.95 for the labelled clips of a head,
0.90 for the AudioSet clips of the sound heads, 0.85 for the MusicCaps label set).
"""
from __future__ import annotations

import json

import numpy as np

from l2r.common import audio, data, work

BENCHMARKS = ("mmau", "mmar", "sakura")


def clips(name: str) -> list[str]:
    """Every audio file of a benchmark (for MMAR also the clips of the items that are not scored)."""
    from l2r import benchmarks
    paths = {it["audio"] for it in benchmarks.load(name)}
    if name == "mmar":
        paths |= {"MMAR/" + d["audio_path"].lstrip("./") for d in json.loads(data("MMAR", "MMAR-meta.json").read_text())}
    return sorted(str(audio(p)) for p in paths if audio(p).exists())


def benchmark_embeddings(clap, log=None, names=BENCHMARKS) -> np.ndarray:
    """L2-normalised CLAP embeddings of every benchmark clip (cached per benchmark in <work>/containment/)."""
    out = []
    for name in names:
        p = work("containment", f"clap_{name}.npy")
        if not p.exists():
            paths = clips(name)
            np.save(p, np.vstack([clap.embed(paths[i:i + 64]) for i in range(0, len(paths), 64)]).astype(np.float32))
            if log:
                log.info("embedded %d %s clips", len(paths), name)
        out.append(np.load(p))
    E = np.vstack(out)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)


def nearest(C: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Per row of C (L2-normalised), the highest cosine to any benchmark clip."""
    return np.concatenate([(C[i:i + 8192] @ E.T).max(1) for i in range(0, len(C), 8192)])
