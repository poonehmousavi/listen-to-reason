"""Feature stores of the frozen encoders.

The encoders are never updated, so the embedding of every chunk is computed once and cached:
`<work>/features/<encoder>__<set>.h5` holds `rows` (chunk row ids) and `x` ([N, D] float32, L2-normalised), one
file per (encoder, set of clips). Training a head on cached features takes minutes.

    python -m l2r.router.features --encoder clap --sets pool,musiccaps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from l2r.common import load_config, setup_run, work
from l2r.dataset import index as I, segment as G

GRIDS = ("g3", "g10")


def store_path(enc: str, name: str) -> Path:
    return work("features", f"{enc}__{name}.h5")


def load(enc: str, name: str) -> tuple[dict[str, int], np.ndarray]:
    """-> (row id -> index, x [N, D]); empty when the store does not exist."""
    import h5py
    p = store_path(enc, name)
    if not p.exists():
        return {}, np.zeros((0, 0), np.float32)
    with h5py.File(p, "r") as h:
        rows = [r.decode() if isinstance(r, bytes) else str(r) for r in h["rows"][:]]
        return {r: i for i, r in enumerate(rows)}, h["x"][:].astype(np.float32)


def _append(enc: str, name: str, rows: list[str], x: np.ndarray):
    import h5py
    vlen = h5py.string_dtype(encoding="utf-8")
    with h5py.File(store_path(enc, name), "a") as h:
        if "x" not in h:
            h.create_dataset("x", data=x, maxshape=(None, x.shape[1]), chunks=(min(1024, len(x)), x.shape[1]))
            h.create_dataset("rows", data=np.array(rows, dtype=object), dtype=vlen, maxshape=(None,), chunks=(1024,))
            h.attrs["encoder"] = enc
        else:
            n = h["x"].shape[0]
            h["x"].resize(n + len(x), 0); h["x"][n:] = x
            h["rows"].resize(n + len(rows), 0); h["rows"][n:] = np.array(rows, dtype=object)


def features(enc: str, name: str, rows: list[dict], cfg=None, log=None, bs: int = 64, encoder=None, by_clip: bool = False) -> np.ndarray:
    """Features of these chunk rows (dicts with row / clip / grid / t0 / t1 / audio_path), in the given order.
    Rows missing from the store of set `name` are computed from their chunk wav and appended.

    Some encoders pad a batch to its longest chunk, so the embedding of a shorter chunk depends on the batch it is
    in. With `by_clip` a batch never mixes clips or grids (inference); otherwise batches follow the row order."""
    idx, X = load(enc, name)
    seen = set(idx)
    todo = [r for r in rows if not (r["row"] in seen or seen.add(r["row"]))]
    if todo:
        if encoder is None:
            from l2r.encoders import build_encoder
            encoder = build_encoder(enc, cfg or load_config())
        if log:
            log.info("%s / %s: %d rows cached, %d to compute", enc, name, len(idx), len(todo))
        if by_clip:
            groups = {}
            for r in todo:
                groups.setdefault((r["clip"], r["grid"]), []).append(r)
            batches = [g[s:s + bs] for g in groups.values() for s in range(0, len(g), bs)]
        else:
            batches = [todo[s:s + bs] for s in range(0, len(todo), bs)]
        pending, n = [], 0
        for k, grp in enumerate(batches):
            e = np.asarray(encoder.embed([str(G.chunk_wav(r)) for r in grp]), dtype=np.float32)
            assert e.shape[0] == len(grp), (e.shape, len(grp))
            pending.append(([r["row"] for r in grp], e)); n += len(grp)
            if sum(len(r) for r, _ in pending) >= 2048 or k == len(batches) - 1:
                _append(enc, name, [x for r, _ in pending for x in r], np.concatenate([e_ for _, e_ in pending]))
                pending = []
                if log:
                    log.info("  %s %d / %d", enc, n, len(todo))
        idx, X = load(enc, name)
    return X[[idx[r["row"]] for r in rows]]


def cache(enc: str, names: list[str], log) -> dict:
    """Fill the stores of one encoder for every chunk row of the given sets."""
    cfg = load_config(); out = {}; encoder = None
    for name in names:
        rows = [r for r in I.rows(name) if r["grid"] in GRIDS]
        if encoder is None and any(r["row"] not in load(enc, name)[0] for r in rows):
            from l2r.encoders import build_encoder
            encoder = build_encoder(enc, cfg)
        x = features(enc, name, rows, cfg, log, encoder=encoder)
        log.info("%s / %s: %d rows, dim %d", enc, name, len(rows), x.shape[1]); out[name] = {"rows": len(rows), "dim": int(x.shape[1])}
    return out


def selftest():
    import shutil

    class Fake:
        def embed(self, paths):
            return np.stack([np.full(4, float(len(p) % 7)) for p in paths]).astype(np.float32)
    name = "_selftest"; p = store_path("fake", name)
    if p.exists():
        p.unlink()
    real = G.chunk_wav; G.chunk_wav = lambda r: Path("x") / r["row"]
    try:
        rows = [{"row": f"clip{i}:g3:{t}", "clip": f"clip{i}", "grid": "g3", "t0": t, "t1": t + 3, "audio_path": ""} for i in range(3) for t in (0, 3, 10.5)]
        a = features("fake", name, rows[:4], cfg={}, encoder=Fake()); assert a.shape == (4, 4)
        b = features("fake", name, rows, cfg={}, encoder=Fake()); assert b.shape == (9, 4) and (b[:4] == a).all()
        idx, x = load("fake", name); assert len(idx) == 9 and "clip2:g3:10.5" in idx and x.shape == (9, 4)
        assert (features("fake", name, rows[::-1], cfg={}, encoder=None) == b[::-1]).all()        # all cached: no encoder needed
    finally:
        G.chunk_wav = real; p.unlink()
    shutil.rmtree(I.sdir(name), ignore_errors=True)
    print("features selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoder", default=""); ap.add_argument("--sets", default="pool"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        _, log = setup_run(f"features_{a.encoder}")
        print(json.dumps(cache(a.encoder, a.sets.split(","), log)))
