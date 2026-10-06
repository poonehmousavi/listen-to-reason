"""Labelled training clips for the environmental-sound identity heads and the event-timeline head.

Three sources of human labels are used: AudioSet strong labels on the WavCaps AudioSet-SL clips (10 s each), FSD50K
and ESC-50. This module builds the AudioSet / FSD50K-eval item list and its frozen features; `sound.py` adds the
FSD50K-dev and ESC-50 clips for the CLAP heads.

    python -m l2r.router.audioset items     # item list, gold labels (with ontology ancestors), CLAP clip vectors
    python -m l2r.router.audioset beats     # BEATs clip vectors of the items
    python -m l2r.router.audioset strong    # PANNs frame features + frame-level strong labels (event-timeline head)

Items: every AudioSet-SL clip with strong labels (split 90 / 10 into train / test by YouTube id) and the FSD50K
evaluation clips (`fsd_eval`, a second held-out set). A clip whose CLAP embedding is within cosine 0.90 of any
benchmark clip is dropped, so no benchmark audio or near-duplicate enters training.

Data (under the data folder):
  AudioSet/ontology.json, AudioSet/audioset_train_strong.tsv, WavCaps/AudioSet_SL/Y<youtube id>.flac
  FSD50K/FSD50K.eval_audio/*.wav, FSD50K/FSD50K.ground_truth/{eval,dev,vocabulary}.csv
Outputs (under <work>/audioset/): items.json, clap_clip.npy, beats_clip.npy, panns_fc1.npy,
strong_y.npy, has_strong.npy, strong_sets.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from l2r import containment
from l2r.common import audio, check, ckpt, data, load_config, setup_run, work

TAU = 0.90                      # containment: CLAP cosine to the nearest benchmark clip
MIN_CLIPS = 20                  # an AudioSet class (or ancestor) is kept with at least this many labelled clips
CLIP_S = 10                     # seconds of audio the frozen features are computed on
BEATS_BINS = 20                 # BEATs frames are averaged into half-second bins before the clip mean
PANNS_T = 31                    # PANNs frames per 10 s
PANNS_FRAME_S = CLIP_S / PANNS_T


def out(name: str) -> Path:
    return work("audioset", name)


def ontology():
    """-> (id -> display name, ancestors(id) -> set of ids)."""
    onto = json.load(open(data("AudioSet", "ontology.json")))
    name = {n["id"]: n["name"] for n in onto}
    parents = defaultdict(set)
    for n in onto:
        for c in n["child_ids"]:
            parents[c].add(n["id"])

    def ancestors(m, seen=None):
        seen = seen if seen is not None else set()
        for p in parents[m]:
            if p not in seen:
                seen.add(p)
                ancestors(p, seen)
        return seen
    return name, ancestors


def strong_segments() -> dict[str, list]:
    """YouTube id -> [(start, end, AudioSet id)] from the strong-label file."""
    seg = defaultdict(list)
    for line in open(data("AudioSet", "audioset_train_strong.tsv")):
        if line.startswith("segment_id"):
            continue
        s, t0, t1, m = line.rstrip("\n").split("\t")
        seg[s.rsplit("_", 1)[0]].append((float(t0), float(t1), m))
    return seg


def readable(path) -> bool:
    import soundfile as sf
    try:
        info = sf.info(str(audio(path)))
        return 0 < info.frames / info.samplerate < 600
    except Exception:
        return False


def embed_clap(clap, paths, log=None) -> np.ndarray:
    E = []
    for i in range(0, len(paths), 64):
        E.append(clap.embed([str(audio(p)) for p in paths[i:i + 64]]))
        if log and (i // 64) % 200 == 0:
            log.info("  CLAP %d / %d", i, len(paths))
    C = np.vstack(E).astype(np.float32)
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)


def items(a, log):
    from l2r.encoders import build_encoder
    name, ancestors = ontology()
    seg = strong_segments()
    files = {p.stem[1:]: p for p in data("WavCaps", "AudioSet_SL").glob("*.flac")}
    ids = sorted(set(files) & set(seg))
    log.info("AudioSet-SL: %d files, %d strongly labelled clips, %d joined", len(files), len(seg), len(ids))
    count = Counter()
    for y in ids:
        count.update({m for _, _, m in seg[y]} | set().union(*[ancestors(m) for _, _, m in seg[y]]))
    classes = sorted(m for m, c in count.items() if c >= MIN_CLIPS and m in name)
    ci = {m: i for i, m in enumerate(classes)}
    log.info("classes: %d", len(classes))
    rel = lambda p: str(Path(p).relative_to(data()))                                       # noqa: E731
    ids = [y for y in ids if readable(files[y])]
    fsd = [(rel(data("FSD50K", "FSD50K.eval_audio", f"{r['fname']}.wav")), r["mids"].split(","))
           for r in csv.DictReader(open(data("FSD50K", "FSD50K.ground_truth", "eval.csv")))]
    fsd = [x for x in fsd if readable(x[0])]
    clap = build_encoder("clap")
    E = containment.benchmark_embeddings(clap, log)
    C = embed_clap(clap, [rel(files[y]) for y in ids] + [p for p, _ in fsd], log)
    best = containment.nearest(C, E)
    log.info("within %.2f of a benchmark clip: AudioSet-SL %d, FSD50K eval %d", TAU, int((best[:len(ids)] >= TAU).sum()), int((best[len(ids):] >= TAU).sum()))
    keep = best < TAU
    its, vec = [], []
    for k, y in enumerate(ids):
        if keep[k]:
            its.append({"path": rel(files[y]), "set": "test" if int(hashlib.md5(y.encode()).hexdigest()[:8], 16) % 10 == 0 else "train",
                        "clip": sorted({ci[mm] for _, _, m in seg[y] for mm in {m} | ancestors(m) if mm in ci})})
            vec.append(C[k])
    for k, (p, mids) in enumerate(fsd):
        lab = sorted({ci[mm] for m in mids for mm in {m} | ancestors(m) if mm in ci})
        if keep[len(ids) + k] and lab:
            its.append({"path": p, "set": "fsd_eval", "clip": lab})
            vec.append(C[len(ids) + k])
    json.dump({"classes": classes, "names": [name[m] for m in classes], "items": its, "tau": TAU}, open(out("items.json"), "w"))
    np.save(out("clap_clip.npy"), np.stack(vec).astype(np.float16))
    sets = Counter(i["set"] for i in its)
    log.info("items: %s", dict(sets))
    check(sets["train"] > 0 and sets["test"] > 0 and sets["fsd_eval"] > 0, "every split has clips", log)


def load_items() -> dict:
    return json.load(open(out("items.json")))


def _centre(y, n):
    if len(y) > n:
        s0 = (len(y) - n) // 2
        y = y[s0:s0 + n]
    return np.pad(y, (0, n - len(y))).astype(np.float32)


def _loader(paths, sr, centre, workers):
    """Batches of 10 s waveforms; an unreadable file is silence."""
    import librosa
    from torch.utils.data import DataLoader, Dataset
    n = sr * CLIP_S

    class Clips(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            try:
                assert readable(paths[i])
                y, _ = librosa.load(str(audio(paths[i])), sr=sr, mono=True, duration=60 if centre else CLIP_S)
            except Exception:
                y = np.zeros(n, np.float32)
            return _centre(y, n) if centre else np.pad(y[:n], (0, n - len(y[:n]))).astype(np.float32)
    return DataLoader(Clips(), batch_size=32, num_workers=workers)


def beats(a, log):
    """BEATs clip vector of every item: the centre 10 s, encoder frames averaged into half-second bins, then over time."""
    import torch
    import torch.nn.functional as F
    import torchaudio
    from l2r.encoders import beats_frames, load_beats
    its = load_items()["items"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_beats(dev)
    X = np.zeros((len(its), 768), np.float32)
    i0 = 0
    with torch.no_grad():
        for b, y32 in enumerate(_loader([x["path"] for x in its], 32000, True, a.workers)):
            y16 = torchaudio.functional.resample(y32.to(dev), 32000, 16000)
            fr = F.adaptive_avg_pool1d(beats_frames(model, y16).transpose(1, 2), BEATS_BINS).transpose(1, 2)
            X[i0:i0 + len(y32)] = fr.mean(1).float().cpu().numpy()
            i0 += len(y32)
            if b % 200 == 0:
                log.info("BEATs %d / %d", i0, len(its))
    np.save(out("beats_clip.npy"), X)
    log.info("wrote BEATs clip vectors for %d items", i0)


def strong(a, log):
    """PANNs frame features (the input of its classification layer, 31 frames per 10 s) and frame-level strong labels
    over PANNs' 527 classes, for the AudioSet-SL items (train and test)."""
    import torch
    from l2r.encoders import load_panns
    its = [it for it in load_items()["items"] if it["set"] in ("train", "test")]
    spec = load_config()["encoders"]["panns"]
    rows = sorted(csv.DictReader(open(ckpt(spec["labels"]))), key=lambda r: int(r["index"]))
    li = {r["mid"]: i for i, r in enumerate(rows)}
    seg = strong_segments()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, box = load_panns(dev)
    X = np.lib.format.open_memmap(out("panns_fc1.npy"), mode="w+", dtype=np.float16, shape=(len(its), PANNS_T, 2048))
    Y = np.zeros((len(its), PANNS_T, 527), np.uint8)
    for k, it in enumerate(its):
        for t0, t1, mid in seg.get(Path(it["path"]).stem[1:], []):
            if mid in li:
                f0 = int(np.floor(t0 / PANNS_FRAME_S))
                Y[k, f0:max(int(np.ceil(t1 / PANNS_FRAME_S)), f0 + 1), li[mid]] = 1
    i0 = 0
    with torch.no_grad():
        for b, y32 in enumerate(_loader([x["path"] for x in its], 32000, False, a.workers)):
            model(y32.to(dev))
            X[i0:i0 + len(y32)] = box["fc1"][:, :PANNS_T].float().cpu().numpy()
            i0 += len(y32)
            if b % 200 == 0:
                log.info("PANNs %d / %d", i0, len(its))
    X.flush()
    np.save(out("strong_y.npy"), np.packbits(Y, axis=2))
    np.save(out("has_strong.npy"), np.array(sorted({li[m] for v in seg.values() for _, _, m in v if m in li})))
    json.dump([it["set"] for it in its], open(out("strong_sets.json"), "w"))
    log.info("wrote PANNs frames and strong labels for %d clips", i0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["items", "beats", "strong"])
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    _, log = setup_run(f"audioset_{a.stage}")
    {"items": items, "beats": beats, "strong": strong}[a.stage](a, log)


if __name__ == "__main__":
    main()
