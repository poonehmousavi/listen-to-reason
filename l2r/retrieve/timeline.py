"""The event timeline: detected sound events with their time spans, named by tree nodes.

Served only for questions about the order, duration or number of events (`l2r.events.asks_timeline`). A linear
head on PANNs frame features, fine-tuned on AudioSet strong labels (l2r/router/event_head.py), scores 527 classes
per frame; the scores are segmented into events and every event is renamed to its node of the sound tree:

    Sound events over time (sound tree nodes): [0.0-2.5s] animal: dog (Bark); [2.4-4.0s] impact: glass (Shatter) ...

    python -m l2r.retrieve.timeline --benchmark mmau     ->  <work>/events/mmau.json  (events per clip)
"""
from __future__ import annotations

import argparse

import numpy as np

from l2r import benchmarks, events as E
from l2r.common import audio, load_config, load_json, save_json, setup_run, work

HEAD = "Sound events over time (sound tree nodes): "
SEGMENT = dict(E.DEFAULTS, smooth=1)              # the frame features are 320 ms apart: no median smoothing


def events_path(benchmark: str):
    return work("events", f"{benchmark}.json")


def node_name(attr: str, value: str, leaf: str) -> str:
    kind = attr.split(".", 1)[1].replace("_", " ")
    return f"{kind}: {value}" + (f" ({leaf})" if leaf and leaf.lower() != value.lower() else "")


def detect(benchmark: str, items: list[dict], cfg=None, log=None) -> dict[str, list[dict]]:
    """Events of every clip that has a timeline question: audio -> [{label, t0, t1, peak}], cached."""
    import librosa
    import torch
    from l2r.encoders import PANNS_SR, load_panns, panns_labels
    from l2r.router import event_head
    cfg = cfg or load_config(); p = events_path(benchmark)
    cache = load_json(p) if p.exists() else {}
    todo = [a for a in dict.fromkeys(it["audio"] for it in items if E.asks_timeline(it["question"])) if a not in cache]
    if todo:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        panns, box = load_panns(dev, cfg); head = event_head.load(dev)
        labels = panns_labels(cfg); ancestors = E.load_ancestors(labels); s = SEGMENT
        for a in todo:
            y, _ = librosa.load(str(audio(a)), sr=PANNS_SR, mono=True, duration=E.MAX_S)
            y = np.pad(y, (0, max(0, PANNS_SR - len(y))))
            with torch.no_grad():
                panns(torch.from_numpy(y[None]).float().to(dev))
                x = box["fc1"]                                             # [1, frames, 2048]
                prob = torch.sigmoid(head(x))[0].cpu().numpy()
            frame_s = len(y) / PANNS_SR / x.shape[1]
            evs = E.segment(prob, labels, frame_s, s["tau_on"], s["tau_off"], s["min_dur"], s["gap"], int(s["k"]), s["smooth"], ancestors=ancestors)
            cache[a] = [e.__dict__ for e in evs]
        save_json(cache, p, indent=None)
    if log:
        log.info("timeline %s: %d clips with a timeline question (%d new)", benchmark, sum(a in cache for a in {it["audio"] for it in items}), len(todo))
    return cache


def lines(items: list[dict], detected: dict[str, list[dict]]) -> dict[str, str]:
    """item id -> the timeline line of its clip, for the items whose question asks for one."""
    from l2r.router import sound
    _, to_nodes = sound.node_index()
    out = {}
    for it in items:
        if not E.asks_timeline(it["question"]):
            continue
        evs = []
        for e in detected.get(it["audio"], []):
            hit = [h for h in to_nodes.get(("audioset", e["label"]), []) if not h[0].startswith("music.")]
            evs.append(E.Event(node_name(*hit[0]) if hit else e["label"], e["t0"], e["t1"], e["peak"]))
        s = E.render(evs, it["question"])
        if s:
            out[str(it["id"])] = HEAD + s
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--benchmark", required=True)
    a = ap.parse_args()
    _, log = setup_run(f"timeline_{a.benchmark}")
    items = benchmarks.load(a.benchmark)
    log.info("%d timeline lines", len(lines(items, detect(a.benchmark, items, log=log))))
