"""Retrieve: route every chunk of a clip into the tree and record the nodes it reaches.

    python -m l2r.retrieve.serve --benchmark mmau        ->  <work>/nodes/mmau.json

A clip is cut into 3 s chunks every 1.5 s (and 10 s chunks for music). The traversal of one clip:

  1. Region gate. One sigmoid per region and chunk; a chunk opens a region at probability >= 0.5, and the clip
     opens it when the mean over its chunks is >= 0.5. A router's attributes are read only for an open region.
  2. Region routers. A clip-level attribute averages the log-probabilities of its chunks and serves its best
     class; a chunk-level attribute decides per chunk. A leaf is added only above its fitted threshold, otherwise
     the router backs off to the class. Attributes with low held-out skill are served only when confident.
  3. Language is predicted from the transcript when there is one.
  4. Music-property heads read one embedding of the clip when the clip-level music gate is open.
  5. Sound-identity heads read the chunks whose own gate heard environmental sound and keep, per attribute, the top
     classes and their top leaves. If no chunk heard environmental sound but a question of the clip asks about a
     sound (question classifier), these heads read the whole clip.
  6. A chunk keeps a node only if its own gate opened the node's region.
  7. A new-domain head (`--domains`) is read on every clip: it serves its best class unless its own `none` class is
     (nearly) certain.

The output holds, per clip, the rows (clip, 10 s chunks, 3 s chunks) with their nodes and probabilities: the
traversal trace. `l2r.retrieve.context` turns it into the text the reader receives.
"""
from __future__ import annotations

import argparse
import collections
import pickle
import re

import numpy as np
import torch

from l2r import benchmarks
from l2r.common import audio, ckpt, load_config, save_json, setup_run, work
from l2r.dataset import index as I, schema as S, segment as G
from l2r.router import calibrate, features as F, model as M

GATE = S.GATE
ENVIRONMENT = "environmental sound present"
MUSIC = "music present"
SPEECH = "speech present"


def nodes_path(benchmark: str):
    return work("nodes", f"{benchmark}.json")


def node(attr: str, value: str, leaf: str | None = "", p=None) -> dict:
    out = {"attr": attr, "value": value, "leaf": leaf or ""}
    if p is not None:
        out["p"] = round(float(p), 3)
    return out


def region_of(attr: str) -> str:
    return attr.split(".")[0]


def chunk_rows(benchmark: str, items: list[dict], hop: float) -> dict[str, list[dict]]:
    """The chunk grid of every clip of a benchmark (set `bench_<benchmark>`): audio path -> rows."""
    import soundfile as sf
    name = f"bench_{benchmark}"
    rows = I.rows(name)
    if not rows:
        schema = S.load_schema()
        for p in sorted({it["audio"] for it in items}):
            try:
                info = sf.info(str(audio(p))); dur = info.frames / info.samplerate
            except Exception:                                     # noqa: BLE001  (a container soundfile cannot read)
                import librosa
                dur = float(librosa.get_duration(path=str(audio(p))))
            rows += G.rows_for(p, dur, schema, source=name, hop=hop)
        I.write_rows(name, rows)
    out = collections.defaultdict(list)
    for r in rows:
        out[r["audio_path"]].append(r)
    return out


def has_audio(row: dict) -> bool:
    """A chunk shorter than 64 ms carries no usable audio."""
    import soundfile as sf
    return sf.info(str(G.chunk_wav(row))).frames >= 1024


class Retriever:
    def __init__(self, cfg=None, log=None, domains: list[str] | None = None):
        self.cfg = cfg or load_config(); self.log = log; c = self.c = self.cfg["serve"]
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        names = list(c["routers"]) + list(domains or [])
        loaded = {n: M.load(n, self.dev) for n in names}
        self.routers = {n: m for n, (m, _) in loaded.items()}
        self.owner = {a: n for n, m in self.routers.items() for a in m.attrs}          # a later router takes an attribute over
        self.scope = {a: ck["space"][a].get("scope", "clip") for m, ck in loaded.values() for a in m.attrs}
        cal = calibrate.load()
        self.temp = {a: r.get("temperature", 1.0) for a, r in cal.items()}
        always = set(c["always"])
        self.floor = {a: (c["weak_conf"] if cal.get(a, {}).get("skill", 0.0) < c["weak_skill"] and a not in always and not self.is_domain(a) else 0.0)
                      for a in self.owner}
        self.per_chunk = list(c["per_chunk"]); self.top_classes = list(c["top_classes"]); self.top_leaves = set(c["top_leaves"])
        self.skip = set(c["replaced_by_sound_heads"])
        self.late = set(self.per_chunk) | set(self.top_classes)
        self.language = pickle.load(open(ckpt("language", "text_head.pkl"), "rb")) if c.get("language_from_transcript", True) else None
        self.with_sound = bool(c.get("sound_heads", True)); self.with_music = bool(c.get("music_property_heads", True))
        if self.with_sound:
            from l2r.router import sound
            self.sound = sound
            self.sound_nodes = sound.load_nodes(); self.sound_encoder = sound.serving_encoder(self.sound_nodes)
            self.sound_heads = {e: sound.load(e) for e in sound.ENCODERS}
        if self.with_music:
            from l2r.router import music_properties
            self.music_properties = music_properties; self.music_heads = music_properties.load()

    # ------------------------------------------------------------------ settings of one attribute
    def grid(self, a: str) -> str:
        return "g3" if a in self.per_chunk else self.routers[self.owner[a]].meta[a]["grid"]

    def local(self, a: str) -> bool:
        return a in self.per_chunk or self.scope.get(a) == "local"

    def is_domain(self, a: str) -> bool:
        """A new-domain head carries a `none` class and decides by itself whether the clip belongs to its domain."""
        return "none" in self.routers[self.owner[a]].meta[a]["values"]

    @torch.no_grad()
    def open_regions(self, gate, X) -> set[str]:
        """The regions a clip opens: mean gate probability over its 3 s chunks >= the threshold."""
        pr = torch.sigmoid(gate.class_logits(GATE, gate.embed(X))).mean(0)
        return {S.REGION_OF[v] for k, v in enumerate(gate.meta[GATE]["values"]) if v in S.REGION_OF and float(pr[k]) >= self.c["gate_tau"]}

    # ------------------------------------------------------------------ one attribute on the chunks of one clip
    @torch.no_grad()
    def decide(self, a: str, Z: torch.Tensor, rows: list[dict]) -> list[tuple[str | None, dict]]:
        """-> [(row id, or None for the clip row; node)]."""
        c = self.c; m = self.routers[self.owner[a]]; vals = m.meta[a]["values"]; lg = m.class_logits(a, Z); out = []
        if m.meta[a]["multi"]:
            pr = torch.sigmoid(lg); thr = c["multi_threshold"].get(a, max(0.5, self.floor[a]))
            for i, r in enumerate(rows):
                for k in range(len(vals)):
                    if float(pr[i, k]) < thr:
                        continue
                    if a in self.top_leaves:
                        leaves = m.top_leaves(a, Z[[i]], k, c["top_k"], c["top_threshold"]) or [None]
                    else:
                        leaves = [m.leaf(a, Z[[i]], k)]
                    out += [(r["row"], node(a, vals[k], lf, pr[i, k])) for lf in leaves]
            return out
        lp = (lg / self.temp.get(a, 1.0)).log_softmax(1)
        for ix in ([[i] for i in range(len(rows))] if self.local(a) else [list(range(len(rows)))]):
            p = lp[ix].mean(0).exp(); k = int(p.argmax()); z = Z[ix]; target = rows[ix[0]]["row"] if self.local(a) else None
            if vals[k] == "none":
                if float(p[k]) >= c["domain_none"]:
                    continue                                     # the domain head abstains
                k = max((j for j in range(len(vals)) if vals[j] != "none"), key=lambda j: float(p[j]))
            if float(p[k]) < self.floor[a]:
                continue
            if a in self.top_classes:
                out += [(target, node(a, v, m.leaf(a, z, vi), pp)) for vi, v, pp in m.top_classes(a, z, c["top_k"], c["top_threshold"])]
            else:
                out.append((target, node(a, vals[k], m.leaf(a, z, k), p[k])))
        return out

    # ------------------------------------------------------------------ one clip
    @torch.no_grad()
    def route(self, rows: list[dict], feat, transcript: str, asked: bool, extra) -> tuple[list[dict], bool]:
        """rows of one clip (the clip row and the chunks that carry audio) -> (rows with their nodes, whether the sound
        nodes come from the question fallback). `feat(encoder, rows)` returns the cached chunk embeddings and
        `extra(kind)` the clip-level features of the sound-identity ("beats") and music-property ("music") heads."""
        c = self.c; tau = c["gate_tau"]
        rec = {r["row"]: {"grid": r["grid"], "t0": r["t0"], "t1": r["t1"], "nodes": []} for r in rows}
        clip = next(r["row"] for r in rows if r["grid"] == "clip")
        by = {g: [r for r in rows if r["grid"] == g] for g in ("g3", "g10")}

        def put(pairs):
            for row, n in pairs:
                rec[row or clip]["nodes"].append(n)

        # 1. region gate
        active = {}
        if GATE in self.owner and by["g3"]:
            m = self.routers[self.owner[GATE]]
            pr = torch.sigmoid(m.class_logits(GATE, m.embed(feat(m.encoder, by["g3"]))))
            for k, v in enumerate(m.meta[GATE]["values"]):
                mean = float(pr[:, k].mean())
                if v in S.REGION_OF:
                    active[S.REGION_OF[v]] = mean
                put([(r["row"], node(GATE, v, p=pr[i, k])) for i, r in enumerate(by["g3"]) if float(pr[i, k]) >= tau])
                if mean >= tau:
                    put([(None, node(GATE, v, p=mean))])

        def open_attrs(n, g, keep):
            m = self.routers[n]
            return [a for a in m.attrs if a != GATE and self.owner[a] == n and a not in self.skip and keep(a) and self.grid(a) == g
                    and not self.is_domain(a) and active.get(region_of(a), 1.0) >= tau]

        def chunk_open(row) -> set:
            return {S.REGION_OF.get(n["value"]) for n in rec[row]["nodes"] if n["attr"] == GATE}

        # 2. region routers: clip-level attributes and the chunk-level ones on their own grid
        for n, m in self.routers.items():
            for g in ("g3", "g10"):
                attrs = open_attrs(n, g, lambda a: a not in self.late)
                if attrs and by[g]:
                    Z = m.embed(feat(m.encoder, by[g]))
                    for a in attrs:
                        put(self.decide(a, Z, by[g]))
        # 3. language from the transcript
        if self.language is not None and active.get("speech", 1.0) >= tau:
            text = re.sub(r"\[[0-9:.]+\]", "", transcript or "").strip().lower()
            if len(text) >= 8:
                for r in rec.values():
                    r["nodes"] = [n for n in r["nodes"] if n["attr"] != "speech.language"]
                T = self.language; pr = T["clf"].predict_proba(T["vec"].transform([text]))[0]; j = int(pr.argmax())
                if float(pr[j]) >= c["language_min"]:
                    leaf = T["clf"].classes_[j]
                    put([(None, node("speech.language", T["leaf2val"][leaf], leaf, pr[j]))])
        clip_regions = {S.REGION_OF.get(n["value"]) for n in rec[clip]["nodes"] if n["attr"] == GATE}
        # 4. music properties (one embedding of the clip)
        if self.with_music and "music" in clip_regions:
            put([(None, node(a, v, p=p)) for a, v, p in self.music_properties.predict(self.music_heads, extra("music"), c["music_property_threshold"])])
        # 5. sound identity
        by_question = False
        if self.with_sound:
            spans = [(r["t0"], r["t1"]) for r in by["g3"] if "sound" in chunk_open(r["row"])]
            if spans or asked:
                by_question = not spans
                put([(None, n) for n in self.sound_identity(spans, asked, by["g3"], feat, extra)])
        # 6. a 3 s chunk keeps a node only if its own gate opened the node's region
        for r in by["g3"]:
            on = chunk_open(r["row"])
            rec[r["row"]]["nodes"] = [n for n in rec[r["row"]]["nodes"] if n["attr"] == GATE or region_of(n["attr"]) in on]
        # the attributes scored per 3 s chunk, then those that serve several classes
        for keep in (lambda a: a in self.per_chunk, lambda a: a in self.top_classes and a not in self.per_chunk):
            for n, m in self.routers.items():
                attrs = open_attrs(n, "g3", keep)
                if attrs and by["g3"]:
                    Z = m.embed(feat(m.encoder, by["g3"]))
                    for a in attrs:
                        put([(row, x) for row, x in self.decide(a, Z, by["g3"]) if region_of(a) in chunk_open(row)])
        # new-domain heads decide by themselves (their `none` class), whatever the region gate says
        for n, m in self.routers.items():
            for g in ("g3", "g10"):
                attrs = [a for a in m.attrs if self.owner[a] == n and self.is_domain(a) and self.grid(a) == g]
                if attrs and by[g]:
                    Z = m.embed(feat(m.encoder, by[g]))
                    for a in attrs:
                        put(self.decide(a, Z, by[g]))
        return sorted(rec.values(), key=lambda r: (r["grid"] != "clip", r["grid"], r["t0"])), by_question

    def sound_identity(self, spans, asked: bool, g3: list[dict], feat, extra) -> list[dict]:
        """The sound-identity nodes of one clip. `spans` are the chunks whose gate heard environmental sound; with
        none (question fallback) the heads read the whole clip and the context attributes are not served."""
        c = self.c; P, TAU = {}, {}
        windows = {"clap": (feat("clap", g3).cpu().numpy() if g3 else np.zeros((0, 512), np.float32), [(r["t0"], r["t1"]) for r in g3]), "beats": extra("beats")}
        for enc in self.sound.ENCODERS:
            X, times = windows[enc]
            keep = [i for i, (u0, u1) in enumerate(times) if any(u0 < b and a0 < u1 for a0, b in spans)]
            if not keep:
                if not asked or not len(X):
                    continue
                keep = list(range(len(X)))
            model, tau, nodes = self.sound_heads[enc]
            o = {k: torch.sigmoid(v).numpy() for k, v in model(torch.tensor(X[keep]).float()).items()}
            for i, a in enumerate(nodes):
                if self.sound_encoder[a] == enc:
                    P[a] = self.sound.joint(o[a], o["kind"][:, i], nodes[a]).max(0); TAU[a] = tau[a]
        out = []
        for a, p in P.items():
            if not spans and a in c["sound_context_attributes"]:
                continue
            nd = self.sound_nodes[a]
            classes = sorted([j for j, (v, l) in enumerate(nd) if not l and p[j] >= TAU[a]], key=lambda j: -p[j])[: c["sound_classes"]]
            for jv in classes:
                v = nd[jv][0]
                leaves = sorted([j for j, (vv, l) in enumerate(nd) if vv == v and l and p[j] >= TAU[a]], key=lambda j: -p[j])[: c["sound_leaves"]]
                out += [node(a, v, nd[j][1], p[j]) for j in leaves] or [node(a, v, p=p[jv])]
        return out

    # ------------------------------------------------------------------ a benchmark
    def run(self, benchmark: str, items: list[dict], transcripts: dict[str, str] | None = None, scores: dict | None = None) -> dict:
        """Route every clip of a benchmark. `transcripts`: audio -> text; `scores`: item id -> question classifier output."""
        c = self.c; log = self.log; name = f"bench_{benchmark}"
        rows = chunk_rows(benchmark, items, c["hop"])
        asked = collections.defaultdict(bool)
        for i, it in enumerate(items):
            asked[it["audio"]] |= any(v >= c["question_fallback"] for v in (scores or {}).get(str(it.get("id", i)), {}).values())
        for p, rs in rows.items():                                # cut the chunk wavs, one read per clip
            if any(not G.wav_path(r).exists() for r in rs if r["grid"] != "clip"):
                G.materialise(p, rs)
        rows = {p: [r for r in rs if r["grid"] == "clip" or has_audio(r)] for p, rs in rows.items()}
        # chunk embeddings, cached per (encoder, set): the gate's encoder on every 3 s chunk, the others only on the
        # chunks of the clips whose region gate is open
        store = {}

        def embed(enc, rs):
            X = F.features(enc, name, rs, self.cfg, log, by_clip=True)
            store[enc] = ({r["row"]: i for i, r in enumerate(rs)}, torch.from_numpy(X).to(self.dev))

        def feat(enc, rs):
            idx, X = store[enc]
            return X[[idx[r["row"]] for r in rs]]

        gate = self.routers[self.owner[GATE]]
        embed(gate.encoder, [r for rs in rows.values() for r in rs if r["grid"] == "g3"])
        need = collections.defaultdict(dict)
        for p, rs in rows.items():
            g3 = [r for r in rs if r["grid"] == "g3"]
            open_ = self.open_regions(gate, feat(gate.encoder, g3)) if g3 else None
            for a_, n in self.owner.items():
                if a_ == GATE or a_ in self.skip or not (open_ is None or region_of(a_) in open_ or self.is_domain(a_)):
                    continue
                need[self.routers[n].encoder].update({r["row"]: r for r in rs if r["grid"] == self.grid(a_)})
            need[gate.encoder].update({r["row"]: r for r in g3})
            if self.with_sound:
                need["clap"].update({r["row"]: r for r in g3})
        for enc, rs in sorted(need.items()):
            embed(enc, list(rs.values()))

        models = {}

        def beats_of(path):
            from l2r.encoders import beats_windows, load_beats
            if "beats" not in models:
                models["beats"] = load_beats(self.dev, self.cfg)
            return beats_windows(models["beats"], audio(path))

        def music_of(path):
            from l2r.encoders import build_encoder
            if "music" not in models:
                models["music"] = build_encoder(self.music_properties.ENCODER, self.cfg)
            return models["music"].embed([str(audio(path))])[0], None

        clips, by_question = {}, []
        for k, (path, rs) in enumerate(sorted(rows.items())):
            def extra(kind, path=path):
                if kind == "beats":
                    return _cached(work("features", f"beats_windows__{name}"), path, lambda: beats_of(path))
                return _cached(work("features", f"{self.music_properties.ENCODER}_clip__{name}"), path, lambda: music_of(path))[0]
            clips[path], q = self.route(rs, feat, (transcripts or {}).get(path, ""), asked[path], extra)
            if q:
                by_question.append(path)
            if log and k % 200 == 0:
                log.info("  %d / %d clips", k, len(rows))
        _flush()
        out = {"benchmark": benchmark, "routers": list(self.routers), "clips": clips, "sound_by_question": by_question,
               "domain_attributes": sorted(a for a in self.owner if self.is_domain(a))}
        save_json(out, nodes_path(benchmark), indent=None)
        if log:
            n = sum(len(r["nodes"]) for rs in clips.values() for r in rs)
            log.info("%s: %d clips, %d nodes (%d with a leaf) -> %s", benchmark, len(clips), n,
                     sum(bool(x["leaf"]) for rs in clips.values() for r in rs for x in r["nodes"]), nodes_path(benchmark))
        return out


# ---------------------------------------------------------------------- a small on-disk cache of clip-level features
_OPEN: dict = {}


def _cached(base, key: str, compute):
    """(array, times) of one clip from `<base>.pkl`, computed and stored when missing."""
    p = base.with_suffix(".pkl")
    if p not in _OPEN:
        _OPEN[p] = [pickle.load(open(p, "rb")) if p.exists() else {}, 0]
    d = _OPEN[p]
    if key not in d[0]:
        d[0][key] = compute(); d[1] += 1
    return d[0][key]


def _flush():
    for p, (d, new) in _OPEN.items():
        if new:
            pickle.dump(d, open(p, "wb"))
    _OPEN.clear()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", required=True, choices=benchmarks.BENCHMARKS)
    ap.add_argument("--domains", default="", help="comma list of new-domain routers to add, e.g. birds_k5")
    a = ap.parse_args()
    cfg = load_config(); _, log = setup_run(f"serve_{a.benchmark}", cfg)
    from l2r.retrieve import asr
    from l2r.router import question
    items = benchmarks.load(a.benchmark)
    transcripts = asr.transcribe([it["audio"] for it in items], a.benchmark, cfg, log)
    scores = question.predict_benchmarks([a.benchmark], log)[a.benchmark] if cfg["serve"].get("sound_heads", True) else {}
    Retriever(cfg, log, [d for d in a.domains.split(",") if d]).run(a.benchmark, items, transcripts, scores)


if __name__ == "__main__":
    main()
