"""Expert classifiers that label training chunks (tree construction only; none of them runs at inference).

    panns        PANNs: which regions a chunk contains, AudioSet sound events, and a few speech / vocal classes
    wavlm_sv     WavLM speaker verification: gender (nearest of two gender centroids)
    emotion2vec  emotion2vec+: emotion
    whisper_lid  Whisper: language of the first 30 s
    librosa      signal analysis: tempo and mode of music, as words, only when the measurement is confident

Each expert takes the rows of one clip and returns `index.fired(...)` records with its score; which label enters
the tree is decided later (`l2r.tree.build.vote`). PANNs runs first: the speech and music experts only look at
rows where it heard that region.

    python -m l2r.dataset.experts --gender-centroids     # once: the two gender centroids, from IEMOCAP
"""
from __future__ import annotations

import collections
import json
import random
import re
from pathlib import Path

import numpy as np

from l2r import events as EV
from l2r.common import audio, ckpt, data, load_config, read_jsonl, setup_run
from l2r.dataset import index as I, segment as G

ORDER = ("panns", "wavlm_sv", "emotion2vec", "whisper_lid", "librosa")
REGION_TAU = {"speech present": 0.30, "music present": 0.30, "environmental sound present": 0.20}
EMOTIONS = {"angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"}       # `other` / `unknown` are not labels
LANGUAGE_ALIAS = {"chinese": "mandarin", "tagalog": "filipino", "moldavian": "romanian", "moldovan": "romanian",
                  "haitian creole": None, "castilian": "spanish", "flemish": "dutch", "panjabi": "punjabi",
                  "pushto": "pashto", "sinhalese": None, "valencian": "catalan", "myanmar": "burmese", "nynorsk": "norwegian"}
VOICE_ROOTS = ("Human voice", "Speech", "Singing")                  # PANNs classes that belong to the speech / music regions
MUSIC_ROOTS = ("Music", "Musical instrument")
# PANNs classes that are a class of a speech / music attribute
PANNS_SPEECH = {"Whispering": ("speech.speaking_style", "whispering"), "Shout": ("speech.speaking_style", "shouting"),
                "Yell": ("speech.speaking_style", "shouting"), "Child speech, kid speaking": ("speech.age_group", "child"),
                "Male speech, man speaking": ("speech.gender", "male voice"), "Female speech, woman speaking": ("speech.gender", "female voice"),
                "Rapping": ("music.vocals", "rap vocals"), "Choir": ("music.vocals", "choir")}
GENDER_CENTROIDS = ("experts", "gender_centroids.npz")              # under the checkpoint folder
GENDER_WIN_S, GENDER_CLIPS = 2.0, 150


# ------------------------------------------------------------------------------------------ pure helpers
def kind_of(label: str, ancestors: dict, rules: list) -> str:
    """The class of `sound.event` an AudioSet label belongs to (schema `audioset_kind`), or `unplaced`."""
    names = {label} | set(ancestors.get(label, ()))
    for keys, kind in rules:
        if names & set(keys):
            return kind
    return "unplaced"


def is_family(label: str, ancestors: dict, roots) -> bool:
    roots = (roots,) if isinstance(roots, str) else roots
    return label in roots or bool(set(roots) & set(ancestors.get(label, ())))


def full_ancestors() -> dict[str, set]:
    """AudioSet class name -> every ontology ancestor name (abstract categories included)."""
    onto = EV.ontology()
    name = {x["id"]: x["name"] for x in onto}; par = collections.defaultdict(set)
    for x in onto:
        for c in x.get("child_ids", []):
            par[c].add(x["id"])
    out = {}
    for x in onto:
        seen, st = set(), list(par[x["id"]])
        while st:
            q = st.pop()
            if q not in seen:
                seen.add(q); st.extend(par[q])
        out[x["name"]] = {name[i] for i in seen}
    return out


def tempo_word(bpm: float) -> str:
    return "slow tempo" if bpm < 80 else "moderate tempo" if bpm < 120 else "fast tempo" if bpm < 160 else "very fast tempo"


def language_groups() -> dict[str, str]:
    """language name -> geographic group (the class), from the FLEURS language table."""
    p = data("FLEURS", "meta.json")
    return {v["language"]: v["group"] for v in json.loads(p.read_text()).values()} if p.exists() else {}


def place_language(name: str, groups: dict) -> tuple[str, str]:
    n = name.lower(); n = LANGUAGE_ALIAS.get(n, n) if n in LANGUAGE_ALIAS else n
    return (groups.get(n, "unplaced"), n) if n else ("unplaced", name.lower())


def regions_of(fired: list[dict]) -> dict[str, set]:
    """row -> the regions PANNs heard on it."""
    fam = collections.defaultdict(set)
    for f in fired:
        if f["attr"] == "cross.families_present":
            fam[f["row"]].add(f["value"])
    return fam


# ------------------------------------------------------------------------------------------ signal analysis (tempo, key)
_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])      # Krumhansl key profiles
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_KEYS = np.concatenate([np.stack([np.roll(b, i) for i in range(12)]) for b in (_MAJ, _MIN)])    # [24, 12]: 12 major, 12 minor


def tempo_and_mode(path, sr: int = 22050) -> dict:
    """{"tempo_bpm", "beat_strength", "mode", "key_conf"} of a recording (empty for less than half a second)."""
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    if len(y) < sr // 2:
        return {}
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=oenv, sr=sr)
    ac = librosa.autocorrelate(oenv)
    strength = float(ac[1:].max() / (ac[0] + 1e-9)) if len(ac) > 1 else 0.0
    v = librosa.feature.chroma_cqt(y=y, sr=sr).mean(1)
    v = v - v.mean(); T = _KEYS - _KEYS.mean(1, keepdims=True)
    c = (T @ v) / (np.linalg.norm(T, axis=1) * np.linalg.norm(v) + 1e-9)
    j = int(np.argmax(c))
    return {"tempo_bpm": round(float(np.atleast_1d(tempo)[0]), 1), "beat_strength": round(strength, 3),
            "mode": "minor key" if j >= 12 else "major key", "key_conf": round(float(c[j]), 3)}


# ------------------------------------------------------------------------------------------ experts
class Panns:
    tool = "panns"

    def __init__(self, cfg, log, schema):
        self.det = EV.Detector(cfg, log)
        self.rules = schema["regions"]["sound"]["attributes"]["event"]["audioset_kind"]
        self.ix = {n: i for i, n in enumerate(self.det.labels)}; self.anc = full_ancestors()
        self.env = [i for n, i in self.ix.items() if n not in EV.STOPLIST and not is_family(n, self.anc, VOICE_ROOTS + MUSIC_ROOTS)]

    def label(self, rows):
        det = self.det
        fw = det.framewise(audio(rows[0]["audio_path"])); out = []
        for r in rows:                                                       # regions per row (and per clip)
            seg = fw[int(r["t0"] * 100): max(int(r["t1"] * 100), int(r["t0"] * 100) + 1)].max(axis=0)
            scores = (("speech present", float(seg[self.ix["Speech"]])), ("music present", float(seg[self.ix["Music"]])),
                      ("environmental sound present", float(seg[self.env].max())))
            for v, s in scores:
                if s >= REGION_TAU[v]:
                    out.append(I.fired(r["row"], "cross.families_present", v, self.tool, s))
        for e in EV.segment(fw, det.labels, ancestors=det.ancestors, **det.params):      # events keep their own span
            hit = G.project((e.t0, e.t1), rows) + [rows[0]]
            if e.label in PANNS_SPEECH:                                       # a speech / vocal attribute, not a sound event
                a, v = PANNS_SPEECH[e.label]
                out += [I.fired(r["row"], a, v, self.tool, e.peak, span=(e.t0, e.t1)) for r in hit]
            if is_family(e.label, self.anc, VOICE_ROOTS + MUSIC_ROOTS):
                continue
            k = kind_of(e.label, self.anc, self.rules)
            out += [I.fired(r["row"], "sound.event", k, self.tool, e.peak, leaf=e.label.lower(), span=(e.t0, e.t1)) for r in hit]
        return out


class WavlmSv:
    tool = "wavlm_sv"; needs = "speech present"

    def __init__(self, cfg, log, schema):
        from l2r.encoders import build_encoder
        self.enc = build_encoder("wavlm_sv", cfg)
        p = ckpt(*GENDER_CENTROIDS)
        assert p.exists(), f"{p} is missing: run `python -m l2r.dataset.experts --gender-centroids`"
        z = np.load(p); self.cen = {"male": z["male"], "female": z["female"]}

    def label(self, rows):
        rows = [r for r in rows if r["grid"] == "g3"]
        if not rows:
            return []
        X = xvectors(self.enc, [_read(G.chunk_wav(r)) for r in rows]); out = []
        for r, x in zip(rows, X):
            m, f = float(x @ self.cen["male"]), float(x @ self.cen["female"])
            out.append(I.fired(r["row"], "speech.gender", "male voice" if m >= f else "female voice", self.tool, abs(m - f)))
        return out


class Emotion2vec:
    tool = "emotion2vec"; needs = "speech present"

    def __init__(self, cfg, log, schema):
        from l2r.encoders import build_encoder
        self.enc = build_encoder("emotion2vec", cfg)

    def label(self, rows):
        rows = [r for r in rows if r["grid"] == "g3"]
        if not rows:
            return []
        res = self.enc.classify([str(G.chunk_wav(r)) for r in rows])
        return [I.fired(r["row"], "speech.emotion", lab, self.tool, s) for r, (lab, s) in zip(rows, res) if lab in EMOTIONS]


class WhisperLid:
    """Language over all of Whisper's language tokens, on the clip's first 30 s."""
    tool = "whisper_lid"; needs = "speech present"

    def __init__(self, cfg, log, schema):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.models.whisper.tokenization_whisper import LANGUAGES
        mid = cfg["models"]["asr"]; self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = WhisperProcessor.from_pretrained(mid)
        self.m = WhisperForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.float16 if self.dev == "cuda" else torch.float32).to(self.dev).eval()
        names = list(LANGUAGES.values())
        ids = [self.proc.tokenizer.convert_tokens_to_ids(f"<|{c}|>") for c in LANGUAGES]
        keep = [i for i, t in enumerate(ids) if t is not None and t != self.proc.tokenizer.unk_token_id]
        self.names = [names[i] for i in keep]; self.ids = torch.tensor([ids[i] for i in keep], device=self.dev)
        self.groups = language_groups()

    def label(self, rows):
        import librosa
        clip = rows[0]; y, _ = librosa.load(str(audio(clip["audio_path"])), sr=16000, mono=True, duration=30.0)
        feats = self.proc([y], sampling_rate=16000, return_tensors="pt").input_features.to(self.dev, self.m.dtype)
        dec = self.torch.full((1, 1), self.m.config.decoder_start_token_id, device=self.dev, dtype=self.torch.long)
        with self.torch.no_grad():
            p = self.m(feats, decoder_input_ids=dec).logits[0, -1, self.ids].float().softmax(-1)
        j = int(p.argmax()); grp, lang = place_language(self.names[j], self.groups)
        return [I.fired(clip["row"], "speech.language", grp, self.tool, float(p[j]), leaf=lang)]


class Librosa:
    """Tempo and mode as words, only when the measurement is confident."""
    tool = "librosa"; needs = "music present"

    def __init__(self, cfg, log, schema):
        pass

    def label(self, rows):
        clip = rows[0]; d = tempo_and_mode(audio(clip["audio_path"])); out = []
        if d.get("tempo_bpm") and d.get("beat_strength", 0) >= 0.3:
            out.append(I.fired(clip["row"], "music.tempo", tempo_word(float(d["tempo_bpm"])), self.tool, d["beat_strength"]))
        if d.get("mode") and d.get("key_conf", 0) >= 0.8:
            out.append(I.fired(clip["row"], "music.mode", d["mode"], self.tool, d["key_conf"]))
        return out


EXPERTS = {c.tool: c for c in (Panns, WavlmSv, Emotion2vec, WhisperLid, Librosa)}


def run(cfg, log, schema, name: str, tools: list[str], limit=0):
    """Label every clip of set `name` with the given experts (resumable per clip and per expert)."""
    by = collections.OrderedDict()
    for r in I.rows(name):
        by.setdefault(r["clip"], []).append(r)
    clips = list(by)[: limit or None]
    for t in [t for t in ORDER if t in tools]:
        sh = I.Shard(name, t); todo = [c for c in clips if c not in sh.done]
        log.info("expert %-12s %d clips to do (%d done)", t, len(todo), len(sh.done))
        if not todo:
            sh.close(); continue
        ex = EXPERTS[t](cfg, log, schema)
        need = getattr(ex, "needs", None)
        fam = regions_of([f for f in read_jsonl(I.sdir(name) / "fired_panns.jsonl") if "done" not in f]) if need else {}
        n_fired = 0
        for k, c in enumerate(todo):
            rows = by[c]
            if need:                                                 # only rows where PANNs heard that region (the clip row
                rows = [r for r in rows if need in fam.get(r["row"], ())]     # is a maximum over its chunks, so it is on whenever a chunk is)
                assert not rows or rows[0]["grid"] == "clip", rows[0]
            try:
                recs = ex.label(rows) if rows else []
            except Exception as e:                                                  # noqa: BLE001
                log.warning("%s failed on %s (%s)", t, by[c][0]["audio_path"], e); continue
            sh.add(c, recs); n_fired += len(recs)
            if k % 50 == 0:
                log.info("  %s %d/%d  labels %d", t, k + 1, len(todo), n_fired)
        sh.close(); del ex
        log.info("expert %-12s %d labels", t, n_fired)


# ------------------------------------------------------------------------------------------ gender centroids
def _read(path):
    import soundfile as sf
    return sf.read(str(path), dtype="float32")[0]


def xvectors(enc, waves: list[np.ndarray]) -> np.ndarray:
    """L2-normalised speaker x-vectors of 16 kHz waveforms."""
    import torch
    out = []
    for i in range(0, len(waves), 16):
        inp = enc.fe(waves[i:i + 16], sampling_rate=G.SR, return_tensors="pt", padding=True).to(enc.dev)
        with torch.no_grad():
            out.append(enc.m(**inp).embeddings.float().cpu().numpy())
    e = np.vstack(out)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)


def iemocap_utterances(root: Path):
    """(wav path, utterance id, emotion code) of every labelled IEMOCAP utterance."""
    head = re.compile(r"^\[\d")
    for sess in range(1, 6):
        evald = root / f"Session{sess}" / "dialog" / "EmoEvaluation"; wavd = root / f"Session{sess}" / "sentences" / "wav"
        for txt in (sorted(evald.glob("Ses*.txt")) if evald.is_dir() else []):
            for line in txt.read_text(errors="ignore").splitlines():
                parts = line.split("\t")
                if not head.match(line) or len(parts) < 3:
                    continue
                utt = parts[1].strip(); wav = wavd / utt.rsplit("_", 1)[0] / f"{utt}.wav"
                if wav.exists():
                    yield str(wav), utt, parts[2].strip()


def gender_centroids(cfg, log, per_gender_cap: int = 300):
    """The mean x-vector of GENDER_CLIPS male and GENDER_CLIPS female IEMOCAP utterances (first 2 s of each)."""
    import librosa
    from l2r.encoders import build_encoder
    utts = list(iemocap_utterances(data("IEMOCAP"))); random.Random(cfg["seed"]).shuffle(utts)
    assert utts, f"no IEMOCAP utterances under {data('IEMOCAP')}"
    by = {"male": [], "female": []}
    for w, utt, _ in utts:
        g = "female" if utt.rsplit("_", 1)[1][0] == "F" else "male"
        if len(by[g]) < per_gender_cap:
            by[g].append(w)
    enc = build_encoder("wavlm_sv", cfg); cent = {}
    for g, paths in by.items():
        waves = [librosa.load(q, sr=G.SR, mono=True, duration=GENDER_WIN_S)[0] for q in paths[:GENDER_CLIPS]]
        c = xvectors(enc, [y for y in waves if len(y) >= G.SR // 2]).mean(0); cent[g] = c / (np.linalg.norm(c) + 1e-8)
    np.savez(ckpt(*GENDER_CENTROIDS), **cent); log.info("gender centroids -> %s", ckpt(*GENDER_CENTROIDS))
    return cent


def selftest(schema):
    rules = schema["regions"]["sound"]["attributes"]["event"]["audioset_kind"]
    anc = {"Bark": {"Dog", "Domestic animals, pets", "Animal"}, "Car alarm": {"Alarm", "Car", "Vehicle"}, "Drill": {"Power tool", "Tools"},
           "Rain": {"Water", "Natural sounds"}, "Cough": {"Respiratory sounds"}, "Male speech, man speaking": {"Speech", "Human voice"}}
    assert [kind_of(x, anc, rules) for x in ("Bark", "Car alarm", "Drill", "Rain", "Cough", "Zipper (clothing)")] == \
        ["animal", "alarm or signal", "tool", "natural element", "human non-speech", "unplaced"]
    assert {k for _, k in rules} <= set(schema["regions"]["sound"]["attributes"]["event"]["values"])
    assert is_family("Male speech, man speaking", anc, VOICE_ROOTS) and not is_family("Bark", anc, MUSIC_ROOTS)
    for a, v in PANNS_SPEECH.values():
        r_, f_ = a.split("."); assert v in schema["regions"][r_]["attributes"][f_]["values"], (a, v)
    assert [tempo_word(x) for x in (60, 100, 140, 180)] == ["slow tempo", "moderate tempo", "fast tempo", "very fast tempo"]
    assert set(map(tempo_word, (60, 100, 140, 180))) <= set(schema["regions"]["music"]["attributes"]["tempo"]["values"])
    g = {"mandarin": "cjk", "swahili": "ssa"}
    assert place_language("Chinese", g) == ("cjk", "mandarin") and place_language("swahili", g) == ("ssa", "swahili")
    assert place_language("latin", g) == ("unplaced", "latin") and place_language("sinhalese", g)[0] == "unplaced"
    assert EMOTIONS <= set(schema["regions"]["speech"]["attributes"]["emotion"]["values"])
    assert regions_of([I.fired("r1", "cross.families_present", "speech present", "panns", 0.9)])["r1"] == {"speech present"}
    if data("AudioSet", "ontology.json").exists():
        fa = full_ancestors()
        assert "Human voice" in fa["Speech"] and is_family("Narration, monologue", fa, VOICE_ROOTS) and is_family("Rapping", fa, VOICE_ROOTS + MUSIC_ROOTS)
        assert kind_of("Children playing", fa, rules) == "human non-speech" and kind_of("Bark", fa, rules) == "animal"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--gender-centroids", action="store_true")
    a = ap.parse_args()
    if a.gender_centroids:
        cfg = load_config(); _, log = setup_run("gender_centroids", cfg); gender_centroids(cfg, log)
