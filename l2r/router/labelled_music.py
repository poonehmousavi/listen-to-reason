"""The `musiccaps` label set: human labels for the music heads.

MusicCaps describes each clip with a list of free-text aspects ("soft female vocal", "low quality", ...) and IRMAS
gives the predominant instrument of each excerpt. Word rules map them onto the tree's music attributes:

    music.instrument    multiple choice   instrument words (MusicCaps), the excerpt's instrument (IRMAS)
    music.vocals        single choice     male / female / choir / rap / mixed / no vocals
    music.mood          single choice     mood words, the most frequent class
    music.production    single choice     live / amateur / lo-fi
    music.fidelity      single choice     clean / low quality / muffled / noisy / reverberant recording
    music.vocal_style   multiple choice   soft / passionate / flat / breathy / operatic / high-pitched / deep / harmonized

MMAU and MMAR draw music from the same sources, so a candidate clip whose CLAP embedding is within `--tau` cosine
of any benchmark clip is left out. One 10 s chunk per clip; train / held-out split by recording.

    python -m l2r.router.labelled_music        ->  <work>/sets/musiccaps/{rows.jsonl, pairs.jsonl, space.json, info.json}

Needs `<data>/MusicCaps/{musiccaps-public.csv, audio/<ytid>.wav}` and `<data>/IRMAS/<instrument>/*.wav` (the IRMAS
training set; folders named by its instrument codes).
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import re
from collections import Counter

import numpy as np

from l2r import containment
from l2r.common import audio, check, data, save_json, setup_run
from l2r.dataset import index as I, schema as S, segment as G
from l2r.router import data as D

NAME = "musiccaps"
MUSICCAPS = "MusicCaps"
IRMAS_DIR = "IRMAS"
# IRMAS instrument code -> (label used in the split key, class of music.instrument); the voice excerpts are not used
IRMAS = {"gac": ("acoustic_guitar", "acoustic guitar"), "cel": ("cello", "cello"), "cla": ("clarinet", "clarinet"),
         "gel": ("electric_guitar", "electric guitar"), "flu": ("flute", "flute"), "org": ("organ", "organ"), "pia": ("piano", "piano"),
         "sax": ("saxophone", "saxophone"), "tru": ("trumpet", "trumpet"), "vio": ("violin", "violin")}


def _words(*ws):
    return re.compile(r"\b(" + "|".join(ws) + r")\b")


INSTRUMENT = [
    ("acoustic guitar", _words("acoustic guitars?", "acoustic rhythm guitar", "rhythmic acoustic guitar", "classical guitar", "nylon string guitar", "spanish guitar")),
    ("electric guitar", _words("electric guitars?", "e-guitars?", "distorted guitar", "distortion guitar", "guitar solo", "amplified guitar", "guitar lead",
                               "lead guitar", "electric guitar melody", "distorted electric guitar", "overdriven guitar")),
    ("bass guitar", _words("bass guitar", "e-bass", "bass line", "bassline", "groovy bass", "upright bass", "double bass", "smooth bass", "strong bass line",
                           "percussive bass line", "bass lines", "bass(?! drum)")),
    ("violin", _words("violins?", "fiddle")), ("cello", _words("cellos?")), ("harp", _words("harp")),
    ("piano", _words("piano", "acoustic piano", "grand piano", "electric piano", "rhodes")), ("organ", _words("organ", "hammond")),
    ("synthesizer", _words("synth", "synths", "synthesizer", "synthesiser", "synth pad", "synth lead", "synth bass", "synthesiser arrangements?", "synth sounds")),
    ("flute", _words("flutes?")), ("clarinet", _words("clarinets?")), ("saxophone", _words("saxophones?", "sax")),
    ("trumpet", _words("trumpets?")), ("trombone", _words("trombones?")),
    ("drum kit", _words("drums", "acoustic drums", "electronic drums", "digital drums", "drum kit", "drumming", "punchy kick", "punchy snare", "kick", "snare",
                        "hi hats?", "hi-hats?", "drum machine", "steady drumming")),
    ("hand drums", _words("tabla", "congas?", "bongos?", "djembe", "hand drums?", "darbuka", "cajon")),
    ("cymbals", _words("cymbals?", "shimmering cymbals")), ("bells", _words("bells", "chimes", "glockenspiel", "xylophone", "marimba", "vibraphone")),
    ("accordion", _words("accordion")), ("harmonica", _words("harmonica")), ("banjo", _words("banjo")),
    ("orchestral strings", _words("strings", "string section", "orchestra", "orchestral", "string ensemble", "sustained strings")),
    ("other plucked or folk instrument", _words("sitar", "mandolin", "ukulele", "oud", "koto", "guzheng", "bouzouki", "lute", "zither", "balalaika", "shamisen", "erhu"))]
VOICE = r"(vocals?|voices?|singers?|singing|vocalists?|vocalisation|rapper|crooner)"
MOOD = {
    "energetic": ["energetic", "upbeat", "exciting", "spirited", "enthusiastic", "vibrant", "lively", "intense", "powerful", "vigorous", "adrenaline", "energy"],
    "calm": ["relaxing", "calming", "soothing", "calm", "peaceful", "meditative", "mellow", "chill", "easygoing", "gentle", "relaxed", "tranquil", "serene"],
    "melancholic": ["sad", "melancholic", "melancholy", "sentimental", "nostalgic", "poignant", "sorrowful", "mournful", "longing"],
    "joyful": ["happy", "cheerful", "joyful", "fun", "festive", "positive", "uplifting", "happy mood", "celebratory", "feel good"],
    "tense": ["suspenseful", "suspense", "tense", "eerie", "ominous", "scary", "spooky", "mysterious", "anxious", "thrilling"],
    "romantic": ["romantic", "love song", "sensual", "romantic mood", "tender", "intimate"],
    "dark": ["dark", "haunting", "grim", "sinister", "menacing", "gloomy", "aggressive", "violent"],
    "triumphant": ["epic", "triumphant", "inspiring", "motivational", "heroic", "anthemic", "majestic", "victorious", "cinematic"],
    "playful": ["playful", "funny", "bouncy", "quirky", "silly", "whimsical", "cheeky", "humorous"],
    "dreamy": ["dreamy", "ethereal", "atmospheric", "hypnotic", "trippy", "psychedelic", "spacey", "floaty"]}
PRODUCTION = [("live performance", _words("live performance", "live recording", "live audience", "concert", "crowd cheering", "live", "audience")),
              ("amateur recording", _words("amateur recording", "home recording", "home video", "amateur", "tutorial")),
              ("lo-fi recording", _words("lo-fi", "lofi", "lo fi"))]
FIDELITY = [("muffled recording", _words("muffled", "muddy", "boomy", "muffled audio", "muffled sound")),
            ("reverberant recording", _words("reverberant", "reverb", "echo", "echoey", "echoing", "roomy", "wide reverb")),
            ("noisy recording", _words("noisy", "static", "hiss", "hissing", "white noise", "crackling", "crushed", "noise")),
            ("low quality recording", _words("low quality", "poor audio quality", "bad audio quality", "inferior audio quality", "low quality audio", "poor quality",
                                             "low quality recording", "average audio quality", "bad quality")),
            ("clean recording", _words("high quality", "good audio quality", "studio quality", "crisp", "clean", "polished", "high fidelity", "clear"))]
VOCAL_STYLE = [("soft vocal", ["soft", "gentle", "tender", "mellow", "sweet"]),
               ("passionate vocal", ["passionate", "powerful", "emotional", "strong", "belting", "intense"]),
               ("flat vocal", ["flat"]), ("breathy vocal", ["breathy", "whispery", "whispered", "airy"]),
               ("operatic vocal", ["operatic", "opera", "soprano", "tenor"]),
               ("high-pitched vocal", ["high pitched", "high-pitched", "falsetto", "higher pitch", "high register"]),
               ("deep vocal", ["deep", "low pitched", "low-pitched", "baritone"]),
               ("harmonized vocals", ["harmonizing", "harmonising", "harmony", "harmonies", "backup", "background vocals", "backing vocals"])]
NO_VOCALS = ("instrumental", "instrumental music", "no voices", "no singer", "no voice", "no vocals", "no singers", "no vocal")


def labels_of(aspects: list[str]) -> dict[str, list[str]]:
    """The MusicCaps aspects of one clip -> {attribute: classes}."""
    a = [x.lower().strip() for x in aspects]
    joined = " | ".join(a)
    out = {}
    ins = [v for v, rx in INSTRUMENT if rx.search(joined)]
    if ins:
        out["music.instrument"] = ins
    voc = [x for x in a if not x.startswith("no ") and re.search(r"\b" + VOICE + r"\b", x) or re.search(r"\b(rap|rapping|choir|choral)\b", x)]
    male = any(re.search(r"\b(male|man|men|boy|boys)\b", x) for x in voc)
    female = any(re.search(r"\b(female|woman|women|girl|girls)\b", x) for x in voc)
    if any(re.search(r"\b(rap|rapping|rapper)\b", x) for x in voc):
        out["music.vocals"] = ["rap vocals"]
    elif any(re.search(r"\b(choir|choral)\b", x) for x in voc):
        out["music.vocals"] = ["choir"]
    elif male and female or any("duet" in x for x in a):
        out["music.vocals"] = ["mixed vocals"]
    elif male:
        out["music.vocals"] = ["male vocals"]
    elif female:
        out["music.vocals"] = ["female vocals"]
    elif any(x in NO_VOCALS for x in a) and not voc:
        out["music.vocals"] = ["no vocals"]
    hits, first = Counter(), {}
    for i, x in enumerate(a):
        for v, ws in MOOD.items():
            if any(re.search(r"\b" + re.escape(w) + r"\b", x) for w in ws) and not re.search(VOICE, x):
                hits[v] += 1
                first.setdefault(v, i)
    if hits:
        top = max(hits.values())
        out["music.mood"] = [min((v for v in hits if hits[v] == top), key=first.get)]
    for v, rx in PRODUCTION:
        if rx.search(joined):
            out["music.production"] = [v]
            break
    for v, rx in FIDELITY:
        if rx.search(joined):
            out["music.fidelity"] = [v]
            break
    else:                                    # MusicCaps names degraded audio consistently: no such word and no live / amateur word = clean
        if "music.production" not in out:
            out["music.fidelity"] = ["clean recording"]
    style = [v for v, ws in VOCAL_STYLE if any(re.search(r"\b" + re.escape(w) + r"\b", x) for x in voc for w in ws)]
    if style:
        out["music.vocal_style"] = style
    return out


def candidates(log) -> list[tuple[str, str, dict, str]]:
    """(audio path relative to the data folder, split key, {attribute: classes}, source) of every labelled clip on disk."""
    out = []
    for r in csv.DictReader(open(data(MUSICCAPS, "musiccaps-public.csv"), newline="")):
        rel = f"{MUSICCAPS}/audio/{r['ytid']}.wav"
        if audio(rel).exists():
            lab = labels_of([str(x).strip() for x in ast.literal_eval(r["aspect_list"]) if str(x).strip()])
            if lab:
                out.append((rel, r["ytid"], lab, "musiccaps"))
    n_mc = len(out)
    for code, (long_name, inst) in sorted(IRMAS.items(), key=lambda kv: kv[1][0]):
        d = data(IRMAS_DIR, code)
        d = d if d.exists() else data(IRMAS_DIR, long_name)
        for w in sorted(d.glob("*.wav")) if d.exists() else []:
            ins = [inst] + (["drum kit"] if "[dru]" in w.name else [])
            out.append((f"{IRMAS_DIR}/{d.name}/{w.name}", w.name.split("__")[0] + "|" + long_name, {"music.instrument": ins}, "irmas"))
    log.info("candidates: %d MusicCaps clips with at least one label, %d IRMAS excerpts", n_mc, len(out) - n_mc)
    return out


def contained(cand: list, names: list[str], tau: float, log) -> np.ndarray:
    """True for the candidates whose best cosine to any benchmark clip is >= tau."""
    from l2r.encoders import build_encoder
    clap = build_encoder("clap")
    E = containment.benchmark_embeddings(clap, log, tuple(names))
    C = np.vstack([clap.embed([str(audio(c[0])) for c in cand[i:i + 64]]) for i in range(0, len(cand), 64)]).astype(np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    best = np.concatenate([(C[i:i + 2048] @ E.T).max(1) for i in range(0, len(C), 2048)])
    for t in (0.80, 0.85, 0.90, 0.95, 0.99):
        log.info("  %d candidates within %.2f of a benchmark clip", int((best >= t).sum()), t)
    return best >= tau


def build(log, tau: float = 0.85, names=("mmau", "mmar", "sakura"), drop: np.ndarray | None = None, name: str = NAME) -> dict:
    import soundfile as sf
    cand = candidates(log)
    drop = contained(cand, list(names), tau, log) if drop is None else drop
    n_drop = int(drop.sum())
    check(n_drop < 0.25 * len(cand), f"containment at {tau} leaves out {n_drop} of {len(cand)} candidates (< 25%)", log)
    rows, pairs = [], []
    for (rel, key, lab, src), out in zip(cand, drop):
        if out:
            continue
        t1 = round(min(sf.info(str(audio(rel))).duration, 10.0), 2)
        cid = G.clip_id(rel)
        rid = f"{cid}:g10:0"
        split = "test" if int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 10 == 0 else "train"
        rows.append({"clip": cid, "audio_path": rel, "source": f"music_{src}", "row": rid, "grid": "g10", "t0": 0.0, "t1": t1})
        for attr, vals in lab.items():
            pairs.append({"row": rid, "clip": cid, "source": f"music_{src}", "grid": "g10", "t0": 0.0, "t1": t1, "attr": attr, "values": vals,
                          "leaves": {}, "from": {v: "own" for v in vals}, "split": split})
    base = D.space(S.load_schema())
    space = {a: dict(base[a]) for a in ("music.instrument", "music.vocals", "music.mood", "music.production")}
    space["music.fidelity"] = {"region": "music", "grid": "g10", "scope": "constant", "multi": False, "values": [v for v, _ in FIDELITY]}
    space["music.vocal_style"] = {"region": "music", "grid": "g10", "scope": "local", "multi": True, "values": [v for v, _ in VOCAL_STYLE]}
    for a, sp in space.items():
        bad = {v for p in pairs if p["attr"] == a for v in p["values"]} - set(sp["values"])
        assert not bad, (a, bad)
    I.write_rows(name, rows)
    D.write(name, pairs, space)
    info = {"tau": tau, "benchmarks": list(names), "candidates": len(cand), "left_out": n_drop, "rows": len(rows), "pairs": len(pairs)}
    save_json(info, I.sdir(name) / "info.json")
    for a in space:
        c = Counter(v for p in pairs if p["attr"] == a and p["split"] == "train" for v in p["values"])
        log.info("%-18s train pairs %5d  %s", a, sum(1 for p in pairs if p["attr"] == a and p["split"] == "train"), dict(c.most_common()))
    log.info("%s: %d clips, %d pairs (%d candidates, %d left out by containment at %.2f)", name, len(rows), len(pairs), len(cand), n_drop, tau)
    return info


def selftest():
    lab = labels_of(["low quality", "sustained strings melody", "soft female vocal", "mellow piano melody", "sad", "soulful", "ballad"])
    assert lab == {"music.instrument": ["piano", "orchestral strings"], "music.vocals": ["female vocals"], "music.mood": ["calm"],
                   "music.fidelity": ["low quality recording"], "music.vocal_style": ["soft vocal"]}, lab
    lab = labels_of(["instrumental", "no voice", "live performance", "electric guitar solo", "energetic drums"])
    assert lab["music.vocals"] == ["no vocals"] and lab["music.production"] == ["live performance"] and "music.fidelity" not in lab
    assert lab["music.instrument"] == ["electric guitar", "drum kit"] and lab["music.mood"] == ["energetic"]
    assert labels_of(["male rapper", "female backing vocals harmonizing"])["music.vocals"] == ["rap vocals"]
    assert labels_of(["calm flute"]) == {"music.instrument": ["flute"], "music.mood": ["calm"], "music.fidelity": ["clean recording"]}
    print("labelled music selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tau", type=float, default=0.85, help="a candidate within this CLAP cosine of a benchmark clip is left out")
    ap.add_argument("--benchmarks", default="mmau,mmar,sakura")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        _, log = setup_run("labelled_music")
        build(log, a.tau, tuple(a.benchmarks.split(",")))
