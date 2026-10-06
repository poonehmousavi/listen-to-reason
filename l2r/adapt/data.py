"""New-domain tasks: bird species (BirdSet, Powdermill subset) and marine mammals (Watkins, via BEANS).

Each domain becomes (a) a multiple-choice species-identification set in the benchmark item shape and (b) a
training pool of clips per species, from which the k labelled clips of a domain head are drawn.

  * Every item has four options: the correct species and three taxonomically close ones (same genus first,
    then the same species group, then any species), so the option list alone does not reveal the answer.
  * Evaluation and pool are disjoint by recordist (birds) or by the BEANS split (marine mammals).
  * A bird evaluation clip is one 10 s window starting 0.5 s before the first detected vocalisation; a marine
    clip is the first 10 s of the recording.

    python -m l2r.adapt.data --domain birds      # -> <work>/domains/birds.json  (+ 10 s cuts under <data>/BirdSet/eval_pow)
    python -m l2r.adapt.data --domain marine     # -> <work>/domains/marine.json (+ wavs under <data>/Watkins/audio)

Expected under the data folder:
    BirdSet/hf/POW/POW_metadata_train.parquet   and   BirdSet/POW_train/<filepath of the metadata>   (HF DBD-research-group/BirdSet, POW)
    Watkins/hf/data/{train,valid,test}-*.parquet                                                     (HF DBD-research-group/beans_watkins)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import random
from collections import Counter, defaultdict
from pathlib import Path

from l2r.common import audio, check, data, load_config, load_json, save_json, setup_run, work

DOMAINS = ("birds", "marine")
WINDOW_S = 10.0

# ----------------------------------------------------------------------------- birds (48 species)
BIRD_META = "BirdSet/hf/POW/POW_metadata_train.parquet"
BIRD_AUDIO = "BirdSet/POW_train"
BIRD_EVAL_DIR = "BirdSet/eval_pow"
BIRD_EVAL_PER_SPECIES = 20
BIRD_EVAL_RECORDIST_FRAC = 5                   # recordists whose hash is 0 modulo 5 go to the evaluation side
BIRD_SR = 32_000

# eBird codes -> common names
BIRD_NAMES = {
    "cangoo": "Canada goose", "wiltur": "wild turkey", "yebcuc": "yellow-billed cuckoo",
    "amgplo": "American golden-plover", "reshaw": "red-shouldered hawk",
    "rebwoo": "red-bellied woodpecker", "dowwoo": "downy woodpecker", "haiwoo": "hairy woodpecker",
    "norfli": "northern flicker", "pilwoo": "pileated woodpecker", "eawpew": "eastern wood-pewee",
    "reevir1": "red-eyed vireo", "buhvir": "blue-headed vireo", "blujay": "blue jay",
    "amecro": "American crow", "comrav": "common raven", "cedwax": "cedar waxwing",
    "tuftit": "tufted titmouse", "bkcchi": "black-capped chickadee", "ruckin": "ruby-crowned kinglet",
    "carwre": "Carolina wren", "buggna": "blue-gray gnatcatcher", "whbnut": "white-breasted nuthatch",
    "brncre": "brown creeper", "woothr": "wood thrush", "amerob": "American robin",
    "eastow": "eastern towhee", "balori": "Baltimore oriole", "rewbla": "red-winged blackbird",
    "bnhcow": "brown-headed cowbird", "ovenbi1": "ovenbird", "louwat": "Louisiana waterthrush",
    "buwwar": "blue-winged warbler", "bawwar": "black-and-white warbler", "naswar": "Nashville warbler",
    "kenwar": "Kentucky warbler", "comyel": "common yellowthroat", "hoowar": "hooded warbler",
    "amered": "American redstart", "babwar": "bay-breasted warbler", "chswar": "chestnut-sided warbler",
    "btnwar": "black-throated green warbler", "scatan": "scarlet tanager",
    "robgro": "rose-breasted grosbeak", "norcar": "northern cardinal", "swathr": "Swainson's thrush",
    "herthr": "hermit thrush", "veery": "veery",
}
BIRD_Q = "Which bird species is vocalising in this recording? Choose the correct option from the following options:"

# ----------------------------------------------------------------------------- marine mammals (31 species)
MARINE_PARQUET = "Watkins/hf/data"
MARINE_AUDIO = "Watkins/audio"
MARINE_EVAL_SPLITS = ("test", "valid")
MARINE_POOL_SPLITS = ("train",)
MARINE_SR = 16_000
MARINE_MIN_S = 0.5                             # shorter recordings are padded with silence

MARINE_GROUP = {
    "dolphin": ["Spinner_Dolphin", "Frasers_Dolphin", "Striped_Dolphin", "Grampus,_Rissos_Dolphin",
                "Clymene_Dolphin", "Pantropical_Spotted_Dolphin", "Atlantic_Spotted_Dolphin",
                "White-beaked_Dolphin", "White-sided_Dolphin", "Common_Dolphin",
                "Rough-Toothed_Dolphin", "Bottlenose_Dolphin"],
    "toothed whale": ["Sperm_Whale", "Long-Finned_Pilot_Whale", "Short-Finned_Pacific_Pilot_Whale",
                      "Melon_Headed_Whale", "False_Killer_Whale", "Killer_Whale", "Narwhal",
                      "Beluga,_White_Whale"],
    "baleen whale": ["Humpback_Whale", "Bowhead_Whale", "Northern_Right_Whale", "Southern_Right_Whale",
                     "Fin,_Finback_Whale", "Minke_Whale"],
    "pinniped": ["Ross_Seal", "Harp_Seal", "Bearded_Seal", "Walrus", "Leopard_Seal"],
}
MARINE_NAMES = {
    "Grampus,_Rissos_Dolphin": "Risso's dolphin", "Fin,_Finback_Whale": "fin whale",
    "Beluga,_White_Whale": "beluga whale", "Frasers_Dolphin": "Fraser's dolphin",
    "Short-Finned_Pacific_Pilot_Whale": "short-finned pilot whale",
    "Melon_Headed_Whale": "melon-headed whale", "Northern_Right_Whale": "North Atlantic right whale",
    "Rough-Toothed_Dolphin": "rough-toothed dolphin", "Long-Finned_Pilot_Whale": "long-finned pilot whale",
}
MARINE_Q = "Which marine mammal species produced the sounds in this recording? Choose the correct option from the following options:"


def marine_name(label: str) -> str:
    return MARINE_NAMES.get(label, label.replace("_", " ").lower())


def domain_file(domain: str) -> Path:
    return work("domains", f"{domain}.json")


# ----------------------------------------------------------------------------- multiple-choice items
def mcq(rng: random.Random, gold_name: str, tiers: list[list[str]], k: int = 4) -> tuple[dict, str]:
    """Distractors from the closest tier first, topped up from the next tiers; options shuffled."""
    chosen: list[str] = []
    for tier in tiers:
        cand = [n for n in tier if n != gold_name and n not in chosen]
        rng.shuffle(cand)
        chosen.extend(cand[: k - 1 - len(chosen)])
        if len(chosen) == k - 1:
            break
    assert len(chosen) == k - 1, (gold_name, chosen)
    opts = chosen + [gold_name]
    rng.shuffle(opts)
    choices = dict(zip("abcd"[:k], opts))
    return choices, next(l for l, o in choices.items() if o == gold_name)


def item(iid, audio_path, group, question, choices, gold, **extra) -> dict:
    it = {"id": iid, "audio": str(audio_path), "group": group, "question": question, "choices": choices, "gold": gold,
          "choices_str": "  ".join(f"({k}) {v}" for k, v in choices.items()),
          "options_norm": {k: " ".join(v.lower().strip().split()) for k, v in choices.items()}}
    it.update(extra)
    return it


# ----------------------------------------------------------------------------- birds
def _events(ev) -> list[list[float]]:
    return [] if ev is None else [[float(a), float(b)] for a, b in ev]


def _recordist_side(recordist: str) -> str:
    h = int(hashlib.md5((recordist or "").encode()).hexdigest(), 16)
    return "eval" if h % BIRD_EVAL_RECORDIST_FRAC == 0 else "pool"


def window_start(events) -> float:
    """Start of a bird clip's 10 s window: 0.5 s before the first detected vocalisation."""
    return max(0.0, events[0][0] - 0.5) if events else 0.0


def cut_window(src: Path, dst: Path, events) -> float | None:
    """Write the 10 s window of a bird recording as a 32 kHz wav; None if the file cannot be read."""
    import soundfile as sf
    try:
        x, sr = sf.read(str(src), dtype="float32", always_2d=True)
    except Exception:
        return None
    x = x.mean(axis=1)
    if sr != BIRD_SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=BIRD_SR)
        sr = BIRD_SR
    dur = len(x) / sr
    start = window_start(_events(events))
    if start + WINDOW_S > dur:
        start = max(0.0, dur - WINDOW_S)
    seg = x[int(start * sr): int((start + WINDOW_S) * sr)]
    if len(seg) < sr:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), seg, sr, subtype="PCM_16")
    return start


def build_birds(cfg, log) -> dict:
    import pandas as pd
    meta = pd.read_parquet(data(BIRD_META))
    check(data(BIRD_AUDIO).exists(), f"the POW recordings are extracted under {data(BIRD_AUDIO)}", log)
    check(set(meta.ebird_code) <= set(BIRD_NAMES), "every POW code has a common name", log)
    by_species: dict[str, list] = defaultdict(list)
    genus_of, group_of = {}, {}
    for r in meta.itertuples(index=False):
        if not data(BIRD_AUDIO, r.filepath).exists():
            continue
        genus_of[r.ebird_code] = r.genus
        group_of[r.ebird_code] = r.species_group
        by_species[r.ebird_code].append(r)
    check(len(by_species) == 48, f"48 species with audio on disk (got {len(by_species)})", log)

    rng = random.Random(cfg.get("seed", 0))
    items, pool, stats = [], {}, Counter()
    for code in sorted(by_species):
        recs = by_species[code]
        ev = sorted((r for r in recs if _recordist_side(r.recordist) == "eval"), key=lambda r: ((r.quality or "Z"), r.filepath))
        pl = [r for r in recs if _recordist_side(r.recordist) == "pool"]
        pool[code] = {"name": BIRD_NAMES[code], "genus": genus_of[code], "species_group": group_of[code],
                      "clips": [{"path": f"{BIRD_AUDIO}/{r.filepath}", "recordist": r.recordist, "quality": r.quality,
                                 "events": _events(r.detected_events)} for r in pl]}
        n_cut = 0
        for r in ev:                                   # best recording quality first
            if n_cut >= BIRD_EVAL_PER_SPECIES:
                break
            xc = Path(r.filepath).stem
            rel = f"{BIRD_EVAL_DIR}/{code}/{xc}.wav"
            if not data(rel).exists() and cut_window(data(BIRD_AUDIO, r.filepath), data(rel), r.detected_events) is None:
                stats["unreadable"] += 1
                continue
            gold_name = BIRD_NAMES[code]
            same_genus = [BIRD_NAMES[c] for c in by_species if genus_of[c] == genus_of[code]]
            same_group = [BIRD_NAMES[c] for c in by_species if group_of[c] == group_of[code]]
            everyone = [BIRD_NAMES[c] for c in by_species]
            choices, gold = mcq(rng, gold_name, [same_genus, same_group, everyone])
            items.append(item(f"pow/{code}/{xc}", rel, group_of[code], BIRD_Q, choices, gold, sub_category=gold_name,
                              source_file=r.filepath, recordist=r.recordist, quality=r.quality, genus=genus_of[code]))
            n_cut += 1
        stats["pool_clips"] += len(pl)
    ev_rec = {it["recordist"] for it in items}
    pl_rec = {c["recordist"] for v in pool.values() for c in v["clips"]}
    check(not (ev_rec & pl_rec), "evaluation and pool recordists are disjoint", log)
    log.info("birds: %d items over %d species, pool %d clips, unreadable %d", len(items), len(by_species), stats["pool_clips"], stats["unreadable"])
    return {"domain": "birds", "n": len(items), "species": sorted(by_species), "names": {c: BIRD_NAMES[c] for c in by_species},
            "gold_letters": dict(Counter(it["gold"] for it in items)), "items": items, "pool": pool}


# ----------------------------------------------------------------------------- marine mammals
def build_marine(cfg, log) -> dict:
    import numpy as np
    import pandas as pd
    import soundfile as sf
    group_of = {lab: g for g, labs in MARINE_GROUP.items() for lab in labs}

    def decode(split):
        """The recordings of one BEANS split, written once as 16 kHz wavs (first 10 s)."""
        rows = []
        for f in sorted(data(MARINE_PARQUET).glob(f"{split}-*.parquet")):
            for r in pd.read_parquet(f).itertuples(index=False):
                idx = int(getattr(r, "_0"))            # the unnamed index column of the parquet
                rel = f"{MARINE_AUDIO}/{split}/{idx:05d}.wav"
                if not data(rel).exists():
                    x, sr = sf.read(io.BytesIO(r.path["bytes"]), dtype="float32", always_2d=True)
                    x = x.mean(axis=1)
                    if sr != MARINE_SR:
                        import librosa
                        x = librosa.resample(x, orig_sr=sr, target_sr=MARINE_SR)
                    x = x[: int(WINDOW_S * MARINE_SR)]
                    if len(x) < int(MARINE_MIN_S * MARINE_SR):
                        x = np.pad(x, (0, int(MARINE_MIN_S * MARINE_SR) - len(x)))
                    data(rel).parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(data(rel)), x, MARINE_SR, subtype="PCM_16")
                rows.append((idx, r.label, rel, r.path.get("path")))
        return rows

    labels, pool = set(), {}
    for split in MARINE_POOL_SPLITS:
        for idx, lab, rel, src in decode(split):
            labels.add(lab)
            pool.setdefault(lab, {"name": marine_name(lab), "group": group_of[lab], "clips": []})
            pool[lab]["clips"].append({"path": rel, "source": src})
    rng = random.Random(cfg.get("seed", 0))
    items = []
    for split in MARINE_EVAL_SPLITS:
        for idx, lab, rel, src in decode(split):
            labels.add(lab)
            gold_name = marine_name(lab)
            same_group = [marine_name(l) for l in MARINE_GROUP[group_of[lab]]]
            everyone = [marine_name(l) for l in group_of]
            choices, gold = mcq(rng, gold_name, [same_group, everyone])
            items.append(item(f"watkins/{split}/{idx}", rel, group_of[lab], MARINE_Q, choices, gold, sub_category=gold_name,
                              source_file=src, split=split))
    check(labels <= set(group_of), f"every Watkins label has a taxonomic group ({labels - set(group_of)})", log)
    check(len(labels) == 31, f"31 marine species (got {len(labels)})", log)
    log.info("marine: %d items over %d species, pool %d clips", len(items), len(labels), sum(len(v["clips"]) for v in pool.values()))
    return {"domain": "marine", "n": len(items), "species": sorted(labels), "names": {l: marine_name(l) for l in labels},
            "groups": MARINE_GROUP, "gold_letters": dict(Counter(it["gold"] for it in items)), "items": items, "pool": pool}


# ----------------------------------------------------------------------------- loaders
def load_domain(domain: str) -> dict:
    """The built task file {items, pool, species, names}; run `python -m l2r.adapt.data --domain <domain>` first."""
    p = domain_file(domain)
    if not p.exists():
        raise SystemExit(f"{p} not found: run `python -m l2r.adapt.data --domain {domain}`")
    return load_json(p)


def candidates(domain: str, dom: dict | None = None) -> dict[str, list[tuple[str, tuple | None]]]:
    """{species label: [(path, 10 s window or None)]}: the pool clips of every species in selection order
    (birds: best recording quality first). The k labelled clips of a domain are the first k that pass the
    containment rule (l2r.adapt.domain)."""
    dom = dom or load_domain(domain)
    out = {}
    for lab in sorted(dom["pool"]):
        clips = dom["pool"][lab]["clips"]
        if domain == "birds":
            clips = sorted(clips, key=lambda c: ((c.get("quality") or "Z"), c["path"]))
            out[lab] = [(c["path"], (window_start(c.get("events") or []), window_start(c.get("events") or []) + WINDOW_S)) for c in clips]
        else:
            out[lab] = [(c["path"], None) for c in sorted(clips, key=lambda c: c["path"])]
    return out


def _items(domain: str, limit=None) -> list[dict]:
    items = load_domain(domain)["items"]
    return items[:limit] if limit else items


def load_birds(limit=None) -> list[dict]:
    """The bird species-identification items (938), `audio` relative to the data folder."""
    return _items("birds", limit)


def load_marine(limit=None) -> list[dict]:
    """The marine-mammal species-identification items (678), `audio` relative to the data folder."""
    return _items("marine", limit)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", choices=DOMAINS, required=True)
    a = ap.parse_args()
    cfg = load_config(); _, log = setup_run(f"adapt_data_{a.domain}", cfg)
    out = (build_birds if a.domain == "birds" else build_marine)(cfg, log)
    for it in out["items"]:
        check(audio(it["audio"]).exists(), f"audio exists: {it['audio']}")
    save_json(out, domain_file(a.domain), indent=None)
    log.info("%s: %d items, gold letters %s -> %s", a.domain, out["n"], out["gold_letters"], domain_file(a.domain))


if __name__ == "__main__":
    main()
