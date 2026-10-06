"""Benchmark loaders. Every benchmark is a list of multiple-choice items in one shape:

    {"id", "audio" (path relative to the data folder), "question", "choices" {letter: text}, "choices_str",
     "options_norm" {letter: normalised text}, "gold" (letter), "group", "category", "sub_category"}

    mmau      MMAU test-mini, 1,000 items                  data/MMAU/           (downloaded and decoded on first use)
    mmar      MMAR, 1,000 items                            data/MMAR/           MMAR-meta.json + audio/
    sakura    SAKURA, 4 tracks x (single + multi hop)      data/SAKURA/         data/<Track>/metadata.json + audio
    mmaupro   MMAU-Pro, single-audio multiple choice       data/MMAU-Pro/       test.parquet + data/
    birds     bird species, 938 items                      (l2r/adapt/data.py)
    marine    marine-mammal species, 678 items             (l2r/adapt/data.py)
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter

from l2r.common import data

BENCHMARKS = ("mmau", "mmar", "sakura", "mmaupro", "birds", "marine")
SAKURA_TRACKS = ("Animal", "Emotion", "Gender", "Language")
LETTERS = "abcdefghijklmnopqrstuvwxyz"
_OPTION = re.compile(r"\(([a-d])\)\s*(.+?)(?=\s*\([a-d]\)|$)", re.IGNORECASE)
_GOLD = re.compile(r"\(([a-dA-D])\)")
SHUFFLE_SEED = 20260925


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def choices_str(opts: dict) -> str:
    return "  ".join(f"({k}) {v}" for k, v in opts.items())


def match_gold(answer: str, opts: dict) -> str:
    """The letter of the option whose text is the answer, or "" when there is none."""
    a = answer.lower().strip()
    for k, v in opts.items():
        if str(v).lower().strip() == a:
            return k
    return ""


def match_gold_mmar(answer: str, opts: dict) -> str:
    """MMAR gives the answer as option text. Exact text first; else the one option that contains the answer (or is
    contained in it); else the option with clearly the highest word overlap. An answer that spans several lines
    names several options and has no single gold."""
    if "\n" in answer.strip():
        return ""
    g = match_gold(answer, opts)
    if g:
        return g
    a = answer.lower().strip()
    hit = [k for k, v in opts.items() if a in str(v).lower() or str(v).lower().strip() in a]
    if len(hit) == 1:
        return hit[0]
    words = lambda s: set(re.findall(r"[a-z0-9]+", s))
    aw = words(a)
    sc = sorted(((len(aw & words(str(v).lower())) / max(1, len(aw)), k) for k, v in opts.items()), reverse=True)
    return sc[0][1] if sc and sc[0][0] >= 0.5 and (len(sc) == 1 or sc[0][0] - sc[1][0] >= 0.2) else ""


def shuffle_options(item_id, opts: dict, gold: str, seed: int = SHUFFLE_SEED):
    """A fixed per-item permutation of the option texts (letters stay a, b, c, ...; the gold letter follows its text)."""
    keys = list(opts); texts = [opts[k] for k in keys]
    perm = list(range(len(keys))); random.Random(f"{seed}:{item_id}").shuffle(perm)
    new = {keys[i]: texts[perm[i]] for i in range(len(keys))}
    return new, keys[perm.index(keys.index(gold))]


def _item(id_, audio, group, question, opts, gold, norm, category="", sub_category="", cstr=None, **extra):
    return {"id": id_, "audio": audio, "group": group, "question": question, "choices": opts, "gold": gold,
            "choices_str": choices_str(opts) if cstr is None else cstr, "options_norm": {k: norm(v) for k, v in opts.items()},
            "category": category, "sub_category": sub_category, **extra}


def load_mmau() -> list[dict]:
    """MMAU test-mini from the Hugging Face hub (gamma-lab-umd/mmau-test-mini); its audio is written once to data/MMAU/audio/."""
    import soundfile as sf
    from datasets import load_dataset
    out = data("MMAU", "audio"); out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("gamma-lab-umd/mmau-test-mini", split="test")
    items = []
    for i in range(len(ds)):
        row = ds[i]
        attr = json.loads(row["other_attributes"])
        wav = out / f"{attr['id']}.wav"
        if not wav.exists():
            s = row["context"].get_all_samples()
            arr = s.data.numpy()
            sf.write(str(wav), arr.mean(0) if arr.ndim == 2 else arr.reshape(-1), s.sample_rate)
        opts = {c[1].lower(): c.split(")", 1)[1].strip() for c in row["choices"]}          # "(A) Man"
        ans = row["answer"]
        gold = ans[1].lower() if ans.startswith("(") else match_gold(ans, opts)
        items.append(_item(attr["id"], f"MMAU/audio/{attr['id']}.wav", attr["task"], row["instruction"], opts, gold, _norm,
                           attr.get("category", ""), attr.get("sub-category", ""), source_dataset=attr.get("dataset", "")))
    return items


def load_mmar() -> list[dict]:
    """MMAR (BoJack/MMAR): data/MMAR/MMAR-meta.json and data/MMAR/audio/. Items whose answer names two options are dropped."""
    items = []
    for d in json.loads(data("MMAR", "MMAR-meta.json").read_text()):
        rel = "MMAR/" + d["audio_path"].lstrip("./")
        if not data(rel).exists():
            continue
        opts = {LETTERS[i]: v for i, v in enumerate(d["choices"])}
        gold = match_gold_mmar(str(d.get("answer") or "").strip(), opts)
        if gold:
            items.append(_item(d["id"], rel, d.get("modality", "?"), d.get("question", ""), opts, gold, lambda v: str(v).lower(),
                               d.get("category", ""), d.get("sub-category", "")))
    return items


def load_sakura() -> list[dict]:
    """SAKURA: per track, every clip with a single-hop and a multi-hop question (the options are part of the question text)."""
    items = []
    for track in SAKURA_TRACKS:
        meta = json.loads(data("SAKURA", "data", track, "metadata.json").read_text())
        for rel, v in meta.items():
            gs, gm = _GOLD.search(v["single_answer"]), _GOLD.search(v["multi_answer"])
            if not (gs and gm):
                continue
            for hop, g in (("single", gs), ("multi", gm)):
                q = v[f"{hop}_instruction"]
                opts = {m.group(1).lower(): _norm(m.group(2)) for m in _OPTION.finditer(q)}
                items.append(_item(f"{track}/{hop}/{rel}", f"SAKURA/{rel}", track, q, opts, g.group(1).lower(), lambda v: v, cstr="",
                                   hop=hop, label=str(v.get("attribute_label", "")).lower()))
    return items


def load_mmaupro() -> list[dict]:
    """MMAU-Pro (gamma-lab-umd/MMAU-Pro): the single-audio multiple-choice items (2 to 11 options). Open-ended,
    instruction-following and multi-audio items are not supported. The gold letters are strongly unbalanced, so the
    options of every item are shuffled with a fixed per-item permutation, the same for every system."""
    import pandas as pd
    df = pd.read_parquet(data("MMAU-Pro", "test.parquet"))
    items = []
    for r in df.itertuples(index=False):
        ch = list(r.choices) if r.choices is not None else []
        paths = list(r.audio_path)
        if r.category in ("open", "instruction following", "multi") or len(paths) != 1 or not 2 <= len(ch) <= len(LETTERS):
            continue
        opts = {LETTERS[i]: str(c).strip() for i, c in enumerate(ch)}
        gold = match_gold(str(r.answer), opts)
        if not gold or not data("MMAU-Pro", paths[0]).exists():
            continue
        opts, gold = shuffle_options(r.id, opts, gold)
        skills = lambda x: " / ".join(list(x) if x is not None else [])
        items.append(_item(r.id, f"MMAU-Pro/{paths[0]}", r.category, r.question, opts, gold, _norm, skills(r.perceptual_skills), skills(r.reasoning_skills)))
    return items


def load(name: str) -> list[dict]:
    """The items of a benchmark."""
    if name in ("birds", "marine"):
        from l2r.adapt import data as domain_data
        return {"birds": domain_data.load_birds, "marine": domain_data.load_marine}[name]()
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r} (known: {BENCHMARKS})")
    items = {"mmau": load_mmau, "mmar": load_mmar, "sakura": load_sakura, "mmaupro": load_mmaupro}[name]()
    assert items, f"no item of {name} found under {data()}"
    return items


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:] or BENCHMARKS:
        its = load(name)
        print(f"{name}: {len(its)} items, {len({i['audio'] for i in its})} clips, groups {dict(Counter(i['group'] for i in its))}")
