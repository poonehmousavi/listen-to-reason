"""In-context baseline: Qwen2.5-Omni-7B is given labelled example clips of each option species in its prompt
(Figure 3, Table "Few-shot domain adaptation").

    python -m baselines.in_context --domain birds --shots 0,1,3,5      # -> <work>/results/in_context_birds.json

For an item, S example clips of each of its four option species precede the clip to classify. The examples are
the labelled clips of the domain head (`l2r.adapt.domain.few_shot`), cut to at most 10 s. The prompt length limits
the method to a few clips per species.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from baselines.audio_llm import ASK, SYSTEM, AudioLLM, extract_letter, load_items, score
from l2r.adapt import data as D
from l2r.adapt.domain import few_shot
from l2r.common import audio, check, data, load_config, save_json, setup_run, work

CUT_DIR = {"birds": "BirdSet/icl_pow", "marine": "Watkins/icl"}       # example clips, under the data folder
SR = 16_000


def cut(path, span, dst: Path) -> str:
    """An example clip as a 16 kHz wav of at most 10 s (birds: the window around the first vocalisation)."""
    if not dst.exists():
        import librosa
        import soundfile as sf
        x, sr = sf.read(str(audio(path)), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if span:
            x = x[int(span[0] * sr): int(span[1] * sr)]
        x = x[: int(D.WINDOW_S * sr)]
        if sr != SR:
            x = librosa.resample(x, orig_sr=sr, target_sr=SR)
        if len(x) < SR // 2:
            x = np.pad(x, (0, SR // 2 - len(x)))
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), x, SR, subtype="PCM_16")
    return str(dst)


def conversation(it: dict, examples: dict, shots: int) -> list[dict]:
    content = []
    if shots:
        content.append({"type": "text", "text": f"Here are {shots} labelled example recording(s) of each option."})
        for letter, name in sorted(it["choices"].items()):
            for j, p in enumerate(examples[name][:shots]):
                content.append({"type": "text", "text": f"Example {j + 1} of ({letter}) {name}:"})
                content.append({"type": "audio", "audio": p})
        content.append({"type": "text", "text": "Now the recording to classify:"})
    content.append({"type": "audio", "audio": it["audio"]})
    content.append({"type": "text", "text": f"{it['question']}\n{it['choices_str']}\n\n{ASK}"})
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}, {"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", choices=D.DOMAINS, required=True); ap.add_argument("--shots", default="0,1,3,5")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B"); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(); _, log = setup_run(f"in_context_{a.domain}", cfg)
    shots = [int(s) for s in a.shots.split(",")]
    items = load_items(a.domain, a.limit)
    names = {lab: v["name"] for lab, v in D.load_domain(a.domain)["pool"].items()}
    examples = {names[lab]: [cut(p, s, data(CUT_DIR[a.domain], lab, f"{i}_{Path(p).stem}.wav")) for i, (p, s) in enumerate(clips)]
                for lab, clips in few_shot(a.domain, max(max(shots), 1), cfg, log).items()}
    missing = {o for it in items for o in it["choices"].values() if o not in examples}
    check(not missing, f"every option species has example clips (missing {sorted(missing)[:5]})", log)
    llm = AudioLLM(a.model)
    out = {"domain": a.domain, "n": len(items), "results": {}, "predictions": {it["id"]: {"gold": it["gold"]} for it in items}}
    for s in shots:
        bs = 8 if s == 0 else max(1, 8 // (4 * s))
        preds, tokens = [], []
        for i in range(0, len(items), bs):
            batch = items[i:i + bs]
            raw = llm.generate([conversation(it, examples, s) for it in batch], max_new_tokens=16)
            preds += [extract_letter(r, it["options_norm"]) for r, it in zip(raw, batch)]
            tokens.append(llm.prompt_tokens)
        out["results"][f"shots{s}"] = {**score(items, preds), "prompt_tokens_p50": int(np.median(tokens))}
        for p, it in zip(preds, items):
            out["predictions"][it["id"]][f"shots{s}"] = p
        log.info("%d example(s) per option: accuracy %.3f (prompt %d tokens)", s, out["results"][f"shots{s}"]["accuracy"], int(np.median(tokens)))
        save_json(out, work("results", f"in_context_{a.domain}.json"))


if __name__ == "__main__":
    main()
