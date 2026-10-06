"""Transcripts with segment timestamps (Whisper), cached per benchmark.

    python -m l2r.retrieve.asr --benchmark mmau        ->  <work>/asr/mmau.json   {audio path: "[0:00.0] text [0:03.0] text ..."}

Whisper reads 30 s per pass, so a longer clip is transcribed window by window (`asr.max_windows` windows, i.e. the
first 3 minutes by default) and each window's segment times are shifted to clip time.
"""
from __future__ import annotations

import argparse
import re

from l2r.common import audio, load_config, load_json, save_json, setup_run, work

WINDOW_S = 30
SR = 16000


def cache_path(benchmark: str):
    return work("asr", f"{benchmark}.json")


def segments(decoded: str) -> list[tuple[float, str]]:
    """[(start seconds, text)] from Whisper's timestamped output `<|0.00|> text <|2.48|> ...`; one segment at 0 when
    the output carries no timestamp."""
    marks = list(re.finditer(r"<\|(\d+\.\d+)\|>", decoded))
    if not marks:
        body = re.sub(r"<\|[^|]*\|>", " ", decoded).strip()
        return [(0.0, body)] if body else []
    out = []
    for m, nxt in zip(marks, marks[1:] + [None]):
        body = re.sub(r"<\|[^|]*\|>", " ", decoded[m.end():(nxt.start() if nxt else len(decoded))]).strip()
        if body:
            out.append((float(m.group(1)), body))
    return out


def transcribe(paths: list[str], benchmark: str, cfg=None, log=None) -> dict[str, str]:
    """Transcripts of these audio files (paths relative to the data folder), from the cache where present."""
    import librosa
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    cfg = cfg or load_config()
    mid = cfg["models"]["asr"]; max_new = int(cfg["asr"]["max_new_tokens"]); max_windows = int(cfg["asr"]["max_windows"])
    cp = cache_path(benchmark)
    cache = load_json(cp) if cp.exists() else {}
    todo = [p for p in dict.fromkeys(paths) if p not in cache]
    if log:
        log.info("asr %s: %d clips, %d cached, %d to transcribe (%s)", benchmark, len(set(paths)), len(set(paths)) - len(todo), len(todo), mid)
    if todo:
        proc = WhisperProcessor.from_pretrained(mid)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        half = dev == "cuda"
        model = WhisperForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.float16 if half else torch.float32).to(dev).eval()
        win = WINDOW_S * SR
        for i, p in enumerate(todo):
            y = librosa.load(str(audio(p)), sr=SR, mono=True)[0]
            parts = []
            for k in range(min(max_windows, max(1, -(-len(y) // win)))):
                chunk = y[k * win:(k + 1) * win]
                if k and len(chunk) < 1600:                      # a tail under 0.1 s
                    continue
                feats = proc(chunk, sampling_rate=SR, return_tensors="pt").input_features.to(dev)
                with torch.no_grad():
                    ids = model.generate(feats.half() if half else feats, max_new_tokens=max_new, return_timestamps=True)
                dec = proc.tokenizer.decode(ids[0].tolist(), skip_special_tokens=False, decode_with_timestamps=True)
                for t0, txt in segments(dec):
                    t = k * WINDOW_S + t0
                    parts.append(f"[{int(t) // 60}:{t % 60:04.1f}] {txt.strip()}")
            cache[p] = " ".join(parts).strip()
            if (i + 1) % 500 == 0:
                save_json(cache, cp, indent=None)
        del model
        save_json(cache, cp, indent=None)
    return {p: cache[p] for p in paths}


if __name__ == "__main__":
    from l2r import benchmarks
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--benchmark", required=True)
    a = ap.parse_args()
    _, log = setup_run(f"asr_{a.benchmark}")
    out = transcribe([it["audio"] for it in benchmarks.load(a.benchmark)], a.benchmark, log=log)
    log.info("%d transcripts -> %s", len(out), cache_path(a.benchmark))
