"""Audio-LLM baselines: an audio LLM hears the clip and answers the same question with the same options and scoring.

    python -m baselines.audio_llm --benchmark mmau                                   # Qwen2.5-Omni-7B (Table 1)
    python -m baselines.audio_llm --benchmark mmaupro --max-sec 180 --batch 4
    python -m baselines.audio_llm --benchmark birds                                  # zero-shot on a new domain
    python -m baselines.audio_llm --benchmark mmau --model Qwen/Qwen3-Omni-30B-A3B-Instruct
    python -m baselines.audio_llm --benchmark birds --adapter <work>/qlora/birds_k5   # the QLoRA baseline (see qlora.py)
    -> <work>/results/audio_llm_<tag>_<benchmark>.json

The model gets the audio first, then the question and options, and is asked for the letter; its answer is mapped
to an option letter by `extract_letter`. `--max-sec` gives the model only the first seconds of a longer clip.
`--adapter PATH` loads the Qwen2.5-Omni thinker in 4-bit with a LoRA adapter, as it was trained; `--adapter 4bit`
is the same quantised thinker without an adapter (the control for a fine-tuned model).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from l2r.common import audio, save_json, setup_run, work

SYSTEM = "You are an expert at reasoning about speech, audio, and the things that produce them."
ASK = "Answer with only the letter of the correct option."
DOMAINS = ("birds", "marine")


def load_items(benchmark: str, limit=None) -> list[dict]:
    if benchmark in DOMAINS:
        from l2r.adapt import data as D
        items = D.load_birds(limit) if benchmark == "birds" else D.load_marine(limit)
    else:
        from l2r.benchmarks import load
        items = load(benchmark)
        items = items[:limit] if limit else items
    return [{**it, "audio": str(audio(it["audio"]))} for it in items]


def user_text(it: dict) -> str:
    return f"{it['question']}\n{it['choices_str']}\n\n{ASK}"


# ----------------------------------------------------------------------------- answer -> option letter
_CONCLUSION = re.compile(r"<Conclu[^>]*>(.*?)(?:</Conclu|$)", re.S | re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def extract_letter(raw: str, options: dict[str, str]) -> str | None:
    """Map a free-text answer to an option letter. `options` is {letter: normalised option text}. Tried in order:
    an explicit "(c)", a bare leading letter, the exact option text, "c." / "c)", an option text contained in the
    answer, the last standalone letter, and finally word overlap with an option."""
    letters = "".join(sorted(options)) or "abcd"
    letters = "abcd" if set(letters) <= set("abcd") else letters
    c = f"{letters[0]}-{letters[-1]}{letters[0].upper()}-{letters[-1].upper()}"
    m = _CONCLUSION.search(raw)
    content = re.sub(r"<[^>]+>", " ", (m.group(1) if m else raw).strip())
    cn = _norm(content)
    m = re.search(rf"\(([{c}])\)", content)
    if m:
        return m.group(1).lower()
    m = re.match(rf"^\s*([{c}])(?:[\.\)\:]|\s*$)", content)
    if m and m.group(1).lower() in options:
        return m.group(1).lower()
    for L, t in options.items():
        if t and cn == t:
            return L
    m = re.search(rf"(?:^|[^a-zA-Z])([{c}])[\.\):]", content)
    if m:
        return m.group(1).lower()
    for L, t in options.items():
        if t and len(cn) >= 3 and (t in cn or cn in t):
            return L
    ms = list(re.finditer(rf"(?:^|[^a-zA-Z])([{c}])(?:[^a-zA-Z]|$)", content))
    if ms:
        return ms[-1].group(1).lower()
    if cn:
        words, best, best_letter = set(cn.split()), 0, None
        for L, t in options.items():
            overlap = len(words & set(t.split()))
            if overlap > best:
                best, best_letter = overlap, L
        if best:
            return best_letter
    return None


# ----------------------------------------------------------------------------- the model
def load_thinker(model_id: str, quant: bool = True, adapter: str | None = None):
    """The Qwen2.5-Omni thinker (audio encoder + language model) alone, optionally 4-bit and with a LoRA adapter."""
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen2_5OmniThinkerForConditionalGeneration
    kw = {"torch_dtype": torch.bfloat16, "device_map": {"": 0}}
    if quant:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    m = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_id, **kw)
    tok = AutoTokenizer.from_pretrained(model_id)                  # the thinker alone has no stop tokens configured
    m.generation_config.eos_token_id = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    m.generation_config.pad_token_id = tok.convert_tokens_to_ids("<|endoftext|>")
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    return m


class AudioLLM:
    """Qwen2.5-Omni or Qwen3-Omni (the class follows the checkpoint name)."""

    def __init__(self, model_id: str, adapter: str | None = None, max_sec: float | None = None, max_new_tokens: int = 128):
        import transformers
        q3 = "qwen3-omni" in model_id.lower()
        self.proc = (transformers.Qwen3OmniMoeProcessor if q3 else transformers.Qwen2_5OmniProcessor).from_pretrained(model_id)
        self.thinker_only = bool(adapter) and not q3
        if self.thinker_only:
            self.m = load_thinker(model_id, adapter=None if adapter == "4bit" else adapter).eval()
        else:
            cls = transformers.Qwen3OmniMoeForConditionalGeneration if q3 else transformers.Qwen2_5OmniForConditionalGeneration
            self.m = cls.from_pretrained(model_id, torch_dtype="auto", device_map="auto").eval()
            self.m.disable_talker()
        self.max_sec, self.max_new_tokens, self.n_cut = max_sec, max_new_tokens, 0

    def _clip(self, path: str) -> str:
        """The clip itself, or its first `max_sec` seconds written once under <work>/audio_llm_cuts/."""
        if not self.max_sec:
            return path
        import soundfile as sf
        info = sf.info(path)
        if info.duration <= float(self.max_sec):
            return path
        out = work("audio_llm_cuts", f"{Path(path).stem}_{int(self.max_sec)}s.wav")
        if not out.exists():
            a, sr = sf.read(path, frames=int(float(self.max_sec) * info.samplerate), dtype="float32")
            sf.write(str(out), a, sr)
        self.n_cut += 1
        return str(out)

    def conversation(self, it: dict) -> list[dict]:
        return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "audio", "audio": self._clip(it["audio"])}, {"type": "text", "text": user_text(it)}]}]

    def generate(self, convs: list[list[dict]], max_new_tokens: int | None = None) -> list[str]:
        import torch
        from qwen_omni_utils import process_mm_info
        texts = [self.proc.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in convs]
        audios = [a for c in convs for a in (process_mm_info(c, use_audio_in_video=False)[0] or [])]
        inp = self.proc(text=texts, audio=audios, return_tensors="pt", padding=True, use_audio_in_video=False).to(self.m.device)
        dtype = next(self.m.parameters()).dtype
        enc = {k: (v.to(dtype) if hasattr(v, "dtype") and v.dtype.is_floating_point else v) for k, v in inp.items()}
        with torch.no_grad():
            g = self.m.generate(**enc, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False, use_audio_in_video=False,
                                **({} if self.thinker_only else {"return_audio": False}))
        ids = g[0] if isinstance(g, (tuple, list)) else g
        self.prompt_tokens = int(enc["input_ids"].shape[1])
        return self.proc.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def score(items: list[dict], preds: list) -> dict:
    hit = [p == it["gold"] for p, it in zip(preds, items)]
    groups = sorted({it["group"] for it in items})
    return {"n": len(items), "accuracy": round(float(np.mean(hit)), 4), "parsed": round(float(np.mean([p is not None for p in preds])), 4),
            "groups": {g: round(float(np.mean([h for h, it in zip(hit, items) if it["group"] == g])), 4) for g in groups}}


def save_result(tag: str, benchmark: str, items: list[dict], preds: list, **meta) -> dict:
    res = score(items, preds)
    save_json({**meta, "benchmark": benchmark, **res, "predictions": [{"id": it["id"], "pred": p, "gold": it["gold"]} for it, p in zip(items, preds)]},
              work("results", f"audio_llm_{tag}_{benchmark}.json"))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True); ap.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--export-manifest", default="", help="write the items for a model that runs in another environment, and stop")
    ap.add_argument("--score-answers", default="", help="score a {id: raw answer} file written by such a model")
    ap.add_argument("--adapter", default=None, help="LoRA adapter folder, or `4bit` for the quantised thinker without an adapter")
    ap.add_argument("--max-sec", type=float, default=None); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tag = a.tag or (Path(a.adapter).name if a.adapter else Path(a.model).name.lower())
    _, log = setup_run(f"audio_llm_{tag}_{a.benchmark}")
    items = load_items(a.benchmark, a.limit)
    if a.export_manifest:
        save_json([{q: it[q] for q in ("id", "audio", "question", "choices")} for it in items], a.export_manifest)
        return
    if a.score_answers:
        from l2r.common import load_json
        raw = load_json(a.score_answers)
        res = save_result(tag, a.benchmark, items, [extract_letter(raw.get(it["id"], ""), it["options_norm"]) for it in items])
        log.info("%s on %s: accuracy %.3f (n=%d, parsed %.3f)", tag, a.benchmark, res["accuracy"], res["n"], res["parsed"])
        return
    llm = AudioLLM(a.model, a.adapter, a.max_sec)
    preds = []
    for i in range(0, len(items), a.batch):
        batch = items[i:i + a.batch]
        raw = llm.generate([llm.conversation(it) for it in batch])
        preds += [extract_letter(r, it["options_norm"]) for r, it in zip(raw, batch)]
        if (i // a.batch) % 20 == 0:
            log.info("  %d / %d", i + len(batch), len(items))
    res = save_result(tag, a.benchmark, items, preds, model=a.model, adapter=a.adapter)
    log.info("%s on %s: accuracy %.3f (n=%d, parsed %.3f)%s", tag, a.benchmark, res["accuracy"], res["n"], res["parsed"],
             f"; {llm.n_cut} clips cut to {a.max_sec:g} s" if a.max_sec else "")


if __name__ == "__main__":
    main()
