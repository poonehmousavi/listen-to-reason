"""QLoRA baseline: fine-tune Qwen2.5-Omni-7B on the same k labelled clips per species (Figure 3, Table "Few-shot
domain adaptation").

    python -m baselines.qlora --domain birds --k 5 --epochs 8          # -> <work>/qlora/birds_k5/
    python -m baselines.audio_llm --benchmark birds --adapter <work>/qlora/birds_k5

Epochs: 8, 6, 4 and 2 for k = 5, 10, 20 and 50 (3 at k = 50 on marine mammals). The training items have the
format of the evaluation items (the clip, four options with taxonomically close distractors, "answer with only
the letter") and are built from the training pool, with the same containment rule as the domain head: a clip
within 0.95 CLAP cosine similarity of an evaluation clip is dropped. The thinker is loaded in 4-bit NF4 and LoRA
adapters (rank 16, alpha 32, dropout 0.05) are trained on the attention and MLP projections of the language
model; the audio encoder stays frozen. AdamW, learning rate 2e-4, batch 4 x 4 accumulation steps, cosine
schedule with 5% warm-up.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np

from baselines.audio_llm import ASK, SYSTEM, load_thinker
from l2r.adapt import data as D
from l2r.adapt.domain import CONTAIN_TAU
from l2r.common import audio, check, data, load_config, save_json, set_seed, setup_run, work

MODEL = "Qwen/Qwen2.5-Omni-7B"
LORA_TARGET = r"^(.*\.)?model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"
CUT_DIR = "BirdSet/ft_pow"                      # 10 s windows of the bird training clips, under the data folder


def train_items(domain: str, k: int, cfg, log, seed: int = 1) -> list[dict]:
    """Up to k training items per species, in the pool's selection order, minus clips too close to the evaluation set."""
    dom = D.load_domain(domain); pool = dom["pool"]
    rng = random.Random(seed)
    items = []
    if domain == "birds":
        genus = {lab: v["genus"] for lab, v in pool.items()}; group = {lab: v["species_group"] for lab, v in pool.items()}
        names = {lab: v["name"] for lab, v in pool.items()}
        for lab in sorted(pool):
            clips = sorted(pool[lab]["clips"], key=lambda c: ((c.get("quality") or "Z"), c["path"]))[:k]
            tiers = [[names[c] for c in pool if genus[c] == genus[lab]], [names[c] for c in pool if group[c] == group[lab]], list(names.values())]
            for c in clips:
                rel = f"{CUT_DIR}/{lab}/{Path(c['path']).stem}.wav"
                if not data(rel).exists() and D.cut_window(audio(c["path"]), data(rel), c.get("events")) is None:
                    continue
                choices, gold = D.mcq(rng, names[lab], tiers)
                items.append(D.item(f"ft/{lab}/{Path(c['path']).stem}", rel, group[lab], D.BIRD_Q, choices, gold, sub_category=names[lab]))
    else:
        group_of = {l: g for g, ls in D.MARINE_GROUP.items() for l in ls}
        for lab in sorted(pool):
            tiers = [[D.marine_name(l) for l in D.MARINE_GROUP[group_of[lab]]], [D.marine_name(l) for l in group_of]]
            for c in sorted(pool[lab]["clips"], key=lambda c: c["path"])[:k]:
                choices, gold = D.mcq(rng, D.marine_name(lab), tiers)
                items.append(D.item(f"ft/{lab}/{Path(c['path']).stem}", c["path"], group_of[lab], D.MARINE_Q, choices, gold, sub_category=D.marine_name(lab)))
    from l2r.encoders import build_encoder                      # the containment rule of the domain head
    clap = build_encoder("clap", cfg)
    unit = lambda e: np.asarray(e, dtype=np.float32) / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
    ev = unit(clap.embed([str(audio(it["audio"])) for it in dom["items"]]))
    tr = unit(clap.embed([str(audio(it["audio"])) for it in items]))
    keep = [it for it, m in zip(items, (tr @ ev.T).max(axis=1)) if m < CONTAIN_TAU]
    log.info("containment: %d / %d training clips within %.2f of an evaluation clip dropped", len(items) - len(keep), len(items), CONTAIN_TAU)
    rng.shuffle(keep)
    return [{**it, "audio": str(audio(it["audio"]))} for it in keep]


def conversation(it: dict) -> list[dict]:
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "audio", "audio": it["audio"]}, {"type": "text", "text": f"{it['question']}\n{it['choices_str']}\n\n{ASK}"}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"({it['gold']})"}]}]


def train(domain: str, k: int, epochs: int, out: Path, cfg, log, model_id=MODEL, lr=2e-4, bs=4, accum=4, rank=16, alpha=32, dropout=0.05, seed=1, max_steps=0):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen2_5OmniProcessor
    set_seed(seed)
    items = train_items(domain, k, cfg, log, seed)
    proc = Qwen2_5OmniProcessor.from_pretrained(model_id)
    tok = proc.tokenizer; tok.padding_side = "right"
    model = prepare_model_for_kbit_training(load_thinker(model_id, quant=True), use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET))
    trainable = [p for p in model.parameters() if p.requires_grad]
    log.info("LoRA rank %d: %.1fM trainable of %.0fM parameters; %d training items", rank, sum(p.numel() for p in trainable) / 1e6,
             sum(p.numel() for p in model.parameters()) / 1e6, len(items))
    check(all("model.layers." in n and "audio_tower" not in n for n, p in model.named_parameters() if p.requires_grad), "LoRA only on the language model; audio encoder frozen", log)
    header = tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]

    def collate(batch):
        convs = [conversation(it) for it in batch]
        texts = [proc.apply_chat_template(c, add_generation_prompt=False, tokenize=False) for c in convs]
        audios = [a for c in convs for a in (process_mm_info(c, use_audio_in_video=False)[0] or [])]
        enc = proc(text=texts, audio=audios, return_tensors="pt", padding=True)
        ids = enc["input_ids"]; labels = ids.clone()
        labels[enc["attention_mask"] == 0] = -100
        for row in range(ids.shape[0]):                          # the loss covers the answer only: mask up to the last assistant header
            seq = ids[row].tolist(); cut = 0
            for j in range(len(seq) - len(header), -1, -1):
                if seq[j:j + len(header)] == header:
                    cut = j + len(header)
                    break
            labels[row, :cut] = -100
        enc["labels"] = labels
        return enc

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    per_epoch = math.ceil(len(items) / (bs * accum))
    total = max_steps or per_epoch * epochs
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    out.mkdir(parents=True, exist_ok=True)
    model.train(); step = 0; t0 = time.time(); losses = []
    rng = random.Random(seed)
    for ep in range(epochs):
        order = list(range(len(items))); rng.shuffle(order)
        for b in range(0, len(order), bs):
            enc = collate([items[i] for i in order[b:b + bs]])
            enc = {q: (v.to(model.device) if hasattr(v, "to") else v) for q, v in enc.items()}
            enc = {q: (v.to(torch.bfloat16) if hasattr(v, "dtype") and v.dtype.is_floating_point else v) for q, v in enc.items()}
            loss = model(**enc).loss / accum
            loss.backward(); losses.append(loss.item() * accum)
            if (b // bs + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 10 == 0 or step == 1:
                    log.info("epoch %d step %d/%d loss %.4f (%.0f s)", ep, step, total, float(np.mean(losses[-10 * accum:])), time.time() - t0)
                if max_steps and step >= max_steps:
                    break
        model.save_pretrained(out)
        if max_steps and step >= max_steps:
            break
    save_json({"domain": domain, "k": k, "items": len(items), "epochs": epochs, "lr": lr, "rank": rank, "alpha": alpha, "model": model_id,
               "steps": step, "final_loss": float(np.mean(losses[-50:])), "seconds": time.time() - t0}, out / "meta.json")
    log.info("adapter -> %s", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", choices=D.DOMAINS, required=True); ap.add_argument("--k", type=int, default=5, help="training clips per species")
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--model", default=MODEL); ap.add_argument("--out", default="")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke test)")
    a = ap.parse_args()
    cfg = load_config(); _, log = setup_run(f"qlora_{a.domain}_k{a.k}", cfg)
    out = Path(a.out) if a.out else work("qlora", f"{a.domain}_k{a.k}", "x").parent
    train(a.domain, a.k, a.epochs, out, cfg, log, model_id=a.model, max_steps=a.max_steps)


if __name__ == "__main__":
    main()
