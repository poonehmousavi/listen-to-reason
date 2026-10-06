"""Reason: a frozen text LLM answers from the cached context. It never receives the audio.

    python -m l2r.reason.reader --model Qwen/Qwen2.5-7B-Instruct --name qwen2.5-7b --template ours
    python -m l2r.reason.reader --reader qwen2.5-7b                    # a reader of configs/readers.yaml

Every reader reads the same contexts (`<work>/contexts/<benchmark>.json`, from l2r.retrieve.context). The answer is
not generated: one forward pass per prompt, and the prediction is the option letter with the highest probability
at the first answer position. The reported score (`ours_poe`) adds a weak prior from the same reader without the
context,

    answer = argmax_letter  log p(letter | context, question) + beta * log p(letter | question),      beta = 0.1

Prompt templates: `ours` (a fixed system line, the context, the question, "Answer with only the letter"),
`native` (the model's default system prompt and an explanation of what the context is), `strict` (native + "deduce
the answer only from the information provided") and `fewshot` (strict + three worked examples). Without
`--template`, the template is selected on 100 development items per benchmark and then locked. The three worked
examples are development items and are never scored.

Output: `<work>/results/<name>.json` (accuracy per benchmark, on all items and on the non-development ones) and
`<work>/results/<name>.<benchmark>.pred.json`; with `--context <variant>` the names are `<name>_<variant>...`.
The development items (`assets/dev_items/`) are the items of 300 clips per benchmark that were used to select
templates and serving rules.
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np
import yaml

from l2r.common import load_config, load_json, resolve, save_json, work

BENCHMARKS = ("mmau", "mmar", "sakura")
TEMPLATES = ("ours", "native", "strict", "fewshot")
LETTERS = "abcdefghijk"
SYSTEM = "You are an expert at reasoning about speech, audio, and the things that produce them."
ASK = "Answer with only the letter of the correct option."
VICUNA_SYSTEM = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
EXPLAIN = ("You cannot hear the audio recording. Instead you are given information that was extracted from it automatically:\n"
           "- a traversal of an audio knowledge tree: lines of the form 'region attribute: value (leaf)', naming what was detected in the "
           "recording, from general to specific (for example 'sound animal: dog (Bark)');\n"
           "- a speech transcript, if the recording contains speech;\n"
           "- for some questions, a timed list of sound events.\n"
           "These are the only evidence about the recording.")
STRICT = "Deduce the answer only from the information provided above; do not assume anything that is not stated there."
ANSWER_PREFIX = " Answer: ("          # --answer-prefix: for readers that do not start their reply with the letter


def dev_split(items: list[dict], benchmark: str, seed: int = 0):
    """-> (100 development items for template selection, 3 worked examples, ids of all development items)."""
    f = resolve(f"assets/dev_items/{benchmark}.json")
    dev_ids = set(load_json(f)) if f.exists() else set()
    pool = [it for it in items if it["id"] in dev_ids]
    idx = np.random.default_rng(seed).permutation(len(pool))
    return [pool[i] for i in idx[:100]], [pool[i] for i in idx[100:103]], {it["id"] for it in pool}


def user_text(it: dict, ctx: str, template: str) -> str:
    if template == "ours":
        return f"{ctx}{it['question']}\n{it['choices_str']}\n\n" + ASK
    info = "Information extracted from the recording:\n" + ctx if ctx.strip() else "No information was extracted from the recording.\n"
    q = f"Question: {it['question']}\n{it['choices_str']}\n\n"
    return EXPLAIN + "\n\n" + info + "\n" + q + (STRICT + " " if template in ("strict", "fewshot") else "") + ASK


def messages(it: dict, ctx: str, template: str, name: str, shots: list[dict]) -> list[dict]:
    system = SYSTEM if template == "ours" else (VICUNA_SYSTEM if "vicuna" in name else "You are a helpful assistant.")
    m = [{"role": "system", "content": system}]
    if template == "fewshot":
        for s in shots:
            m += [{"role": "user", "content": user_text(s, s["ctx"]["ours"], "strict")}, {"role": "assistant", "content": f"({s['gold']})"}]
    m.append({"role": "user", "content": user_text(it, ctx, template)})
    return m


def to_prompt(tok, msgs: list[dict], name: str) -> str:
    if tok.chat_template is None or "vicuna" in name:          # Vicuna v1.5 uses a plain-text format
        s = msgs[0]["content"] + " "
        for x in msgs[1:]:
            s += ("USER: " if x["role"] == "user" else "ASSISTANT: ") + x["content"] + ("</s>" if x["role"] == "assistant" else " ")
        return s + "ASSISTANT:"
    kw = {"reasoning_effort": "low"} if "gpt-oss" in name else {}
    if name.startswith("qwen3-") and "a3b" not in name and "2507" not in name:
        kw["enable_thinking"] = False                            # hybrid-thinking Qwen3 models answer directly, like every other reader
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    except Exception:                                            # noqa: BLE001  (a template without a system role)
        m2 = [dict(x) for x in msgs[1:]]
        m2[0]["content"] = msgs[0]["content"] + "\n\n" + m2[0]["content"]
        return tok.apply_chat_template(m2, tokenize=False, add_generation_prompt=True)


class Reader:
    def __init__(self, model: str, name: str, four_bit=False, generate=False, prefix=""):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch; self.name = name.lower(); self.gen = generate; self.prefix = prefix
        self.tok = AutoTokenizer.from_pretrained(model); self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = {}
        if four_bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        self.m = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype=torch.bfloat16, **kw).eval()
        self.letter_ids = {}                                     # every single-token way of writing a letter: a, A, " a", "(a", ...
        for L in LETTERS:
            ids = set()
            for f in (L, L.upper(), f" {L}", f" {L.upper()}", f"({L}", f"({L.upper()}", f" ({L}", f" ({L.upper()}"):
                t = self.tok.encode(f, add_special_tokens=False)
                if len(t) == 1:
                    ids.add(t[0])
            self.letter_ids[L] = sorted(ids)
        assert all(self.letter_ids.values()), f"no single-token form for some letter: {self.letter_ids}"

    def prompts(self, items, cond: str, template: str, shots) -> list[str]:
        return [to_prompt(self.tok, messages(it, it["ctx"].get(cond, ""), template, self.name, shots), self.name) + self.prefix for it in items]

    def score(self, prompts: list[str], bs=8) -> list[dict]:
        """Log-probability of every option letter at the first answer position."""
        torch = self.torch; out = []
        for i in range(0, len(prompts), bs):
            enc = self.tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(self.m.device)
            with torch.no_grad():
                try:
                    lg = self.m(**enc, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
                except TypeError:
                    lg = self.m(**enc).logits[:, -1].float().log_softmax(-1)
            for r in range(lg.shape[0]):
                out.append({L: float(max(lg[r, t] for t in ids)) for L, ids in self.letter_ids.items()})
        return out

    def generate(self, prompts: list[str], bs=4, max_new=512) -> list[dict]:
        """For readers that reason before answering: generate, then parse the letter."""
        torch = self.torch; out = []
        for i in range(0, len(prompts), bs):
            enc = self.tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(self.m.device)
            with torch.no_grad():
                g = self.m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.pad_token_id)
            for t in self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
                t = t.split("final")[-1]                           # the answer follows the final-channel marker
                m = re.findall(r"\(([a-kA-K])\)|\b([a-kA-K])\b", t)
                L = (m[-1][0] or m[-1][1]).lower() if m else "a"
                out.append({x: (0.0 if x == L else -10.0) for x in LETTERS})
        return out


def answer(reader: Reader, items: list[dict], template: str, shots: list[dict], bs: int, beta: float) -> list[dict]:
    lp = {}
    for cond in ("ours", "blind"):
        prompts = reader.prompts(items, cond, template, shots)
        lp[cond] = reader.generate(prompts) if reader.gen else reader.score(prompts, bs)
    out = []
    for it, a, b in zip(items, lp["ours"], lp["blind"]):
        on = it.get("options_norm")
        opts = sorted(on) if isinstance(on, dict) and on else list("abcd")
        out.append({"id": it["id"], "gold": it["gold"], "group": it.get("group"),
                    "ours": max(opts, key=lambda L: a[L]), "ours_poe": max(opts, key=lambda L: a[L] + beta * b[L]), "blind": max(opts, key=lambda L: b[L])})
    return out


def accuracy(res: list[dict], key: str, keep=None):
    r = [x for x in res if keep is None or x["id"] in keep]
    return round(float(np.mean([x[key] == x["gold"] for x in r])), 4) if r else None


def run(model: str, name: str, context: str = "", benchmarks=BENCHMARKS, template: str = "", four_bit=False, generate=False, answer_prefix=False,
        bs: int = 8, tag: str = "") -> dict:
    cfg = load_config(); beta = cfg["reader"]["beta"]; t0 = time.time()
    reader = Reader(model, name, four_bit, generate, ANSWER_PREFIX if answer_prefix else "")
    data = {b: load_json(work("contexts", f"{b}{'_' + context if context else ''}.json")) for b in benchmarks}
    split = {b: dev_split(data[b], b) for b in benchmarks}
    out_path = work("results", f"{name}{'_' + context if context else ''}{tag}.json")
    rep = {"model": model, "name": name, "context": context, "dev": {}, "full": {}}
    if not template:                                             # select the template on the development items, then lock it
        for tpl in TEMPLATES:
            rep["dev"][tpl] = {}
            for b in benchmarks:
                dev, shots, _ = split[b]
                r = answer(reader, dev, tpl, shots, bs, beta)
                rep["dev"][tpl][b] = {k: accuracy(r, k) for k in ("ours", "ours_poe", "blind")}
            print(name, "dev", tpl, rep["dev"][tpl], flush=True)
        template = max(rep["dev"], key=lambda t: np.mean([rep["dev"][t][b]["ours_poe"] for b in benchmarks]))
    rep["template"] = template
    print(name, "template:", template, flush=True)
    for b in benchmarks:
        _, shots, dev_ids = split[b]
        shot_ids = {s["id"] for s in shots}
        r = answer(reader, [it for it in data[b] if it["id"] not in shot_ids], template, shots, bs, beta)
        test = {x["id"] for x in r if x["id"] not in dev_ids}
        rep["full"][b] = {k: {"ALL": accuracy(r, k), "TEST": accuracy(r, k, test)} for k in ("ours", "ours_poe", "blind")}
        rep["full"][b]["n"] = len(r); rep["full"][b]["n_test"] = len(test)
        rep["full"][b]["groups"] = {g: accuracy([x for x in r if x["group"] == g], "ours_poe") for g in sorted({x["group"] for x in r if x["group"]})}
        save_json([{k: x[k] for k in ("id", "ours_poe")} for x in r], out_path.with_suffix(f".{b}.pred.json"), indent=None)
        print(name, b, rep["full"][b], flush=True)
        save_json(rep, out_path, indent=1)
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    save_json(rep, out_path, indent=1)
    print("done", out_path)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reader", default="", help="a reader of configs/readers.yaml (model, template and flags as in the paper)")
    ap.add_argument("--model", default="", help="Hugging Face id or local path"); ap.add_argument("--name", default="")
    ap.add_argument("--context", default="", help="context variant: reads <work>/contexts/<benchmark>_<variant>.json")
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    ap.add_argument("--template", default="", choices=("",) + TEMPLATES, help="skip the selection on development items")
    ap.add_argument("--four-bit", action="store_true", help="4-bit NF4 quantization (the 70B / 72B readers)")
    ap.add_argument("--generate", action="store_true", help="generate and parse the answer (readers that reason first)")
    ap.add_argument("--answer-prefix", action="store_true", help='start the reply with " Answer: (" (readers that do not answer with the letter)')
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--tag", default="", help="suffix of the result file")
    a = ap.parse_args()
    spec = {"model": a.model, "name": a.name, "template": a.template, "four_bit": a.four_bit, "generate": a.generate, "answer_prefix": a.answer_prefix}
    if a.reader:
        known = yaml.safe_load(resolve("configs/readers.yaml").read_text())["readers"]
        assert a.reader in known, f"unknown reader {a.reader!r} (known: {sorted(known)})"
        spec = {**spec, "name": a.reader, **known[a.reader], **({"template": a.template} if a.template else {})}
    assert spec["model"] and spec["name"], "--reader, or --model and --name"
    run(context=a.context, benchmarks=tuple(a.benchmarks.split(",")), bs=a.bs, tag=a.tag, **spec)


if __name__ == "__main__":
    main()
