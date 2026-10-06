"""The identity attributes of environmental sound, from the AudioSet ontology (tree construction only).

Environmental sound has many more identities than speech or music, so its identity attributes (animal, vehicle, machine,
tool, household, human non-speech, nature, alarm and signal, impact, plus environment, sound type and activity) take
their classes and leaves from AudioSet: a class is an AudioSet class or a small union of them
(`configs/sound_classes.yaml`), and its leaves are the finer AudioSet classes below it.

  plan    AudioSet classes are placed under a class by ontology ancestry (the deepest class root wins).
  llm     a text LLM places the rest (AudioSet classes the ontology does not place, and the VGGSound classes that label
          the activity attribute); every answer is validated against the lists of the schema.
  verify  the LLM checks each of its own placements (yes / no); a "no" is left out of the tree.
  build   the tree -> <checkpoints>/tree/sound_tree.json: {attribute: {class: [{leaf, src, how, same_as, ...}]}}.

    python -m l2r.tree.sound plan | llm | verify | build

Needs the AudioSet ontology (<data>/AudioSet/ontology.json), the list of AudioSet classes that have training clips
(`names` in <work>/audioset/items.json, written when the sound heads' training set is built) and the class list of
the VGGSound classifier (its `label_encoder` file; `--vggsound-labels`).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

import yaml

from l2r import events as EV
from l2r.common import ckpt, load_config, resolve as repo_path, setup_run, work

CLASSES = "configs/sound_classes.yaml"
# AudioSet lists these classes under two equally deep class roots (Growling under Cat and Dog): the class they are placed under
TIE_BREAK = {"Growling": "cat", "Howl": "wild canid", "Bleat": "sheep"}
VGGSOUND_LABELS = "pretrained/vggsound_ecapa/label_encoder.ckpt"       # under the checkpoint folder


def step_file(name: str):
    """Intermediate files of the steps live in <checkpoints>/tree/sound/."""
    return ckpt("tree", "sound", name)


SPLIT = {"animal": "sound.animal", "tool": "sound.tool", "household object": "sound.household", "human non-speech": "sound.human_nonspeech",
         "natural element": "sound.nature", "alarm or signal": "sound.alarm_signal", "impact or texture": "sound.impact"}
VEHICLE = {"car", "truck", "bus", "emergency vehicle", "motorcycle", "traffic noise", "rail transport", "aircraft", "boat", "non-motorized vehicle"}
ACTIVITY = ["travelling or driving", "caring for animals", "office or desk work", "cleaning or housework", "personal care", "construction or repair",
            "sport or exercise", "playing or gaming", "celebration or ceremony", "cooking or eating", "shopping or trading", "military or weapons",
            "craft or workshop", "water activity"]
ENV = ["inside, small room", "inside, large room or hall", "inside, public space", "outside, urban or manmade", "outside, rural or natural",
       "reverberant", "echo", "noisy background", "silence"]
SOUND_TYPE = ["whoosh or swish", "thump or thud", "clicking", "buzzing", "humming", "beeping", "rumbling", "hissing", "squeaking", "rattling",
              "tapping", "scraping", "ringing tone", "static or noise", "sound effect"]
INSTR = ["acoustic guitar", "bass guitar", "drum kit", "piano", "electric guitar", "synthesizer", "harp", "saxophone", "trumpet", "orchestral strings",
         "violin", "trombone", "banjo", "harmonica", "other plucked or folk instrument", "hand drums", "accordion", "bells", "flute", "cymbals", "cello",
         "organ", "clarinet", "other brass", "other woodwind", "mallet percussion", "other percussion"]


def ontology():
    O = EV.ontology(); by = {o["id"]: o for o in O}; nm = {o["name"]: o["id"] for o in O}
    par = defaultdict(set)
    for o in O:
        for c in o["child_ids"]:
            par[c].add(o["id"])

    def anc(m):
        out, st = set(), [m]
        while st:
            x = st.pop()
            for p in par[x]:
                if p not in out:
                    out.add(p); st.append(p)
        return out
    return by, nm, par, anc


def schema():
    """-> ({attribute: [classes]}, {AudioSet id of a class root: (attribute, class)})."""
    S = yaml.safe_load(repo_path(CLASSES).read_text()); sch = defaultdict(list); roots = {}
    by, nm, par, anc = ontology()
    for kind, cl in S["kinds"].items():
        for c in cl:
            attr = ("sound.vehicle" if c["name"] in VEHICLE else "sound.machine") if kind == "machine or vehicle" else SPLIT[kind]
            sch[attr].append(c["name"])
            for r in c["audioset"]:
                if r in nm:
                    roots[nm[r]] = (attr, c["name"])
    sch["sound.activity"] = ACTIVITY; sch["sound.environment"] = ENV; sch["sound.sound_type"] = SOUND_TYPE; sch["music.instrument"] = INSTR
    return dict(sch), roots


def vggsound_labels():
    """The VGGSound classifier's label file (fetched from the hub when the classifier has not been downloaded yet)."""
    p = ckpt(VGGSOUND_LABELS)
    if not p.exists():
        from huggingface_hub import hf_hub_download
        spec = load_config()["encoders"].get("vggsound") or {"source": "Ubenwa/sb-ecapa-vggsound", "dir": "pretrained/vggsound_ecapa"}
        hf_hub_download(spec["source"], p.name, local_dir=str(ckpt(spec["dir"])))
    return p


def vggsound_classes(path) -> list[str]:
    """Class names of the VGGSound classifier, in output order, from its label-encoder file (lines `'name' => index`)."""
    out = {}
    for line in open(path):
        m = re.match(r"'(.*)' => (\d+)\s*$", line)
        if m and m.group(1) != "starting_index":
            out[int(m.group(2))] = m.group(1)
    return [out[i] for i in sorted(out)]


def plan(a, cfg, log):
    by, nm, par, anc = ontology(); sch, roots = schema()
    names = json.load(open(work("audioset", "items.json")))["names"]; placed, rest = {}, []
    for n in names:
        m = nm.get(n); cands = [(roots[x], len(anc(x))) for x in ({m} | anc(m)) if m and x in roots] if m else []
        if cands:
            cands.sort(key=lambda t: (-t[1], t[0]))                                 # the deepest root first, then alphabetical
            tied = [c[0] for c in cands if c[1] == cands[0][1]]
            attr, val = next((c for c in tied if c[1] == TIE_BREAK.get(n)), tied[0])
            placed[n] = {"attr": attr, "value": val, "leaf": n, "how": "ontology"}
        else:
            rest.append(n)
    json.dump({"schema": sch, "audioset_placed": placed, "audioset_rest": rest}, open(step_file("plan.json"), "w"), indent=1)
    log.info("schema: %s", {k: len(v) for k, v in sch.items()}); log.info("AudioSet classes placed by the ontology %d, left for the LLM %d", len(placed), len(rest))


PROMPT = """You are organising a taxonomy of sounds for an audio knowledge graph. The taxonomy is FIXED below: attributes, each with its allowed VALUES.

{schema}

Special destinations:
- "speech": the class is human speech or talking (man / woman / child speaking, whispering, babbling, radio chatter). It is handled elsewhere.
- "music": the class is music as such (a genre, singing, a song) and not a playable instrument.
- "drop": too generic to be useful (e.g. "Sound", "Vehicle" with no specific type).

For EACH input class below, choose exactly one destination:
  {{"name": <input class>, "attr": <attribute or speech|music|drop>, "value": <one of that attribute's VALUES, exactly as written>, "new_value": <only if NO listed value fits: a short new value name for that attribute, else null>, "same_as": <if the input class is the same sound as another input or an AudioSet class listed as KNOWN LEAVES, that exact name, else null>}}
Rules: a sound PRODUCED BY an animal/vehicle/tool/object goes under that source (dog barking -> sound.animal/dog); an activity with no single source (playing tennis, skiing, sailing) goes to sound.activity; "playing <instrument>" goes to music.instrument; generic acoustic types with no source (whoosh, thump, buzz) go to sound.sound_type; room / place acoustics go to sound.environment. Never put two different animals or two different instruments under the same leaf.

KNOWN LEAVES already placed (for same_as): {known}

Input classes:
{items}

Return ONLY a JSON list, one object per input class, in order."""


def _llm(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_id); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)).eval()

    def ask(prompt: str, max_new: int) -> str:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
    return ask


def llm(a, cfg, log):
    P = json.load(open(step_file("plan.json"))); sch = P["schema"]
    vgg = vggsound_classes(repo_path(a.vggsound_labels) if a.vggsound_labels else vggsound_labels())
    todo = [("audioset", n) for n in P["audioset_rest"]] + [("vggsound", n) for n in vgg]
    part = step_file("llm_part.json"); done = json.load(open(part)) if part.exists() else {}
    todo = [t for t in todo if f"{t[0]}|{t[1]}" not in done]
    ask = _llm(a.model or cfg["models"]["tree_llm"])
    sch_txt = "\n".join(f"- {k}: {', '.join(v)}" for k, v in sch.items())
    known = ", ".join(sorted(P["audioset_placed"]))
    for i in range(0, len(todo), a.bs):
        batch = todo[i:i + a.bs]
        txt = ask(PROMPT.format(schema=sch_txt, known=known, items="\n".join(f"- {n}" for _, n in batch)), 3000); m = re.search(r"\[.*\]", txt, re.S)
        try:
            ans = json.loads(m.group(0))
        except Exception:                                         # noqa: BLE001
            ans = []
        byname = {str(x.get("name", "")).strip().lower(): x for x in ans if isinstance(x, dict)}
        for src, n in batch:
            done[f"{src}|{n}"] = byname.get(n.lower(), {"name": n, "attr": None, "unparsed": True})
        json.dump(done, open(part, "w"), indent=1); log.info("%d / %d placed", i + len(batch), len(todo))
    json.dump(done, open(step_file("llm.json"), "w"), indent=1)


STOP = {"the", "a", "an", "of", "and", "or", "with", "on", "in", "etc", "sound", "sounds", "noise", "people", "playing"}
def stems(x):
    return {w[:4] for w in re.findall(r"[a-z]+", x.lower()) if len(w) > 3 and w not in STOP}


def stem2(w):
    """suffix stemmer for the in-class merge (exact match only): hissing -> hiss, purring -> purr, rattle / rattling -> rattl, cries -> cry."""
    w = w.lower()
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3: w = w[: -len(suf)]; break
    if len(w) > 3 and w[-1] == w[-2]: w = w[:-1]
    if len(w) > 3 and w.endswith("e"): w = w[:-1]
    if w.endswith("i"): w = w[:-1] + "y"
    return w


def words2(x):
    return {stem2(w) for w in re.findall(r"[a-z]+", x.lower()) if len(w) > 2 and w not in STOP}


def resolve():
    """LLM answers -> candidate placements, under three checks:
    (1) an AudioSet class that is an ontology ancestor of a class root (Animal, Vehicle, Tools ...) is dropped: too generic;
    (2) a merge (`same_as`) is kept only when the target's ontology placement equals the LLM's, or the two names share a
        content word (then an ontology placement of the target wins); otherwise the class stays its own leaf;
    (3) an explicit new class wins over a listed class given with it; an attribute outside the schema is unresolved."""
    P = json.load(open(step_file("plan.json"))); sch = P["schema"]; L = json.load(open(step_file("llm.json"))); placed = P["audioset_placed"]
    by, nm, par, anc = ontology(); _, roots = schema()
    generic = set()
    for r in roots:
        generic |= {by[x]["name"] for x in anc(r)}
    cand, routed, unresolved, merges, notes = [], Counter(), [], [], Counter()
    for key, r in L.items():
        src, n = key.split("|", 1); at = r.get("attr"); val = r.get("value"); nv = r.get("new_value"); same = r.get("same_as")
        if src == "audioset" and n in generic:
            routed["drop (generic parent)"] += 1; notes["generic parent dropped"] += 1; continue
        if at in ("speech", "music", "drop"): routed[at] += 1; continue
        if at not in sch: unresolved.append({"name": n, "src": src, "why": f"attribute {at!r} not in schema"}); continue
        if nv: val = nv.strip().lower()
        if not val: unresolved.append({"name": n, "src": src, "why": "no value"}); continue
        if same:
            T = placed.get(same); overlap = bool(stems(n) & stems(same))
            if T and (T["attr"], T["value"]) == (at, val): merges.append((n, same)); notes["merge, placements agree"] += 1
            elif overlap:
                if T: at, val = T["attr"], T["value"]
                merges.append((n, same)); notes["merge, shared word (ontology placement)" if T else "merge, shared word"] += 1
            else: notes["merge refused"] += 1; same = None
        cand.append({"name": n, "src": src, "attr": at, "value": val, "same_as": same, "new_value": val not in sch[at]})
    return P, cand, routed, unresolved, merges, notes


VPROMPT = """Check each placement of a sound class in a taxonomy. Answer "yes" only if the sound class really is a sound OF / BELONGING TO the category
(e.g. "dog barking" under animal / dog: yes; "computer mouse clicking" under animal / rodent: no; "gibbon howling" under animal / roaring big cat: no).
If the class could have two meanings, judge the everyday audio meaning.
{items}
Answer with one line per item and nothing else, exactly in the form
<number>: yes
or
<number>: no"""


def verify(a, cfg, log):
    _, cand, *_ = resolve()
    ask = _llm(a.model or cfg["models"]["tree_llm"]); V = {}
    for i in range(0, len(cand), 30):
        b = cand[i:i + 30]; items = "\n".join(f"{j + 1}. \"{c['name']}\" under {c['attr'].split('.')[-1]} / {c['value']}" for j, c in enumerate(b))
        txt = ask(VPROMPT.format(items=items), 400)
        got = {int(k): v for k, v in re.findall(r"(?m)^\D{0,3}(\d+)\s*[:.)-]\s*\**\s*(yes|no)\b", txt.lower())}   # answers keyed by item number
        for j, c in enumerate(b):
            V[f"{c['src']}|{c['name']}"] = got.get(j + 1, "unparsed")
        log.info("verified %d / %d", i + len(b), len(cand))
    json.dump(V, open(step_file("verify.json"), "w"), indent=1)
    log.info("verify: %s", Counter(V.values()))


def build(a, cfg, log):
    P, cand, routed, unresolved, merges, notes = resolve()
    V = json.load(open(step_file("verify.json"))) if step_file("verify.json").exists() else {}
    tree = defaultdict(lambda: defaultdict(list)); rows = []; review = []
    for n, r in P["audioset_placed"].items():
        tree[r["attr"]][r["value"]].append({"leaf": n, "src": "audioset", "how": "ontology"})
    for c in cand:
        v = V.get(f"{c['src']}|{c['name']}", "not checked")
        if v == "no": review.append({**c, "verify": v}); continue
        tree[c["attr"]][c["value"]].append({"leaf": c["name"], "src": c["src"], "how": "llm", "same_as": c["same_as"], "new_value": c["new_value"], "verify": v})
        rows.append(c)
    # second merge pass, deterministic, inside ONE class: leaves whose content words are identical once the class's own name and filler words are
    # removed name the same sound ("cat hissing" / "Hiss" under cat -> {hiss}); an empty residue never absorbs anything ("Dog" under dog)
    auto = []
    for at, vd in tree.items():
        for v, leaves in vd.items():
            grp = defaultdict(list)
            for l in leaves:
                key = frozenset(words2(l["leaf"]) - words2(v))
                if key: grp[key].append(l)
            for key, ls in grp.items():
                if len(ls) < 2: continue
                canon = next((l for l in ls if l["src"] == "audioset"), ls[0])
                for l in ls:
                    if l is not canon and not l.get("same_as"):
                        l["same_as"] = canon["leaf"]; auto.append((l["leaf"], canon["leaf"]))
    out = {at: {v: lv for v, lv in vd.items()} for at, vd in tree.items()}
    json.dump(out, open(ckpt("tree", "sound_tree.json"), "w"), indent=1)
    newv = defaultdict(set)
    for c in rows:
        if c["new_value"]: newv[c["attr"]].add(c["value"])
    rev = {"routed": dict(routed), "notes": dict(notes), "new_values": {k: sorted(v) for k, v in newv.items()}, "unresolved": unresolved,
           "needs_review": review, "merges": merges + auto, "auto_merges": auto, "counts": {at: {"values": len(vd), "leaves": sum(len(x) for x in vd.values())} for at, vd in out.items()}}
    json.dump(rev, open(step_file("review.json"), "w"), indent=1)
    log.info("sound tree: %s", rev["counts"]); log.info("routed %s | notes %s | new values %s | merges %d | needs review %d | unresolved %d",
             dict(routed), dict(notes), {k: len(v) for k, v in newv.items()}, len(merges) + len(auto), len(review), len(unresolved))
    log.info("auto merges inside a class: %d, e.g. %s", len(auto), auto[:12])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("stage", choices=["plan", "llm", "verify", "build"])
    ap.add_argument("--model", default="", help="text LLM (default: models.tree_llm of the config)"); ap.add_argument("--bs", type=int, default=20)
    ap.add_argument("--vggsound-labels", default="", help="label-encoder file of the VGGSound classifier")
    a = ap.parse_args(); cfg = load_config(); _, log = setup_run(f"sound_tree_{a.stage}", cfg)
    {"plan": plan, "llm": llm, "verify": verify, "build": build}[a.stage](a, cfg, log)
