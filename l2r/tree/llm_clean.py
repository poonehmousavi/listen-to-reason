"""Clean the leaf vocabulary of the tree with an external text LLM (tree construction only).

Under each (attribute, class) the LLM reads the admitted and the refused leaves and proposes
  - groups of names that denote the same category (siren / siren wailing; thunderclap / clap of thunder),
  - generic -> specific links (car > car passing), which are linked, never merged,
  - leaves that belong under another class or are not a leaf of the attribute at all (kept as proposals only).
The LLM never hears audio and its proposals are not trusted. A group is applied only if
  1. lexical check: every member is an input name, the group name is a member, and no pair conflicts in number,
     negation or polarity (`l2r.tree.guards`);
  2. identity check: on attributes that name what something is (sound event, instrument, genre, language, emotion),
     two names merge only if they share a source word (`goat bleating` and `sheep bleating` stay apart);
  3. acoustic check: in the embedding space of an encoder suited to the attribute, the chunks of the group are
     closer to each other than to the other leaves of the class.
Then deterministic folds merge a leaf that only adds a filler word into its sibling (`alarm sound` -> `alarm`), and
the attributes in PRUNE_ATTRS are removed from the tree.

    python -m l2r.tree.publish --raw             # the tree before cleaning -> <checkpoints>/tree_raw/
    python -m l2r.tree.llm_clean propose         # GPU: the LLM proposals            -> tree_raw/llm_groups_raw.json (resumable)
    python -m l2r.tree.llm_clean check           # GPU: the acoustic check           -> tree_raw/llm_groups_audio.json
    python -m l2r.tree.llm_clean apply           # CPU: the checks -> the apply table -> tree_raw/llm_groups.json (+ a report)
    python -m l2r.tree.publish                   # the final tree, with the table applied -> <checkpoints>/tree/
"""
from __future__ import annotations

import argparse
import collections
import json
import re

import numpy as np

from l2r.common import audio, check, ckpt, load_config, read_jsonl, setup_run, work
from l2r.dataset import index as I, schema as S, segment as G
from l2r.tree import build as T
from l2r.tree.guards import may_merge, merge_features

PROMPT_VERSION = "g1"
RAW_TREE = "tree_raw"                               # folder (under the checkpoint folder) of the tree before cleaning


def raw_file(name: str):
    return ckpt(RAW_TREE, name)


MAX_LEAVES = 140                                    # per prompt; a value with more is split into alphabetical halves

SYS_LEAF = """You curate the leaf vocabulary of an audio knowledge graph. Every leaf is a short name an audio annotator wrote for a sound heard in a 3-10 s recording. Leaves under one value are supposed to be DISTINCT sound categories that a listener could tell apart. Return JSON only."""

USER_LEAF = """Region: {region}. Attribute `{attr}` = {hint}
Value (closed parent): "{value}". Other values of this attribute: {others}.

Leaves under this value, as "name (clips, status)". Status: admitted = a node today; refused = kept out only because it was seen on too few clips (it may still be a good synonym of an admitted leaf):
{leaves}

Tasks:
1. groups: group names that denote the SAME sound category -- the same source doing the same action, or two names of one thing (spelling, plural, verb form, word order, an AudioSet-style name like "fixed-wing aircraft, airplane" vs "airplane", "engine running" / "motor running"). Name each group by its clearest plain everyday member name; the name MUST be one of the members. A name that is in no group stays its own leaf: do not list singletons.
   Never merge: a generic with its specifics ("car" vs "car passing", "dog" vs "dog barking" -> use `generic`), different sources ("dog barking" vs "cat meowing"), different actions of one source ("dog barking" vs "dog howling"), opposite or negated descriptions, names with different numbers.
2. generic: a name that is a broader category of other listed names -> link it to those specifics (the generic keeps its own node; the specifics stay separate nodes under it).
3. misplaced: leaves that are not a sound of THIS value: give "move_to" = one of the other values listed above, or "drop" if the name is not a {attr_short} leaf at all (a place, an activity, a judgement, a non-sound).
Be conservative: when unsure, leave the leaf alone.

Return exactly:
{{"groups": [{{"name": "...", "members": ["...", "..."]}}],
 "generic": [{{"generic": "...", "specifics": ["...", "..."]}}],
 "misplaced": [{{"leaf": "...", "move_to": "<value or drop>", "reason": "..."}}]}}"""

def load_tree():
    d = ckpt(RAW_TREE, "tree.json").parent
    return tuple(json.loads((d / f).read_text()) for f in ("tree.json", "refused.json", "members.json", "MANIFEST.json"))


def hint_of(schema, attr):
    r, a = attr.split("."); at = schema["regions"][r]["attributes"][a]
    return at.get("leaf_hint") or at.get("hint") or at.get("na_hint") or "(no definition in the schema)"


def pairs(tree, refused):
    """(attr, value) -> [(leaf, clips, status, why)] over admitted + refused leaves."""
    out = collections.defaultdict(list)
    for r, A in tree.items():
        for a, V in A.items():
            for v, q in V.items():
                for l, ql in q["leaves"].items():
                    out[(f"{r}.{a}", v)].append((l, ql["clips"], "admitted", ""))
    for x in refused:
        if x["why"].startswith(("seen on one clip", "seen in one source")):
            out[(x["attr"], x["value"])].append((x["leaf"], x["clips"], "refused", x["why"]))
    return out


def _json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    for cand in (m.group(0), m.group(0).rsplit("}", 1)[0] + "}"):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def _llm(model_id, log, four_bit=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if four_bit:                                                    # one 80 GB card: NF4 weights, bf16 compute (the `--reader-4bit` recipe)
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    log.info("LLM %s on %s", model_id, {str(p.device) for p in model.parameters()})

    def ask(prompts, max_new):
        texts = [tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": u}], tokenize=False, add_generation_prompt=True) for s, u in prompts]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        return [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True) for g in gen]
    return ask


def stage_propose(a, log):
    schema = S.load_schema(); tree, refused, _, man = load_tree(); P = pairs(tree, refused); p = raw_file("llm_groups_raw.json")
    raw = json.loads(p.read_text()) if p.exists() and not a.fresh else {"prompt_version": PROMPT_VERSION, "built": man["built"], "leaf": {}}
    check(raw.get("prompt_version") == PROMPT_VERSION and raw.get("built") == man["built"], "the cached proposals match the prompt version and the tree (else use --fresh)", log)
    ask = _llm(a.model or load_config()["models"]["tree_llm"], log, a.four_bit)
    jobs = []                                                       # one prompt per (attribute, class[, half])
    for (attr, value), L in sorted(P.items()):
        if len(L) < 2:
            continue
        r = attr.split(".")[0]; others = ", ".join(f'"{v}"' for v in tree[r][attr.split(".")[1]] if v != value)
        L = sorted(L, key=lambda t: (t[2] != "admitted", -t[1], t[0]))
        halves = [L] if len(L) <= MAX_LEAVES else [sorted(L)[: len(L) // 2], sorted(L)[len(L) // 2:]]
        for hi, H in enumerate(halves):
            key = f"{attr}={value}" + (f"#{hi}" if len(halves) > 1 else "")
            if key in raw["leaf"] and raw["leaf"][key].get("parsed") is not None:
                continue
            leaves = "\n".join(f'- {l} ({c}, {s})' for l, c, s, _ in H)
            jobs.append((key, SYS_LEAF, USER_LEAF.format(region=r, attr=attr, hint=hint_of(schema, attr), value=value, others=others, leaves=leaves,
                                                        attr_short=attr.split(".")[1].replace("_", " "))))
    log.info("%d prompts to run (%d cached)", len(jobs), len(raw["leaf"]))
    for b in range(0, len(jobs), a.batch):
        grp = jobs[b:b + a.batch]
        for (key, _, u), txt in zip(grp, ask([(s, u) for _, s, u in grp], a.max_new)):
            raw["leaf"][key] = {"raw": txt, "parsed": _json(txt), "prompt_chars": len(u)}
        p.write_text(json.dumps(raw, indent=1, ensure_ascii=False))
        log.info("  %d/%d prompts, parse failures so far %d", min(b + a.batch, len(jobs)), len(jobs), sum(1 for d in raw["leaf"].values() if d.get("parsed") is None))
    ok = sum(1 for d in raw["leaf"].values() if d.get("parsed") is not None)
    check(ok >= 0.9 * max(len(raw["leaf"]), 1), f"{ok}/{len(raw['leaf'])} prompts parsed as JSON", log)


# ---- lexical and identity checks
SOUND_WORDS = {"sound", "sounds", "noise", "noises", "call", "calls", "cry", "cries", "vocalization", "vocalisation", "vocal", "voice", "vocals",
               "music", "playing", "hit", "hits", "beat", "beats", "loud", "quiet", "soft", "heavy", "light", "distant", "multiple", "single"}


def source_tokens(name: str, attr: str) -> set:
    """Tokens that can name a SOURCE: not an -ing verb (barking, wailing), not a generic word, not a sound word."""
    out = set()
    for t in re.split(r"[\s,/-]+", name.lower()):
        if not t or t.endswith("ing") or t in T.GENERIC or t in SOUND_WORDS:
            continue
        t = T.stem_word(t)
        if len(t) >= 4 and t.endswith("s") and not t.endswith("ss"):   # `bees` -> bee: stem_word keeps stems under 4 letters whole
            t = t[:-1]
        out.add(t)
    return out


def shares_source(a: str, b: str, attr: str) -> bool:
    """IDENTITY attributes (sound.event, music.instrument, music.genre, speech.language, speech.emotion): the leaf name IS the
    answer token, so two names may only merge when they name the same source -- `goat bleating` and `sheep bleating` sound
    alike (CLAP sep +0.26) and are different answers. Rule: share at least one source token (`alarm siren` / `siren`,
    `engine idling` / `engine`), or one name contains the other."""
    if attr not in T.IDENTITY:
        return True
    na, nb = T.norm_key(a, attr), T.norm_key(b, attr)
    if na in nb or nb in na:
        return True
    return bool(source_tokens(a, attr) & source_tokens(b, attr))


def guarded(tree, refused, raw, log=None):
    """Apply the guards to the LLM's leaf proposals -> the apply table + a per-group record for the report."""
    P = pairs(tree, refused); groups, generic, moved, dropped, rej = [], [], [], [], collections.Counter()
    for key, rec in raw["leaf"].items():
        p = rec.get("parsed") or {}; attr, value = key.split("#")[0].split("=", 1)
        names = {l: (c, s) for l, c, s, _ in P.get((attr, value), [])}; low = {l.lower(): l for l in names}
        r = attr.split(".")[0]; values = set(tree[r][attr.split(".")[1]])

        def known(x):
            return low.get(str(x).strip().lower().strip(" ."))
        for g in p.get("groups") or []:
            mem = [m for m in (known(x) for x in g.get("members") or []) if m]
            mem = list(dict.fromkeys(mem)); name = known(g.get("name"))
            if len(mem) < 2:
                rej["group: <2 known members"] += 1; continue
            if name not in mem:                                     # the name must be a member; fall back to the best-supported admitted member
                rej["group: name not a member (renamed)"] += 1
                name = sorted(mem, key=lambda m: (names[m][1] != "admitted", -names[m][0], m))[0]
            F = {m: merge_features(T.norm_key(m, attr)) for m in mem}; keep, why = [], []
            for m in mem:
                if m == name:
                    keep.append(m); continue
                ok, w = may_merge(F[m], F[name])
                if ok and not shares_source(m, name, attr):
                    ok, w = False, "identity: no shared source token with the group name"
                (keep if ok else why).append(m if ok else f"{m}: {w}")
            for w in why:
                rej["group member: " + w.split(": ", 1)[1]] += 1
            if len(keep) >= 2:
                groups.append({"attr": attr, "value": value, "name": name, "members": keep,
                               "admitted": [m for m in keep if names[m][1] == "admitted"], "clips": sum(names[m][0] for m in keep), "refused_by_guard": why})
        for g in p.get("generic") or []:
            gn = known(g.get("generic")); sp = [s for s in (known(x) for x in g.get("specifics") or []) if s and s != gn]
            if gn and sp:
                generic.append({"attr": attr, "value": value, "generic": gn, "specifics": sp})
            else:
                rej["generic: unknown names"] += 1
        for m in p.get("misplaced") or []:
            l = known(m.get("leaf")); to = str(m.get("move_to", "")).strip().lower()
            if not l:
                rej["misplaced: unknown leaf"] += 1; continue
            if to == "drop":
                dropped.append({"attr": attr, "value": value, "leaf": l, "clips": names[l][0], "reason": m.get("reason", "")})
            elif to in values and to != value:
                moved.append({"attr": attr, "value": value, "leaf": l, "to": to, "clips": names[l][0], "reason": m.get("reason", "")})
            else:
                rej["misplaced: target not a value"] += 1
    # a generic link wins over a merge of the same names: `cuban music > cuban son` means cuban son stays its own node
    gsp = {(x["attr"], x["value"], m) for x in generic for m in (x["generic"], *x["specifics"])}
    for g in groups:
        if any((g["attr"], g["value"], m) in gsp for m in g["members"]):
            before = len(g["members"]); g["members"] = [m for m in g["members"] if (g["attr"], g["value"], m) not in gsp or m == g["name"]]
            rej["group member: is a generic-link node"] += before - len(g["members"])
            if g["name"] not in g["members"]:
                g["members"] = []
    groups = [g for g in groups if len(g["members"]) >= 2]
    # a leaf in two groups: keep it in the first (larger) one
    seen = set(); out = []
    for g in sorted(groups, key=lambda g: -g["clips"]):
        g["members"] = [m for m in g["members"] if (g["attr"], g["value"], m) not in seen]
        if g["name"] in g["members"] and len(g["members"]) >= 2:
            seen |= {(g["attr"], g["value"], m) for m in g["members"]}; out.append(g)
        else:
            rej["group: members already taken"] += 1
    if log:
        log.info("guards: %d groups (%d leaves folded), %d generic links, %d moves, %d drops; rejected %s", len(out),
                 sum(len(g["members"]) - 1 for g in out), len(generic), len(moved), len(dropped), dict(rej))
    return {"groups": out, "generic": generic, "moved": moved, "dropped": dropped, "rejected": dict(rej)}


# ---- acoustic check. The encoder that can judge a merge under each attribute: CLAP for sound and music, a speech encoder for
# speech attributes; attributes whose content lies in the words have no audio space and are applied on the lexical checks alone.
ENC_FOR = {"speech.emotion": "emotion2vec", "speech.speaking_style": "emotion2vec", "speech.voice_quality": "emotion2vec",
           "speech.gender": "wavlm_sv", "speech.age_group": "wavlm_sv", "speech.accent_origin": "wavlm_sv", "speech.channel": "wavlm_sv",
           "speech.language": "whisper"}
TEXT_ATTRS = {"speech.intent", "speech.topic", "speech.scenario", "speech.speaker_role"}


def enc_for(attr):
    return None if attr in TEXT_ATTRS else ENC_FOR.get(attr, "clap")


def stage_check(a, log):
    """One centroid per leaf (the embeddings of at most 12 of its chunks); per group: the mean cosine among the member
    centroids (within) against their mean cosine to the other leaves of the class (between). sep = within - between;
    a group with sep <= 0 is not applied."""
    from l2r.encoders import build_encoder
    tree, refused, members, _ = load_tree(); raw = json.loads(raw_file("llm_groups_raw.json").read_text()); Gd = guarded(tree, refused, raw, log)
    rows = {r["row"]: r for r in I.rows(a.set)}
    said = collections.defaultdict(list)                            # (attr, class, name) -> rows, for leaves that are not nodes: the table keeps what was written
    for r in read_jsonl(I.sdir(a.set) / "index.jsonl"):
        for l in r.get("labels", []):
            for nm in {l.get("leaf"), l.get("said")} - {None, ""}:
                said[(l["attr"], l["value"], nm)].append(r["row"])

    def rows_of(attr, value, l):
        ids = members.get("/".join((*attr.split("."), value, l))) or said.get((attr, value, l), [])
        rs = [rows[x] for x in ids if x in rows]
        return ([x for x in rs if x["grid"] != "clip"] or rs)[:12]
    cfg = load_config(); encs = {}; cent = {}
    need = collections.defaultdict(set)
    for g in Gd["groups"]:
        need[(g["attr"], g["value"])].update(g["members"])
    for (attr, value) in need:                                      # every admitted leaf of a touched class, for the "between" side
        r, an = attr.split(".")
        need[(attr, value)].update(tree[r][an][value]["leaves"])
    todo = sorted(((attr, value, l) for (attr, value), ls in need.items() for l in ls), key=lambda t: enc_for(t[0]) or "")
    log.info("acoustic check: %d leaves over %d classes", len(todo), len(need))
    for attr, value, l in todo:
        name = enc_for(attr); rs = rows_of(attr, value, l)
        if name is None:
            continue
        # a chunk row is read from its chunk wav; a clip row (a leaf of an attribute decided per clip) from the clip itself, by CLAP only
        paths = [str(G.chunk_wav(x)) if x["grid"] != "clip" else str(audio(x["audio_path"])) for x in rs if x["grid"] != "clip" or name == "clap"]
        if not paths:
            continue
        if name not in encs:
            encs[name] = build_encoder(name, cfg); log.info("encoder %s loaded", name)
        v = np.asarray(encs[name].embed(paths), dtype="float32").mean(0); cent[(attr, value, l)] = v / (np.linalg.norm(v) + 1e-8)
    res = []
    for g in Gd["groups"]:
        M = [m for m in g["members"] if (g["attr"], g["value"], m) in cent]; base = {k: g[k] for k in ("attr", "value", "name")}
        if len(M) < 2:
            res.append({**base, "within": None, "between": None, "sep": None, "encoder": enc_for(g["attr"])}); continue
        C = np.stack([cent[(g["attr"], g["value"], m)] for m in M]); within = float(((C @ C.T).sum() - len(M)) / (len(M) * (len(M) - 1)))
        others = [cent[k] for k in cent if k[:2] == (g["attr"], g["value"]) and k[2] not in g["members"]]
        between = float(np.mean(C @ np.stack(others).T)) if others else None
        res.append({**base, "within": round(within, 3), "between": None if between is None else round(between, 3),
                    "sep": None if between is None else round(within - between, 3), "n_scored": len(M), "encoder": enc_for(g["attr"])})
    raw_file("llm_groups_audio.json").write_text(json.dumps({"built": raw["built"], "groups": res}, indent=1))
    sc = [x["sep"] for x in res if x["sep"] is not None]
    log.info("%d groups scored, median sep %.3f, not separated %d", len(sc), float(np.median(sc)) if sc else 0.0, sum(1 for s in sc if s <= 0))


# ---- deterministic folds: the LLM is asked to be conservative, so filler variants (`overlapping` / `overlapping speech`) remain
FILLER = {"speech", "speaking", "talk", "talking", "voice", "voices", "vocal", "vocals", "sound", "sounds", "noise", "noises", "audio",
          "delivery", "tone", "style", "recording", "chatter", "murmur", "a", "an", "the", "of", "in", "on", "at", "with", "some", "background",
          "effect", "effects", "quality", "manner", "type", "kind", "general", "clear", "very", "quite", "somewhat", "slightly", "heavy", "light"}
# Attributes removed from the tree: their content lies in the spoken words (topic, intent, scenario, speaker role), or they
# carried no usable information (recording channel, voice quality, accent, weather, number of sources).
PRUNE_ATTRS = {"speech.topic", "speech.intent", "speech.scenario", "speech.speaker_role", "speech.channel", "speech.voice_quality",
               "speech.accent_origin", "sound.weather", "sound.num_sources"}


def rule_folds(tree, refused, canon, generic, log=None):
    """Under one value: a leaf that is a sibling leaf PLUS one word folds into that sibling (`overlapping talk` -> `overlapping`),
    on identity attributes only when the extra word is a filler (`alarm bell` stays; `alarm sound` folds). On non-identity
    attributes a linked specific folds into its generic (the link becomes a merge). Same merge guards as the LLM groups."""
    P = pairs(tree, refused); out = []; n_link = 0
    done = {(x["attr"], x["value"], x["leaf"]) for x in canon}; to = {(x["attr"], x["value"], x["leaf"]): x["to"] for x in canon}
    for (attr, value), L in P.items():
        names = [l for l, c, st, _ in L if (attr, value, l) not in done]
        keys = {l: T.norm_key(l, attr).split() for l in names}; byset = {frozenset(k): l for l, k in keys.items() if k}
        F = {l: merge_features(T.norm_key(l, attr)) for l in names}
        for l in names:
            k = keys[l]
            if len(k) < 2:
                continue
            for w in dict.fromkeys(k):                              # drop one word (in name order): is the rest a sibling?
                rest = frozenset(x for x in k if x != w)
                base = byset.get(rest)
                if not base or base == l or (attr, value, base) in done:
                    continue
                if attr in T.IDENTITY and w not in {T.stem_word(f) for f in FILLER}:
                    continue
                ok, why = may_merge(F[l], F[base])
                if ok:
                    out.append({"attr": attr, "value": value, "leaf": l, "to": base, "rule": "one extra word"}); done.add((attr, value, l)); break
    if True:                                                        # links -> merges on non-identity attributes
        for x in generic:
            if x["attr"] in T.IDENTITY:
                continue
            for sp in x["specifics"]:
                if (x["attr"], x["value"], sp) not in done:
                    out.append({"attr": x["attr"], "value": x["value"], "leaf": sp, "to": x["generic"], "rule": "specific into generic"}); done.add((x["attr"], x["value"], sp)); n_link += 1
    # resolve chains a -> b -> c to a -> c
    m = {(x["attr"], x["value"], x["leaf"]): x["to"] for x in out}; m.update(to)
    for x in out:
        t, seen = x["to"], set()
        while (x["attr"], x["value"], t) in m and t not in seen:
            seen.add(t); t = m[(x["attr"], x["value"], t)]
        x["to"] = t
    if log:
        log.info("rule folds: %d (%d specific-into-generic), over %d attributes", len(out), n_link, len({x["attr"] for x in out}))
    return out


def stage_apply(a, log):
    """The checks -> the apply table read by `l2r.tree.publish` (leaf -> group name, generic links, pruned attributes)."""
    tree, refused, _, man = load_tree(); raw = json.loads(raw_file("llm_groups_raw.json").read_text()); Gd = guarded(tree, refused, raw, log)
    ap_ = raw_file("llm_groups_audio.json")
    aud = {(x["attr"], x["value"], x["name"]): x for x in json.loads(ap_.read_text())["groups"]} if ap_.exists() else {}
    for g in Gd["groups"]:
        g.update({k: aud.get((g["attr"], g["value"], g["name"]), {}).get(k) for k in ("within", "between", "sep")}); g["encoder"] = enc_for(g["attr"])
        g["flag"] = ("" if g["encoder"] is None                                      # no audio space can judge the attribute: applied on the lexical checks
                     else f"{g['encoder']}: members are not closer to each other than to the rest of the class" if g["sep"] is not None and g["sep"] <= 0
                     else "unscored: no audio rows for the members" if g["sep"] is None else "")
    canon = [{"attr": g["attr"], "value": g["value"], "leaf": m, "to": g["name"]} for g in Gd["groups"] if not g["flag"] for m in g["members"] if m != g["name"]]
    folds = rule_folds(tree, refused, canon, Gd["generic"], log)
    table = {"built": man["built"], "prompt_version": raw["prompt_version"], "canon": canon + [{k: x[k] for k in ("attr", "value", "leaf", "to")} for x in folds],
             "rule_folds": folds, "generic": [x for x in Gd["generic"] if x["attr"] in T.IDENTITY], "pruned_attrs": sorted(PRUNE_ATTRS),
             "moved": [], "dropped": [], "proposed_moves": Gd["moved"], "proposed_drops": Gd["dropped"]}
    raw_file("llm_groups.json").write_text(json.dumps(table, indent=1, ensure_ascii=False))
    by = collections.defaultdict(list)
    for g in Gd["groups"]:
        by[g["attr"]].append(g)
    md = ["# LLM cleaning of the leaf vocabulary", "",
          f"{len(Gd['groups'])} groups proposed and lexically valid, {sum(1 for g in Gd['groups'] if g['flag'])} held back by the acoustic check; "
          f"{len(folds)} rule folds; {len(table['generic'])} generic links; pruned attributes: {', '.join(sorted(PRUNE_ATTRS))}.", "",
          "Refused by the lexical / identity checks: " + (", ".join(f"{k} {v}" for k, v in sorted(Gd["rejected"].items())) or "none"), "",
          "## Groups (name <- members; sep = within - between cosine)", ""]
    for attr in sorted(by):
        md.append(f"### {attr}")
        for g in sorted(by[attr], key=lambda g: (g["value"], -g["clips"])):
            sep = " · no audio check" if g["encoder"] is None else "" if g["sep"] is None else f" · {g['encoder']} sep {g['sep']:+.2f}"
            md.append(f"- `{g['value']}`: **{g['name']}** <- " + ", ".join(m for m in g["members"] if m != g["name"]) + sep + (" **NOT APPLIED**" if g["flag"] else ""))
    md += ["", "## Generic -> specifics (linked, not merged)", ""] + [f"- `{x['attr']}={x['value']}`: {x['generic']} > " + ", ".join(x["specifics"]) for x in Gd["generic"]]
    out = work("reports", "llm_groups.md"); out.write_text("\n".join(md) + "\n")
    log.info("wrote %s and %s (%d leaves folded into a group)", raw_file("llm_groups.json"), out, len(table["canon"]))


def selftest():
    tree = {"sound": {"event": {"animal": {"clips": 40, "chunks": 0, "leaves": {"bee buzzing": {"clips": 31}, "thunderclap": {"clips": 5}, "clap of thunder": {"clips": 3}, "dog barking": {"clips": 20}, "goat bleating": {"clips": 4}, "sheep bleating": {"clips": 4}, "dog": {"clips": 9}, "two dogs barking": {"clips": 3}, "loud bark": {"clips": 3}, "quiet bark": {"clips": 3}}},
                                "tool": {"clips": 5, "chunks": 0, "leaves": {}},
                                "alarm or signal": {"clips": 9, "chunks": 0, "leaves": {"siren": {"clips": 5}, "alarm siren": {"clips": 2}, "siren wailing": {"clips": 2}}}}}}
    refused = [{"attr": "sound.event", "value": "animal", "leaf": "bees buzzing", "clips": 1, "why": "seen on one clip"},
               {"attr": "sound.event", "value": "animal", "leaf": "not bark", "clips": 1, "why": "seen on one clip"}]
    raw = {"leaf": {"sound.event=animal": {"parsed": {
        "groups": [{"name": "bee buzzing", "members": ["bee buzzing", "bees buzzing"]}, {"name": "thunder", "members": ["thunderclap", "clap of thunder"]},
                   {"name": "goat bleating", "members": ["goat bleating", "sheep bleating"]}, {"name": "siren", "members": ["siren", "alarm siren", "siren wailing"]},
                   {"name": "dog barking", "members": ["dog barking", "two dogs barking", "not bark"]}, {"name": "loud bark", "members": ["loud bark", "quiet bark"]},
                   {"name": "x", "members": ["unknown leaf", "dog barking"]}],
        "generic": [{"generic": "dog", "specifics": ["dog barking", "ghost"]}, {"generic": "thunderclap", "specifics": ["clap of thunder"]}],
        "misplaced": [{"leaf": "thunderclap", "move_to": "natural element"}, {"leaf": "dog", "move_to": "tool"}, {"leaf": "bee buzzing", "move_to": "drop"}]}}}}
    G = guarded(tree, refused, raw)
    names = {g["name"]: g for g in G["groups"]}
    assert "bee buzzing" in names and names["bee buzzing"]["members"] == ["bee buzzing", "bees buzzing"], names           # refused leaf rescued
    assert "thunderclap" not in names                                        # the generic link on the same pair wins over the merge
    assert "dog barking" not in names                                        # negation + numeric members refused -> the group collapses to nothing
    assert "loud bark" not in names or len(names["loud bark"]["members"]) < 2                                              # polarity refused -> no group
    assert "x" not in names and all("unknown leaf" not in g["members"] for g in G["groups"])
    assert "goat bleating" not in names, names                                # different sources that sound alike: refused on an identity attribute
    raw["leaf"]["sound.event=alarm or signal"] = {"parsed": {"groups": [{"name": "siren", "members": ["siren", "alarm siren", "siren wailing"]}]}}
    G2 = guarded(tree, refused, raw); n2 = {g["name"]: g for g in G2["groups"]}
    assert set(n2["siren"]["members"]) == {"siren", "alarm siren", "siren wailing"}, n2
    assert G["generic"][0] == {"attr": "sound.event", "value": "animal", "generic": "dog", "specifics": ["dog barking"]} and len(G["generic"]) == 2
    assert [m["leaf"] for m in G["moved"]] == ["dog"] and [d["leaf"] for d in G["dropped"]] == ["bee buzzing"]           # unknown target value rejected
    print("selftest ok:", {k: len(v) if isinstance(v, list) else v for k, v in G.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["propose", "check", "apply", "selftest"]); ap.add_argument("--set", default="pool")
    ap.add_argument("--model", default="", help="text LLM (default: models.tree_llm of the config)"); ap.add_argument("--four-bit", action="store_true", help="load the LLM in 4-bit")
    ap.add_argument("--batch", type=int, default=6); ap.add_argument("--max-new", type=int, default=1800); ap.add_argument("--fresh", action="store_true", help="discard cached proposals")
    a = ap.parse_args()
    if a.stage == "selftest":
        selftest()
    else:
        _, log = setup_run(f"tree_llm_clean_{a.stage}")
        {"propose": stage_propose, "check": stage_check, "apply": stage_apply}[a.stage](a, log)
