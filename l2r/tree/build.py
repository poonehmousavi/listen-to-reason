"""Tree construction: label table -> vote -> normalize -> merge -> admit -> the tree and the membership of its nodes.

  region -> attribute -> class (closed) -> leaf (open, descriptive)

The tree is always re-derived from the label table (`<work>/sets/<set>/index.jsonl`), so the vote rule, the merge
thresholds and the admission rule can change without re-annotating. Nothing is a node unless chunks support it:
classes that were never labelled are absent, and a leaf must pass `admit()`.

Files written to `<checkpoints>/tree/`: tree.json (the tree), canon.json (surface form -> leaf), aliases.json (expert
leaf -> annotator leaf), moves.json / reparent.json (leaves placed under another attribute / class), members.json
(node -> chunk rows), refused.json (candidate leaves that did not become nodes, with the reason), merges.json.
"""
from __future__ import annotations

import collections
from pathlib import Path
import json
import re

from l2r.common import ckpt, read_jsonl
from l2r.dataset import index as I

LALM = "qwen3_omni"
# single-choice attributes: the expert classifier of the attribute, and the score below which the annotator's label is used instead
EXPERT = {"speech.emotion": ("emotion2vec", 0.60), "speech.gender": ("wavlm_sv", 0.05), "speech.language": ("whisper_lid", 0.50),
          "music.tempo": ("librosa", 0.30), "music.mode": ("librosa", 0.80)}
MULTI_EXPERT = {"sound.event": ("panns", 0.30), "cross.families_present": ("panns", 0.30)}     # multiple-choice: union with the annotator
ADMIT = {"min_clips": 2, "min_sources": 2}


def vote(fired: list[dict]) -> tuple[list[dict], collections.Counter]:
    """One row's fired labels -> the labels that enter the tree: [{attr, value, leaf, tools, agree}]."""
    by = collections.defaultdict(list); stat = collections.Counter(); out = []
    for f in fired:
        if f["value"] != "unplaced":
            by[f["attr"]].append(f)
    for attr, fs in by.items():
        lal = [f for f in fs if f["tool"] == LALM]
        if attr in MULTI_EXPERT:
            tool, floor = MULTI_EXPERT[attr]
            keep = lal + [f for f in fs if f["tool"] == tool and (f["score"] or 0) >= floor]
            seen = {}
            for f in keep:                                                  # one node per (value, leaf) per row
                k = (f["value"], f["leaf"]); seen.setdefault(k, {"attr": attr, "value": f["value"], "leaf": f["leaf"], "tools": set(), "agree": None})
                seen[k]["tools"].add(f["tool"])
            out += list(seen.values()); continue
        ex = None
        if attr in EXPERT:
            tool, floor = EXPERT[attr]
            c = [f for f in fs if f["tool"] == tool and (f["score"] or 0) >= floor]
            ex = max(c, key=lambda f: f["score"]) if c else None
        if ex is None and not lal:
            stat["expert_below_floor"] += 1; continue
        parent = (ex or lal[0])["value"]; agree = None
        if ex is not None and lal:
            agree = lal[0]["value"] == parent; stat["agree" if agree else "disagree"] += 1
        leaves = [f["leaf"] for f in fs if f["leaf"] and f["value"] == parent]   # a leaf stays under the parent its author gave
        stat["leaf_orphaned"] += sum(1 for f in lal if f["leaf"] and f["value"] != parent)
        for leaf in (dict.fromkeys(leaves) or [""]):
            out.append({"attr": attr, "value": parent, "leaf": leaf, "tools": {f["tool"] for f in fs if f["value"] == parent}, "agree": agree})
    return out, stat


FILL = {"a", "an", "the", "of", "some", "something", "someone", "sound", "sounds", "noise", "being", "on", "about", "in", "at", "for", "to"}
TAIL = {"mood", "tone", "vibe", "style", "music", "voice", "feel", "feeling"}     # `relaxed mood` = `relaxed`, `rock music` = `rock` (only as the LAST of several words)
# Word-level synonyms, curated: an explicit list is safer than a looser embedding threshold,
# which would also merge `animal growling` ~ `dog growling` and `soprano sax` ~ `tenor sax`.
WORD_SYN = {"sax": "saxophone", "providing": "giving", "simple": "basic", "educator": "teacher", "repetitive": "repeating", "airplane": "aircraft",
            "plane": "aircraft", "movie": "film", "telephone": "phone", "theatrical": "dramatic", "geese": "goose", "laughter": "laughing",
            "hen": "chicken", "hens": "chicken", "chickens": "chicken", "fear": "fearful", "disgust": "disgusted", "anger": "angry", "sadness": "sad", "happiness": "happy", "surprise": "surprised", "kids": "child", "children": "child", "kid": "child", "person": "someone", "automobile": "car", "tv": "television", "mic": "microphone"}


def stem_word(w: str) -> str:
    """Suffix fold, no model, applied UNTIL STABLE (`sandpapering` -> sandpaper -> sandpap, the same place `sandpaper` lands;
    one pass left them apart): announcement / announcing / announcer / announces -> announc; chatting -> chat."""
    w = WORD_SYN.get(w, w)
    while True:
        for suf in ("ements", "ement", "ments", "ment", "ations", "ation", "ings", "ing", "ers", "er", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]; break
        else:
            break
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouls":
        w = w[:-1]                                                # chatt -> chat, runn -> run
    return w[:-1] if len(w) > 4 and w[-1] == "e" else w


ATTR_FILL = {"speech.accent_origin": {"standard", "native", "neutral", "accent", "english", "speech", "variety"}}   # american english = standard american = american accent


def norm_key(name: str, attr: str = "") -> str:
    raw = [w for w in re.findall(r"[a-z]+", name.lower()) if w not in FILL and w not in ATTR_FILL.get(attr, ())]
    if len(raw) > 1 and raw[-1] in TAIL:
        raw = raw[:-1]
    ws = [stem_word(w) for w in raw]
    return " ".join(sorted(ws)) or name                          # `shattering glass` = `glass shattering`


def light_canon(names: list[str], attr: str = "") -> dict[str, str]:
    """names are ordered best-supported first, so the canonical form is the best-supported surface form."""
    first = {}
    return {n: first.setdefault(norm_key(n, attr), n) for n in names}


# AudioSet's onomatopoeic class names, as PANNs emits them, -> the descriptive name a person would use.
# Static because the co-fire alias needs the annotator to have described the SAME chunk, and `moo` stayed a leaf next to a
# one-clip `cow mooing`. Applied to expert leaves only, before the co-fire vote.
ONOMATOPOEIA = {"moo": "cow mooing", "meow": "cat meowing", "oink": "pig oinking", "bow-wow": "dog barking", "quack": "duck quacking",
                "neigh, whinny": "horse neighing", "cluck": "chicken clucking", "croak": "frog croaking", "caw": "crow cawing",
                "bleat": "sheep bleating", "purr": "cat purring", "yip": "dog barking", "howl": "dog howling", "hoot": "owl hooting",
                "coo": "pigeon cooing", "gobble": "turkey gobbling", "crowing, cock-a-doodle-doo": "rooster crowing", "honk": "goose honking",
                "roar": "animal roaring", "hiss": "hissing", "buzz": "insect buzzing", "chirp, tweet": "bird chirping", "squawk": "bird squawking"}


def cofire_aliases(table, ancestors: dict) -> dict:
    """An expert CLASS name and the annotator's description of the SAME event on the SAME chunk are one leaf:
    PANNs `bow-wow` (ontology: under Dog) + LALM `dog barking` -> `dog barking`. Evidence = same row, same parent,
    and a shared word between the description and the class name or one of its ontology ancestors. Majority over rows."""
    low = {k.lower(): {a.lower() for a in v} for k, v in ancestors.items()}; votes = collections.defaultdict(collections.Counter)
    for r in table:
        by = collections.defaultdict(lambda: ([], []))
        for f in r["fired"]:
            if f["attr"] in MULTI_EXPERT and f["leaf"] and f["value"] != "unplaced":
                by[(f["attr"], f["value"])][f["tool"] == LALM].append(f["leaf"])
        for (attr, value), (exp, lal) in by.items():
            for e in set(exp) - set(lal):
                ew = {stem_word(w) for n in {e} | low.get(e, set()) for w in re.findall(r"[a-z]+", n)} - {stem_word(w) for w in re.findall(r"[a-z]+", value)}
                sc = {m: len(ew & {stem_word(w) for w in re.findall(r"[a-z]+", m)}) for m in set(lal)}
                best = sorted(sc.items(), key=lambda t: -t[1])
                if best and best[0][1] > 0 and (len(best) == 1 or best[0][1] > best[1][1]):
                    votes[(attr, value, e)][best[0][0]] += 1
    out = {}
    for k, c in votes.items():                               # ties broken by NAME, never by row order
        m, n = sorted(c.items(), key=lambda t: (-t[1], t[0]))[0]
        if n >= 0.5 * sum(c.values()):
            out[k] = m
    return out   # a generic class (`engine`) that co-fires with many descriptions stays itself


NO_EMBED_MERGE = {"speech.language", "speech.accent_origin"}       # names of DIFFERENT things that embed alike
#                                                                     (lithuanian ~ latvian, 0.856); only the suffix fold applies here


def embed_may_merge(a: str, b: str, cos: float, tau: float) -> bool:
    """Synonyms only (`news announcer` and `sports announcer` are different things). An embedding calls
    `car horn` ~ `train horn` (0.81), `animal growling` ~ `dog growling` (0.91) and `sports announcer` ~ `announcer`
    (0.92) alike, so similarity merges only two names of the SAME length at >= 0.93 (movie / film reviewer, phone /
    telephone ringing, city / urban street). A name that generalises another is its PARENT, never its synonym: that
    pair is linked (`Merger.generic`), not merged."""
    wa, wb = norm_key(a).split(), norm_key(b).split()
    return cos >= 0.93 and len(wa) == len(wb)


META_HEAD = {"announcement", "discussion", "talk", "review", "lesson", "commentary", "report", "advertisement", "class", "broadcast",
             "interview", "tutorial", "lecture", "story", "debate"}


class Merger:
    """Synonym merge of the leaves under ONE parent: MiniLM cosine >= tau, never across a conflict of `guards.may_merge`
    (numeric / negation / polarity). The canonical name is the best-supported surface form, never invented."""

    def __init__(self, tau=0.80, model=True):
        self.tau = tau; self.st = None; self.log = []; self.generic = {}
        if model:
            from sentence_transformers import SentenceTransformer
            from l2r.common import load_config
            self.st = SentenceTransformer(load_config()["models"]["text_embedder"])

    def canon(self, where: str, support: dict[str, int]) -> dict[str, str]:
        names = sorted(support, key=lambda n: (-support[n], n))
        if len(names) < 2:
            return {n: n for n in names}
        pre = light_canon(names, where.split("=")[0])
        if self.st is None or where.split("=")[0] in NO_EMBED_MERGE:
            return pre
        from l2r.tree.guards import may_merge, merge_features
        full, names = names, [n for n in names if pre[n] == n]
        E = self.st.encode(names, normalize_embeddings=True, show_progress_bar=False); F = [merge_features(n) for n in names]
        out, heads = {}, []
        for i, n in enumerate(names):
            for j in heads:
                cos = float(E[i] @ E[j])
                if self.tau - 0.12 <= cos < self.tau:
                    self.log.append({"where": where, "a": n, "b": names[j], "cos": round(cos, 3), "merged": False, "why": "near miss (below tau)"})
                if cos >= self.tau:
                    ok, why = may_merge(F[i], F[j])
                    if ok and not embed_may_merge(n, names[j], cos, self.tau):
                        ok, why = False, "similar but neither generalises the other (cos < 0.90)"
                    self.log.append({"where": where, "a": n, "b": names[j], "cos": round(cos, 3), "merged": ok, "why": why})
                    if ok:
                        out[n] = names[j]; break
            else:
                out[n] = n; heads.append(i)
        # GENERIC -> SPECIFIC links among the surviving names: `announcer` > `sports announcer`, `news announcer`.
        # The generic's words are a strict subset of the specific's and the two are similar; the closest generic wins.
        keys = {i: set(norm_key(names[i]).split()) for i in heads}
        for i in heads:
            last = names[i].split()[-1]
            if last in META_HEAD:                                           # an "X announcement" is ABOUT X: never linked under X
                continue
            best = max(((float(E[i] @ E[j]), j) for j in heads if j != i and keys[j] < keys[i]), default=(0.0, -1))
            if best[0] >= self.tau:
                self.generic[(where, names[i])] = names[best[1]]
                self.log.append({"where": where, "a": names[i], "b": names[best[1]], "cos": round(best[0], 3), "merged": False, "why": "child of the generic (linked, not merged)"})
        return {n: out[pre[n]] for n in full}


GENERIC = {"clear", "clean", "normal", "standard", "regular", "general", "plain", "typical", "good", "simple", "basic", "audio", "voice",
           "sound", "speech", "music", "tone", "style", "quality", "weather", "recording", "background", "noise"}


# Example details that the template of the first annotation rounds showed the annotator, which it copied as fillers.
# Such an example is refused when it is also a hub of its attribute (>= PARROT_SHARE of the attribute's leaf mass);
# a rare `sarcastic` stays.
PARROT = {"speech.emotion": {"flat", "sarcastic", "tired", "matter-of-fact"}, "music.genre": {"garage rock"},
          "speech.voice_quality": {"stuttering", "hoarse", "slurred", "nasal", "out of breath", "monotone"},
          "music.mood": {"wistful", "meditative", "lullaby-like"}, "sound.weather": {"heavy rain", "distant thunder", "strong gusts"},
          "sound.event": {"dog barking", "car alarm", "door slamming", "small dog yapping"},
          "sound.activity": {"chopping vegetables", "playing table tennis", "typing"}}
PARROT_SHARE, HUB_SHARE, HUB_MIN = 0.04, 0.40, 25
HUB_EXEMPT = {"sound.event", "music.instrument", "music.genre", "speech.language"}   # there the dominant leaf IS the node (english, drums)
EXPERT_BACKED = {"sound.event"}                       # here an example survives when an expert names the same thing (co-fire alias)


IDENTITY = ("sound.event", "music.instrument", "music.genre", "speech.language", "speech.emotion")   # WHAT it is outranks HOW / WHERE


AGENT = re.compile(r"(er|or|ist|ian|ant|ee|man|woman|person|host|guide|coach|judge|chef|pilot)$")


def leaf_home(by_attr: dict, leaf: str = "") -> str:
    """The one attribute a leaf lives in. The annotator writes `door slamming` as the event AND as its action and its
    scene; the identity attribute wins whenever it holds the leaf at all (first build: support alone sent door slamming,
    glass breaking, footsteps to action_material and deleted them from sound.event). Otherwise most clips, ties by name."""
    for a in IDENTITY:
        if a in by_attr:
            return a
    if "speech.speaker_role" in by_attr and leaf and AGENT.search(leaf.split()[-1]):
        return "speech.speaker_role"                           # `announcer`, `narrator`, `commentator` name WHO speaks, not a style
    return sorted(by_attr.items(), key=lambda t: (-t[1], t[0]))[0][0]


SHARED_VOCAB = {frozenset(("music.mood", "speech.emotion"))}     # the same affect word legitimately names a node in BOTH


def members(value: str) -> list[str]:
    """`chinese, japanese and korean` -> its members; a plain value -> [value]."""
    return [m.strip() for m in re.split(r",| and | or ", value) if m.strip()]


def plan_moves(voted: list[tuple[str, list[dict]]], vocab: dict[str, set]) -> dict:
    """(attr, value, leaf) -> (home attr, value there, leaf there): a leaf the annotator wrote under the WRONG attribute is
    MOVED to its home, not dropped (`thunderstorm` under sound.event, `door slamming` under action_material). Homes are decided INSIDE a
    region: weather `calm` is not speech.emotion `calm`, music.mood `playful` is not speech.emotion `playful`."""
    cnt = collections.defaultdict(lambda: collections.defaultdict(set)); parv = collections.defaultdict(lambda: collections.defaultdict(set))
    form = collections.defaultdict(collections.Counter); keys = set()
    for clip, labs in voted:
        for l in labs:
            if l["leaf"]:
                k = norm_key(l["leaf"]); reg = l["attr"].split(".")[0]
                cnt[(reg, k)][l["attr"]].add(clip); parv[(l["attr"], k)][l["value"]].add(clip); form[(l["attr"], k)][l["leaf"]] += 1
                keys.add((l["attr"], l["value"], l["leaf"]))
    move = {}
    for attr, value, leaf in keys:
        reg = attr.split(".")[0]; k = norm_key(leaf); c = cnt[(reg, k)]
        h = leaf_home({a: len(x) for a, x in c.items()}, leaf) if len(c) > 1 else attr
        if h != attr:
            hv = sorted(parv[(h, k)].items(), key=lambda t: (-len(t[1]), t[0]))[0][0]
            move[(attr, value, leaf)] = (h, hv, form[(h, k)].most_common(1)[0][0]); continue
        for a2, vs in vocab.items():                            # the leaf IS a closed value of a sibling attribute
            if a2 != attr and a2.split(".")[0] == reg and leaf in vs:
                move[(attr, value, leaf)] = (a2, leaf, ""); break
    return move


def apply_moves(labs: list[dict], move: dict, multi: set) -> list[dict]:
    """A moved leaf lands at its home only when that cannot contradict the row: the home is multi-valued, or the row
    says nothing there, or it already says the same parent. The original label keeps its PARENT (without the leaf)."""
    if not move:
        return labs
    out = {}
    def put(l):
        k = (l["attr"], l["value"], l["leaf"])
        if k in out:
            out[k]["tools"] = set(out[k]["tools"]) | set(l["tools"])
        else:
            out[k] = {**l, "tools": set(l["tools"])}
    for l in labs:
        t = move.get((l["attr"], l["value"], l["leaf"])) if l["leaf"] else None
        if not t:
            put(l); continue
        a2, v2, l2 = t
        cur = {x["value"] for x in labs if x["attr"] == a2 and not (x["leaf"] and (x["attr"], x["value"], x["leaf"]) in move)}
        if a2 in multi or not cur or cur == {v2}:
            put({**l, "attr": a2, "value": v2, "leaf": l2})
        put({**l, "leaf": ""})
    return list(out.values())


def admit(path, node, vocab: dict[str, set], home: dict | None = None, ctx: dict | None = None) -> str:
    """'' = admitted, else the refusal reason. `vocab[attr]` = closed values of every attribute (leak check);
    `home[leaf]` = the attribute under which that leaf has the most clips (a leaf lives in ONE attribute)."""
    attr, value, leaf = path
    if not leaf:
        return ""
    reg = attr.split(".")[0]
    if home and home.get((reg, leaf), attr) != attr:
        return f"belongs to {home[(reg, leaf)]}"
    if ctx and attr == "speech.accent_origin":
        core = [w for w in re.findall(r"[a-z]+", leaf) if w not in ATTR_FILL[attr] - {"english"}]
        if len(core) <= 1 and (not core or core[0] in ctx.get("languages", ())):
            return "names the language, not an accent"
    if ctx:
        forms = {leaf} | set(node.get("forms") or ())
        if forms & PARROT.get(attr, set()) and ctx["attr_share"] >= PARROT_SHARE and not (attr in EXPERT_BACKED and ctx["expert_backed"]):
            return "prompt example copied as a filler"
        if attr not in HUB_EXEMPT and ctx["parent_clips"] >= HUB_MIN and len(node["clips"]) >= HUB_SHARE * ctx["parent_clips"]:
            return "covers most of its parent"
    member = len(members(value)) > 1 and leaf in members(value)      # `japanese` under `chinese, japanese and korean` NARROWS the parent
    if not member and set(re.findall(r"[a-z]+", leaf)) <= GENERIC | set(re.findall(r"[a-z]+", value)) | set(re.findall(r"[a-z]+", attr)):
        return "says no more than its parent"
    if re.search(r"\d", leaf):
        return "numeric"
    if not member and (leaf == value or set(leaf.split()) <= set(value.split())):
        return "says no more than its parent"
    for a, vs in vocab.items():
        if a != attr and leaf in vs and frozenset((a, attr)) not in SHARED_VOCAB:   # `playful` speech is not `playful` music
            return f"belongs to {a}"
    expert_only = node["tools"] and LALM not in node["tools"]                       # PANNs class / Whisper language: a fixed vocabulary
    if len(node["clips"]) < ADMIT["min_clips"]:
        return "seen on one clip"
    if not expert_only and len(node["sources"]) < ADMIT["min_sources"]:
        return "seen in one source"
    return ""


def load_llm_groups(path) -> dict:
    """The apply table written by `l2r.tree.llm_clean` (its checks already passed there):
    canon (attr, value, leaf) -> group name; generic (attr, value, specific) -> generic; moved (attr, value, leaf) -> value; dropped set.
    Keys are matched on `norm_key`, so a surface form of a grouped leaf follows its group."""
    G = json.loads(Path(path).read_text())
    k = lambda a, v, l: (a, v, norm_key(l, a))
    return {"canon": {k(x["attr"], x["value"], x["leaf"]): x["to"] for x in G["canon"]},
            "generic": {k(x["attr"], x["value"], s): x["generic"] for x in G["generic"] for s in x["specifics"]},
            "moved": {k(x["attr"], x["value"], x["leaf"]): x["to"] for x in G["moved"]},
            "dropped": {k(x["attr"], x["value"], x["leaf"]) for x in G["dropped"]}, "pruned": set(G.get("pruned_attrs", [])), "built": G.get("built")}


def build(name: str, schema: dict, model=True, log=None, llm_groups=None, out: str = "tree") -> dict:
    """Build the tree from the label table of set `name` and write it to `<checkpoints>/<out>/`. `model` = merge
    near-synonyms with a sentence embedder; `llm_groups` = the apply table of the LLM cleaning step (leaf groups,
    generic links, pruned attributes)."""
    table = read_jsonl(I.sdir(name) / "index.jsonl") or I.assemble(name)
    LG = load_llm_groups(llm_groups) if llm_groups else None
    vocab = {f"{r}.{a}": {v.lower() for v in at["values"]} for r, R in schema["regions"].items()
             for a, at in R["attributes"].items() if isinstance(at.get("values"), list)}
    raw = collections.defaultdict(lambda: {"clips": set(), "rows": [], "sources": set(), "tools": set(), "forms": set()})
    stat = collections.Counter()
    try:
        from l2r.dataset.experts import full_ancestors
        alias = cofire_aliases(table, full_ancestors())
    except FileNotFoundError:
        alias = {}
    stat["cofire_aliases"] = len(alias)
    voted = []
    for r in table:
        labs, st = vote(r["fired"]); stat.update(st)
        if LG and LG["pruned"]:
            n0 = len(labs); labs = [l for l in labs if l["attr"] not in LG["pruned"]]; stat["pruned_attr_labels"] += n0 - len(labs)
        for l in labs:
            if LALM not in l["tools"]:
                l["leaf"] = ONOMATOPOEIA.get(l["leaf"], l["leaf"])
            l["leaf"] = alias.get((l["attr"], l["value"], l["leaf"]), l["leaf"])
            if LG and l["leaf"]:
                kk = (l["attr"], l["value"], norm_key(l["leaf"], l["attr"]))
                if kk in LG["dropped"]:
                    l["leaf"] = ""; stat["llm_dropped_rows"] += 1                       # the row keeps its parent value
                elif kk in LG["moved"]:
                    l["value"] = LG["moved"][kk]; stat["llm_moved_rows"] += 1
        voted.append(list({(x["attr"], x["value"], x["leaf"]): x for x in labs}.values()))
    multi = {f"{r}.{a}" for r, R in schema["regions"].items() for a, at in R["attributes"].items() if at.get("multi")}
    move = plan_moves([(r["clip"], labs) for r, labs in zip(table, voted)], vocab)
    voted = [apply_moves(labs, move, multi) for labs in voted]; stat["moved_leaves"] = len(move)
    # ONE PATH PER LEAF: the annotator files `door slamming` under household object, tool AND human non-speech on
    # different clips. The leaf's parent is where most clips put it (ties by name); the other rows are re-parented.
    par = collections.defaultdict(lambda: collections.defaultdict(set))
    for r, labs in zip(table, voted):
        for l in labs:
            if l["leaf"]:
                par[(l["attr"], norm_key(l["leaf"], l["attr"]))][l["value"]].add(r["clip"])        # `door slam` and `door slamming` move together
    best = {k: sorted(vs.items(), key=lambda t: (-len(t[1]), t[0]))[0][0] for k, vs in par.items() if len(vs) > 1}
    reparent = {(l["attr"], l["value"], l["leaf"]): best[(l["attr"], norm_key(l["leaf"], l["attr"]))] for labs in voted for l in labs
                if l["leaf"] and (l["attr"], norm_key(l["leaf"], l["attr"])) in best}
    stat["reparented_leaves"] = len(best)
    for r, labs in zip(table, voted):
        for l in labs:
            l["value"] = reparent.get((l["attr"], l["value"], l["leaf"]), l["value"])
        for l in {(x["attr"], x["value"], x["leaf"]): x for x in labs}.values():
            for leaf in {"", l["leaf"]}:                                           # the parent counts every row under it
                n = raw[(l["attr"], l["value"], leaf)]
                n["clips"].add(r["clip"]); n["sources"].add(r["source"]); n["tools"] |= l["tools"]
                if r["grid"] != "clip":
                    n["rows"].append(r["row"])
    mg = Merger(model=model); nodes = {}; leafmap = {}                  # (attr, value, leaf as voted) -> canonical leaf
    for (attr, value) in sorted({(a, v) for a, v, _ in raw}):
        leaves = {l: len(raw[(attr, value, l)]["clips"]) for a, v, l in raw if (a, v) == (attr, value) and l}
        can = mg.canon(f"{attr}={value}", leaves)
        if LG:                                                          # LLM groups compose on top of the synonym merge; generic links join Merger.generic
            for l in can:
                g = LG["canon"].get((attr, value, norm_key(can[l], attr)), LG["canon"].get((attr, value, norm_key(l, attr))))
                if g and g != can[l]:
                    can[l] = g; stat["llm_grouped_leaves"] += 1
            for l in set(can.values()):
                g = LG["generic"].get((attr, value, norm_key(l, attr)))
                if g and g != l and (f"{attr}={value}", l) not in mg.generic:
                    mg.generic[(f"{attr}={value}", l)] = g; stat["llm_generic_links"] += 1
        nodes[(attr, value, "")] = raw[(attr, value, "")]
        for l, c in can.items():
            leafmap[(attr, value, l)] = c
            n = nodes.setdefault((attr, value, c), {"clips": set(), "rows": [], "sources": set(), "tools": set(), "forms": set()})
            s = raw[(attr, value, l)]
            n["clips"] |= s["clips"]; n["rows"] += s["rows"]; n["sources"] |= s["sources"]; n["tools"] |= s["tools"]; n["forms"].add(l)
    gen = {(w.split("=", 1)[0], w.split("=", 1)[1], c): g for (w, c), g in mg.generic.items()}     # (attr, value, specific) -> generic
    own = {k: {"clips": set(n["clips"]), "rows": list(n["rows"]), "sources": set(n["sources"]), "tools": set(n["tools"])} for k, n in nodes.items()}
    for k in gen:                                                   # a node's OWN support goes to EVERY ancestor on its chain. Order-free:
        a_, v_, x = k; seen = set()                                 # the first version relied on "most specific first" by word count, and a
        while (a_, v_, x) in gen and x not in seen:                 # tie (`metal parts hitting` / `metal hitting metal`, 3 words each) left
            seen.add(x); x = gen[(a_, v_, x)]                       # the grandparent one clip short -- caught by the graph/table gate.
            g = nodes.get((a_, v_, x))
            if g is not None and k in own:
                g["clips"] |= own[k]["clips"]; g["rows"] += own[k]["rows"]; g["sources"] |= own[k]["sources"]; g["tools"] |= own[k]["tools"]
    tree, refused, members = {}, [], {}
    sup = collections.defaultdict(collections.Counter)                          # (region, leaf) -> attr -> clips: `coughing` under action_material
    for (attr, value, leaf), n in nodes.items():                                # AND under sound.event lives where it is best supported
        if leaf:
            sup[(attr.split(".")[0], leaf)][attr] += len(n["clips"])
    home = {k: leaf_home(c, k[1]) for k, c in sup.items() if len(c) > 1}
    mass = collections.Counter()
    for (attr, value, leaf), n in nodes.items():
        if leaf:
            mass[attr] += len(n["clips"])
    languages = {l for (a, v, l) in nodes if a == "speech.language" and l}
    for (attr, value, leaf), n in sorted(nodes.items()):
        if LG and attr in LG["pruned"]:                              # a pruned attribute never becomes a node (leaf homing can re-create one)
            stat["pruned_attr_nodes"] += 1; continue
        ctx = {"languages": languages, "attr_share": len(n["clips"]) / max(mass[attr], 1), "parent_clips": len(nodes[(attr, value, "")]["clips"]),
               "expert_backed": bool(n["tools"] - {LALM})} if leaf else None
        why = admit((attr, value, leaf), n, vocab, home, ctx)
        if why:
            refused.append({"attr": attr, "value": value, "leaf": leaf, "clips": len(n["clips"]), "why": why}); stat[f"refused: {why}"] += 1
            continue
        region, a = attr.split(".")
        v = tree.setdefault(region, {}).setdefault(a, {}).setdefault(value, {"clips": 0, "chunks": 0, "leaves": {}})
        rec = {"clips": len(n["clips"]), "chunks": len(set(n["rows"])), "sources": sorted(n["sources"]), "tools": sorted(n["tools"])}
        if leaf:
            v["leaves"][leaf] = {**rec, "forms": sorted(n["forms"] - {leaf})}; stat["leaves"] += 1
        else:
            v.update(rec); stat["values"] += 1
        members["/".join((region, a, value, leaf)).rstrip("/")] = sorted(set(n["rows"]))
    def up(attr, value, leaf):                                   # nearest ADMITTED generic of a leaf, or ""
        seen = set()
        while (attr, value, leaf) in gen and leaf not in seen:
            seen.add(leaf); leaf = gen[(attr, value, leaf)]
            if leaf in tree.get(attr.split(".")[0], {}).get(attr.split(".")[1], {}).get(value, {}).get("leaves", {}):
                return leaf
        return ""
    for region in tree:
        for a in tree[region]:
            for value, v in tree[region][a].items():
                for l, q in v["leaves"].items():
                    g = up(f"{region}.{a}", value, l)
                    if g:
                        q["parent"] = g; v["leaves"][g].setdefault("children", []).append(l); stat["linked_leaves"] += 1
    # invariants: every leaf has one path; nothing without support; no digit in a leaf
    for region in tree:
        for a in tree[region]:
            for value, v in tree[region][a].items():
                assert v["clips"] > 0, (region, a, value)
                assert all(not re.search(r"\d", l) and q["clips"] >= ADMIT["min_clips"] for l, q in v["leaves"].items())
    for region in tree:
        for a in tree[region]:
            ls = [l for v in tree[region][a].values() for l in v["leaves"]]
            assert len(ls) == len(set(ls)), (region, a, [l for l, k in collections.Counter(ls).items() if k > 1])
    d = ckpt(out, "tree.json").parent
    (d / "tree.json").write_text(json.dumps(tree, indent=1)); (d / "members.json").write_text(json.dumps(members))
    (d / "refused.json").write_text(json.dumps(refused, indent=1)); (d / "merges.json").write_text(json.dumps(mg.log, indent=1))
    (d / "aliases.json").write_text(json.dumps([{"attr": a, "value": v, "expert_leaf": e, "leaf": m} for (a, v, e), m in sorted(alias.items())], indent=1))
    (d / "moves.json").write_text(json.dumps([{"attr": a, "value": v, "leaf": l, "to": list(t)} for (a, v, l), t in sorted(move.items())], indent=1))
    (d / "reparent.json").write_text(json.dumps([{"attr": a, "value": v, "leaf": l, "to": t} for (a, v, l), t in sorted(reparent.items())], indent=1))
    if log:
        log.info("tree: %s", dict(stat))
    admitted = {k for k in nodes if "/".join((*k[0].split("."), k[1], k[2])).rstrip("/") in members}
    (d / "canon.json").write_text(json.dumps([{"attr": a, "value": v, "leaf": l, "canon": c, "admitted": (a, v, c) in admitted}
                                              for (a, v, l), c in sorted(leafmap.items())], indent=1))
    return {"tree": tree, "stat": stat, "refused": refused, "merges": mg.log, "alias": alias, "leafmap": leafmap, "admitted": admitted, "reparent": reparent, "move": move, "multi": multi,
            "pruned": set(LG["pruned"]) if LG else set(),
            "generic": {k: g for k, g in gen.items()},
            "node_clips": {"/".join((*k[0].split("."), k[1], k[2])).rstrip("/"): set(n["clips"]) for k, n in nodes.items()}}


def clean_labels(fired: list[dict], res: dict) -> list[dict]:
    """One row's fired labels -> the POST-PROCESSED labels (voted, expert names aliased, synonyms merged, generics linked):
    [{attr, value, leaf, node, admitted, generic, tools}]. `leaf` is the most specific ADMITTED node the row reaches: a
    `sports announcer` seen once trains `announcer`; with no admitted generic the row trains its parent value. `generic`
    = the admitted generics above that leaf, nearest first. This is what the index table carries."""
    out = {}; gen = res.get("generic", {}); labs = vote(fired)[0]
    for l in labs:
        leaf = l["leaf"] if LALM in l["tools"] else ONOMATOPOEIA.get(l["leaf"], l["leaf"])
        l["leaf"] = res["alias"].get((l["attr"], l["value"], leaf), leaf)
    for l in apply_moves(labs, res.get("move", {}), res.get("multi", set())):
        leaf = l["leaf"]
        l["value"] = res.get("reparent", {}).get((l["attr"], l["value"], leaf), l["value"]); leaf = res["leafmap"].get((l["attr"], l["value"], leaf), leaf)
        if leaf and (l["attr"], l["value"], leaf) not in res["admitted"]:          # a clip the tree never saw filed a KNOWN leaf under another parent
            hit = res.get("byname", {}).get((l["attr"], norm_key(leaf, l["attr"])))
            if hit:
                l["value"], leaf = hit
        if (l["attr"], l["value"], "") not in res["admitted"]:
            continue
        chain, x, seen = [], leaf, set()
        while x and x not in seen:
            seen.add(x)
            if (l["attr"], l["value"], x) in res["admitted"]:
                chain.append(x)
            x = gen.get((l["attr"], l["value"], x), "")
        r, a = l["attr"].split("."); served = chain[0] if chain else ""
        node = "/".join((r, a, l["value"]) + ((served,) if served else ()))
        out[(l["attr"], l["value"], served)] = {"attr": l["attr"], "value": l["value"], "leaf": served, "said": leaf, "node": node,
                                                 "admitted": True, "generic": chain[1:], "tools": sorted(l["tools"])}
    return list(out.values())


def selftest(schema):
    F = I.fired
    lab, st = vote([F("r", "speech.emotion", "angry", "emotion2vec", 0.9), F("r", "speech.emotion", "neutral", LALM, leaf="sarcastic"),
                    F("r", "speech.gender", "male voice", LALM), F("r", "speech.gender", "female voice", "wavlm_sv", 0.01),
                    F("r", "music.genre", "rock", LALM, leaf="garage rock"),
                    F("r", "sound.event", "animal", "panns", 0.8, leaf="bark"), F("r", "sound.event", "animal", LALM, leaf="dog barking"),
                    F("r", "sound.event", "unplaced", "panns", 0.9, leaf="zipper"), F("r", "sound.event", "tool", "panns", 0.1, leaf="drill")])
    got = {(l["attr"], l["value"], l["leaf"]) for l in lab}
    assert got == {("speech.emotion", "angry", ""), ("speech.gender", "male voice", ""), ("music.genre", "rock", "garage rock"),
                   ("sound.event", "animal", "bark"), ("sound.event", "animal", "dog barking")}, got
    assert st["disagree"] == 1 and st["leaf_orphaned"] == 1                          # sarcastic was written under neutral
    vocab = {"speech.emotion": {"angry"}, "music.genre": {"rock"}}
    n = lambda c, s, t=(LALM,): {"clips": set(range(c)), "sources": set(range(s)), "tools": set(t)}
    assert admit(("speech.emotion", "angry", "sarcastic"), n(2, 2), vocab) == ""
    assert admit(("speech.emotion", "angry", "sarcastic"), n(1, 1), vocab) == "seen on one clip"
    assert admit(("speech.emotion", "angry", "sarcastic"), n(3, 1), vocab) == "seen in one source"
    assert admit(("speech.language", "south asian", "hindi"), n(3, 1, ("whisper_lid",)), vocab) == ""
    assert admit(("speech.speaking_style", "singing", "rock"), n(5, 3), vocab) == "belongs to music.genre"
    assert admit(("music.tempo", "fast tempo", "fast"), n(5, 3), vocab).startswith("says no more")
    assert norm_key("announcement") == norm_key("announcing") == norm_key("announcer") and norm_key("describing a situation") == norm_key("describing situation")
    assert norm_key("sandpapering") == norm_key("sandpaper") and norm_key("chatting") == norm_key("chat") and norm_key("relaxed mood") == norm_key("relaxed")
    assert norm_key("rock music") == norm_key("rock") and norm_key("music") == "music" and norm_key("doorbell being pressed") == norm_key("doorbell pressed")
    assert norm_key("lesson on history") == norm_key("history lesson") and norm_key("soprano sax") == norm_key("soprano saxophone") != norm_key("tenor sax")
    assert norm_key("giving information") == norm_key("providing information") and norm_key("dog growling") != norm_key("animal growling")
    assert norm_key("sports announcer") != norm_key("news announcer") and norm_key("car horn") != norm_key("train horn")
    assert norm_key("fear") == norm_key("fearful") and norm_key("disgust") == norm_key("disgusted")
    assert norm_key("bark") != norm_key("barn") and norm_key("station") != norm_key("stationary") and norm_key("the") == "the"
    assert light_canon(["announcing", "announcement", "explaining"]) == {"announcing": "announcing", "announcement": "announcing", "explaining": "explaining"}
    rowf = [F("r", "sound.event", "animal", "panns", 0.8, leaf="bow-wow"), F("r", "sound.event", "animal", LALM, leaf="dog barking"),
            F("r", "sound.event", "animal", LALM, leaf="cat meowing"), F("r", "sound.event", "animal", "panns", 0.8, leaf="crow")]
    al = cofire_aliases([{"fired": rowf}], {"Bow-wow": {"Dog", "Animal", "Domestic animals, pets"}, "Crow": {"Bird", "Animal"}})
    assert al == {("sound.event", "animal", "bow-wow"): "dog barking"}, al        # `animal` (the parent) is not evidence; crow has no match
    res = {"alias": {("sound.event", "animal", "bow-wow"): "dog barking"}, "leafmap": {("sound.event", "animal", "dog barking"): "dog bark", ("sound.event", "animal", "cat meowing"): "cat meowing"},
           "admitted": {("sound.event", "animal", ""), ("sound.event", "animal", "dog bark")}}
    cl = {(x["leaf"], x["node"]) for x in clean_labels(rowf, res)}
    assert cl == {("dog bark", "sound/event/animal/dog bark"), ("", "sound/event/animal")}, cl
    res2 = {"alias": {}, "leafmap": {}, "generic": {("speech.speaker_role", "professional expert", "sports announcer"): "announcer"},
            "admitted": {("speech.speaker_role", "professional expert", ""), ("speech.speaker_role", "professional expert", "announcer")}}
    x = clean_labels([F("r", "speech.speaker_role", "professional expert", LALM, leaf="sports announcer")], res2)[0]
    assert (x["leaf"], x["said"], x["node"]) == ("announcer", "sports announcer", "speech/speaker_role/professional expert/announcer"), x   # seen once -> trains its generic
    assert not embed_may_merge("car horn", "train horn", 0.806, 0.8) and not embed_may_merge("train passing", "train approaching", 0.812, 0.8)
    assert not embed_may_merge("slight wind noise", "wind noise", 0.904, 0.8) and not embed_may_merge("sports announcer", "announcer", 0.919, 0.8)
    assert not embed_may_merge("animal growling", "dog growling", 0.911, 0.8)               # same length, but below the synonym bar
    assert embed_may_merge("movie reviewer", "film reviewer", 0.95, 0.8) and not embed_may_merge("lithuanian", "latvian", 0.856, 0.8)
    assert norm_key("shattering glass") == norm_key("glass shattering") and norm_key("bird wings flapping") == norm_key("bird flapping wings")
    assert Merger(model=False).canon("speech.language=eastern european", {"lithuanian": 2, "latvian": 1}) == {"lithuanian": "lithuanian", "latvian": "latvian"}
    nf = lambda c, t=(LALM,), forms=(): {"clips": set(range(c)), "sources": {1, 2}, "tools": set(t), "forms": set(forms)}
    assert admit(("speech.emotion", "neutral", "matter-of-fact"), nf(200), vocab, None, {"attr_share": 0.6, "parent_clips": 400, "expert_backed": False}).startswith("prompt example")
    assert admit(("speech.emotion", "neutral", "sarcastic"), nf(3), vocab, None, {"attr_share": 0.01, "parent_clips": 400, "expert_backed": False}) == ""
    assert admit(("sound.event", "animal", "dog barking"), nf(30, (LALM, "panns")), vocab, None, {"attr_share": 0.1, "parent_clips": 200, "expert_backed": True}) == ""
    assert admit(("sound.event", "household object", "door slamming"), nf(16), vocab, None, {"attr_share": 0.05, "parent_clips": 200, "expert_backed": False}).startswith("prompt example")
    assert admit(("speech.language", "western european", "english"), nf(178, ("whisper_lid",)), vocab, None, {"attr_share": 0.5, "parent_clips": 200, "expert_backed": True}) == ""
    assert admit(("speech.emotion", "neutral", "calm"), nf(113), vocab, None, {"attr_share": 0.1, "parent_clips": 250, "expert_backed": False}) == "covers most of its parent"
    assert light_canon(["american english", "standard american", "american accent", "british english"], "speech.accent_origin") == {
        "american english": "american english", "standard american": "american english", "american accent": "american english", "british english": "british english"}
    cx = {"languages": {"english", "french", "uzbek"}, "attr_share": 0.1, "parent_clips": 10, "expert_backed": False}
    assert admit(("speech.accent_origin", "native accent of the language", "standard english"), nf(40), vocab, None, cx).startswith("names the language")
    assert admit(("speech.accent_origin", "native accent of the language", "uzbek accent"), nf(7), vocab, None, cx).startswith("names the language")
    assert admit(("speech.accent_origin", "native accent of the language", "american english"), nf(70), vocab, None, cx) == ""
    assert admit(("speech.accent_origin", "regional accent", "castilian spanish"), nf(4), vocab, None, cx) == ""
    xx = clean_labels([F("r", "sound.event", "animal", "panns", 0.8, leaf="moo")], {"alias": {}, "leafmap": {}, "admitted": {("sound.event", "animal", ""), ("sound.event", "animal", "cow mooing")}})
    assert xx[0]["leaf"] == "cow mooing", xx                                                  # the expert's onomatopoeia lands on the descriptive node
    assert leaf_home({"sound.action_material": 9, "sound.event": 2}) == "sound.event" and leaf_home({"speech.intent": 3, "speech.scenario": 3}) == "speech.intent"
    assert leaf_home({"speech.topic": 1, "speech.scenario": 4}) == "speech.scenario"
    assert leaf_home({"speech.speaking_style": 31, "speech.speaker_role": 8}, "announcer") == "speech.speaker_role"
    assert leaf_home({"speech.speaking_style": 31, "speech.speaker_role": 8}, "formal") == "speech.speaking_style"
    assert admit(("sound.action_material", "air flow", "coughing"), n(3, 2), vocab, {("sound", "coughing"): "sound.event"}) == "belongs to sound.event"
    assert admit(("sound.event", "human non-speech", "coughing"), n(3, 2), vocab, {("sound", "coughing"): "sound.event"}) == ""
    assert admit(("speech.channel", "studio quality", "clear audio"), n(6, 3), vocab).startswith("says no more")
    assert admit(("sound.weather", "calm and dry", "clear weather"), n(3, 2), vocab).startswith("says no more") and admit(("speech.emotion", "neutral", "flat"), n(3, 2), vocab) == ""
    class _St:                                                        # stub encoder: names sharing their HEAD word sit at cosine 0.83
        def encode(self, names, **k):
            import numpy as np
            heads = sorted({n.split()[-1] for n in names}); M = np.zeros((len(names), len(heads) + len(names)))
            for i, n in enumerate(names):
                M[i, heads.index(n.split()[-1])] = 1.0; M[i, len(heads) + i] = 0.45
            return M / np.linalg.norm(M, axis=1, keepdims=True)
    mgx = Merger(model=False); mgx.st = _St()
    got = mgx.canon("speech.speaker_role=x", {"announcer": 9, "sports announcer": 5, "news announcer": 4, "heavy breathing": 3, "breathing": 2})
    assert all(k == v for k, v in got.items()), got                                          # nothing merges: none of these are synonyms
    assert mgx.generic == {("speech.speaker_role=x", "sports announcer"): "announcer", ("speech.speaker_role=x", "news announcer"): "announcer",
                           ("speech.speaker_role=x", "heavy breathing"): "breathing"}, mgx.generic
    mgy = Merger(model=False); mgy.st = _St(); mgy.canon("speech.topic=x", {"concert": 5, "concert announcement": 2, "announcement": 1})
    assert ("speech.topic=x", "concert announcement") not in mgy.generic, mgy.generic          # an announcement ABOUT a concert is not a kind of concert
    assert light_canon(["bird chirping", "birds chirping", "bird"]) == {"bird chirping": "bird chirping", "birds chirping": "bird chirping", "bird": "bird"}
