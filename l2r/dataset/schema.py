"""Tree schema: the closed part of the tree (region -> attribute -> class) and the annotation template.

`configs/tree_schema.yaml` lists, per region, the attributes and their closed classes. Attributes marked
`leaf: open` are answered as {"is": <class>, "detail": <open descriptive leaf>}. Nothing here loads a model.
"""
from __future__ import annotations

import json
import re

import yaml

from l2r.common import resolve

SCHEMA = "configs/tree_schema.yaml"
REGIONS = ("speech", "music", "sound")
GATE = "cross.families_present"
REGION_OF = {"speech present": "speech", "music present": "music", "environmental sound present": "sound"}
REGION_WORD = {"speech": "the SPEECH", "music": "the MUSIC", "sound": "the NON-SPEECH, NON-MUSIC SOUNDS",
               "cross": "the MIXTURE as a whole"}
REGION_PROMPT = ('Listen to the audio. Which of these are clearly audible? Answer with a JSON object only, e.g. '
                 '{"speech": true, "music": false, "sound": true}. "sound" means environmental or object sounds '
                 'other than speech and music. SINGING is part of music, not speech: set "speech" to true only when '
                 'someone is TALKING.')


def load_schema(path=None) -> dict:
    return yaml.safe_load(resolve(path or SCHEMA).read_text())


def fields(schema: dict, region: str, local_only=False) -> list[tuple[str, dict]]:
    """The attributes of a region; `local_only` keeps those decided per chunk."""
    return [(name, a) for name, a in schema["regions"][region]["attributes"].items()
            if not (local_only and a.get("scope") != "local")]


def form_prompt(schema: dict, region: str, local_only=False) -> str:
    """The template the annotator fills for one region."""
    lines = []
    for name, a in fields(schema, region, local_only):
        v = a["values"]
        if a.get("leaf") == "open":
            multi = " [a LIST of such objects, one per thing you hear]" if a.get("multi") else ""
            hint = f' -- detail = {a["leaf_hint"]}' if a.get("leaf_hint") else ""
            na = f' ({a["na_hint"]})' if a.get("na_hint") else ""
            lines.append(f'- "{name}"{multi}{na}: {{"is": one of ' + " | ".join(v) + f', "detail": "..."}}{hint}')
        else:
            multi = " [list ALL that apply]" if a.get("multi") else ""
            lines.append(f'- "{name}"{multi}: ' + " | ".join(v))
    return (f"Listen to the audio and fill in this form about {REGION_WORD[region]} in it. For each field choose "
            f"exactly one of the listed values (copy it verbatim), or \"unsure\" if you cannot tell from the audio, "
            f"or \"none\" if the field does not apply. Do not guess.\n"
            f"Where a field is an object, \"is\" is the listed value and \"detail\" is the SINGLE most specific term "
            f"(1-3 plain words of YOUR OWN, no \"and\") for what you hear under that value. A detail describes "
            f"THAT FIELD ONLY -- never the background, the instruments or the words when the field is about "
            f"something else. No numbers in a detail. An empty detail \"\" is the RIGHT answer whenever you can add nothing "
            f"specific -- never fill it with a generic word.\n" + "\n".join(lines) +
            '\n- "notable": one short sentence about anything important in the audio that this form does not '
            'cover, or "" if nothing.\nAnswer with one JSON object only, no other text.')


def parse_json(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def clean_leaf(leaf, parent: str) -> str:
    """An open leaf is descriptive: one term of at most four words, no digits, and it must say more than its
    class. Anything else is dropped, never repaired."""
    x = re.split(r" and | with |[,;/]", str(leaf or "").lower())[0]
    x = re.sub(r"[^a-z' -]", " ", x.replace("_", " "))
    x = re.sub(r"\s+", " ", x).strip(" -'")
    if not x or x in ("unsure", "none", "n a", parent) or re.search(r"\d", str(leaf)) or len(x.split()) > 4:
        return ""
    return x


def _pairs(got) -> list[tuple[str, str]]:
    """A field's raw answer -> [(class, detail)]. Accepts a string, an object, or a list of either."""
    out = []
    for g in (got if isinstance(got, list) else [got]):
        if isinstance(g, dict):
            out.append((str(g.get("is") or "").strip().lower(), g.get("detail") or ""))
        else:
            out += [(x.strip().lower(), "") for x in str(g).split(",") if x.strip()]
    return out


def validate(schema: dict, region: str, d: dict, local_only=False) -> dict:
    """A filled template -> {"values": {field: class | [classes]}, "leaves": {field: [[class, leaf]]},
    "unsure": [...], "off": {field: answers outside the listed classes}, "missing": [...], "notable": str}."""
    out = {"values": {}, "leaves": {}, "unsure": [], "off": {}, "missing": [], "notable": str(d.get("notable") or "").strip()}
    for name, a in fields(schema, region, local_only):
        if name not in d:
            out["missing"].append(name); continue
        pairs = _pairs(d[name])
        lst = [v for v, _ in pairs if v]
        allowed = {v.lower() for v in a["values"]}
        ok = [x for x in lst if x in allowed]
        bad = [x for x in lst if x not in allowed and x not in ("unsure", "none")]
        if bad:
            out["off"][name] = bad
        if "unsure" in lst and not ok:
            out["unsure"].append(name)
        if ok:
            out["values"][name] = ok if a.get("multi") else ok[0]
            lv = [[v, clean_leaf(det, v)] for v, det in pairs if v in allowed and (a.get("multi") or v == ok[0])]
            lv = [q for q in lv if q[1]]
            if lv:
                out["leaves"][name] = lv
    return out
