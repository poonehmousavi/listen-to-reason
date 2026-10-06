"""The reader's context: the served nodes of a clip as text, the event timeline and the transcript.

    python -m l2r.retrieve.context --benchmark mmau      ->  <work>/contexts/mmau.json

Nodes are collected over all rows of the clip (clip, 10 s chunks, 3 s chunks), each node once, in the order it
was first reached, and grouped by attribute:

    Audio analysis of the recording:
    Audio graph nodes matched to this recording: speech gender: female voice; sound animal: dog (Bark).
    Sound events over time (sound tree nodes): ...            (only for questions about order, duration or counts)
    Transcript: "[0:00.0] ..."                                (only when speech or singing is heard)

Time spans and node probabilities are not written into the context; they stay in the traversal trace
(`<work>/nodes/<benchmark>.json`).

`python -m l2r.retrieve.context --benchmark mmau --run` runs every step that is missing first (transcripts,
question classifier, traversal, event timeline).
"""
from __future__ import annotations

import argparse
import collections
import re

from l2r import benchmarks
from l2r.common import load_config, load_json, save_json, setup_run, work

HEAD = "Audio analysis of the recording:"
NODES = "Audio graph nodes matched to this recording:"
MAX_NODES_CHARS = 2400
MAX_TIMELINE_CHARS = 600
_TRANSCRIPT = re.compile(r"\n?Transcript:.*", re.S)
_VOCALS = re.compile(r"music vocals: ([^;\n]*)")


def context_path(benchmark: str, variant: str = ""):
    return work("contexts", f"{benchmark}{'_' + variant if variant else ''}.json")


def node_text(n: dict) -> str:
    """`class`, or `class (leaf)` when a leaf is served."""
    return f"{n['value']} ({n['leaf']})" if n["leaf"] and n["leaf"] != n["value"] else n["value"]


def clause_head(attr: str) -> str:
    """`sound.action_material` -> `sound action material`."""
    region, name = attr.split(".")
    return f"{region} {name.replace('_', ' ')}"


def nodes_line(rows: list[dict], last=()) -> str:
    """The nodes of a clip: `region attribute: class (leaf), class; region attribute: ...`. The attributes in
    `last` (new-domain attributes) are listed after all others."""
    by_attr = collections.OrderedDict()
    seen = set()
    for r in rows:
        for n in r["nodes"]:
            k = (n["attr"], n["value"], n["leaf"])
            if k not in seen:
                seen.add(k)
                by_attr.setdefault(n["attr"], []).append(n)
    clause = lambda a: f"{clause_head(a)}: {', '.join(dict.fromkeys(node_text(n) for n in by_attr[a]))}"
    s = "; ".join(clause(a) for a in by_attr if a not in last)[:MAX_NODES_CHARS].strip()
    domain = [clause(a) for a in by_attr if a in last]
    return "; ".join(([s[:-1] if s.endswith(".") else s] if s else []) + domain) if domain else s


def drop_clauses(text: str, attrs) -> str:
    """Remove the clauses of these attributes from the nodes line of a context."""
    if NODES not in text:
        return text
    heads = tuple(clause_head(a) + ":" for a in attrs)
    a0 = text.index(NODES) + len(NODES); e0 = text.find("\n", a0); e0 = len(text) if e0 < 0 else e0
    line = text[a0:e0]; dot = line.rstrip().endswith(".")
    clauses = line.rstrip().rstrip(".").split("; ")
    keep = [c for c in clauses if not c.strip().startswith(heads)]
    if len(keep) == len(clauses):
        return text
    new = "; ".join(keep)
    new = (new if new.startswith(" ") else " " + new.lstrip()) + ("." if dot and keep else "")
    return text[:a0] + new + text[e0:]


def hears_voice(text: str) -> bool:
    """The router heard speech, or singing (a served vocals node other than `no vocals`)."""
    if "speech present" in text:
        return True
    return any(v.strip(" .") and v.strip(" .") != "no vocals" for m in _VOCALS.finditer(text) for v in m.group(1).split(","))


def render(rows: list[dict], timeline: str = "", transcript: str | None = None, gate: bool = True, last=()) -> str:
    """The context of one item. The transcript is kept only when speech or singing is heard (`gate`)."""
    lines = [HEAD]
    s = nodes_line(rows, last)
    if s:
        lines.append(f"{NODES} {s}" + ("" if s.endswith(".") else "."))
    if timeline:
        lines.append(timeline.strip()[:MAX_TIMELINE_CHARS] + ".")
    if transcript is not None:
        lines.append(f'Transcript: "{transcript}"')
    if len(lines) == 1:
        return ""
    text = "\n".join(lines) + "\n\n"
    if gate and not hears_voice(text):
        text = _TRANSCRIPT.sub("", text)
    return text


def build(items: list[dict], nodes: dict, transcripts: dict, timelines: dict, scores: dict, cfg=None, sound_attrs=None, gate: bool = True) -> list[dict]:
    """One record per item: the item's fields and `ctx = {"ours": context, "blind": ""}`."""
    cfg = cfg or load_config()
    if sound_attrs is None:
        from l2r.router import sound
        sound_attrs = list(sound.load_nodes())
    fallback = set(nodes.get("sound_by_question", ())); tau = cfg["serve"]["question_fallback"]; last = set(nodes.get("domain_attributes", ()))
    out = []
    for i, it in enumerate(items):
        text = render(nodes["clips"].get(it["audio"], []), timelines.get(str(it["id"]), ""), transcripts.get(it["audio"]), gate, last)
        # sound nodes opened by the question fallback are served only to the items whose own question asks about a sound
        if it["audio"] in fallback and max(scores.get(str(it.get("id", i)), {}).values() or [0]) < tau:
            text = drop_clauses(text, sound_attrs)
        rec = {k: it.get(k) for k in ("id", "question", "choices_str", "options_norm", "gold", "group", "category", "sub_category", "audio")}
        out.append({**rec, "ctx": {"ours": text, "blind": ""}})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", required=True, choices=benchmarks.BENCHMARKS)
    ap.add_argument("--run", action="store_true", help="run the steps whose output is missing (transcripts, question classifier, traversal, timeline)")
    ap.add_argument("--domains", default="", help="with --run: new-domain routers to add to the traversal, e.g. birds_k5")
    ap.add_argument("--tree-only", action="store_true", help="also write the context without the transcript -> <benchmark>_tree_only.json")
    a = ap.parse_args()
    cfg = load_config(); _, log = setup_run(f"context_{a.benchmark}", cfg)
    from l2r.retrieve import asr, serve, timeline
    from l2r.router import question
    items = benchmarks.load(a.benchmark); b = a.benchmark
    transcripts = asr.transcribe([it["audio"] for it in items], b, cfg, log) if a.run else load_json(asr.cache_path(b))
    qp = work("question", f"{b}.json")
    scores = question.predict_benchmarks([b], log)[b] if a.run and not qp.exists() else load_json(qp)
    if a.run and not serve.nodes_path(b).exists():
        serve.Retriever(cfg, log, [d for d in a.domains.split(",") if d]).run(b, items, transcripts, scores)
    nodes = load_json(serve.nodes_path(b))
    detected = timeline.detect(b, items, cfg, log) if a.run else load_json(timeline.events_path(b))
    out = build(items, nodes, transcripts, timeline.lines(items, detected), scores, cfg)
    path = context_path(b)
    save_json(out, path, indent=None)
    if a.tree_only:
        save_json([{**x, "ctx": {**x["ctx"], "ours": _TRANSCRIPT.sub("", x["ctx"]["ours"]).rstrip()}} for x in out], context_path(b, "tree_only"), indent=None)
    log.info("%s: %d contexts (%d with a transcript, %d with a timeline) -> %s", b, len(out), sum("Transcript:" in x["ctx"]["ours"] for x in out),
             sum(timeline.HEAD in x["ctx"]["ours"] for x in out), path)


if __name__ == "__main__":
    main()
