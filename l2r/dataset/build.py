"""Dataset generation: label the pool of training clips with the expert classifiers and the audio LLM annotator.

    python -m l2r.dataset.build chunk                 # assets/pool.jsonl -> chunk rows + chunk wavs      (<work>/sets/pool/rows.jsonl)
    python -m l2r.dataset.build experts               # expert classifiers (GPU)                           (fired_<expert>.jsonl)
    python -m l2r.dataset.build annotate              # audio LLM annotator: regions, then one form each   (fired_qwen3_omni.jsonl)
    python -m l2r.dataset.build music-properties      # audio LLM annotator: the music-property form       (music_form.jsonl)
    python -m l2r.dataset.build assemble              # the label table                                    (index.jsonl)
    python -m l2r.dataset.build report                # what each annotator labelled, agreement with free labels

`experts`, `annotate` and `music-properties` are resumable per clip; `--limit N` runs the first N clips; `--set NAME`
and `--manifest FILE` label another set of clips with the same pipeline. The manifest has one JSON object per
line: {"path": audio file relative to the data folder, "source": dataset name, "gold": labels that come for free}.
"""
from __future__ import annotations

import argparse
import collections

from l2r.common import audio, check, load_config, read_jsonl, resolve, setup_run, work
from l2r.dataset import experts as X, index as I, schema as S, segment as G

MANIFEST = "assets/pool.jsonl"


def stage_chunk(log, name: str, manifest: str, limit=0):
    import librosa
    schema = S.load_schema(); sample = read_jsonl(resolve(manifest))[: limit or None]
    rows, bad = [], 0
    for k, it in enumerate(sample):
        try:
            dur = librosa.get_duration(path=str(audio(it["path"])))
            rs = G.rows_for(it["path"], dur, schema, it.get("source", ""), {"gold": it.get("gold", {})})
            G.materialise(it["path"], rs)
        except Exception as e:                                                      # noqa: BLE001
            bad += 1; log.warning("unreadable %s (%s)", it["path"], e); continue
        rows += rs
        if k % 100 == 0:
            log.info("  chunk %d/%d  rows %d", k + 1, len(sample), len(rows))
    I.write_rows(name, rows)
    n = collections.Counter(r["grid"] for r in rows)
    log.info("rows %s from %d clips (%d unreadable)", dict(n), n["clip"], bad)
    check(n["clip"] >= 0.95 * len(sample), f"chunked >= 95% of the manifest ({n['clip']}/{len(sample)})", log)
    check(len({r["row"] for r in rows}) == len(rows), "row ids are unique", log)


def clip_votes(table):
    """(clip, annotator, attribute) -> Counter of classes / leaves over that clip's rows."""
    val = collections.defaultdict(collections.Counter); leaf = collections.defaultdict(collections.Counter)
    for r in table:
        for f in r["fired"]:
            val[(r["clip"], f["tool"], f["attr"])][f["value"]] += 1
            if f["leaf"]:
                leaf[(r["clip"], f["tool"], f["attr"])][f["leaf"]] += 1
    return val, leaf


def stage_report(log, name: str):
    table = I.assemble(name); val, leaf = clip_votes(table)
    gold = {r["clip"]: r.get("gold") or {} for r in table if r["grid"] == "clip"}
    tools = sorted({k[1] for k in val})
    L = [f"# Label table of set `{name}`\n", f"- clips {len(gold)} · rows {len(table)} · annotators {', '.join(tools)}",
         f"- rows with no label at all: {sum(not r['fired'] for r in table)}\n", "## What each annotator labelled\n",
         "| annotator | attribute | labels | clips | most frequent classes |", "|---|---|---|---|---|"]
    per = collections.defaultdict(collections.Counter); clips = collections.defaultdict(set)
    for (c, t, a), cn in val.items():
        per[(t, a)].update(cn); clips[(t, a)].add(c)
    for (t, a) in sorted(per):
        L.append(f"| {t} | {a} | {sum(per[(t, a)].values())} | {len(clips[(t, a)])} | " + ", ".join(f"{v} {n}" for v, n in per[(t, a)].most_common(6)) + " |")
    L += ["\n## Agreement with the labels that come with the datasets (clip level: the annotator's majority class over the clip's rows)\n",
          "| attribute | annotator | class accuracy | leaf accuracy | clips |", "|---|---|---|---|---|"]
    for a in sorted({a for g in gold.values() for a in g}):
        for t in tools:
            ok = okl = n = nl = 0
            for c, g in gold.items():
                if a not in g or (c, t, a) not in val:
                    continue
                gv = g[a] if isinstance(g[a], dict) else {"is": g[a]}
                top = val[(c, t, a)].most_common(1)[0][0]; n += 1; ok += top == gv.get("is", top)
                if "leaf" in gv:
                    lf = leaf[(c, t, a)].most_common(1); nl += 1; okl += bool(lf) and gv["leaf"] in lf[0][0]
            if n:
                L.append(f"| {a} | {t} | {ok / n:.2f} | {f'{okl / nl:.2f}' if nl else ''} | {n} |")
    out = work("reports", f"labels_{name}.md"); out.write_text("\n".join(L) + "\n"); log.info("wrote %s", out)


def selftest():
    from l2r.dataset import annotator
    from l2r.tree import build as T, guards
    s = S.load_schema(); G.selftest(s); I.selftest(); X.selftest(s); annotator.selftest(s); guards.selftest(); T.selftest(s)
    for r in S.REGIONS:
        assert '"notable"' in S.form_prompt(s, r)
    print("dataset + tree selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["chunk", "experts", "annotate", "music-properties", "assemble", "report", "selftest"])
    ap.add_argument("--set", default="pool"); ap.add_argument("--manifest", default=MANIFEST); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--experts", default=",".join(X.ORDER)); ap.add_argument("--chunks", type=int, default=4, help="chunk forms per region and clip")
    a = ap.parse_args()
    if a.stage == "selftest":
        return selftest()
    cfg = load_config(); _, log = setup_run(f"dataset_{a.set}_{a.stage}", cfg)
    if a.stage == "chunk":
        stage_chunk(log, a.set, a.manifest, a.limit)
    elif a.stage == "experts":
        X.run(cfg, log, S.load_schema(), a.set, a.experts.split(","), a.limit)
    elif a.stage == "annotate":
        from l2r.dataset import annotator
        annotator.run(cfg, log, S.load_schema(), a.set, a.chunks, a.limit)
    elif a.stage == "music-properties":
        from l2r.dataset import annotator
        annotator.music_properties(cfg, log, a.set, a.limit)
    elif a.stage == "assemble":
        log.info("label table: %d rows", len(I.assemble(a.set)))
    elif a.stage == "report":
        stage_report(log, a.set)


if __name__ == "__main__":
    main()
