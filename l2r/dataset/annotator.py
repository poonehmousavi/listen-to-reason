"""The audio LLM annotator (tree construction only; it never runs at inference).

It hears the first 30 s of a clip and fills the template in two steps: which regions are audible, then one form per
audible region (`schema.form_prompt`): for every attribute a listed class, "unsure" or "none", and optionally a short
free-text leaf. Chunk forms cover the attributes decided per chunk, on up to `n_chunks` chunks of each region.
The free-text `notable` answers are kept in `notes_qwen3_omni.jsonl`; they are not labels.

`music_properties` fills a second, clip-level form (period, vocal style, form, rhythm feel, dynamics) that
supervises the music-property heads.
"""
from __future__ import annotations

import collections
import json

from l2r.common import audio, read_jsonl
from l2r.dataset import index as I, schema as S, segment as G

TOOL = "qwen3_omni"
SYSTEM = "You are a careful audio annotator. You describe only what can be heard, and you say unsure when you cannot tell."
CLIP_S = 30.0

MUSIC_PROPERTIES = {
    "period": ["baroque", "classical era", "romantic era", "modern or contemporary classical", "contemporary popular music", "traditional or folk music", "not music"],
    "vocal_style": ["operatic or classical singing", "pop or rock singing", "rap or spoken rhythm", "choir or group singing", "falsetto or head voice",
                    "belting or powerful singing", "chant or recitation", "screaming or growling", "humming or wordless vocals", "no vocals"],
    "form": ["verse and chorus", "strophic (same music repeated)", "through-composed", "loop-based or repetitive", "theme and variations", "call and response",
             "improvised solo", "unclear"],
    "rhythm_feel": ["straight", "swung or shuffle", "syncopated", "free rhythm", "waltz-like triple", "march-like"],
    "dynamics": ["soft throughout", "loud throughout", "builds up", "fades out", "sudden changes", "steady medium level"],
}
MUSIC_PROMPT = ("Listen to the whole recording and fill this JSON. Choose EXACTLY one of the listed options where options are given; write \"unsure\" when you cannot tell. "
                "No text outside the JSON.\n{\n"
                + "".join(f' "music_{k}": "one of: {" | ".join(v)}",\n' for k, v in MUSIC_PROPERTIES.items()) +
                ' "music_lead_instrument": "the instrument or voice that carries the main melody, 1-3 words, or not music",\n'
                ' "music_key": "the key if you can tell it, e.g. A minor, F# major; else unsure",\n'
                ' "events": [{"sound": "a distinct non-speech sound event, 1-3 words", "position": "beginning | middle | end | throughout", "count": "once | twice | 3-5 times | many times", "duration": "brief | sustained"}]\n}'
                "\nList the events in the order they FIRST appear. Count separate occurrences carefully.")
MUSIC_FILE = "music_form.jsonl"                  # in the set folder: {"audio": path, "new": the parsed answer}


class Annotator:
    def __init__(self, cfg, log, max_new=900):
        import torch, transformers
        model_id = cfg["models"]["annotator"]
        self.torch = torch; self.max_new = max_new
        log.info("loading annotator %s", model_id)
        self.proc = transformers.Qwen3OmniMoeProcessor.from_pretrained(model_id)
        self.m = transformers.Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto").eval()
        self.m.disable_talker(); self.dt = next(self.m.parameters()).dtype

    def ask(self, wavs: list, prompts: list[str]) -> list[str]:
        convs = [[{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                  {"role": "user", "content": [{"type": "audio", "audio": "x"}, {"type": "text", "text": p}]}] for p in prompts]
        texts = [self.proc.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in convs]
        inp = self.proc(text=texts, audio=list(wavs), return_tensors="pt", padding=True, use_audio_in_video=False).to(self.m.device)
        enc = {k: (v.to(self.dt) if hasattr(v, "dtype") and v.dtype.is_floating_point else v) for k, v in inp.items()}
        with self.torch.no_grad():
            g = self.m.generate(**enc, max_new_tokens=self.max_new, do_sample=False, return_audio=False, use_audio_in_video=False)
        ids = g[0] if isinstance(g, (tuple, list)) else g
        return [r.strip() for r in self.proc.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]


def to_fired(row: str, region: str, v: dict) -> list[dict]:
    """One validated form -> label records. A multiple-choice field yields one record per (class, leaf) pair."""
    out = []
    for f, val in v["values"].items():
        pairs = [tuple(q) for q in (v.get("leaves") or {}).get(f, [])]
        vals = val if isinstance(val, list) else [val]
        with_leaf = collections.Counter(p[0] for p in pairs)
        for x, leaf in pairs:
            out.append(I.fired(row, f"{region}.{f}", x, TOOL, leaf=leaf))
        for x in collections.Counter(vals) - with_leaf:                       # classes that came without a leaf
            out.append(I.fired(row, f"{region}.{f}", x, TOOL))
    return out


def run(cfg, log, schema, name: str, n_chunks=4, limit=0, batch=4):
    """Annotate every clip of set `name` (resumable per clip)."""
    import librosa
    by = collections.OrderedDict()
    for r in I.rows(name):
        by.setdefault(r["clip"], []).append(r)
    sh = I.Shard(name, TOOL); todo = [c for c in list(by)[: limit or None] if c not in sh.done]
    log.info("annotator: %d clips to do (%d done)", len(todo), len(sh.done))
    if not todo:
        return
    A = Annotator(cfg, log); notes = (I.sdir(name) / f"notes_{TOOL}.jsonl").open("a"); stat = collections.Counter()
    region_value = {"speech": "speech present", "music": "music present", "sound": "environmental sound present"}
    for k, c in enumerate(todo):
        rows = by[c]; clip = rows[0]
        try:
            y, _ = librosa.load(str(audio(clip["audio_path"])), sr=G.SR, mono=True, duration=CLIP_S)
        except Exception as e:                                                          # noqa: BLE001
            log.warning("unreadable %s (%s)", clip["audio_path"], e); continue
        fam = S.parse_json(A.ask([y], [S.REGION_PROMPT])[0]) or {}
        regions = [r for r in S.REGIONS if fam.get(r) is True]
        recs = [I.fired(clip["row"], S.GATE, region_value[r], TOOL) for r in regions]
        stat["region_fail"] += not fam
        jobs = [(clip["row"], r, y, False) for r in regions]                           # clip forms: every attribute
        for r in regions:                                                               # chunk forms: attributes decided per chunk
            if not S.fields(schema, r, local_only=True):
                continue
            g = [x for x in rows if x["grid"] == G.GRID_OF[r] and x["t1"] <= CLIP_S + 1e-6][:n_chunks]
            if len(g) > 1:                                                              # a one-chunk clip is its own clip form
                jobs += [(x["row"], r, y[int(x["t0"] * G.SR): int(x["t1"] * G.SR)], True) for x in g]
        for i in range(0, len(jobs), batch):
            part = jobs[i:i + batch]
            raws = A.ask([j[2] for j in part], [S.form_prompt(schema, j[1], local_only=j[3]) for j in part])
            for (row, r, _y, loc), raw in zip(part, raws):
                d = S.parse_json(raw); stat["forms"] += 1
                if not d:
                    stat["parse_fail"] += 1; continue
                v = S.validate(schema, r, d, local_only=loc); recs += to_fired(row, r, v)
                stat["off"] += len(v["off"]); stat["unsure"] += len(v["unsure"])
                if v["notable"] or v["off"]:
                    notes.write(json.dumps({"row": row, "region": r, "notable": v["notable"], "off": v["off"]}, ensure_ascii=False) + "\n")
        sh.add(c, recs); notes.flush()
        if k % 10 == 0:
            log.info("  annotator %d/%d  %s  %s", k + 1, len(todo), clip.get("source"), dict(stat))
    sh.close(); log.info("annotator done: %s", dict(stat))


def music_properties(cfg, log, name: str, limit=0, batch=2, max_sec=60.0):
    """The music-property form on every clip of set `name` (resumable) -> <set>/music_form.jsonl."""
    import librosa
    out = I.sdir(name) / MUSIC_FILE
    done = {r["audio"] for r in read_jsonl(out)}
    todo = [r["audio_path"] for r in I.rows(name) if r["grid"] == "clip" and r["audio_path"] not in done][: limit or None]
    log.info("music-property form: %d clips to do (%d done)", len(todo), len(done))
    if not todo:
        return
    A = Annotator(cfg, log, max_new=600); fail = 0
    with out.open("a") as fh:
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]; ys = [librosa.load(str(audio(p)), sr=G.SR, mono=True, duration=max_sec)[0] for p in part]
            try:
                raws = A.ask(ys, [MUSIC_PROMPT] * len(part))
            except Exception as e:                                # noqa: BLE001  (e.g. out of memory on two long clips): one by one
                log.warning("batch failed (%s); one by one", str(e)[:120]); raws = []
                for y in ys:
                    try:
                        raws.append(A.ask([y], [MUSIC_PROMPT])[0])
                    except Exception:                             # noqa: BLE001
                        raws.append("")
            for p, raw in zip(part, raws):
                d = S.parse_json(raw) or {}; fail += not d
                fh.write(json.dumps({"audio": p, "new": d}, ensure_ascii=False) + "\n"); fh.flush()
            if (i // batch) % 10 == 0:
                log.info("  %d/%d  parse failures %d", i + len(part), len(todo), fail)
    log.info("music-property form done: %d clips, %d parse failures", len(todo), fail)


def selftest(schema):
    v = S.validate(schema, "sound", {"event": [{"is": "animal", "detail": "dog barking"}, {"is": "animal", "detail": "cat meowing"},
                                               {"is": "tool", "detail": ""}], "proximity": "approaching"})
    f = to_fired("r", "sound", v)
    assert sorted((x["attr"], x["value"], x["leaf"]) for x in f) == [("sound.event", "animal", "cat meowing"), ("sound.event", "animal", "dog barking"),
                                                                       ("sound.event", "tool", ""), ("sound.proximity", "approaching", "")], f
    v = S.validate(schema, "speech", {"emotion": {"is": "neutral", "detail": "sarcastic"}, "gender": "female voice"})
    assert {(x["attr"], x["value"], x["leaf"]) for x in to_fired("r", "speech", v)} == {("speech.emotion", "neutral", "sarcastic"), ("speech.gender", "female voice", "")}
    assert '"music_period": "one of: baroque | classical era' in MUSIC_PROMPT
