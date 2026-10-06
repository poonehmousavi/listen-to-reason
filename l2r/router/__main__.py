"""Router training: every head of the traversal, on the frozen encoders' cached chunk embeddings.

    python -m l2r.router                     # every stage in order; a finished stage is skipped
    python -m l2r.router --from train:gate

Stages (heads under <checkpoints>/, everything else under <work>/):
  pairs                 (chunk, attribute) training pairs of the pool (l2r.router.data)          -> sets/pool/pairs.jsonl
  musiccaps             the MusicCaps + IRMAS label set (l2r.router.labelled_music)               -> sets/musiccaps/
  features:<encoder>    chunk embeddings of one frozen encoder (l2r.router.features)              -> features/<encoder>__<set>.h5
  train:<router>        one region router of the `routers:` config section (l2r.router.train)     -> routers/<router>.pt
  calibrate             temperature and held-out skill per attribute                              -> routers/calibration.json
  language              language from the transcript's text (l2r.router.language)                -> language/text_head.pkl
  audioset-*            AudioSet / FSD50K / ESC-50 clips, BEATs and PANNs features, strong labels -> audioset/
  sound-*               sound identity heads: VGGSound teacher, targets, extra clips, training     -> sound/heads_{clap,beats}.pt
  event-head            the event-timeline head on PANNs frames (l2r.router.event_head)           -> events/head.pt
  music-*               music-property heads on MuQ-MuLan (l2r.router.music_properties)           -> music_properties/heads.pt
  question-*            the question classifier of the sound fallback (l2r.router.question)       -> question/model.pkl
"""
from __future__ import annotations

from l2r.common import ckpt, load_config, work
from l2r.dataset import index as I
from l2r.pipeline import Step, parser, run
from l2r.router import features as F, labelled_music as LM, model
from l2r.router.data import PAIRS


def steps() -> list[Step]:
    cfg = load_config(); R = cfg["routers"]
    sets = {}                                     # encoder -> the sets it embeds (the pool, plus the labelled music set)
    for n, s in R.items():
        for x in s.get("sets", "pool").split(","):
            sets.setdefault(s["encoder"], []).append(x.split(":")[0]) if x.split(":")[0] not in sets.get(s["encoder"], []) else None
    S = [Step("pairs", ["l2r.router.data", "--set", "pool"], [I.sdir("pool") / PAIRS]),
         Step("musiccaps", ["l2r.router.labelled_music"], [I.sdir(LM.NAME) / "info.json"])]
    S += [Step(f"features:{e}", ["l2r.router.features", "--encoder", e, "--sets", ",".join(ss)], [F.store_path(e, x) for x in ss]) for e, ss in sets.items()]
    S += [Step(f"train:{n}", ["l2r.router.train", "--name", n], [model.path(n)]) for n in R]
    A = work("audioset", "x").parent
    S += [Step("calibrate", ["l2r.router.calibrate"], [ckpt("routers", "calibration.json")]),
          Step("language", ["l2r.router.language"], [ckpt("language", "text_head.pkl")]),
          Step("audioset-items", ["l2r.router.audioset", "items"], [A / "items.json"]),
          Step("audioset-beats", ["l2r.router.audioset", "beats"], [A / "beats_clip.npy"]),
          Step("audioset-strong", ["l2r.router.audioset", "strong"], [A / "strong_y.npy", A / "panns_fc1.npy"]),
          Step("sound-teacher", ["l2r.router.sound", "teacher"], [A / "vggsound_teacher.npz"]),
          Step("sound-targets", ["l2r.router.sound", "targets"], [A / "sound_targets.npz", ckpt("sound", "nodes.json")]),
          Step("sound-extra", ["l2r.router.sound", "extra"], [A / "sound_extra.npz"]),
          Step("sound-train:clap", ["l2r.router.sound", "train", "--encoder", "clap"], [ckpt("sound", "heads_clap.pt")]),
          Step("sound-train:beats", ["l2r.router.sound", "train", "--encoder", "beats"], [ckpt("sound", "heads_beats.pt")]),
          Step("event-head", ["l2r.router.event_head"], [ckpt("events", "head.pt")]),
          Step("music-embed", ["l2r.router.music_properties", "embed"], [work("music_properties", "pool.npz")]),
          Step("music-train", ["l2r.router.music_properties", "train"], [ckpt("music_properties", "heads.pt")]),
          Step("question-data", ["l2r.router.question", "data"], [work("question", "data.json")]),
          Step("question-train", ["l2r.router.question", "train"], [ckpt("question", "model.pkl")])]
    return S


if __name__ == "__main__":
    run(steps(), parser("router", __doc__).parse_args())
