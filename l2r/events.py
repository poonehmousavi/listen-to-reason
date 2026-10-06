"""Sound events over time: PANNs frame scores -> events -> the timeline line given to the reader.

Three parts:
  * the wording gate: the timeline is served only for questions about the order, duration or number of events;
  * `segment`: frame-level class probabilities -> events (hysteresis, gap merging, ontology ancestors dropped);
  * `render`: events -> one line, e.g.
        [0.0-2.5s] Dog; [2.4-4.0s] Shatter Occurrences: Dog x1, Shatter x1 (2 distinct sounds).
    with the derived line (longest / order / occurrences) the question asks for, so the reader does not have to
    compute durations or counts from the spans.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from l2r.common import data, load_config

FRAME_S = 0.01                      # PANNs frame hop (320 samples at 32 kHz)
MAX_S = 60.0                        # only the first minute of a clip is scored

# Classes that describe the room or the recording rather than an event.
STOPLIST = {
    "Sound", "Noise", "Environmental noise", "Inside, small room", "Inside, large room or hall",
    "Inside, public space", "Outside, urban or manmade", "Outside, rural or natural",
    "Reverberation", "Echo", "Static", "Mains hum", "Distortion", "Sidetone", "Cacophony",
    "White noise", "Pink noise", "Throbbing", "Vibration", "Silence", "Sound effect",
    "Field recording", "Background noise", "Hum",
}

# Segmentation: enter an event at tau_on, leave it at tau_off, merge gaps < gap s, drop events < min_dur s,
# keep the k strongest events, median-smooth the scores over `smooth` frames.
DEFAULTS = {"tau_on": 0.10, "tau_off": 0.05, "min_dur": 0.20, "gap": 0.30, "k": 10, "smooth": 5}


@dataclass(frozen=True)
class Event:
    label: str
    t0: float
    t1: float
    peak: float

    @property
    def dur(self) -> float:
        return self.t1 - self.t0


# --------------------------------------------------------------------------- the wording gate
_SPAN = re.compile(
    r"(?:from|between)\s+(?P<a>\d+(?::\d+)?(?:\.\d+)?)\s*(?:to|and|until|-|–)\s*"
    r"(?P<b>\d+(?::\d+)?(?:\.\d+)?)", re.I)
_BARE = re.compile(r"\b(?P<a>\d+(?::\d+)?(?:\.\d+)?)\s*(?:to|-|–)\s*(?P<b>\d+(?::\d+)?(?:\.\d+)?)"
                   r"\s*(?:second|sec|s)\b", re.I)
_FIRST_N = re.compile(r"\b(?:first|opening)\s+(?P<n>\d+(?:\.\d+)?)\s*(?:second|sec|s)\b", re.I)
_LAST_N = re.compile(r"\b(?:last|final)\s+(?P<n>\d+(?:\.\d+)?)\s*(?:second|sec|s)\b", re.I)
_REL = re.compile(
    r"toward(?:s)?\s+the\s+end|\bat\s+the\s+(?:very\s+)?end\b|\bends?\s+(?:with|by)\b|\bin\s+the\s+end\b"
    r"|\bfinal\s+(?:part|section|portion)\b|\bclosing\s+(?:part|section|portion)\b|\blate\s+in\s+the\b"
    r"|\bat\s+the\s+(?:beginning|start)\b|\bbegins?\s+(?:with|by)\b|\bstarts?\s+with\b|\bin\s+the\s+beginning\b"
    r"|\binitial\s+(?:part|section|portion)\b|\bopening\s+(?:part|section|portion)\b|\bearly\s+in\s+the\b"
    r"|\bright\s+at\s+the\s+beginning\b|\bin\s+the\s+middle\b|\bmiddle\s+(?:part|section|portion)\b"
    r"|\bfirst\s+half\b|\bsecond\s+half\b|\blast\s+half\b", re.I)
_TEMPORAL = re.compile(
    r"\b(order|sequence|sequential(?:ly)?|followed by|follows|how many times|"
    r"at what (?:point|time|moment)|when (?:does|do|is|did|was)|time ?frames?|timestamps?|"
    r"(?:first|last|final|second|third|next|previous|earlier|later|opening|closing)\s+"
    r"(?:sound|sounds|noise|event|speaker|voice|note|chord|instrument|half|part|segment|section|"
    r"thing|word|sentence|phrase|clip|beat|measure|bar)|"
    r"(?:at|towards?|near|by|in)\s+the\s+(?:beginning|start|end|middle|outset|close)|"
    r"(?:beginning|start|end|middle)\s+of\s+the\s+(?:audio|clip|recording|track|song|sound)|"
    r"(?:before|after|prior to|subsequent to)\s+(?:the|that|this|a|an|hearing|it)\b|"
    r"transition(?:s|ed|ing)?|precede[sd]?|subsequent(?:ly)?|"
    r"\d+(?:\.\d+)?\s*(?:s|sec|seconds?)\b|\d+:\d\d)\b", re.I)
_COUNT = re.compile(r"\b(how many|number of|count(?:ed|ing)?|occurrences?|"
                    r"times (?:does|do|is|are|did|was|were|can))\b", re.I)
_EVENT = re.compile(r"\b(longest|shortest|longer|shorter|duration|"
                    r"cannot be heard|can not be heard|can't be heard|not (?:be )?heard|"
                    r"not present|absent|background|foreground|"
                    r"simultaneous(?:ly)?|at the same time|overlap(?:ping|s)?|"
                    r"(?:first|last|final)\s+(?:sound|noise|event|thing)|"
                    r"(?:begin|start|end)(?:s|ning|ing)?\s+(?:of|with)\s+the)\b", re.I)
# Wording that names something a sound-event detector cannot see: a musical object or a linguistic one.
_NOT_EVENT = re.compile(
    r"\b(chords?|notes?|note durations?|key change|keys?|modulation|tonic|scale|tempo|bpm|"
    r"beats?|bars?|measures?|rhythm|melod\w*|harmon\w*|texture|inversion|12-tone|passage|"
    r"strings?|phonemes?|syllables?|stressed|unstressed|words?|sentences?|speakers?|"
    r"contributors?|people|persons?|voices|conversation|discussion|understand|sarcastic|"
    r"said|says|asking|asked)\b", re.I)


def _sec(t: str) -> float:
    if ":" in t:
        m, s = t.split(":", 1)
        return int(m) * 60 + float(s)
    return float(t)


def _names_time(q: str) -> bool:
    """The question names a time span or a position in the clip ("from 2.9 to 4.3", "the first 5 seconds", "at the end")."""
    m = _SPAN.search(q) or _BARE.search(q)
    if m:
        a, b = _sec(m.group("a")), _sec(m.group("b"))
        if b > a >= 0 and (b - a) <= 600:
            return True
    return bool(_FIRST_N.search(q) or _LAST_N.search(q) or _REL.search(q))


def is_event_question(question: str) -> bool:
    """The wording asks about time, order, counts or the set of events."""
    q = question or ""
    return bool(_names_time(q) or _TEMPORAL.search(q) or _COUNT.search(q) or _EVENT.search(q))


def asks_timeline(question: str) -> bool:
    """The gate of the event timeline: event wording, and no musical or linguistic object named."""
    return is_event_question(question) and not _NOT_EVENT.search(question or "")


def _wants(question: str) -> set[str]:
    """Which derived line(s) the wording asks for."""
    q = (question or "").lower()
    out = set()
    if re.search(r"\b(longest|shortest|longer|shorter|duration|lasts|lasted|how long)\b", q):
        out.add("duration")
    if re.search(r"\b(order of|in (?:what|which) order|sequence|sequential(?:ly)?|"
                 r"(?:first|last|final|next|previous|earlier|later)\s+(?:sound|noise|event|thing|"
                 r"speaker|voice|one)|(?:occur|come|hear|appear|play|happen)\w*\s+(?:first|last|next)|"
                 r"before|after|followed|follows|precede|subsequent|"
                 r"(?:at|towards?|near) the (?:beginning|start|end)|"
                 r"(?:beginning|start|end) of the)\b", q):
        out.add("order")
    if _COUNT.search(q):
        out.add("count")
    return out


# --------------------------------------------------------------------------- frame scores -> events
def _medfilt(p: np.ndarray, size: int) -> np.ndarray:
    if size <= 1 or len(p) < size:
        return p
    from scipy.ndimage import median_filter
    return median_filter(p, size=size, mode="nearest")


def _iou(a: Event, b: Event) -> float:
    inter = min(a.t1, b.t1) - max(a.t0, b.t0)
    return max(0.0, inter) / max(1e-6, max(a.t1, b.t1) - min(a.t0, b.t0))


def collapse_ancestors(events: list[Event], ancestors: dict[str, set] | None) -> list[Event]:
    """Drop an event whose label is an ontology ancestor of a co-occurring event's label: `Dog; Animal; Bark`
    over one span is one sound, and the more specific label is kept."""
    if not ancestors or len(events) < 2:
        return events
    keep = []
    for a in events:
        if not any(b is not a and a.label in ancestors.get(b.label, ()) and _iou(a, b) >= 0.5 for b in events):
            keep.append(a)
    return keep


def segment(probs: np.ndarray, labels: list[str], frame_s: float = FRAME_S, tau_on: float = 0.30,
            tau_off: float = 0.15, min_dur: float = 0.20, gap: float = 0.30, k: int = 10,
            smooth: int = 5, stop: set | None = None, ancestors: dict | None = None) -> list[Event]:
    """Frame-level class probabilities [T, C] -> at most k events, sorted by onset."""
    stop = STOPLIST if stop is None else stop
    T, _ = probs.shape
    out: list[Event] = []
    for c in np.where(probs.max(axis=0) >= tau_on)[0]:
        lab = labels[c]
        if lab in stop:
            continue
        p = _medfilt(probs[:, c].astype(np.float32), smooth)
        evs, on, start = [], False, 0
        for t in range(T):
            if not on and p[t] >= tau_on:
                on, start = True, t
            elif on and p[t] < tau_off:
                evs.append((start, t)); on = False
        if on:
            evs.append((start, T))
        merged: list[list[int]] = []
        for a, b in evs:
            if merged and (a - merged[-1][1]) * frame_s < gap:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        for a, b in merged:
            if (b - a) * frame_s >= min_dur:
                out.append(Event(lab, round(a * frame_s, 2), round(b * frame_s, 2), round(float(p[a:b].max()), 3)))
    out = collapse_ancestors(out, ancestors)
    out.sort(key=lambda e: -e.peak)
    out = out[:k] if k else out
    out.sort(key=lambda e: (e.t0, e.t1))
    return out


def ontology() -> list[dict]:
    """The AudioSet ontology (`<data>/AudioSet/ontology.json`)."""
    return json.loads(data("AudioSet", "ontology.json").read_text())


def load_ancestors(labels: list[str]) -> dict[str, set]:
    """class name -> its ancestor class names in the AudioSet ontology, restricted to `labels`."""
    onto = ontology()
    name = {x["id"]: x["name"] for x in onto}
    parents: dict[str, set] = {}
    for x in onto:
        for c in x.get("child_ids", []):
            parents.setdefault(c, set()).add(x["id"])
    names = set(labels)
    out: dict[str, set] = {}
    for x in onto:
        if x["name"] not in names:
            continue
        seen, stack = set(), list(parents.get(x["id"], ()))
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid); stack.extend(parents.get(pid, ()))
        out[x["name"]] = {name[i] for i in seen if name.get(i) in names}
    return out


# --------------------------------------------------------------------------- rendering
def label_durations(events: list[Event]) -> dict[str, float]:
    """Seconds each label is audible: the union of its intervals."""
    by: dict[str, list[tuple[float, float]]] = {}
    for e in events:
        by.setdefault(e.label, []).append((e.t0, e.t1))
    out = {}
    for lab, iv in by.items():
        iv.sort(); tot, cs, ce = 0.0, iv[0][0], iv[0][1]
        for a, b in iv[1:]:
            if a > ce:
                tot += ce - cs; cs, ce = a, b
            else:
                ce = max(ce, b)
        out[lab] = tot + (ce - cs)
    return out


def group_occurrences(events: list[Event], iou: float = 0.5) -> list[Event]:
    """Labels that co-occur over one span are one occurrence, named by all of them (sorted, so the same set of
    labels always reads the same): three clinks tagged `Clang; Chink, clink; Ding` each time are three
    occurrences of one sound, not nine events."""
    groups: list[list[Event]] = []
    for e in sorted(events, key=lambda e: -e.peak):
        for g in groups:
            if any(_iou(e, k) >= iou for k in g):
                g.append(e); break
        else:
            groups.append([e])
    out = [Event(" / ".join(sorted(k.label for k in g)), min(k.t0 for k in g), max(k.t1 for k in g), max(k.peak for k in g)) for g in groups]
    out.sort(key=lambda e: (e.t0, e.t1))
    return out


def derived(events: list[Event], wants: set[str]) -> list[str]:
    """The lines computed for the reader: durations, order, counts."""
    if not events:
        return []
    lines = []
    if "duration" in wants:
        dur = label_durations(events)
        hi = max(dur, key=dur.get); lo = min(dur, key=dur.get)
        s = f"Longest: {hi} ({dur[hi]:.1f} s)"
        if lo != hi:
            s += f"; shortest: {lo} ({dur[lo]:.1f} s)"
        lines.append(s + ".")
    if "order" in wants:
        seen, seq = set(), []
        for e in sorted(events, key=lambda e: e.t0):
            if e.label not in seen:
                seen.add(e.label); seq.append(e.label)
        if len(seq) >= 2:
            lines.append("Order: " + ", then ".join(seq) + ".")
    if "count" in wants:
        n: dict[str, int] = {}
        for e in events:
            n[e.label] = n.get(e.label, 0) + 1
        lines.append("Occurrences: " + ", ".join(f"{l} x{c}" for l, c in n.items()) + f" ({len(n)} distinct sound{'s' if len(n) != 1 else ''}).")
    return lines


def render(events: list[Event], question: str = "") -> str:
    """The timeline of a clip and the derived line(s) its question asks for, as one line of text."""
    if not events:
        return ""
    events = group_occurrences(events)
    parts = ["; ".join(f"[{e.t0:.1f}-{e.t1:.1f}s] {e.label}" for e in events)]
    parts += derived(events, _wants(question))
    return " ".join(parts)


# --------------------------------------------------------------------------- the detector
class Detector:
    """PANNs Cnn14_DecisionLevelMax: waveform -> frame-level probabilities [T, 527] every 10 ms (first MAX_S seconds)."""

    def __init__(self, cfg=None, log=None, device=None):
        from l2r.encoders import load_panns, panns_labels
        self.cfg = cfg or load_config()
        self.m, _ = load_panns(device, self.cfg)
        self.dev = next(self.m.parameters()).device
        self.labels = panns_labels(self.cfg)
        self.ancestors = load_ancestors(self.labels)
        self.params = dict(DEFAULTS)

    def framewise(self, path) -> np.ndarray:
        import librosa
        import torch
        from l2r.encoders import PANNS_SR
        y, _ = librosa.load(str(path), sr=PANNS_SR, mono=True, duration=MAX_S)
        if len(y) < PANNS_SR:                    # the pooling stack needs about a second of audio
            y = np.pad(y, (0, PANNS_SR - len(y)))
        with torch.no_grad():
            out = self.m(torch.from_numpy(y[None, :].astype(np.float32)).to(self.dev))
        return out["framewise_output"][0].detach().cpu().numpy()

    def events(self, path) -> list[Event]:
        p = self.params
        return segment(self.framewise(path), self.labels, FRAME_S, p["tau_on"], p["tau_off"], p["min_dur"], p["gap"], int(p["k"]), int(p["smooth"]), ancestors=self.ancestors)


def selftest():
    labels = ["Dog", "Shatter", "Speech", "Inside, small room", "Music"]
    T = 1000; P = np.zeros((T, 5), np.float32)
    P[0:250, 0] = 0.9; P[240:400, 1] = 0.8; P[410:980, 2] = 0.7; P[:, 3] = 0.95
    ev = segment(P, labels, tau_on=0.3, tau_off=0.15)
    assert [(e.label, e.t0, e.t1) for e in ev] == [("Dog", 0.0, 2.5), ("Shatter", 2.4, 4.0), ("Speech", 4.1, 9.8)], ev
    assert render(ev) == "[0.0-2.5s] Dog; [2.4-4.0s] Shatter; [4.1-9.8s] Speech"
    assert "Longest: Speech (5.7 s); shortest: Shatter (1.6 s)." in render(ev, "Which sound is the longest?")
    assert "Order: Dog, then Shatter, then Speech." in render(ev, "What is the order of the sounds?")
    assert "Occurrences: Dog x1, Shatter x1, Speech x1 (3 distinct sounds)." in render(ev, "How many times does the dog bark?")
    assert collapse_ancestors([Event("Animal", 0, 2, 0.5), Event("Dog", 0, 2, 0.9)], {"Dog": {"Animal"}}) == [Event("Dog", 0, 2, 0.9)]
    three = [Event(l, t, t + 0.5, p) for t in (0.5, 2.6, 5.0) for l, p in (("Ding", 0.9 - t / 10), ("Clang", 0.8), ("Chink, clink", 0.7 + t / 10))]
    assert "Chink, clink / Clang / Ding x3 (1 distinct sound)." in render(three, "how many times")
    assert asks_timeline("Which sound lasts the longest?") and asks_timeline("How many times does the dog bark?")
    assert not asks_timeline("Which chord is played from 1.83 to 3.66?") and not asks_timeline("How many speakers are there?")
    assert not asks_timeline("What animal is this?") and is_event_question("What happens at the end?")
    print("events selftest ok")


if __name__ == "__main__":
    selftest()
