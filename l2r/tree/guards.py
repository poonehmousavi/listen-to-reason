"""Lexical checks on merging two names into one node.

A sentence embedding scores topical similarity, and opposites share their topic: `fast tempo` / `slow tempo` and
`gunshot` / `no gunshot` embed closer than many true synonyms, so no similarity threshold separates them. Two names
are therefore merged only if they do not conflict in

  number     `3/4 time signature` vs `4/4 time signature`
  negation   `dog bark` vs `no bark`
  polarity   `high-pitched` vs `low-pitched`, `male voice` vs `female voice`

A generalisation is not a conflict: `synth` may absorb `dark synth` and `bright synth`.
"""
from __future__ import annotations

import re
from collections import defaultdict

_ARTICLE = re.compile(r"^(a|an|the|some|any)\s+", re.I)
_PLURAL = re.compile(r"(?<=[a-z])(?:ies|es|s)$")
_GERUND = re.compile(r"(?<=[a-z]{3})ing$")
_DOUBLED = re.compile(r"([bdfgklmnprt])\1$")          # "chatting" -> "chatt" -> "chat"


def normalise(m: str) -> str:
    """Surface form -> comparison key: lower case, no article, plural and -ing endings stripped."""
    s = re.sub(r"\s+", " ", str(m or "")).strip().lower().strip(".,;:!?")
    s = _ARTICLE.sub("", s)
    toks = []
    for t in s.split():
        if _GERUND.search(t) and len(t) > 5:
            t = _DOUBLED.sub(r"\1", _GERUND.sub("", t))
        t = _PLURAL.sub("", t) if _PLURAL.search(t) and len(t) > 3 else t
        toks.append(t)
    return " ".join(toks)


_NEG = re.compile(r"(?:^|[\s-])(no|not|non|without|absent|absence|lack|lacking|zero|never|silent)"
                  r"(?=$|[\s-])")
_NUM = re.compile(r"\d+(?:\.\d+)?")

# Contrastive axes. Each axis has two sides, and a side is a set of words that take it: `soft` and `quiet` both
# oppose `loud`. A conflict is two different sides of ONE axis; two different axes are independent descriptors.
_AXES_RAW = [
    ({"high"}, {"low"}), ({"fast", "quick", "rapid"}, {"slow"}),
    ({"loud"}, {"quiet", "soft", "faint"}),
    ({"male", "man", "men", "boy", "masculine"}, {"female", "woman", "women", "girl", "feminine"}),
    ({"major"}, {"minor"}), ({"indoor", "indoors", "inside"}, {"outdoor", "outdoors", "outside"}),
    ({"rising", "increasing", "ascending"}, {"falling", "decreasing", "descending"}),
    ({"near", "close", "nearby"}, {"distant", "far", "faraway"}), ({"single"}, {"multiple"}),
    ({"short"}, {"long"}), ({"happy", "joyful", "cheerful"}, {"sad", "sorrowful"}),
    ({"bright"}, {"dark"}), ({"wet"}, {"dry"}), ({"clean"}, {"distorted"}),
    ({"mono"}, {"stereo"}), ({"start", "beginning"}, {"end", "ending"}), ({"before"}, {"after"}),
    ({"open"}, {"closed"}), ({"up", "upward"}, {"down", "downward"}),
    ({"forward"}, {"backward"}), ({"warm", "hot"}, {"cold", "cool"}), ({"young"}, {"old"}),
    ({"in"}, {"out"}),
]

# Number words, so `four-on-the-floor` and `4 on the floor` carry the same numeric token. The
# first guarded build refused that pair as a numeric conflict. `one` is NOT in the table: it is

# Number words, so `four-on-the-floor` and `4 on the floor` carry the same numeric token. `one` is left out:
# it is a determiner far more often than a count.
_NUM_WORDS = {w: str(i + 2) for i, w in enumerate(
    "two three four five six seven eight nine ten eleven twelve".split())}


def _axes() -> dict[str, tuple[int, int]]:
    """{normalised word -> (axis, side)}. Built through `normalise`, the same function the names go through."""
    out: dict[str, tuple[int, int]] = {}
    for ax, (side_a, side_b) in enumerate(_AXES_RAW):
        for side, words in ((0, side_a), (1, side_b)):
            for w in words:
                nw = normalise(w)
                if nw:
                    out.setdefault(nw, (ax, side))
    return out


_AXES = None


def merge_features(stem: str) -> tuple:
    """(numeric tokens, is-negated, frozenset of (axis, side)) of one normalised name."""
    global _AXES
    if _AXES is None:
        _AXES = _axes()
    toks = stem.replace("-", " ").split()
    poles = frozenset(_AXES[t] for t in toks if t in _AXES)
    nums = tuple(_NUM.findall(" ".join(_NUM_WORDS.get(t, t) for t in toks)))
    return (nums, bool(_NEG.search(stem)), poles)


def may_merge(fa: tuple, fb: tuple) -> tuple[bool, str]:
    """May two names (given as `merge_features`) become one node? -> (ok, reason of the refusal)."""
    if fa[0] != fb[0]:
        return False, "numeric"
    if fa[1] != fb[1]:
        return False, "negation"
    pa, pb = fa[2], fb[2]
    if pa != pb:
        # compare the SETS of sides per axis: a name can stand on both sides of one axis ("low-pitched high-volume"),
        # and two names may share an axis only if they take exactly the same sides of it
        sa, sb = defaultdict(set), defaultdict(set)
        for ax, sd in pa:
            sa[ax].add(sd)
        for ax, sd in pb:
            sb[ax].add(sd)
        if any(ax in sb and sa[ax] != sb[ax] for ax in sa):
            return False, "polarity"
    return True, ""


def selftest():
    f = lambda s: merge_features(normalise(s))
    assert may_merge(f("fast tempo"), f("slow tempo")) == (False, "polarity")
    assert may_merge(f("3/4 time signature"), f("4/4 time signature")) == (False, "numeric")
    assert may_merge(f("gunshot"), f("no gunshot")) == (False, "negation")
    assert may_merge(f("synth"), f("dark synth"))[0] and may_merge(f("soft murmur"), f("quiet murmur"))[0]
    assert may_merge(f("four-on-the-floor"), f("4 on the floor"))[0]
