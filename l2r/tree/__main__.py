"""Tree generation: vote the labels, admit the leaves, clean them with a text LLM, add the sound identity classes.

    python -m l2r.tree                       # every stage in order; a finished stage is skipped
    python -m l2r.tree --list

Stages (outputs under <checkpoints>/):
  raw             vote, normalise and admit the leaves (python -m l2r.tree.publish --raw)      -> tree_raw/tree.json
  propose         the text LLM proposes groups of synonymous leaves (l2r.tree.llm_clean)       -> tree_raw/llm_groups_raw.json
  check           the acoustic check of every group                                              -> tree_raw/llm_groups_audio.json
  apply           keep the groups that pass every guard, prune                                   -> tree_raw/llm_groups.json
  publish         the final tree (python -m l2r.tree.publish)                                    -> tree/tree.json
  audioset-items  the AudioSet clips and classes of the sound identity attributes (l2r.router.audioset items) -> <work>/audioset/items.json
  sound-plan      place the AudioSet classes under the sound attributes (l2r.tree.sound)         -> tree/sound/plan.json
  sound-llm       the text LLM places the remaining AudioSet and VGGSound classes                -> tree/sound/llm.json
  sound-verify    the placements are checked against the ontology                                -> tree/sound/verify.json
  sound-build     the sound identity tree                                                        -> tree/sound_tree.json
With the paper's checkpoints in place (python -m l2r.checkpoints), the LLM stages are already done and only `raw`
and `publish` rebuild the tree from the label table.
"""
from __future__ import annotations

from l2r.common import ckpt, work
from l2r.pipeline import Step, parser, run

STEPS = [Step("raw", ["l2r.tree.publish", "--raw"], [ckpt("tree_raw", "tree.json")]),
         Step("propose", ["l2r.tree.llm_clean", "propose"], [ckpt("tree_raw", "llm_groups_raw.json")]),
         Step("check", ["l2r.tree.llm_clean", "check"], [ckpt("tree_raw", "llm_groups_audio.json")]),
         Step("apply", ["l2r.tree.llm_clean", "apply"], [ckpt("tree_raw", "llm_groups.json")]),
         Step("publish", ["l2r.tree.publish"], [ckpt("tree", "tree.json")]),
         Step("audioset-items", ["l2r.router.audioset", "items"], [work("audioset", "items.json")]),
         Step("sound-plan", ["l2r.tree.sound", "plan"], [ckpt("tree", "sound", "plan.json")]),
         Step("sound-llm", ["l2r.tree.sound", "llm"], [ckpt("tree", "sound", "llm.json")]),
         Step("sound-verify", ["l2r.tree.sound", "verify"], [ckpt("tree", "sound", "verify.json")]),
         Step("sound-build", ["l2r.tree.sound", "build"], [ckpt("tree", "sound_tree.json")])]

if __name__ == "__main__":
    run(STEPS, parser("tree", __doc__).parse_args())
