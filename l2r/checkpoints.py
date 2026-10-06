"""Download the tree and the trained heads of the paper into the checkpoint folder.

    python -m l2r.checkpoints                      # from the Hugging Face hub (models.checkpoints in the config)
    python -m l2r.checkpoints --birdnet-export     # add the embedding output to a downloaded BirdNET ONNX model (bird domain only)

What arrives, under <checkpoints>/:
  tree/tree.json, tree/sound_tree.json       the tree (speech, music, sound attributes; the sound identity classes)
  tree_raw/llm_groups.json, tree/sound/      the text LLM's answers used to build it (python -m l2r.tree then rebuilds it without the LLM)
  routers/*.pt, routers/calibration.json     the region routers (+ birds_k5 / marine_k5, the two domains of the paper)
  sound/, events/, music_properties/, question/, language/     the other heads of the traversal
  experts/gender_centroids.npz               the gender expert of the annotation
The PANNs, BEATs and BirdNET weights are downloaded by hand from the sources named under `encoders:` in the config,
which also holds the PATH/TO placeholders for where they are put (README).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from l2r.common import ckpt, load_config

REQUIRED = ("tree/tree.json", "tree/sound_tree.json", "routers/calibration.json", "language/text_head.pkl", "sound/nodes.json",
            "sound/heads_clap.pt", "sound/heads_beats.pt", "events/head.pt", "music_properties/heads.pt", "question/model.pkl")
BIRDNET_POOL = "GLOBAL_AVG_POOL"


def download(repo: str | None = None) -> None:
    from huggingface_hub import snapshot_download
    repo = repo or load_config()["models"]["checkpoints"]
    snapshot_download(repo, repo_type="model", local_dir=str(ckpt()))
    verify()


def verify() -> None:
    cfg = load_config(); missing = [f for f in REQUIRED if not ckpt(f).exists()]
    missing += [f"routers/{n}.pt" for n in cfg.get("routers", {}) if not ckpt("routers", f"{n}.pt").exists()]
    assert not missing, f"missing under {ckpt()}: {missing}"
    print(f"checkpoints complete under {ckpt()}")
    E = cfg["encoders"]
    for name, path in (("panns.weights", E["panns"]["weights"]), ("panns.labels", E["panns"]["labels"]), ("beats.dir", E["beats"]["dir"]),
                       ("birdnet.onnx", E["birdnet"]["onnx"])):
        state = "set it in the config and download the file (README)" if "PATH/TO" in str(path) else ("ok" if ckpt(path).exists() else f"not found: {ckpt(path)}")
        print(f"encoders.{name:16s} {state}")


def birdnet_export(src: Path, dst: Path) -> None:
    """BirdNET's ONNX graph with its global-average-pool tensor (the 1,024-d embedding) added as a second output."""
    import onnx
    model = onnx.load(str(src))
    names = [o for node in model.graph.node for o in node.output if BIRDNET_POOL in o]
    assert names, f"no tensor named *{BIRDNET_POOL}* in {src}"
    if names[-1] not in {o.name for o in model.graph.output}:
        model.graph.output.append(onnx.helper.make_tensor_value_info(names[-1], onnx.TensorProto.FLOAT, [None, 1024]))
    onnx.save(model, str(dst)); print(f"embedding output `{names[-1]}` -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--repo", default="", help="hub id (default: models.checkpoints)")
    ap.add_argument("--verify", action="store_true", help="only check that every file is present")
    ap.add_argument("--birdnet-export", action="store_true", help="write encoders.birdnet.onnx from birdnet.onnx in the same folder")
    a = ap.parse_args()
    if a.birdnet_export:
        dst = ckpt(load_config()["encoders"]["birdnet"]["onnx"]); birdnet_export(dst.with_name("birdnet.onnx"), dst)
    elif a.verify:
        verify()
    else:
        download(a.repo or None)
