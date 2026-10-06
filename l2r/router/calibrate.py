"""Calibrate the trained routers on held-out chunks of the training pool.

For every attribute this writes what inference reads (`<checkpoints>/routers/calibration.json`):

    temperature   single-choice attributes: the temperature that minimises the held-out negative log-likelihood;
                  it rescales the class probabilities and never reorders them
    skill         balanced accuracy above chance, (balanced - 1/classes) / (1 - 1/classes); an attribute with low
                  skill is served only when the router is confident

    python -m l2r.router.calibrate                      # the routers trained on the pool
    python -m l2r.router.calibrate --routers gate,sound
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from l2r.common import ckpt, load_config, save_json, setup_run
from l2r.router import model as M
from l2r.router.train import DEFAULTS, Data

TEMPERATURES = (0.5, 1, 2, 3, 5, 8, 12, 20, 30)


def path():
    return ckpt("routers", "calibration.json")


def skill(balanced: float, classes: int) -> float:
    c = max(classes, 1)
    return (balanced - 1 / c) / max(1 - 1 / c, 1e-6)


@torch.no_grad()
def class_logits(data: Data, model: M.Router, attr: str, prs: list[dict], dev) -> torch.Tensor:
    out = [model.class_logits(attr, model.embed(data.x([p["row"] for p in prs[s:s + 128]], dev))) for s in range(0, len(prs), 128)]
    return torch.cat(out)


def calibrate(routers: list[str], name: str, log) -> dict:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = {n: M.load(n, dev)[0] for n in routers}
    owner = {a: n for n, m in models.items() for a in m.attrs}             # a later router takes an attribute over
    datas = {}
    out = {}
    for attr, n in sorted(owner.items()):
        m = models[n]
        if m.encoder not in datas:
            datas[m.encoder] = Data([name], m.encoder, log)
        data = datas[m.encoder]
        vi = {v: i for i, v in enumerate(m.meta[attr]["values"])}
        prs = [p for p in data.by_attr[attr]["test"] if any(v in vi for v in p["values"])]
        if not prs:
            continue
        lg = class_logits(data, m, attr, prs, dev)
        pred = lg.argmax(1).cpu().numpy()
        gold = [{vi[v] for v in p["values"] if v in vi} for p in prs]
        hit = np.array([pred[i] in g for i, g in enumerate(gold)])
        recall = [hit[[i for i, g in enumerate(gold) if k in g]].mean() for k in range(len(vi)) if any(k in g for g in gold)]
        r = {"router": n, "chunks": len(prs), "classes": len(recall), "accuracy": float(hit.mean()), "balanced": float(np.mean(recall))}
        r["skill"] = skill(r["balanced"], r["classes"])
        if not m.meta[attr]["multi"]:
            g1 = torch.tensor([min(g) for g in gold], device=lg.device)
            nll = [float(torch.nn.functional.cross_entropy(lg / t, g1)) for t in TEMPERATURES]
            r["temperature"] = TEMPERATURES[int(np.argmin(nll))]
        out[attr] = r
        log.info("%-26s %-16s chunks %5d  accuracy %.3f  balanced %.3f  skill %.3f  temperature %s", attr, n, r["chunks"], r["accuracy"],
                 r["balanced"], r["skill"], r.get("temperature", "-"))
    save_json({"attributes": out}, path())
    log.info("wrote %s (%d attributes)", path(), len(out))
    return out


def load() -> dict[str, dict]:
    """attribute -> {skill, temperature, ...}; an attribute without an entry has skill 0 and temperature 1."""
    p = path()
    return json.loads(p.read_text())["attributes"] if p.exists() else {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--routers", default="", help="default: every configured router trained on the pool alone")
    ap.add_argument("--set", default="pool")
    a = ap.parse_args()
    cfg = load_config()
    _, log = setup_run("router_calibrate", cfg)
    names = a.routers.split(",") if a.routers else [n for n, s in (cfg.get("routers") or {}).items() if s.get("sets", DEFAULTS["sets"]) == a.set]
    calibrate(names, a.set, log)
