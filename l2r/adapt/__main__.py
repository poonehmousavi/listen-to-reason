"""Adding a domain: one attribute with one head on a frozen encoder, from k labelled clips per class.

    python -m l2r.adapt --domain birds --k 5             # also: marine; the reader with --reader
    python -m l2r.adapt --domain birds --k 5 --list

Stages:
  data        evaluation items and the training pool of the domain (l2r.adapt.data)        -> <work>/domains/<domain>.json
  build       the label set <domain>_k<k>: k clips per class and a `none` class (l2r.adapt.domain build)
  features    embeddings of the set on the domain's encoder (adapt.encoder in the config)
  train       the head (l2r.router.train; settings `adapt:` in the config)                  -> <checkpoints>/routers/<domain>_k<k>.pt
  probe       routing accuracy without a reader (l2r.adapt.domain probe)                    -> <work>/results/probe_<domain>_k<k>.json
  context     the traversal with the new head; the context with and without the transcript  -> <work>/contexts/<domain>[_tree_only].json
  reader      the frozen reader on the tree alone (l2r.reason)                              -> <work>/results/<reader>_tree_only_<domain>_k<k>.json
A benchmark's traversal is rebuilt when it was last served with another head (another k).
"""
from __future__ import annotations

from l2r.adapt.data import domain_file
from l2r.adapt.domain import set_name
from l2r.common import load_config, load_json, work
from l2r.dataset import index as I
from l2r.pipeline import Step, parser, run
from l2r.retrieve.context import context_path
from l2r.retrieve.serve import nodes_path
from l2r.router import features as F, model


def steps(domain: str, k: int, reader: str) -> list[Step]:
    cfg = load_config(); c = cfg["adapt"]; name = set_name(domain, k); enc = c["encoder"][domain]
    train = ["l2r.router.train", "--name", name, "--sets", name, "--regions", "sound", "--encoder", enc, "--layers", str(c["layers"]),
             "--lr", str(c["lr"]), "--balanced", str(c["balanced"]), "--min-clips", str(c["min_clips"]), "--val-frac", str(c["val_frac"]),
             "--epochs", str(c["epochs"]), "--steps", str(c["steps"])]

    def served_with_head() -> bool:
        return nodes_path(domain).exists() and name in load_json(nodes_path(domain)).get("routers", [])

    def drop_stale():
        if nodes_path(domain).exists() and not served_with_head():
            for p in (nodes_path(domain), context_path(domain), context_path(domain, "tree_only")):
                p.unlink(missing_ok=True)

    return [Step("data", ["l2r.adapt.data", "--domain", domain], [domain_file(domain)]),
            Step("build", ["l2r.adapt.domain", "build", "--domain", domain, "--k", str(k)], [I.sdir(name) / "pairs.jsonl"]),
            Step("features", ["l2r.router.features", "--encoder", enc, "--sets", name], [F.store_path(enc, name)]),
            Step("train", train, [model.path(name)]),
            Step("probe", ["l2r.adapt.domain", "probe", "--domain", domain, "--k", str(k)], [work("results", f"probe_{name}.json")]),
            Step("context", ["l2r.retrieve.context", "--benchmark", domain, "--run", "--domains", name, "--tree-only"],
                 done=lambda: served_with_head() and context_path(domain, "tree_only").exists(), before=drop_stale),
            Step("reader", ["l2r.reason.reader", "--reader", reader, "--benchmarks", domain, "--context", "tree_only", "--tag", f"_{name}"],
                 [work("results", f"{reader}_tree_only_{name}.json")])]


if __name__ == "__main__":
    ap = parser("adapt", __doc__); ap.add_argument("--domain", required=True, choices=("birds", "marine"))
    ap.add_argument("--k", type=int, default=5, help="labelled clips per class"); ap.add_argument("--reader", default="qwen2.5-7b")
    a = ap.parse_args(); run(steps(a.domain, a.k, a.reader), a)
