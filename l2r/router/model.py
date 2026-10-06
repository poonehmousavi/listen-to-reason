"""The router model: a small trainable body on a frozen encoder's embedding, and one head per attribute.

    h = E(x)                         frozen encoder embedding of a chunk (cached, see features.py)
    z = body(h)                      LayerNorm -> MLP -> LayerNorm, shared by the attributes of the router
    g_a(z)                           one head per attribute a, one output per NODE of the attribute:
                                     every class v (read as "v, nothing finer") and every leaf under it

The score of a class is the log-sum-exp of its own node and its leaves (Eq. 1 of the paper), so a confident leaf
also supports its class. A single-choice attribute takes a softmax over class scores, a multiple-choice attribute
one sigmoid per class. A leaf is served only when its probability inside its class clears a threshold fitted on
held-out data; otherwise the router backs off to the class.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from l2r.common import ckpt

DIM = 256
NEG = -1e4                                    # "not under this class" inside a log-sum-exp


def key(attr: str) -> str:
    """`nn.ModuleDict` keys cannot hold a "."."""
    return attr.replace(".", "__")


def head_dim(n_nodes: int) -> int:
    """A head's hidden width follows its number of nodes: 3 nodes -> 64, 400 nodes -> 512."""
    return int(min(512, max(64, 2 ** math.ceil(math.log2(max(4 * n_nodes, 2))))))


class Body(nn.Module):
    """The shared projection of a frozen embedding into the router's space."""

    def __init__(self, in_dim: int, dim: int = DIM, layers: int = 2):
        super().__init__()
        blocks = [nn.LayerNorm(in_dim), nn.Linear(in_dim, dim), nn.GELU(), nn.Dropout(0.1)]
        for _ in range(layers - 1):
            blocks += [nn.Linear(dim, dim), nn.GELU(), nn.Dropout(0.1)]
        self.net = nn.Sequential(*blocks, nn.LayerNorm(dim))

    def forward(self, x):
        return self.net(x)


class Router(nn.Module):
    """One router = the attributes that share an encoder. `classes[a]` lists the nodes of attribute a as
    (class, leaf) pairs, with leaf "" for the class's own node."""

    def __init__(self, space: dict[str, dict], classes: dict[str, list], encoder: str, in_dim: int, layers: int = 2, hdim: dict | None = None):
        super().__init__()
        self.attrs = sorted(classes); self.layers = layers; self.encoder = encoder; self.in_dim = in_dim
        self.tau: dict[str, float] = {}                   # per attribute: the probability a leaf needs to be served
        self.meta, self.cix = {}, {}
        for a in self.attrs:
            cl = [tuple(c) for c in classes[a]]; vals = [v for v in space[a]["values"] if (v, "") in cl]
            assert vals and all(c[0] in vals for c in cl), a
            self.meta[a] = {"values": vals, "classes": cl, "multi": space[a]["multi"], "grid": space[a]["grid"], "region": space[a]["region"]}
            self.cix[a] = {c: i for i, c in enumerate(cl)}
        self.hdim = {a: (hdim or {}).get(a) or head_dim(len(self.meta[a]["classes"])) for a in self.attrs}
        self.enc = Body(in_dim, layers=layers)
        self.head = nn.ModuleDict({key(a): self._head(a) for a in self.attrs})
        for a in self.attrs:
            self._masks(a)

    def _head(self, a):
        return nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, self.hdim[a]), nn.GELU(), nn.Dropout(0.1), nn.Linear(self.hdim[a], len(self.meta[a]["classes"])))

    def _masks(self, a):
        m = self.meta[a]; vi = {v: i for i, v in enumerate(m["values"])}
        under = torch.full((len(m["values"]), len(m["classes"])), NEG)
        for j, (v, _) in enumerate(m["classes"]):
            under[vi[v], j] = 0.0
        self.register_buffer("under_" + key(a), under, persistent=False)
        self.register_buffer("vof_" + key(a), torch.tensor([vi[v] for v, _ in m["classes"]]), persistent=False)

    def under(self, a: str) -> torch.Tensor:
        """[n classes, n nodes]: 0 where the node belongs to the class, NEG elsewhere."""
        return getattr(self, "under_" + key(a))

    def class_of(self, a: str) -> torch.Tensor:
        """[n nodes]: the class index of every node."""
        return getattr(self, "vof_" + key(a))

    # ---------------------------------------------------------------- forward
    def embed(self, x) -> torch.Tensor:
        """[B, in_dim] frozen embeddings -> [B, DIM]."""
        return self.enc(x)

    def node_logits(self, a: str, z) -> torch.Tensor:
        return self.head[key(a)](z)

    def class_score(self, a: str, s) -> torch.Tensor:
        """[B, n nodes] node logits -> [B, n classes]: the log-sum-exp of the nodes under each class."""
        return torch.logsumexp(s[:, None, :] + self.under(a)[None], -1)

    def class_logits(self, a: str, z) -> torch.Tensor:
        return self.class_score(a, self.node_logits(a, z))

    @torch.no_grad()
    def within(self, a: str, z, vi: int) -> tuple[torch.Tensor, list[int]]:
        """Posterior over the nodes under class `vi`, pooled over the chunks `z` (mean log-probability) -> (p, node ids)."""
        ids = [j for j, (v, _) in enumerate(self.meta[a]["classes"]) if v == self.meta[a]["values"][vi]]
        lp = self.node_logits(a, z)[:, ids].log_softmax(1).mean(0)
        return lp.softmax(0), ids

    @torch.no_grad()
    def leaf(self, a: str, z, vi: int, tau: float | None = None):
        """The leaf under class `vi` for these chunks, or None: the class's own node won, or the best leaf is
        under the fitted threshold (back-off to the class)."""
        p, ids = self.within(a, z, vi)
        if len(ids) < 2:
            return None
        j = int(p.argmax()); leaf = self.meta[a]["classes"][ids[j]][1]
        return leaf if leaf and float(p[j]) >= (self.tau.get(a, 1.01) if tau is None else tau) else None

    @torch.no_grad()
    def top_nodes(self, a: str, z, k: int, thr: float) -> list[tuple[str, str, float]]:
        """Single-choice attribute: up to k (class, leaf or "", p) from one softmax over all nodes, pooled over the
        chunks; the best is always served, the others need p >= thr. A class's own node is dropped when one of its
        leaves is served."""
        p = self.node_logits(a, z).log_softmax(1).mean(0).softmax(0); out = []
        for n, j in enumerate(p.argsort(descending=True).tolist()[:k]):
            if n and float(p[j]) < thr:
                break
            out.append((*self.meta[a]["classes"][j], float(p[j])))
        leafed = {v for v, lf, _ in out if lf}
        return [(v, lf, q) for v, lf, q in out if lf or v not in leafed]

    @torch.no_grad()
    def top_classes(self, a: str, z, k: int, thr: float) -> list[tuple[int, str, float]]:
        """Single-choice attribute: up to k (class index, class, p) by class probability, pooled over the chunks;
        the best is always served, the others need p >= thr."""
        p = self.class_logits(a, z).log_softmax(1).mean(0).softmax(0); out = []
        for n, j in enumerate(p.argsort(descending=True).tolist()[:k]):
            if n and float(p[j]) < thr:
                break
            out.append((j, self.meta[a]["values"][j], float(p[j])))
        return out

    @torch.no_grad()
    def top_leaves(self, a: str, z, vi: int, k: int, thr: float) -> list[str]:
        """Multiple-choice attribute, class `vi` is on: up to k leaves under it with within-class probability >= thr."""
        p, ids = self.within(a, z, vi)
        out = [(float(p[j]), self.meta[a]["classes"][ids[j]][1]) for j in range(len(ids)) if self.meta[a]["classes"][ids[j]][1] and float(p[j]) >= thr]
        return [lf for _, lf in sorted(out, reverse=True)[:k]]


# ------------------------------------------------------------------------------------- io
def path(name: str):
    return ckpt("routers", f"{name}.pt")


def save(model: Router, name: str, space: dict, extra: dict | None = None):
    torch.save({"model": model.state_dict(), "space": {a: space[a] for a in model.attrs},
                "classes": {a: [list(c) for c in model.meta[a]["classes"]] for a in model.attrs}, "layers": model.layers, "hdim": model.hdim,
                "encoder": model.encoder, "in_dim": model.in_dim, "tau": model.tau, **(extra or {})}, path(name))


def load(name: str, dev="cpu") -> tuple[Router, dict]:
    ck = torch.load(path(name), map_location=dev, weights_only=False)
    m = Router(ck["space"], ck["classes"], ck["encoder"], ck["in_dim"], layers=ck["layers"], hdim=ck["hdim"]).to(dev)
    m.load_state_dict(ck["model"]); m.tau = dict(ck.get("tau") or {}); m.eval()
    return m, ck
