"""
models/gnn.py
=============
Graph neural network baselines for patient-trial matching, implemented in plain
PyTorch (no torch_geometric / torch_scatter dependency, which are the two
packages that most often refuse to build on a cluster).

Graph design -- and why patient-trial edges are absent
------------------------------------------------------
Nodes:  patient  |  code (diagnosis / medication / lab)  |  trial
Edges:  patient --has--> code
        trial --requires--> code      (inclusion criteria)
        trial --excludes--> code      (exclusion criteria)
        plus the reverse of each; six relations in total.

There is deliberately **no patient-trial edge anywhere in the message-passing
graph**. This is the single most important detail in the file. In GNN link
prediction it is standard -- and a standard source of inflated results -- to
place the training edges in the graph and then predict held-out edges. Here the
supervision edges are exactly the thing being predicted, and because eligibility
is transitive through shared codes, leaving even the training pairs in the graph
lets a two-layer model reach a test patient's label through
patient -> trial -> similar patient. Excluding them entirely makes that
structurally impossible and keeps the encoder inductive: a brand-new trial is
encoded purely from its criterion codes, with no retraining.

Patients and trials carry the *same* SVD input features the tabular models get,
so any difference in performance is attributable to message passing rather than
to one model quietly receiving richer inputs. Code nodes get learnable
embeddings, which is what gives the graph something to propagate.

Variants
--------
    gcn         symmetric-normalised convolution over the merged graph
    sage        mean-aggregation GraphSAGE with a separate self-transform
    gat         single/multi-head attention over edges (scatter-softmax)
    rgcn        relation-specific weights (heterogeneous; closest to the
                upstream HeteroConv encoder)
    sage_gcl    GraphSAGE with unsupervised NT-Xent pretraining under edge
                dropout, then supervised fine-tuning -- the direct structural
                analogue of the upstream Stage A -> Stage B recipe
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..features import subsample_pairs
from ..metrics import safe_pr_auc
from ..splits import PairSet
from .base import BaseMatcher, TrainContext, positive_class_weight, register

log = logging.getLogger(__name__)

MAX_VAL_PAIRS = 400_000

RELATIONS = (
    "patient_has_code",
    "code_of_patient",
    "trial_requires_code",
    "code_required_by_trial",
    "trial_excludes_code",
    "code_excluded_by_trial",
)


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
@dataclass
class HeteroGraph:
    n_patients: int
    n_codes: int
    n_trials: int
    edges: Dict[str, np.ndarray]      # relation -> (2, E) in global node index space

    @property
    def n_nodes(self) -> int:
        return self.n_patients + self.n_codes + self.n_trials

    @property
    def patient_offset(self) -> int:
        return 0

    @property
    def code_offset(self) -> int:
        return self.n_patients

    @property
    def trial_offset(self) -> int:
        return self.n_patients + self.n_codes

    def describe(self) -> str:
        e = {k: int(v.shape[1]) for k, v in self.edges.items()}
        return (
            f"HeteroGraph(nodes={self.n_nodes}: {self.n_patients}p/"
            f"{self.n_codes}c/{self.n_trials}t, edges={e})"
        )


def build_graph(ctx: TrainContext, code_vocab_from_train_only: bool = True) -> HeteroGraph:
    """Build the message-passing graph.

    The code vocabulary is taken from training patients and training trials
    only. Codes that appear exclusively in held-out entities are mapped to
    nothing (the entity keeps its own feature vector but gains no neighbours),
    which is the honest inductive setting: at deployment a new trial may cite a
    code the model has never seen.
    """
    ds = ctx.dataset
    patients, trials = ds.patients, ds.trials

    if code_vocab_from_train_only:
        src_patients = [patients[i] for i in ctx.split.patients["train"]]
        src_trials = [trials[i] for i in ctx.split.trials["train"]]
    else:
        src_patients, src_trials = patients, trials

    vocab: Dict[str, int] = {}
    for p in src_patients:
        for c in p.diagnosis_codes:
            vocab.setdefault(f"di:{c}", len(vocab))
        for c in p.medication_codes:
            vocab.setdefault(f"me:{c}", len(vocab))
        for c in p.lab_values:
            vocab.setdefault(f"la:{c}", len(vocab))
    for t in src_trials:
        for c in t.criteria:
            if c.is_resolved:
                vocab.setdefault(f"{c.entity_type[:2]}:{c.entity_code}", len(vocab))

    n_p, n_c, n_t = len(patients), max(len(vocab), 1), len(trials)
    code_off, trial_off = n_p, n_p + n_c

    pc_src, pc_dst = [], []
    for i, p in enumerate(patients):
        keys = (
            [f"di:{c}" for c in p.diagnosis_codes]
            + [f"me:{c}" for c in p.medication_codes]
            + [f"la:{c}" for c in p.lab_values]
        )
        for k in keys:
            j = vocab.get(k)
            if j is not None:
                pc_src.append(i)
                pc_dst.append(code_off + j)

    ti_src, ti_dst, te_src, te_dst = [], [], [], []
    for i, t in enumerate(trials):
        for c in t.criteria:
            if not c.is_resolved:
                continue
            j = vocab.get(f"{c.entity_type[:2]}:{c.entity_code}")
            if j is None:
                continue
            if c.is_inclusion:
                ti_src.append(trial_off + i); ti_dst.append(code_off + j)
            else:
                te_src.append(trial_off + i); te_dst.append(code_off + j)

    def pack(src, dst):
        if not src:
            return np.zeros((2, 0), dtype=np.int64)
        return np.stack([np.asarray(src, np.int64), np.asarray(dst, np.int64)])

    e_pc = pack(pc_src, pc_dst)
    e_ti = pack(ti_src, ti_dst)
    e_te = pack(te_src, te_dst)

    edges = {
        "patient_has_code": e_pc,
        "code_of_patient": e_pc[::-1].copy(),
        "trial_requires_code": e_ti,
        "code_required_by_trial": e_ti[::-1].copy(),
        "trial_excludes_code": e_te,
        "code_excluded_by_trial": e_te[::-1].copy(),
    }

    g = HeteroGraph(n_p, n_c, n_t, edges)
    log.info("%s", g.describe())
    _assert_no_patient_trial_edge(g)
    return g


def _assert_no_patient_trial_edge(g: HeteroGraph) -> None:
    """Guard the central anti-leakage invariant."""
    p_hi = g.n_patients
    t_lo = g.trial_offset
    for rel, e in g.edges.items():
        if e.shape[1] == 0:
            continue
        src, dst = e[0], e[1]
        p2t = ((src < p_hi) & (dst >= t_lo)).any()
        t2p = ((src >= t_lo) & (dst < p_hi)).any()
        if p2t or t2p:
            raise AssertionError(
                f"Relation '{rel}' contains a direct patient<->trial edge. "
                "That leaks supervision into message passing."
            )


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
def _build_sparse_adj(edge_index: np.ndarray, n_nodes: int, mode: str, device):
    """Normalised sparse adjacency, indexed [dst, src] so A @ H aggregates."""
    import torch

    if edge_index.shape[1] == 0:
        idx = torch.zeros((2, 0), dtype=torch.long, device=device)
        val = torch.zeros((0,), dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(idx, val, (n_nodes, n_nodes)).coalesce()

    src = torch.tensor(edge_index[0], dtype=torch.long, device=device)
    dst = torch.tensor(edge_index[1], dtype=torch.long, device=device)

    deg_dst = torch.zeros(n_nodes, device=device).index_add_(
        0, dst, torch.ones_like(dst, dtype=torch.float32)
    )
    if mode == "mean":
        val = 1.0 / deg_dst.clamp(min=1)[dst]
    elif mode == "sym":
        deg_src = torch.zeros(n_nodes, device=device).index_add_(
            0, src, torch.ones_like(src, dtype=torch.float32)
        )
        val = 1.0 / torch.sqrt(deg_dst.clamp(min=1)[dst] * deg_src.clamp(min=1)[src])
    else:
        val = torch.ones_like(dst, dtype=torch.float32)

    idx = torch.stack([dst, src])
    return torch.sparse_coo_tensor(idx, val, (n_nodes, n_nodes)).coalesce()


def _make_layer(kind: str, in_dim: int, out_dim: int, n_rel: int, heads: int):
    import torch
    import torch.nn as nn

    class GCNLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(in_dim, out_dim)

        def forward(self, h, adjs, edges=None):
            agg = torch.sparse.mm(adjs["_merged"], h)
            return self.lin(agg + h)

    class SAGELayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin_self = nn.Linear(in_dim, out_dim)
            self.lin_neigh = nn.Linear(in_dim, out_dim)

        def forward(self, h, adjs, edges=None):
            agg = torch.sparse.mm(adjs["_merged"], h)
            return self.lin_self(h) + self.lin_neigh(agg)

    class RGCNLayer(nn.Module):
        """Relation-specific weights -- the heterogeneous variant."""

        def __init__(self):
            super().__init__()
            self.lin_self = nn.Linear(in_dim, out_dim)
            self.lin_rel = nn.ModuleList(
                [nn.Linear(in_dim, out_dim, bias=False) for _ in range(n_rel)]
            )

        def forward(self, h, adjs, edges=None):
            out = self.lin_self(h)
            for r, rel in enumerate(RELATIONS[:n_rel]):
                A = adjs.get(rel)
                if A is None or A._nnz() == 0:
                    continue
                out = out + self.lin_rel[r](torch.sparse.mm(A, h))
            return out

    class GATLayer(nn.Module):
        """Multi-head attention with a scatter softmax over incoming edges."""

        def __init__(self):
            super().__init__()
            assert out_dim % heads == 0, "out_dim must be divisible by heads"
            self.h = heads
            self.d = out_dim // heads
            self.lin = nn.Linear(in_dim, out_dim, bias=False)
            self.att_src = nn.Parameter(torch.empty(1, heads, self.d))
            self.att_dst = nn.Parameter(torch.empty(1, heads, self.d))
            self.lin_self = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(self.att_src)
            nn.init.xavier_uniform_(self.att_dst)

        def forward(self, h, adjs, edges=None):
            n = h.size(0)
            src, dst = edges
            z = self.lin(h).view(n, self.h, self.d)
            alpha = (z * self.att_src).sum(-1)[src] + (z * self.att_dst).sum(-1)[dst]
            alpha = torch.nn.functional.leaky_relu(alpha, 0.2)

            # Numerically stable segment softmax over each destination node.
            amax = torch.full((n, self.h), -1e30, device=h.device)
            amax = amax.index_reduce_(0, dst, alpha, "amax", include_self=True)
            ex = torch.exp(alpha - amax[dst])
            denom = torch.zeros((n, self.h), device=h.device).index_add_(0, dst, ex)
            att = ex / denom[dst].clamp(min=1e-16)

            msg = z[src] * att.unsqueeze(-1)
            agg = torch.zeros((n, self.h, self.d), device=h.device).index_add_(0, dst, msg)
            return agg.reshape(n, -1) + self.lin_self(h)

    return {"gcn": GCNLayer, "sage": SAGELayer, "rgcn": RGCNLayer, "gat": GATLayer}[kind]()


# ---------------------------------------------------------------------------
# Encoder + decoder
# ---------------------------------------------------------------------------
def _make_encoder(kind, graph, d_p, d_t, hidden, out, n_layers, dropout, heads):
    import torch
    import torch.nn as nn

    n_rel = len(RELATIONS)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_p = nn.Linear(d_p, hidden)
            self.proj_t = nn.Linear(d_t, hidden)
            self.code_emb = nn.Embedding(graph.n_codes, hidden)
            nn.init.normal_(self.code_emb.weight, std=0.1)

            dims = [hidden] * n_layers
            dims[-1] = out
            self.layers = nn.ModuleList()
            d_in = hidden
            for d_out in dims:
                self.layers.append(_make_layer(kind, d_in, d_out, n_rel, heads))
                d_in = d_out
            self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims])
            self.dropout = nn.Dropout(dropout)
            # Projection head used only by the contrastive pretraining stage.
            self.projector = nn.Sequential(
                nn.Linear(out, out), nn.ReLU(), nn.Linear(out, out)
            )

        def input_features(self, Xp, Xt):
            h = torch.cat(
                [
                    self.proj_p(Xp),
                    self.code_emb.weight,
                    self.proj_t(Xt),
                ],
                dim=0,
            )
            return h

        def forward(self, Xp, Xt, adjs, edges):
            h = self.input_features(Xp, Xt)
            for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
                h = layer(h, adjs, edges)
                h = norm(h)
                if i < len(self.layers) - 1:
                    h = self.dropout(torch.relu(h))
            return h

    return Encoder()


def _make_decoder(kind: str, dim: int):
    import torch
    import torch.nn as nn

    class DotDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.bias = nn.Parameter(torch.tensor(0.0))

        def forward(self, zp, zt):
            return (self.scale * (zp * zt).sum(-1) + self.bias).unsqueeze(-1)

    class MLPDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3 * dim, dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(dim, 1)
            )

        def forward(self, zp, zt):
            return self.net(torch.cat([zp, zt, zp * zt], dim=-1))

    return DotDecoder() if kind == "dot" else MLPDecoder()


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------
class _GNNBase(BaseMatcher):
    conv_kind = "sage"
    contrastive_pretrain = False
    uses_pair_features = False

    def _prepare(self, ctx: TrainContext):
        import torch

        cfgg = ctx.cfg.gnn
        self.device = torch.device(ctx.device)
        self.graph = build_graph(ctx)

        Xp = ctx.bundle.P.astype(np.float32)
        Xt = ctx.bundle.T.astype(np.float32)
        self.Xp = torch.tensor(Xp, device=self.device)
        self.Xt = torch.tensor(Xt, device=self.device)

        norm = "sym" if self.conv_kind == "gcn" else "mean"
        self.adjs = {
            rel: _build_sparse_adj(e, self.graph.n_nodes, norm, self.device)
            for rel, e in self.graph.edges.items()
        }
        merged = np.concatenate(
            [e for e in self.graph.edges.values() if e.shape[1] > 0]
            or [np.zeros((2, 0), np.int64)],
            axis=1,
        )
        self.adjs["_merged"] = _build_sparse_adj(merged, self.graph.n_nodes, norm, self.device)
        self.merged_edges_np = merged
        self.edges = (
            torch.tensor(merged[0], dtype=torch.long, device=self.device),
            torch.tensor(merged[1], dtype=torch.long, device=self.device),
        )

        hidden = int(self.params.get("hidden_dim", cfgg.hidden_dim))
        out = int(self.params.get("out_dim", cfgg.out_dim))
        heads = int(self.params.get("heads", cfgg.heads))
        if self.conv_kind == "gat":
            hidden = max(heads, (hidden // heads) * heads)
            out = max(heads, (out // heads) * heads)

        self.encoder = _make_encoder(
            self.conv_kind, self.graph, Xp.shape[1], Xt.shape[1], hidden, out,
            int(self.params.get("num_layers", cfgg.num_layers)),
            float(self.params.get("dropout", cfgg.dropout)), heads,
        ).to(self.device)
        self.decoder = _make_decoder(
            str(self.params.get("decoder", cfgg.decoder)), out
        ).to(self.device)

    # -- forward helpers ------------------------------------------------
    def _embed(self, drop_edge: float = 0.0):
        import torch

        if drop_edge <= 0:
            return self.encoder(self.Xp, self.Xt, self.adjs, self.edges)

        # Edge dropout: the augmentation used by the upstream Stage A GCL.
        keep = np.random.random(self.merged_edges_np.shape[1]) > drop_edge
        e = self.merged_edges_np[:, keep]
        adjs = dict(self.adjs)
        norm = "sym" if self.conv_kind == "gcn" else "mean"
        adjs["_merged"] = _build_sparse_adj(e, self.graph.n_nodes, norm, self.device)
        edges = (
            torch.tensor(e[0], dtype=torch.long, device=self.device),
            torch.tensor(e[1], dtype=torch.long, device=self.device),
        )
        return self.encoder(self.Xp, self.Xt, adjs, edges)

    def _pair_logits(self, H, p_idx, t_idx):
        import torch

        p = torch.as_tensor(p_idx, dtype=torch.long, device=self.device)
        t = torch.as_tensor(t_idx, dtype=torch.long, device=self.device) + self.graph.trial_offset
        return self.decoder(H[p], H[t])

    # -- contrastive pretraining ---------------------------------------
    def _pretrain(self, ctx: TrainContext, epochs: int = 60, temperature: float = 0.1):
        """NT-Xent over two edge-dropped views (upstream Stage A analogue)."""
        import torch
        import torch.nn.functional as F

        opt = torch.optim.AdamW(self.encoder.parameters(), lr=1e-3, weight_decay=1e-5)
        n_sample = min(1024, self.graph.n_patients)

        for ep in range(epochs):
            self.encoder.train()
            h1 = self.encoder.projector(self._embed(drop_edge=0.15))
            h2 = self.encoder.projector(self._embed(drop_edge=0.20))

            idx = torch.randperm(self.graph.n_patients, device=self.device)[:n_sample]
            z1 = F.normalize(h1[idx], dim=-1)
            z2 = F.normalize(h2[idx], dim=-1)

            logits = z1 @ z2.T / temperature
            target = torch.arange(z1.size(0), device=self.device)
            loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5.0)
            opt.step()
            if ep % 20 == 0:
                log.debug("[%s] pretrain epoch %d NT-Xent=%.4f", self.name, ep, float(loss))

    # -- supervised fit -------------------------------------------------
    def _fit(self, ctx: TrainContext) -> None:
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        cfgg = ctx.cfg.gnn
        self._prepare(ctx)

        if self.contrastive_pretrain:
            log.info("[%s] contrastive pretraining...", self.name)
            self._pretrain(ctx)

        train = ctx.split.train
        val = subsample_pairs(ctx.split.val, MAX_VAL_PAIRS, seed=self.seed)
        y_tr = torch.tensor(train.y.astype(np.float32), device=self.device)

        pw = torch.tensor(
            positive_class_weight(train.y, cap=50.0), dtype=torch.float32, device=self.device
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.AdamW(
            params,
            lr=float(self.params.get("lr", cfgg.lr)),
            weight_decay=float(self.params.get("weight_decay", cfgg.weight_decay)),
        )

        batch = int(self.params.get("batch_size", cfgg.batch_size))
        steps = int(min(32, max(1, np.ceil(len(train) / batch))))
        drop_edge = float(self.params.get("drop_edge", 0.1))
        rng = np.random.default_rng(self.seed)

        best, best_state, bad = -np.inf, None, 0
        for epoch in range(int(self.params.get("max_epochs", cfgg.max_epochs))):
            self.encoder.train(); self.decoder.train()
            total = 0.0
            for _ in range(steps):
                idx = rng.choice(len(train), size=min(batch, len(train)), replace=False)
                H = self._embed(drop_edge=drop_edge)
                logits = self._pair_logits(H, train.p_idx[idx], train.t_idx[idx]).view(-1)
                loss = loss_fn(logits, y_tr[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                total += float(loss)

            val_scores = self._score_with_graph(val)
            val_prauc = safe_pr_auc(val.y, val_scores)
            if np.isfinite(val_prauc) and val_prauc > best + 1e-5:
                best, bad = val_prauc, 0
                best_state = (
                    copy.deepcopy(self.encoder.state_dict()),
                    copy.deepcopy(self.decoder.state_dict()),
                )
            else:
                bad += 1
            if epoch % 10 == 0:
                log.debug("[%s] epoch %3d loss=%.4f val_pr_auc=%.4f", self.name, epoch,
                          total / steps, val_prauc)
            if bad >= int(self.params.get("patience", cfgg.patience)):
                log.info("[%s] early stop at epoch %d (best val PR-AUC %.4f)", self.name, epoch, best)
                break

        if best_state is not None:
            self.encoder.load_state_dict(best_state[0])
            self.decoder.load_state_dict(best_state[1])
        self.best_val_pr_auc = float(best)
        self._cached_H = None

    # -- inference ------------------------------------------------------
    def _score_with_graph(self, pairs: PairSet, batch: int = 100_000) -> np.ndarray:
        import torch

        self.encoder.eval(); self.decoder.eval()
        out = np.empty(len(pairs), dtype=np.float32)
        with torch.no_grad():
            H = self._embed(drop_edge=0.0)
            for s in range(0, len(pairs), batch):
                e = min(s + batch, len(pairs))
                logits = self._pair_logits(H, pairs.p_idx[s:e], pairs.t_idx[s:e])
                out[s:e] = logits.view(-1).cpu().numpy()
        return out

    def score(self, ctx: TrainContext, pairs: PairSet) -> np.ndarray:
        self.check_fitted()
        return self._score_with_graph(pairs)

    def proba(self, ctx: TrainContext, pairs: PairSet) -> Optional[np.ndarray]:
        return 1.0 / (1.0 + np.exp(-np.clip(self.score(ctx, pairs), -30, 30)))

    def export_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Patient and trial node embeddings, for retrieval or visualisation."""
        import torch

        self.check_fitted()
        with torch.no_grad():
            H = self._embed(0.0).cpu().numpy()
        g = self.graph
        return H[: g.n_patients], H[g.trial_offset :]

    def hyperparameter_space(self, rng: np.random.Generator) -> Dict:
        return {
            "hidden_dim": int(rng.choice([64, 128, 256])),
            "out_dim": int(rng.choice([32, 64, 128])),
            "num_layers": int(rng.choice([2, 3])),
            "dropout": float(rng.uniform(0.0, 0.5)),
            "lr": float(10 ** rng.uniform(-3.3, -2.0)),
            "drop_edge": float(rng.uniform(0.0, 0.3)),
            "decoder": str(rng.choice(["dot", "mlp"])),
        }


@register("gcn")
class GCNMatcher(_GNNBase):
    name = "GCN"
    conv_kind = "gcn"


@register("graphsage")
class GraphSAGEMatcher(_GNNBase):
    name = "GraphSAGE"
    conv_kind = "sage"


@register("gat")
class GATMatcher(_GNNBase):
    name = "GAT"
    conv_kind = "gat"


@register("rgcn")
class RGCNMatcher(_GNNBase):
    name = "R-GCN (hetero)"
    conv_kind = "rgcn"


@register("graphsage_gcl")
class GraphSAGEGCLMatcher(_GNNBase):
    """Stage A contrastive pretraining, then Stage B supervised fine-tuning."""

    name = "GraphSAGE+GCL"
    conv_kind = "sage"
    contrastive_pretrain = True
