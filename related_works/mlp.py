"""
models/mlp.py
=============
Two neural baselines on the tabular features.

`MLPMatcher`
    A plain feed-forward net over the concatenated pair vector. This is the
    "deep learning without inductive bias" control: if it matches the GNN, the
    graph structure was not contributing anything the features didn't already
    carry.

`TwoTowerMatcher`
    Separate patient and trial encoders whose outputs are combined by a dot
    product (optionally an MLP head). This is the closest non-graph analogue of
    the upstream trial-aware alignment model: it produces a genuine embedding
    per entity, so unseen trials can be encoded at inference without retraining,
    and the same embeddings can be indexed for fast retrieval. It is therefore
    the fairest architectural comparison to the GCL approach, whereas the
    concat-MLP is not (it has no reusable entity representation at all).

Training details that matter for reproducibility
------------------------------------------------
* Early stopping tracks validation **PR-AUC**, not loss. With ~2% positives,
  BCE is minimised well before the ranking is any good.
* `pos_weight` is capped. Uncapped, on a fold where positives are very rare,
  the weight explodes and the net collapses to predicting everything positive.
* The best weights are restored at the end of training. Without restoration
  "early stopping with patience 8" means you keep the model from 8 epochs
  *past* the optimum, which is not what the phrase implies.
* All randomness is seeded, and the DataLoader uses a seeded generator, so a
  rerun with the same seed reproduces the number exactly.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..features import materialize, pair_features, subsample_pairs
from ..metrics import safe_pr_auc
from ..splits import PairSet
from .base import BaseMatcher, TrainContext, positive_class_weight, register

log = logging.getLogger(__name__)

MAX_VAL_PAIRS = 400_000


def _torch():
    import torch

    return torch


def _set_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mlp_stack(in_dim: int, hidden: Sequence[int], out_dim: int, dropout: float):
    import torch.nn as nn

    layers: List[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class _TorchMatcherMixin:
    """Shared fit loop: minibatch BCE, PR-AUC early stopping, weight restore."""

    def _iterate_scores(self, ctx: TrainContext, pairs: PairSet, batch: int = 8192) -> np.ndarray:
        torch = _torch()
        self._net.eval()
        out = np.empty(len(pairs), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(pairs), batch):
                e = min(s + batch, len(pairs))
                logits = self._forward_pairs(ctx, pairs.p_idx[s:e], pairs.t_idx[s:e])
                out[s:e] = logits.detach().cpu().numpy().ravel()
        return out

    def _run_training(
        self,
        ctx: TrainContext,
        max_epochs: int,
        patience: int,
        lr: float,
        weight_decay: float,
        batch_size: int,
        pos_weight_cap: float = 50.0,
    ) -> None:
        torch = _torch()
        import torch.nn as nn

        device = torch.device(ctx.device)
        self._net.to(device)

        train = ctx.split.train
        val = subsample_pairs(ctx.split.val, MAX_VAL_PAIRS, seed=self.seed)

        y_tr = torch.tensor(train.y.astype(np.float32), device=device)
        pw = torch.tensor(
            positive_class_weight(train.y, cap=pos_weight_cap),
            dtype=torch.float32, device=device,
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        opt = torch.optim.AdamW(self._net.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=max(2, patience // 3)
        )

        n = len(train)
        rng = np.random.default_rng(self.seed)
        best, best_state, bad = -np.inf, None, 0

        for epoch in range(max_epochs):
            self._net.train()
            perm = rng.permutation(n)
            total = 0.0
            for s in range(0, n, batch_size):
                idx = perm[s : s + batch_size]
                if idx.size < 2:      # BatchNorm needs >1 row
                    continue
                logits = self._forward_pairs(ctx, train.p_idx[idx], train.t_idx[idx])
                loss = loss_fn(logits.view(-1), y_tr[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 5.0)
                opt.step()
                total += float(loss) * idx.size

            val_scores = self._iterate_scores(ctx, val)
            val_prauc = safe_pr_auc(val.y, val_scores)
            sched.step(val_prauc if np.isfinite(val_prauc) else 0.0)

            if np.isfinite(val_prauc) and val_prauc > best + 1e-5:
                best, bad = val_prauc, 0
                best_state = copy.deepcopy(self._net.state_dict())
            else:
                bad += 1

            if epoch % 5 == 0 or bad == 0:
                log.debug(
                    "[%s] epoch %3d loss=%.4f val_pr_auc=%.4f (best %.4f)",
                    self.name, epoch, total / max(n, 1), val_prauc, best,
                )
            if bad >= patience:
                log.info("[%s] early stop at epoch %d (best val PR-AUC %.4f)", self.name, epoch, best)
                break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self.best_val_pr_auc = float(best)


@register("mlp")
class MLPMatcher(_TorchMatcherMixin, BaseMatcher):
    name = "MLP"

    def _forward_pairs(self, ctx, p_idx, t_idx):
        torch = _torch()
        X = pair_features(ctx.bundle, p_idx, t_idx)
        xb = torch.tensor(X, dtype=torch.float32, device=next(self._net.parameters()).device)
        xb = (xb - self._mu) / self._sd
        return self._net(xb)

    def _fit(self, ctx: TrainContext) -> None:
        torch = _torch()
        _set_seed(self.seed)
        mc = ctx.cfg.mlp

        X_sample, _ = materialize(ctx.bundle, subsample_pairs(ctx.split.train, 100_000, self.seed))
        device = torch.device(ctx.device)
        self._mu = torch.tensor(X_sample.mean(0), dtype=torch.float32, device=device)
        self._sd = torch.tensor(
            np.maximum(X_sample.std(0), 1e-6), dtype=torch.float32, device=device
        )

        hidden = tuple(self.params.get("hidden", mc.hidden))
        self._net = _mlp_stack(
            X_sample.shape[1], hidden, 1, float(self.params.get("dropout", mc.dropout))
        )
        self._run_training(
            ctx,
            max_epochs=mc.max_epochs,
            patience=mc.patience,
            lr=float(self.params.get("lr", mc.lr)),
            weight_decay=float(self.params.get("weight_decay", mc.weight_decay)),
            batch_size=int(self.params.get("batch_size", mc.batch_size)),
            pos_weight_cap=mc.pos_weight_cap,
        )

    def score(self, ctx: TrainContext, pairs: PairSet) -> np.ndarray:
        self.check_fitted()
        return self._iterate_scores(ctx, pairs)

    def proba(self, ctx: TrainContext, pairs: PairSet) -> Optional[np.ndarray]:
        return 1.0 / (1.0 + np.exp(-np.clip(self.score(ctx, pairs), -30, 30)))

    def hyperparameter_space(self, rng: np.random.Generator) -> Dict:
        return {
            "hidden": tuple(rng.choice([(128,), (256, 128), (512, 256), (256, 128, 64)], 1)[0]),
            "dropout": float(rng.uniform(0.0, 0.5)),
            "lr": float(10 ** rng.uniform(-4, -2.3)),
            "weight_decay": float(10 ** rng.uniform(-6, -3)),
            "batch_size": int(rng.choice([512, 1024, 2048])),
        }


@register("two_tower")
class TwoTowerMatcher(_TorchMatcherMixin, BaseMatcher):
    """Independent patient / trial encoders combined by a dot product."""

    name = "TwoTowerMLP"

    def _forward_pairs(self, ctx, p_idx, t_idx):
        torch = _torch()
        dev = next(self._net.parameters()).device
        p = torch.tensor(ctx.bundle.P[p_idx], dtype=torch.float32, device=dev)
        t = torch.tensor(ctx.bundle.T[t_idx], dtype=torch.float32, device=dev)
        return self._net(p, t)

    def _fit(self, ctx: TrainContext) -> None:
        torch = _torch()
        import torch.nn as nn

        _set_seed(self.seed)
        tc = ctx.cfg.two_tower
        d_p = ctx.bundle.P.shape[1]
        d_t = ctx.bundle.T.shape[1]
        k = int(self.params.get("embed_dim", tc.embed_dim))
        hidden = tuple(self.params.get("tower_hidden", tc.tower_hidden))
        dropout = float(self.params.get("dropout", tc.dropout))
        use_head = bool(self.params.get("use_head", True))

        class TwoTower(nn.Module):
            def __init__(self):
                super().__init__()
                self.p_tower = _mlp_stack(d_p, hidden, k, dropout)
                self.t_tower = _mlp_stack(d_t, hidden, k, dropout)
                self.head = (
                    nn.Sequential(nn.Linear(3 * k, k), nn.ReLU(), nn.Linear(k, 1))
                    if use_head
                    else None
                )
                self.scale = nn.Parameter(torch.tensor(1.0))
                self.bias = nn.Parameter(torch.tensor(0.0))

            def forward(self, p, t):
                zp = self.p_tower(p)
                zt = self.t_tower(t)
                if self.head is not None:
                    return self.head(torch.cat([zp, zt, zp * zt], dim=-1))
                return (self.scale * (zp * zt).sum(-1) + self.bias).unsqueeze(-1)

            def embed_patients(self, p):
                return self.p_tower(p)

            def embed_trials(self, t):
                return self.t_tower(t)

        self._net = TwoTower()
        self._run_training(
            ctx,
            max_epochs=tc.max_epochs,
            patience=tc.patience,
            lr=float(self.params.get("lr", tc.lr)),
            weight_decay=float(self.params.get("weight_decay", tc.weight_decay)),
            batch_size=int(self.params.get("batch_size", tc.batch_size)),
        )

    def score(self, ctx: TrainContext, pairs: PairSet) -> np.ndarray:
        self.check_fitted()
        return self._iterate_scores(ctx, pairs)

    def proba(self, ctx: TrainContext, pairs: PairSet) -> Optional[np.ndarray]:
        return 1.0 / (1.0 + np.exp(-np.clip(self.score(ctx, pairs), -30, 30)))

    def export_embeddings(self, ctx: TrainContext) -> Tuple[np.ndarray, np.ndarray]:
        """Entity embeddings, so retrieval can be done with an ANN index."""
        torch = _torch()
        self.check_fitted()
        dev = next(self._net.parameters()).device
        self._net.eval()
        with torch.no_grad():
            zp = self._net.embed_patients(
                torch.tensor(ctx.bundle.P, dtype=torch.float32, device=dev)
            ).cpu().numpy()
            zt = self._net.embed_trials(
                torch.tensor(ctx.bundle.T, dtype=torch.float32, device=dev)
            ).cpu().numpy()
        return zp, zt

    def hyperparameter_space(self, rng: np.random.Generator) -> Dict:
        return {
            "embed_dim": int(rng.choice([32, 64, 128])),
            "tower_hidden": tuple(rng.choice([(128,), (256, 128), (512, 256)], 1)[0]),
            "dropout": float(rng.uniform(0.0, 0.4)),
            "lr": float(10 ** rng.uniform(-4, -2.3)),
            "use_head": bool(rng.random() < 0.5),
        }
