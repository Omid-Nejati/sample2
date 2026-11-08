"""Spectral MoE adapters with exact initialization centering and projected conflict statistics.

Important distinctions from GOAT:
 * The default uses a per-expert frozen anchor to preserve the original layer's
   output at step zero, for ANY routing selection (not just in expectation).
 * Projected full-weight gradients are captured through autograd on the layer's
   output using frozen SVD bases, avoiding dependence on the A/B factorization.
 * Adaptation capacity is allocated by routing mass; rank dimensions are FIXED.
   This prototype does NOT claim to implement adaptive rank-budget redistribution.

SVD initialization follows a GOAT-inspired nonoverlapping spectral partition.
No GOAT source code is reproduced here. Cite GOAT and MedQwen when using it.
"""
from __future__ import annotations
import math
from typing import Dict, Optional
import torch
from torch import nn
from torch.nn import functional as F


class SpectralMoELinear(nn.Module):
    def __init__(self, linear: nn.Linear, experts: int = 4, rank_per_expert: int = 2,
                 top_k: int = 2, rho: float = 10.0, svd_init: bool = True,
                 route_balance: float = 1.0, anchor: bool = True, ema_beta: float = 0.9):
        super().__init__()
        if not 1 <= top_k <= experts:
            raise ValueError("Expected 1 <= top_k <= experts")
        if rank_per_expert < 1 or rho <= 0:
            raise ValueError("rank_per_expert must be positive and rho > 0")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        if svd_init and experts * rank_per_expert > min(self.in_features, self.out_features):
            raise ValueError("SVD initialization requires E*r <= min(input,output) dimension")
        self.experts, self.rank_per_expert, self.top_k = experts, rank_per_expert, top_k
        self.ema_beta, self.route_balance = ema_beta, route_balance
        self.scale = math.sqrt(3.0 * self.in_features / rank_per_expert)
        self.anchor = anchor
        # Keep the original pretrained map; freeze only this map, not the inserted adapters.
        self.base = nn.Linear(self.in_features, self.out_features, bias=linear.bias is not None,
                              device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            self.base.weight.copy_(linear.weight)
            if linear.bias is not None:
                self.base.bias.copy_(linear.bias)
        self.base.requires_grad_(False)
        device, dtype = linear.weight.device, linear.weight.dtype
        A = torch.empty(experts, rank_per_expert, self.in_features, device=device, dtype=dtype)
        B = torch.empty(experts, self.out_features, rank_per_expert, device=device, dtype=dtype)
        # We retain the reference basis for conflict calculations even after A/B change.
        Uproj = torch.zeros(experts, self.out_features, rank_per_expert, device=device, dtype=torch.float32)
        Vproj = torch.zeros(experts, self.in_features, rank_per_expert, device=device, dtype=torch.float32)
        if svd_init:
            with torch.no_grad():
                u, s, vh = torch.linalg.svd(linear.weight.detach().float().cpu(), full_matrices=False)
                stride = min(self.in_features, self.out_features) // experts
                for e in range(experts):
                    ids = torch.arange(e * stride, e * stride + rank_per_expert)
                    singular = s[ids].sqrt().to(device=device, dtype=dtype)
                    factor = singular / math.sqrt(self.scale * rho)
                    B[e] = (u[:, ids].to(device=device, dtype=dtype) * factor).contiguous()
                    A[e] = (factor[:, None] * vh[ids].to(device=device, dtype=dtype)).contiguous()
                    Uproj[e] = u[:, ids].to(device=device)
                    Vproj[e] = vh[ids].T.to(device=device)
        else:
            # Random warm-start, only for smoke tests; still exact at initialization.
            nn.init.normal_(A, std=0.02)
            nn.init.normal_(B, std=0.02)
            for e in range(experts):
                # Random orthonormal bases (use CPU for portable QR).
                uq, _ = torch.linalg.qr(torch.randn(self.out_features, rank_per_expert))
                vq, _ = torch.linalg.qr(torch.randn(self.in_features, rank_per_expert))
                Uproj[e], Vproj[e] = uq.to(device), vq.to(device)
        self.A, self.B = nn.Parameter(A), nn.Parameter(B)
        self.register_buffer("anchor_A", A.detach().clone(), persistent=True)
        self.register_buffer("anchor_B", B.detach().clone(), persistent=True)
        self.register_buffer("basis_U", Uproj, persistent=True)
        self.register_buffer("basis_V", Vproj, persistent=True)
        self.router = nn.Linear(self.in_features, experts, device=device, dtype=dtype)
        nn.init.normal_(self.router.weight, std=1e-4)
        nn.init.zeros_(self.router.bias)
        # Role scores: [shared, language-specialist, segmentation-specialist].
        self.register_buffer("role_score", torch.zeros(experts, 3, device=device, dtype=torch.float32))
        self.register_buffer("ema_cosine", torch.zeros(experts, device=device, dtype=torch.float32))
        self.register_buffer("ema_vqa_norm", torch.zeros(experts, device=device, dtype=torch.float32))
        self.register_buffer("ema_seg_norm", torch.zeros(experts, device=device, dtype=torch.float32))
        self.register_buffer("usage_vqa", torch.full((experts,), 1.0/experts, device=device))
        self.register_buffer("usage_seg", torch.full((experts,), 1.0/experts, device=device))
        self.register_buffer("stats_steps", torch.zeros((), device=device, dtype=torch.long))
        self.register_buffer("difficulty_steps", torch.zeros((), device=device, dtype=torch.long))
        # Difficulty is inferred from activations, supervised separately using spatial error/uncertainty.
        self.difficulty = nn.Linear(self.in_features, 1, device=device, dtype=dtype)
        nn.init.zeros_(self.difficulty.weight)
        nn.init.zeros_(self.difficulty.bias)
        self.capture_task: Optional[str] = None
        self.recent_prob: Dict[str, torch.Tensor] = {}
        self.recent_difficulty: Dict[str, torch.Tensor] = {}
        self._projections: Dict[str, torch.Tensor] = {}

    def _projected_grad(self, x: torch.Tensor, dy: torch.Tensor, task: str) -> None:
        # G = dL/dW = Σ_n (dL/dy_n) x_n^T.
        # U_e^T G V_e = (dy @ U_e)^T (x @ V_e), without materializing G.
        with torch.no_grad():
            xx = x.reshape(-1, self.in_features).detach().float()
            yy = dy.reshape(-1, self.out_features).detach().float()
            u = self.basis_U
            v = self.basis_V
            pu = torch.einsum("no,eor->enr", yy, u)
            pv = torch.einsum("ni,eir->enr", xx, v)
            proj = torch.einsum("enr,ens->ers", pu, pv)
            self._projections[task] = proj.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        # Router is task-ID-free: it infers spatial relevance from token activations.
        diff = self.difficulty(x2).sigmoid()  # N,1
        role_shared = self.role_score[:, 0]
        role_language = self.role_score[:, 1]
        role_spatial = self.role_score[:, 2]
        role = (1 - diff) * (role_shared + role_language).unsqueeze(0) + diff * (role_shared + role_spatial).unsqueeze(0)
        logits = self.router(x2) + role.to(dtype=x2.dtype)
        probabilities = logits.softmax(dim=-1)
        values, ids = torch.topk(probabilities, k=self.top_k, dim=-1)
        values = values / values.sum(-1, keepdim=True).clamp_min(1e-9)
        result = self.base(x2)
        # Sparse token-to-expert dispatch: memory scales with assigned tokens, not N*E*out_dim.
        updates = torch.zeros_like(result)
        for e in range(self.experts):
            which = (ids == e).nonzero(as_tuple=False)
            if which.numel() == 0:
                continue
            rows, slots = which[:, 0], which[:, 1]
            xe = x2.index_select(0, rows)
            current = F.linear(F.linear(xe, self.A[e]), self.B[e])
            if self.anchor:
                initial = F.linear(F.linear(xe, self.anchor_A[e]), self.anchor_B[e])
                current = current - initial  # EXACT zero initially, any top-k or router weights.
            contribution = self.scale * current * values[rows, slots].unsqueeze(-1)
            updates.index_add_(0, rows, contribution)
        result = result + updates
        if self.capture_task is not None:
            task = self.capture_task
            self.recent_prob[task] = probabilities.mean(0)   # differentiable regularizer
            self.recent_difficulty[task] = diff.reshape(*shape[:-1]).reshape(shape[0], -1).mean(-1)
            if result.requires_grad:
                result.register_hook(lambda grad, inp=x2.detach(), name=task:
                                     self._projected_grad(inp, grad, name))
        return result.reshape(*shape[:-1], self.out_features)

    def task_regularizer(self, task: str, conflict_strength: float = 1.0,
                         balance_strength: float = 0.01) -> torch.Tensor:
        if task not in self.recent_prob:
            return self.A.sum() * 0
        p = self.recent_prob[task]
        other = self.usage_seg if task == "vqa" else self.usage_vqa
        conflict = (-self.ema_cosine).relu().detach()
        # Lagged cross-task load: avoids retaining both VQA and segmentation graphs.
        interference = (conflict * p * other).sum()
        balance = self.experts * p.square().sum() - 1.0
        return conflict_strength * interference + balance_strength * balance

    @torch.no_grad()
    def finish_task(self, task: str, beta: float = 0.9) -> None:
        usage = self.usage_vqa if task == "vqa" else self.usage_seg
        if task in self.recent_prob:
            usage.mul_(beta).add_(self.recent_prob[task].detach().float(), alpha=1-beta)
            self.recent_prob.pop(task)
        self.recent_difficulty.pop(task, None)

    @torch.no_grad()
    def update_affinity(self) -> Optional[torch.Tensor]:
        if not all(k in self._projections for k in ("vqa", "seg")):
            return None
        a = self._projections.pop("vqa").flatten(1)
        b = self._projections.pop("seg").flatten(1)
        an, bn = a.norm(dim=1), b.norm(dim=1)
        valid = (an > 1e-9) & (bn > 1e-9)
        cosine = F.cosine_similarity(a, b, dim=1).clamp(-1, 1)
        beta = self.ema_beta
        self.ema_cosine[valid] = beta * self.ema_cosine[valid] + (1-beta) * cosine[valid]
        self.ema_vqa_norm.mul_(beta).add_(an, alpha=1-beta)
        self.ema_seg_norm.mul_(beta).add_(bn, alpha=1-beta)
        # Shared on positive cosine; specialist bias on conflicting directions.
        common = self.ema_cosine.relu()
        conflict = (-self.ema_cosine).relu()
        total = (self.ema_vqa_norm + self.ema_seg_norm).clamp_min(1e-8)
        ratio_vqa = self.ema_vqa_norm / total
        ratio_seg = self.ema_seg_norm / total
        scores = torch.stack((common, conflict * ratio_vqa, conflict * ratio_seg), dim=-1)
        self.role_score.mul_(beta).add_(scores, alpha=1-beta)
        self.stats_steps.add_(1)
        return cosine


def spectral_layers(model: nn.Module):
    return [(name, module) for name, module in model.named_modules() if isinstance(module, SpectralMoELinear)]


def install_spectral_experts(model: nn.Module, targets=("q_proj", "v_proj"),
                             only_language: bool = True, **kwargs) -> int:
    """Replace selected nn.Linear modules, preserving their pretrained function at init.

    Assumes Qwen3VL path includes `.language_model.`; for a toy model use
    `only_language=False`. `svd_init=True` can be costly on 8B weights.
    """
    candidates = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and name.split(".")[-1] in targets:
            if not only_language or ".language_model." in ("." + name + "."):
                candidates.append((name, m))
    for name, layer in candidates:
        owner_name, key = name.rsplit(".", 1) if "." in name else ("", name)
        owner = model.get_submodule(owner_name) if owner_name else model
        setattr(owner, key, SpectralMoELinear(layer, **kwargs))
    return len(candidates)
