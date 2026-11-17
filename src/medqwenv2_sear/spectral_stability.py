"""Optional, auditable control of SVD-initialized spectral experts.

This module intentionally DOES NOT modify routing or the fixed-basis gradient
conflict proxy. All controls are disabled in train_pair_step by default.

Do not confuse rank_per_expert (the factors' inner dimension) with the rank
of BA - B0 A0, which can reach 2 * rank_per_expert in the unconstrained model.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from .spectral import SpectralMoELinear, spectral_layers


def _orthogonal_component(layer: SpectralMoELinear):
    """Differentiable residual factors outside their initial SVD subspaces.

    For an SVD-initialized expert e:
      B_perp = B - U_e (U_e^T B)
      A_perp = A - (A V_e) V_e^T.
    A/B are promoted to float32 for numerical stability (grads still flow).
    """
    a = layer.A.float()
    b = layer.B.float()
    u = layer.basis_U.float()
    v = layer.basis_V.float()
    b_projected = u @ (u.transpose(-2, -1) @ b)
    a_projected = (a @ v) @ v.transpose(-2, -1)
    return (a - a_projected), (b - b_projected), a_projected, b_projected


def layer_subspace_penalty(layer: SpectralMoELinear, eps: float = 1e-12) -> Tensor:
    """Scale-independent soft penalty for drifting from frozen SVD bases.

    Returns average over experts of normalized squared perpendicular factor
    energy, averaged over A and B; zero for perfectly basis-aligned factors.
    Does not guarantee separation, and has no effect unless its weight > 0.
    """
    a_perp, b_perp, _, _ = _orthogonal_component(layer)
    a, b = layer.A.float(), layer.B.float()
    pa = a_perp.square().sum(dim=(-2, -1)) / a.square().sum(dim=(-2, -1)).clamp_min(eps)
    pb = b_perp.square().sum(dim=(-2, -1)) / b.square().sum(dim=(-2, -1)).clamp_min(eps)
    return 0.5 * (pa + pb).mean()


def model_subspace_penalty(model: nn.Module) -> Tensor:
    """Unweighted mean across spectral layers, not a sum that grows with depth."""
    layers = spectral_layers(model)
    if not layers:
        raise ValueError("No SpectralMoELinear layers found")
    return torch.stack([layer_subspace_penalty(layer) for _, layer in layers]).mean()


def _anchor_misalignment(layer: SpectralMoELinear, eps: float = 1e-12) -> float:
    """Check that the FROZEN anchors lie in the selected SVD subspaces."""
    with torch.no_grad():
        a0, b0 = layer.anchor_A.float(), layer.anchor_B.float()
        u, v = layer.basis_U.float(), layer.basis_V.float()
        ab_err = (a0 - (a0 @ v) @ v.transpose(-2, -1)).norm() / a0.norm().clamp_min(eps)
        bb_err = (b0 - u @ (u.transpose(-2, -1) @ b0)).norm() / b0.norm().clamp_min(eps)
        return float(torch.maximum(ab_err, bb_err))


@torch.no_grad()
def project_model_to_initial_subspaces_(model: nn.Module, *, anchor_tolerance: float = 1e-2) -> None:
    """Hard projection of current A/B back into each expert's INITIAL span.

    This is projected AdamW when called just after optimizer.step(). For
    genuinely SVD-aligned anchors, each effective correction then belongs to
    U_e (r x r) V_e^T, hence has mathematical rank <= r rather than <= 2r.
    Different experts' corrections are spectrally separate, up to rounding.

    IMPORTANT: This changes the method and is intended as an ABLATION, not the
    default SEAR configuration. Call only with SVD initialized adapters; the
    function rejects random-initialization anchors that do not match U/V.
    """
    layers = spectral_layers(model)
    if not layers:
        raise ValueError("No SpectralMoELinear layers found")
    # Validate ALL layers before changing ANY tensors.
    for name, layer in layers:
        if not layer.anchor:
            raise ValueError(f"{name}: hard projection requires anchor=True")
        err = _anchor_misalignment(layer)
        if err > anchor_tolerance:
            raise ValueError(
                f"{name}: anchors are not aligned with the fixed bases "
                f"(relative error={err:.4g}); use svd_init=True."
            )
    for _, layer in layers:
        _, _, a_projected, b_projected = _orthogonal_component(layer)
        layer.A.copy_(a_projected.to(dtype=layer.A.dtype))
        layer.B.copy_(b_projected.to(dtype=layer.B.dtype))


@torch.no_grad()
def summarize_factor_geometry(model: nn.Module) -> dict:
    """Cheap in-memory summary (requires no dense out_dim x in_dim products).

    `*_drift` is normalized off-basis factor energy. `*_overlap` is the mean
    squared cosine of pairwise factor subspace angles; computed via reduced QR.
    The full anchored-correction energy/overlap analysis is in
    scripts/audit_sear_checkpoint.py; do not substitute factor metrics for it.
    """
    def span(x: Tensor):
        # Thin SVD handles rank deficiency and near-zero factors robustly.
        u, s, _ = torch.linalg.svd(x.float(), full_matrices=False)
        if not s.numel():
            return u[:, :0]
        tolerance = max(x.shape) * torch.finfo(s.dtype).eps * s.max()
        return u[:, s > tolerance]

    def overlap(q1: Tensor, q2: Tensor) -> float | None:
        r = min(q1.shape[1], q2.shape[1])
        if r == 0:
            return None
        return float(((q1.T @ q2).square().sum() / r).clamp(0, 1))

    def avg(xs):
        return sum(xs) / len(xs) if xs else None

    drift_a, drift_b, overlaps_a, overlaps_b = [], [], [], []
    layers = spectral_layers(model)
    for _, layer in layers:
        a_perp, b_perp, _, _ = _orthogonal_component(layer)
        a, b = layer.A.float(), layer.B.float()
        drift_a.extend((a_perp.square().sum((-2, -1)) /
                        a.square().sum((-2, -1)).clamp_min(1e-12)).tolist())
        drift_b.extend((b_perp.square().sum((-2, -1)) /
                        b.square().sum((-2, -1)).clamp_min(1e-12)).tolist())
        spans_a = [span(a[e].T) for e in range(layer.experts)]
        spans_b = [span(b[e]) for e in range(layer.experts)]
        for e in range(layer.experts):
            for f in range(e + 1, layer.experts):
                oa, ob = overlap(spans_a[e], spans_a[f]), overlap(spans_b[e], spans_b[f])
                if oa is not None:
                    overlaps_a.append(oa)
                if ob is not None:
                    overlaps_b.append(ob)
    return {
        "num_spectral_layers": len(layers),
        "num_experts": len(drift_a),
        "right_factor_drift_mean": avg(drift_a),
        "left_factor_drift_mean": avg(drift_b),
        "right_factor_overlap_mean": avg(overlaps_a),
        "left_factor_overlap_mean": avg(overlaps_b),
    }
