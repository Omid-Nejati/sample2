"""CPU regression tests for the optional spectral subspace controls."""
import torch
from torch import nn
import pytest
from medqwenv2_sear.spectral import SpectralMoELinear
from medqwenv2_sear.spectral_stability import (
    layer_subspace_penalty,
    model_subspace_penalty,
    project_model_to_initial_subspaces_,
    summarize_factor_geometry,
)


def test_initial_svd_alignment_and_zero_effective_update():
    torch.manual_seed(123)
    original = nn.Linear(12, 10)
    layer = SpectralMoELinear(original, experts=2, rank_per_expert=2, top_k=1, svd_init=True)
    x = torch.randn(3, 12)
    torch.testing.assert_close(layer(x), original(x), atol=1e-6, rtol=1e-6)
    assert layer_subspace_penalty(layer).item() < 1e-9
    metrics = summarize_factor_geometry(layer)
    assert metrics['num_experts'] == 2
    assert metrics['left_factor_drift_mean'] < 1e-9
    assert metrics['right_factor_drift_mean'] < 1e-9


def test_unrestricted_factors_drift_and_projection_restores_separation():
    torch.manual_seed(5)
    layer = SpectralMoELinear(nn.Linear(12, 12), experts=2,
                              rank_per_expert=2, top_k=1, svd_init=True)
    # Rotate expert 0 toward expert 1: unrestricted A and B permit this.
    with torch.no_grad():
        layer.B[0].add_(0.1 * layer.basis_U[1].to(layer.B.dtype))
        layer.A[0].add_(0.1 * layer.basis_V[1].T.to(layer.A.dtype))
    assert layer_subspace_penalty(layer).item() > 1e-5
    layer_subspace_penalty(layer).backward()
    assert layer.A.grad is not None and torch.isfinite(layer.A.grad).all()
    assert layer.B.grad is not None and torch.isfinite(layer.B.grad).all()
    project_model_to_initial_subspaces_(layer)
    assert layer_subspace_penalty(layer).item() < 1e-8
    # Both the trained and the frozen products now live in U_e [r,r] V_e^T.
    for e in range(layer.experts):
        correction = layer.B[e] @ layer.A[e] - layer.anchor_B[e] @ layer.anchor_A[e]
        u, v = layer.basis_U[e], layer.basis_V[e]
        reconstructed = u @ (u.T @ correction @ v) @ v.T
        torch.testing.assert_close(correction, reconstructed, atol=1e-5, rtol=1e-5)


def test_soft_regularizer_is_differentiable():
    torch.manual_seed(12)
    layer = SpectralMoELinear(nn.Linear(10, 10), experts=2,
                              rank_per_expert=2, svd_init=True)
    with torch.no_grad():
        layer.B[0].add_(0.1 * layer.basis_U[1].to(layer.B.dtype))
    value = model_subspace_penalty(layer)
    assert value.item() > 0
    value.backward()
    assert layer.B.grad is not None and layer.B.grad.isfinite().all()
    assert layer.B.grad.norm() > 0


def test_random_init_is_rejected_by_hard_projection():
    torch.manual_seed(15)
    layer = SpectralMoELinear(nn.Linear(8, 8), experts=2,
                              rank_per_expert=2, svd_init=False)
    with pytest.raises(ValueError, match='svd_init=True'):
        project_model_to_initial_subspaces_(layer)


def test_anchored_correction_can_have_twice_factor_rank():
    b0 = torch.tensor([[1.0], [0.0]])
    a0 = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[0.0], [1.0]])
    a = torch.tensor([[0.0, 1.0]])
    difference = b @ a - b0 @ a0
    assert torch.linalg.matrix_rank(b @ a).item() == 1
    assert torch.linalg.matrix_rank(difference).item() == 2


def test_training_integration_projected_ablation():
    from medqwenv2_sear.toy import make_toy_sear, make_synthetic_batch
    from medqwenv2_sear.trainer import train_pair_step
    from medqwenv2_sear.spectral import spectral_layers
    torch.manual_seed(42)
    model = make_toy_sear(hidden=16, experts=2, rank=2)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    vqa, _ = make_synthetic_batch(2, size=16)
    seg, target = make_synthetic_batch(2, size=16)
    with torch.no_grad():
        features = model.vlm.vision(seg['images'])
    info = train_pair_step(model, optimizer, vqa, seg, target,
                           spatial_features=features,
                           spectral_constraint_weight=0.02,
                           project_spectral_subspaces=True)
    assert info['conflict_cosine'] is not None
    assert all(layer_subspace_penalty(layer).item() < 1e-7
               for _, layer in spectral_layers(model))
