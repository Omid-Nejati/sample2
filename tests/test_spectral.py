import torch
from torch import nn
from medqwenv2_sear.spectral import SpectralMoELinear,install_spectral_experts


def test_exact_anchor_and_gradient():
    torch.manual_seed(0)
    original=nn.Linear(16,12)
    layer=SpectralMoELinear(original,experts=4,rank_per_expert=2,top_k=2,svd_init=True)
    x=torch.randn(2,3,16)
    torch.testing.assert_close(layer(x),original(x),atol=1e-6,rtol=1e-6)
    target=torch.randn(2,3,12)
    layer.capture_task="vqa"
    (layer(x)-target).square().mean().backward()
    assert torch.isfinite(layer.A.grad).all()
    assert layer._projections["vqa"].shape==(4,2,2)


def test_conflict_projection_and_role_update():
    torch.manual_seed(1)
    layer=SpectralMoELinear(nn.Linear(8,8),experts=2,rank_per_expert=2,svd_init=True)
    for name in ["vqa","seg"]:
        layer.zero_grad(set_to_none=True)
        layer.capture_task=name
        layer(torch.randn(3,8)).square().sum().backward()
        assert name in layer._projections
        layer.finish_task(name)
    cosine=layer.update_affinity()
    assert cosine.shape==(2,)
    assert torch.isfinite(cosine).all()
    assert layer.stats_steps.item()==1
    assert layer.role_score.shape==(2,3)


def test_injection_respects_language_filter():
    m=nn.Sequential(nn.Linear(8,8))
    assert install_spectral_experts(m,targets=("0",),only_language=True)==0
