"""Tests the injection interface using a stub, NOT actual SAM3 internals/weights."""
from types import SimpleNamespace
import torch
from torch import nn
from medqwenv2_sear.sam3_bridge import NativeSAM3Bridge
from medqwenv2_sear.prompts import RegionPrompts


class _Backbone(nn.Module):
    def forward_text(self,text_batch,*args,**kwargs):
        b=len(text_batch)
        return {"language_features":torch.zeros(4,b,8),
                "language_mask":torch.zeros(b,4,dtype=torch.bool),
                "language_embeds":torch.ones(4,b,16)}


class _NativeSAM3Stub(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=_Backbone()
        self.mask_layer=nn.Linear(8,5*5)

    def forward(self,native):
        prompt=self.backbone.forward_text(native.find_text_batch)
        text=prompt["language_features"].mean(0)
        mask=self.mask_layer(text).reshape(-1,1,5,5)
        return [{"pred_masks":mask,"pred_logits":text.mean(-1)[:,None,None]}]


def test_stub_bridge_gradient_to_prompts():
    stub=_NativeSAM3Stub()
    bridge=NativeSAM3Bridge(stub)
    tokens=torch.randn(2,3,8,requires_grad=True)
    prompts=RegionPrompts(tokens,torch.rand(2,3,4),torch.rand(2,3))
    native=SimpleNamespace(find_text_batch=["lesion","lesion"])
    out=bridge(native,prompts)
    assert out["pred_masks"].shape==(2,1,5,5)
    out["pred_masks"].square().sum().backward()
    assert tokens.grad is not None
    assert tokens.grad.abs().sum()>0
    # Original text encoder is restored after bridge.forward.
    assert stub.backbone.forward_text(["lesion"])["language_features"].shape==(4,1,8)
