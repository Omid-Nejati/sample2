"""Regression tests: supervised-only affinity probes, complete optimizer gradients.

These CPU tests use deliberately stacked spectral layers. Parallel q_proj/v_proj
in the existing toy model can hide contamination because no later adapter's
regularizer depends on an earlier adapter's *output*.
"""
import copy
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from medqwenv2_sear.spectral import SpectralMoELinear, spectral_layers
from medqwenv2_sear.trainer import (
    _difficulty_loss,
    _regularizer,
    _spatial_difficulty_target,
    train_pair_step,
)


class StackedSpectralTasks(nn.Module):
    """Small deterministic network with downstream regularizer coupling."""

    def __init__(self):
        super().__init__()
        self.first = SpectralMoELinear(
            nn.Linear(8, 8), experts=2, rank_per_expert=2,
            top_k=2, svd_init=True
        )
        self.second = SpectralMoELinear(
            nn.Linear(8, 8), experts=2, rank_per_expert=2,
            top_k=2, svd_init=True
        )
        with torch.no_grad():
            self.second.router.weight[0].copy_(
                torch.tensor([0.40, -0.35, 0.25, 0.10, -0.20, 0.15, 0.30, -0.10])
            )
            self.second.router.weight[1].copy_(
                torch.tensor([-0.20, 0.10, -0.30, 0.15, 0.35, -0.10, -0.25, 0.20])
            )
            self.second.ema_cosine.fill_(-0.75)
            self.second.usage_seg.copy_(torch.tensor([0.8, 0.2]))
            self.second.usage_vqa.copy_(torch.tensor([0.3, 0.7]))

    @contextmanager
    def task(self, task_name):
        layers = [layer for _, layer in spectral_layers(self)]
        previous = [layer.capture_task for layer in layers]
        for layer in layers:
            layer.capture_task = task_name
        try:
            yield
        finally:
            for layer, old in zip(layers, previous):
                layer.capture_task = old

    def _forward(self, batch):
        return self.second(self.first(batch["x"]).tanh())

    def forward_vqa(self, vqa_inputs):
        output = self._forward(vqa_inputs)
        return {"loss": F.mse_loss(output, vqa_inputs["target"])}

    def forward_seg(self, seg_inputs, mask, *, spatial_features=None,
                    native_sam_batch=None, predicted_box=False):
        output = self._forward(seg_inputs)
        mask_logits = output[:, :1].reshape(-1, 1, 1, 1).expand(-1, 1, 4, 4)
        loss = F.mse_loss(mask_logits, mask) + 0.1 * output.square().mean()
        return {"loss": loss, "mask_logits": mask_logits}


def _inputs():
    x = torch.tensor([
        [-1.1, 0.4, -0.3, 0.8, 0.2, -0.5, 0.6, 0.9],
        [0.8, -0.9, 0.5, -0.7, 1.0, 0.1, -0.4, 0.3],
        [-0.4, 0.3, 1.1, 0.2, -0.8, 0.7, -0.6, 0.4],
    ])
    vqa = {"x": x, "target": x.flip(-1) / 2}
    seg = {"x": x.flip(0)}
    mask = torch.zeros(3, 1, 4, 4)
    mask[0, :, 1:3, 1:3] = 1.0
    mask[1, :, 0:2, 2:4] = 1.0
    return vqa, seg, mask


def _new_model():
    torch.manual_seed(119)
    return StackedSpectralTasks()


def _run_one_step(model, vqa, seg, mask, *, route_weight,
                  balance_weight, difficulty_weight, update_conflict=True):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
    return train_pair_step(
        model, opt, vqa, seg, mask,
        route_weight=route_weight, balance_weight=balance_weight,
        difficulty_weight=difficulty_weight, update_conflict=update_conflict,
        grad_clip=1e9,
    )


def test_conflict_measurement_is_invariant_to_auxiliary_weights():
    base = _new_model()
    zero_aux = copy.deepcopy(base)
    heavy_aux = copy.deepcopy(base)
    vqa, seg, mask = _inputs()

    info0 = _run_one_step(zero_aux, vqa, seg, mask,
                          route_weight=0.0, balance_weight=0.0,
                          difficulty_weight=0.0)
    info1 = _run_one_step(heavy_aux, vqa, seg, mask,
                          route_weight=10.0, balance_weight=7.0,
                          difficulty_weight=9.0)
    assert info0["conflict_cosine"] is not None
    assert info1["conflict_cosine"] is not None
    for (name_a, la), (name_b, lb) in zip(spectral_layers(zero_aux),
                                         spectral_layers(heavy_aux)):
        assert name_a == name_b
        # Compare both the signed affinity and the *norms* because cosine can
        # remain very close while its underlying gradient is contaminated.
        for name in ("ema_cosine", "ema_vqa_norm", "ema_seg_norm", "role_score"):
            torch.testing.assert_close(getattr(la, name), getattr(lb, name),
                                       atol=1e-6, rtol=1e-5)
        assert not la._projections and not lb._projections
        assert la.collect_projection and lb.collect_projection


def test_split_backward_matches_full_objective_parameter_gradients():
    split = _new_model()
    reference = copy.deepcopy(split)
    vqa, seg, mask = _inputs()
    route_weight, balance_weight, difficulty_weight, seg_weight = 2.0, 3.0, 0.7, 1.6

    # Production training path, with split backwards and hooks isolated.
    opt = torch.optim.SGD([p for p in split.parameters() if p.requires_grad], lr=0.0)
    train_pair_step(split, opt, vqa, seg, mask, seg_weight=seg_weight,
                    route_weight=route_weight, balance_weight=balance_weight,
                    difficulty_weight=difficulty_weight, grad_clip=1e9)

    # Independent reference: original complete objective backpropagation.
    reference.zero_grad(set_to_none=True)
    with reference.task("vqa"):
        v = reference.forward_vqa(vqa)
        (v["loss"] + _regularizer(reference, "vqa", route_weight,
                                   balance_weight)).backward()
    for _, layer in spectral_layers(reference):
        layer.finish_task("vqa")
    with reference.task("seg"):
        s = reference.forward_seg(seg, mask)
        target = _spatial_difficulty_target(s["mask_logits"], mask)
        loss = (seg_weight * s["loss"]
                + _regularizer(reference, "seg", route_weight, balance_weight)
                + difficulty_weight * _difficulty_loss(reference, target))
        loss.backward()
    for _, layer in spectral_layers(reference):
        layer.finish_task("seg")
    torch.nn.utils.clip_grad_norm_((p for p in reference.parameters()
                                    if p.requires_grad), 1e9)

    any_nonzero = False
    for (name_a, pa), (name_b, pb) in zip(split.named_parameters(),
                                         reference.named_parameters()):
        assert name_a == name_b
        assert (pa.grad is None) == (pb.grad is None), name_a
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad, atol=1e-5, rtol=1e-5,
                                       msg=name_a)
            any_nonzero |= bool(pa.grad.abs().max() > 0)
    assert any_nonzero


def test_disabled_conflict_measurement_uses_no_stale_probes():
    model = _new_model()
    vqa, seg, mask = _inputs()
    for _, layer in spectral_layers(model):
        layer._projections["vqa"] = torch.ones(2, 2, 2)
        layer._projections["seg"] = torch.ones(2, 2, 2)
    info = _run_one_step(model, vqa, seg, mask, route_weight=1.0,
                         balance_weight=1.0, difficulty_weight=1.0,
                         update_conflict=False)
    assert info["conflict_cosine"] is None
    for _, layer in spectral_layers(model):
        assert not layer._projections
        assert layer.collect_projection
