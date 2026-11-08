"""Memory-conscious sequential two-task update (one optimizer step per pair)."""
from contextlib import nullcontext
import torch
from torch import nn
from .spectral import spectral_layers
from torch.nn import functional as F


def _spatial_difficulty_target(mask_logits,mask_target):
    """Detached, per-image foreground/boundary error; no GT at inference."""
    p = mask_logits.detach().float().sigmoid()
    gt = mask_target.float()
    if gt.ndim == 3:
        gt = gt[:,None]
    if p.shape[-2:] != gt.shape[-2:]:
        p = F.interpolate(p,gt.shape[-2:],mode="bilinear",align_corners=False)
    error = (p-gt).abs()
    dilation = F.max_pool2d(gt,3,1,1)
    erosion = -F.max_pool2d(-gt,3,1,1)
    edge = (dilation-erosion).clamp(0,1)
    boundary_error = (error*edge).flatten(1).sum(1)/edge.flatten(1).sum(1).clamp_min(1)
    return (0.5*error.flatten(1).mean(1)+0.5*boundary_error).clamp(0,1)


def _difficulty_loss(model, target):
    losses=[]
    for _,layer in spectral_layers(model):
        if "seg" in layer.recent_difficulty:
            pred=layer.recent_difficulty["seg"]
            if pred.shape == target.shape:
                losses.append(F.mse_loss(pred,target.detach()))
    return torch.stack(losses).mean() if losses else target.sum()*0


def _regularizer(model,task,route_weight,balance_weight):
    reg = None
    for _,layer in spectral_layers(model):
        term = layer.task_regularizer(task,conflict_strength=route_weight,
                                      balance_strength=balance_weight)
        reg = term if reg is None else reg + term
    return reg if reg is not None else next(model.parameters()).sum()*0


def train_pair_step(model, optimizer, vqa_inputs, seg_inputs, mask,
                    *, spatial_features=None, native_sam_batch=None,
                    seg_weight=1.0, route_weight=0.05, balance_weight=0.01,
                    grad_clip=1.0, predicted_box=False, update_conflict=True,
                    difficulty_weight=0.1):
    """One language minibatch + one segmentation minibatch, with memory reuse.

    `seg_inputs` goes through the VLM separately. Capture gradients in a *fixed*
    spectral coordinate system. Route penalty uses lagged usage from the other
    task, avoiding retention of both task graphs. For DDP/FSDP, customize this.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with model.task("vqa"):
        vqa = model.forward_vqa(vqa_inputs)
        vqa_objective = vqa["loss"] + _regularizer(model,"vqa",route_weight,balance_weight)
        vqa_objective.backward()
    for _,layer in spectral_layers(model):
        layer.finish_task("vqa")
    with model.task("seg"):
        seg = model.forward_seg(seg_inputs,mask,spatial_features=spatial_features,
                                native_sam_batch=native_sam_batch,predicted_box=predicted_box)
        spatial_target = _spatial_difficulty_target(seg["mask_logits"],mask)
        difficulty_term = _difficulty_loss(model,spatial_target)
        seg_objective = (seg_weight*seg["loss"] +
                         _regularizer(model,"seg",route_weight,balance_weight) +
                         difficulty_weight*difficulty_term)
        seg_objective.backward()
    for _,layer in spectral_layers(model):
        layer.finish_task("seg")
    torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),grad_clip)
    optimizer.step()
    conflicts=[]
    if update_conflict:
        for _,layer in spectral_layers(model):
            cosine = layer.update_affinity()
            if cosine is not None:
                conflicts.append(cosine.mean().item())
    return {"vqa_loss":float(vqa["loss"].detach()),
            "seg_loss":float(seg["loss"].detach()),
            "conflict_cosine":sum(conflicts)/len(conflicts) if conflicts else None,
            "difficulty_loss": float(difficulty_term.detach()),
            "mask_logits":seg["mask_logits"].detach()}
