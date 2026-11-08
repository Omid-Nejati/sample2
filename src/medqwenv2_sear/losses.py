"""Binary semantic/foreground mask losses and presence-grounding diagnostics.

For true multi-instance SAM3, use the native matching loss, NOT the semantic
union loss below. The loss here targets one conceptual union mask per image.
"""
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F


def _mask_inputs(logits, target):
    if logits.ndim == 3:
        logits = logits[:, None]
    if target.ndim == 3:
        target = target[:, None]
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, target.shape[-2:], mode="bilinear", align_corners=False)
    return logits.float(), target.float()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0):
    logits, target = _mask_inputs(logits, target)
    prob = logits.sigmoid()
    axes = tuple(range(1, prob.ndim))
    dice = (2*(prob*target).sum(axes)+eps) / (prob.sum(axes)+target.sum(axes)+eps)
    return 1-dice.mean()


def _edge(x: torch.Tensor):
    # Differentiable soft morphological gradient for predicted probabilities.
    high = F.max_pool2d(x, 3, stride=1, padding=1)
    low = -F.max_pool2d(-x, 3, stride=1, padding=1)
    return (high-low).clamp(0, 1)


def boundary_loss(logits: torch.Tensor, target: torch.Tensor, eps=1.0):
    logits, target = _mask_inputs(logits, target)
    p, t = _edge(logits.sigmoid()), _edge(target)
    # Optional soft edge Dice; nonempty and empty edge cases both well-defined.
    axes = tuple(range(1,p.ndim))
    edge_dice = (2*(p*t).sum(axes)+eps) / (p.sum(axes)+t.sum(axes)+eps)
    return 1-edge_dice.mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor,
                      bce_weight: float = 1.0, dice_weight: float = 1.0,
                      boundary_weight: float = 0.2):
    logits, target = _mask_inputs(logits, target)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = soft_dice_loss(logits, target)
    boundary = boundary_loss(logits, target)
    total = bce_weight*bce + dice_weight*dice + boundary_weight*boundary
    return {"loss": total, "bce": bce, "dice": dice, "boundary": boundary}


@torch.no_grad()
def dice_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    logits,target = _mask_inputs(logits,target)
    pred, gt = logits.sigmoid() > threshold, target > 0.5
    reduce = tuple(range(1,pred.ndim))
    tp = (pred & gt).sum(reduce)
    denom = pred.sum(reduce) + gt.sum(reduce)
    return torch.where(denom==0, torch.ones_like(denom).float(), 2*tp.float()/denom.clamp_min(1))


@torch.no_grad()
def boundary_f1(logits: torch.Tensor, target: torch.Tensor, threshold=0.5, tolerance=2):
    logits,target = _mask_inputs(logits,target)
    p = (logits.sigmoid()>threshold).float()
    t = (target>0.5).float()
    eb = _edge(p)>0
    tb = _edge(t)>0
    k = 2*tolerance+1
    pd = F.max_pool2d(eb.float(),k,stride=1,padding=tolerance)>0
    td = F.max_pool2d(tb.float(),k,stride=1,padding=tolerance)>0
    dims = tuple(range(1,p.ndim))
    prec = (eb & td).sum(dims).float() / eb.sum(dims).clamp_min(1)
    recall = (tb & pd).sum(dims).float() / tb.sum(dims).clamp_min(1)
    result = 2*prec*recall/(prec+recall).clamp_min(1e-7)
    both_empty = (~eb).flatten(1).all(-1) & (~tb).flatten(1).all(-1)
    return torch.where(both_empty,torch.ones_like(result),result)


def grounding_consistency_loss(presence_logit: torch.Tensor, mask_logits: torch.Tensor,
                              answer_presence_logit: Optional[torch.Tensor] = None):
    """Align presence with mask evidence. Do not use as a surrogate for supervised labels.

    `presence_logit`: [B], `mask_logits`: [B,1,H,W].
    Mask evidence is detached so the consistency loss cannot make both heads
    agree by simply erasing all masks.
    """
    evidence = mask_logits.sigmoid().flatten(1).amax(-1).detach().clamp(1e-5,1-1e-5)
    loss = F.binary_cross_entropy_with_logits(presence_logit.float(), evidence.float())
    if answer_presence_logit is not None:
        loss = loss + F.binary_cross_entropy_with_logits(answer_presence_logit.float(), evidence.float())
    return loss
