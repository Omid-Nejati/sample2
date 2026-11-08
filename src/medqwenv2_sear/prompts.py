"""Multi-query image-grounded prompt head for SAM3.

Prompt vectors are [B,Q,256]; `sam3_language_features` converts to [Q,B,256].
Boxes are normalized (cx,cy,w,h) and must be evaluated as PREDICTED boxes;
ground-truth boxes are oracle baselines only.
"""
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RegionPrompts:
    tokens: torch.Tensor              # [B,Q,D_prompt]
    boxes: torch.Tensor               # [B,Q,4] normalized cxcywh
    presence_logits: torch.Tensor     # [B,Q]
    coarse_mask_logits: Optional[torch.Tensor] = None # [B,Q,H,W]

    def sam3_language_features(self):
        tokens = self.tokens.transpose(0, 1).contiguous()
        mask = torch.zeros(self.tokens.shape[:2], device=self.tokens.device, dtype=torch.bool)
        return {"language_features": tokens, "language_mask": mask}


class RegionPromptHead(nn.Module):
    def __init__(self, hidden_dim: int, prompt_dim: int = 256, num_queries: int = 4,
                 spatial_dim: Optional[int] = None, heads: int = 4):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by attention heads")
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.cross = nn.MultiheadAttention(hidden_dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.prompt = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                    nn.Linear(hidden_dim, prompt_dim))
        self.box = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, 4))
        self.presence = nn.Linear(hidden_dim, 1)
        self.spatial_proj = nn.Conv2d(spatial_dim, prompt_dim, 1) if spatial_dim is not None else None
        self.coarse_scale = prompt_dim ** -0.5

    def forward(self, language_states: torch.Tensor, spatial_feats: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None) -> RegionPrompts:
        batch = language_states.shape[0]
        query = self.queries.unsqueeze(0).expand(batch, -1, -1).to(language_states.dtype)
        attended, _ = self.cross(query, language_states, language_states,
                                 key_padding_mask=padding_mask, need_weights=False)
        attended = self.norm(attended + query)
        tokens = self.prompt(attended)
        boxes = torch.sigmoid(self.box(attended))
        # Restrict w/h to sensible positive values; still fully learned and normalized.
        boxes = torch.cat((boxes[..., :2], 0.05 + 0.90 * boxes[..., 2:]), dim=-1)
        presence = self.presence(attended).squeeze(-1)
        coarse = None
        if spatial_feats is not None:
            if self.spatial_proj is None:
                raise ValueError("spatial_dim must be set to enable coarse localization")
            features = self.spatial_proj(spatial_feats)
            coarse = torch.einsum("bqc,bchw->bqhw", tokens, features) * self.coarse_scale
        return RegionPrompts(tokens, boxes, presence, coarse)


def gt_boxes_from_masks(masks: torch.Tensor, min_size: float = 1e-3) -> torch.Tensor:
    """Compute cxcywh for supervised box loss; NEVER feed these as predicted boxes at eval.

    masks [B,H,W] or [B,1,H,W]. Empty masks produce zeros, and should be
    excluded from box regression using the returned foreground mask.
    """
    if masks.ndim == 4:
        masks = masks[:, 0]
    b, h, w = masks.shape
    present = masks > 0.5
    yy = torch.arange(h, device=masks.device)[None, :, None].expand(b,h,w)
    xx = torch.arange(w, device=masks.device)[None, None, :].expand(b,h,w)
    x_min = torch.where(present, xx, w).flatten(1).min(1).values.float()
    x_max = torch.where(present, xx, 0).flatten(1).max(1).values.float()
    y_min = torch.where(present, yy, h).flatten(1).min(1).values.float()
    y_max = torch.where(present, yy, 0).flatten(1).max(1).values.float()
    cx = (x_min+x_max+1) / (2*w)
    cy = (y_min+y_max+1) / (2*h)
    bw = (x_max-x_min+1) / w
    bh = (y_max-y_min+1) / h
    exists = present.flatten(1).any(1)
    return torch.where(exists.unsqueeze(-1), torch.stack((cx,cy,bw,bh),dim=-1),
                       torch.zeros((b,4),device=masks.device))
