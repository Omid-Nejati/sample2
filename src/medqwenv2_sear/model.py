"""Joint VQA + segmentation wrapper, with a toy decoder or native SAM3 backend."""
from contextlib import contextmanager
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .spectral import spectral_layers
from .prompts import RegionPromptHead, gt_boxes_from_masks
from .losses import segmentation_loss, grounding_consistency_loss
from .sam3_bridge import NativeSAM3Bridge, sam3_semantic_union


def semantic_mask_union(logits: torch.Tensor, presence_logits: torch.Tensor):
    # Independent query OR. Single-class semantic mask, not instance matching.
    p = logits.float().sigmoid() * presence_logits.float().sigmoid()[...,None,None]
    union = 1 - (1-p).prod(1,keepdim=True)
    return torch.logit(union.clamp(1e-5,1-1e-5))


class SEARModel(nn.Module):
    """`vlm` must return `.hidden_states[-1]` and `.loss` if labels are passed.

    `spatial_features` is a multiscale encoder FPN feature map (single level
    in this prototype). For real Qwen+SAM3, use native_sam_batch and optionally
    spatial_features extracted from SAM3 (not automatically extracted here).
    """
    def __init__(self, vlm: nn.Module, hidden_dim: int, spatial_dim=None,
                 num_region_queries: int = 4, prompt_dim: int = 256,
                 sam3_model: nn.Module | None = None):
        super().__init__()
        self.vlm = vlm
        self.region_head = RegionPromptHead(hidden_dim, prompt_dim, num_region_queries,
                                            spatial_dim=spatial_dim)
        self.answer_presence = nn.Linear(hidden_dim, 1)
        # Explicit mask-conditioned feedback into the evidence/answer verification head.
        # This does not yet modify autoregressively generated Qwen answer tokens.
        self.evidence_projection = nn.Linear(spatial_dim, hidden_dim) if spatial_dim else None
        self.evidence_gate = nn.Parameter(torch.tensor(-2.0))
        self.sam3_bridge = NativeSAM3Bridge(sam3_model) if sam3_model is not None else None

    @contextmanager
    def task(self, name):
        layers = [m for _,m in spectral_layers(self)]
        old = [m.capture_task for m in layers]
        for layer in layers:
            layer.capture_task = name
        try:
            yield
        finally:
            for layer,prev in zip(layers,old):
                layer.capture_task = prev

    def run_vlm(self, vlm_inputs):
        return self.vlm(**vlm_inputs, output_hidden_states=True, return_dict=True)

    def forward_vqa(self, vlm_inputs):
        result = self.run_vlm(vlm_inputs)
        if result.loss is None:
            raise ValueError("VQA inputs must include labels so the LM supplies a loss")
        return {"loss": result.loss, "vlm_output": result}

    def forward_seg(self, vlm_inputs, mask_target, spatial_features=None, native_sam_batch=None,
                    predicted_box=False, box_weight=0.2, presence_weight=0.3,
                    grounding_weight=0.1):
        output = self.run_vlm(vlm_inputs)
        states = output.hidden_states[-1]
        # Pad mask should be True at positions to ignore.
        attention_mask = vlm_inputs.get("attention_mask")
        padding = attention_mask == 0 if attention_mask is not None else None
        prompts = self.region_head(states, spatial_features, padding)
        if native_sam_batch is not None:
            if self.sam3_bridge is None:
                raise ValueError("native_sam_batch supplied without sam3_model")
            pred = self.sam3_bridge(native_sam_batch, prompts, use_predicted_box=predicted_box)
            masks = pred["pred_masks"]
            scores = pred["pred_logits"].squeeze(-1)
            if masks.ndim != 4 or scores.ndim != 2:
                raise ValueError("SAM3 mask and score shapes need validation")
            semantic = sam3_semantic_union(masks,scores)
        elif prompts.coarse_mask_logits is not None:
            semantic = semantic_mask_union(prompts.coarse_mask_logits, prompts.presence_logits)
        else:
            raise ValueError("Need `spatial_features` for toy coarse masks or a native SAM3 batch")
        mask_loss = segmentation_loss(semantic,mask_target)
        has_fg = (mask_target.flatten(1).sum(1)>0).float()
        joint_presence = 1 - (1-prompts.presence_logits.sigmoid()).prod(-1)
        joint_presence_logits = torch.logit(joint_presence.clamp(1e-5,1-1e-5))
        pres_loss = F.binary_cross_entropy_with_logits(joint_presence_logits,has_fg)
        # Box supervision is only meaningful for the first query in this prototype.
        # Multi-instance training should replace this with bipartite matching.
        gt_box = gt_boxes_from_masks(mask_target)
        box_loss = F.smooth_l1_loss(prompts.boxes[has_fg.bool(),0],gt_box[has_fg.bool()]) \
                   if has_fg.any() else semantic.sum()*0
        pooled_states = states.mean(1)
        if spatial_features is not None and self.evidence_projection is not None:
            # Soft region pooling conditions the presence answer on predicted mask evidence.
            # IMPORTANT: keep gradients through the mask for joint grounding learning.
            mask_prob = F.interpolate(semantic.sigmoid(),spatial_features.shape[-2:],
                                      mode="bilinear",align_corners=False)
            region_vec = (spatial_features*mask_prob).flatten(2).sum(-1) / \
                         mask_prob.flatten(2).sum(-1).clamp_min(1e-5)
            pooled_states = pooled_states + self.evidence_gate.sigmoid()*self.evidence_projection(region_vec)
        answer_presence = self.answer_presence(pooled_states).squeeze(-1)
        # Supervise answer head using actual presence labels, not just pseudo-evidence.
        answer_loss = F.binary_cross_entropy_with_logits(answer_presence,has_fg)
        consistency = grounding_consistency_loss(joint_presence_logits,semantic,answer_presence)
        loss = (mask_loss["loss"] + presence_weight*(pres_loss+answer_loss) +
                box_weight*box_loss + grounding_weight*consistency)
        return {"loss":loss,"mask_logits":semantic,"prompts":prompts,"mask_losses":mask_loss,
                "presence_loss":pres_loss,"box_loss":box_loss,"grounding_loss":consistency,
                "answer_presence_logits":answer_presence}
