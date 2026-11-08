"""SAM3 native adapter -- requires local facebookresearch/sam3 plus a real BatchedDatapoint.

Unlike the differentiable PyTorch core, this file has NOT been verified against
real SAM3 weights here. Keep batch size one per prompt set during initial checks.
The SAM3 image processor's `@torch.inference_mode()` API is NOT for training.
Use the native SAM3 model forward with an appropriately preprocessed batch.
"""
from contextlib import contextmanager
from typing import Any
import torch
from torch import nn
from .prompts import RegionPrompts


@contextmanager
def temporary_sam3_prompt_override(sam3_model: nn.Module, prompts: RegionPrompts):
    """Inject differentiable learned text features through backbone.forward_text.

    Original SAM3's model.forward calls `backbone.forward_text` before grounding.
    `language_mask=True` is padding. `language_embeds` is adjusted in token length
    if present, but should be checked for compatibility on the installed commit.
    This monkeypatch is for a single nonconcurrent training worker only.
    """
    backbone = sam3_model.backbone
    original = backbone.forward_text
    features = prompts.sam3_language_features()

    def injected(text_batch, *args, **kwargs):
        baseline = original(text_batch, *args, **kwargs)
        token_batch = features["language_features"].shape[1]
        if len(text_batch) != token_batch:
            raise ValueError("SAM3 native prompt batch differs from Qwen prompt batch; "
                             "initial integration should use one prompt per image")
        # SAM3 usually runs float32; Qwen hidden/prompt projections can be bf16.
        # `to()` retains the gradient back to the prompt projector.
        reference = baseline["language_features"]
        baseline["language_features"] = features["language_features"].to(
            device=reference.device,dtype=reference.dtype)
        baseline["language_mask"] = features["language_mask"].to(device=reference.device)
        if "language_embeds" in baseline:
            old = baseline["language_embeds"]
            # Match sequence length and batch for APIs that still refer to embeddings.
            baseline["language_embeds"] = old.mean(0,keepdim=True).expand(
                features["language_features"].shape[0],-1,-1).contiguous()
        return baseline
    backbone.forward_text = injected
    try:
        yield
    finally:
        backbone.forward_text = original


def apply_predicted_box_to_native_input(native_input: Any, prompts: RegionPrompts,
                                        selected_query: int = 0):
    """Override boxes with PREDICTED normalized cxcywh; mutates native_input.

    SAM3's native find_input input_boxes_before_embed is expected [B,N_box,4]
    in the current SAM3 formulation. Check the specific SAM3 version after install.
    Single target per image; multi-instance matching must be added for full tasks.
    """
    if not hasattr(native_input, "find_inputs") or len(native_input.find_inputs)!=1:
        raise ValueError("Expected a single-frame SAM3 BatchedDatapoint")
    find_input = native_input.find_inputs[0]
    b = prompts.boxes.shape[0]
    selected = prompts.boxes[:,selected_query,:].unsqueeze(1)
    find_input.input_boxes_before_embed = selected
    find_input.input_boxes_mask = torch.zeros((b,1), dtype=torch.bool,device=selected.device)
    find_input.input_boxes_label = torch.ones((b,1), dtype=torch.bool,device=selected.device)
    return native_input


class NativeSAM3Bridge(nn.Module):
    def __init__(self, sam3_model: nn.Module):
        super().__init__()
        self.model = sam3_model

    def forward(self, native_input, prompts: RegionPrompts, use_predicted_box=False):
        if use_predicted_box:
            apply_predicted_box_to_native_input(native_input,prompts)
        with temporary_sam3_prompt_override(self.model,prompts):
            outputs = self.model(native_input)
        # Native SAM3 returns a list of stage outputs in published inference API.
        last = outputs[-1] if isinstance(outputs,(list,tuple)) else outputs
        if not isinstance(last,dict) or "pred_masks" not in last:
            raise RuntimeError("SAM3 output schema differs; inspect your installed SAM3 commit")
        return last


def sam3_semantic_union(logits_per_instance: torch.Tensor, scores: torch.Tensor,
                        max_instances: int = 5):
    """Differentiable score-weighted union of top-scoring SAM3 instances.

    This is a *semantic foreground* approximation for a single concept, NOT an
    alternative to SAM3's native instance/Hungarian loss. Limiting to top scores
    prevents the product union from saturating across ~200 DETR queries.
    Inputs: masks [B,Q,H,W], score logits [B,Q]. Top-k IDs are discrete; gradients
    still flow to selected masks and scores.
    """
    if max_instances < 1:
        raise ValueError("max_instances must be >= 1")
    k=min(max_instances,scores.shape[1])
    values,indices = scores.float().topk(k,dim=1)
    selected = logits_per_instance.float().gather(
        1,indices[:,:,None,None].expand(-1,-1,*logits_per_instance.shape[-2:]))
    p=selected.sigmoid()*values.sigmoid()[...,None,None]
    return torch.logit((1-(1-p).prod(dim=1,keepdim=True)).clamp(1e-5,1-1e-5))
