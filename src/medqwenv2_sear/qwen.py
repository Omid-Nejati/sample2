"""Optional real Qwen3-VL-8B initialization. Requires downloaded model weights.

Use ONLY after testing against your cluster's transformers commit. Native SAM3
loading, preprocessing and loss matching are separate from this loader.
"""
import torch
from .spectral import install_spectral_experts
from .model import SEARModel


def load_qwen3_vl_sear(model_id="Qwen/Qwen3-VL-8B-Instruct", dtype="bfloat16",
                       rank_per_expert=2, experts=4, top_k=2, svd_init=True,
                       device_map=None, **from_pretrained_kwargs):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    ddtype = getattr(torch,dtype)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,torch_dtype=ddtype,device_map=device_map,**from_pretrained_kwargs)
    if device_map is None and torch.cuda.is_available():
        qwen = qwen.to("cuda")  # single-GPU path; use explicit device_map for sharding
    processor = AutoProcessor.from_pretrained(model_id)
    # Freeze first so all new adapter params are the only LM trainables.
    qwen.requires_grad_(False)
    count = install_spectral_experts(qwen,targets=("q_proj","v_proj"),
                                     only_language=True,experts=experts,
                                     rank_per_expert=rank_per_expert,top_k=top_k,
                                     svd_init=svd_init)
    if count == 0:
        raise RuntimeError("No Qwen3-VL language projections matched: inspect named_modules")
    hidden_dim = qwen.config.text_config.hidden_size
    joint = SEARModel(qwen,hidden_dim,num_region_queries=4,prompt_dim=256,spatial_dim=256)
    # The region head needs to match the Qwen device and dtype.
    joint.region_head.to(device=next(qwen.parameters()).device,dtype=ddtype)
    joint.answer_presence.to(device=next(qwen.parameters()).device,dtype=ddtype)
    return joint,processor,count
