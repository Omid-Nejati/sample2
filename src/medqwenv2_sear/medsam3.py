"""Optional MedSAM3-v1 native loading and batch preprocessing.

Adapted from the PUBLIC API of MedSAM3's infer_sam.py, not validated here
against real SAM3 weights. Install SAM3 and MedSAM3 under their own licenses.
"""
from pathlib import Path


def load_medsam3(config_path, weights_path, device="cuda"):
    import yaml
    from sam3.model_builder import build_sam3_image_model
    from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
    with open(config_path,encoding="utf-8") as f:
        cfg=yaml.safe_load(f)
    c=cfg["lora"]
    model=build_sam3_image_model(device=device,compile=False,load_from_HF=True,
                                 bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                                 eval_mode=False)
    lora=LoRAConfig(
        rank=c["rank"],alpha=c["alpha"],dropout=0.0,
        target_modules=c["target_modules"],
        apply_to_vision_encoder=c["apply_to_vision_encoder"],
        apply_to_text_encoder=c["apply_to_text_encoder"],
        apply_to_geometry_encoder=c["apply_to_geometry_encoder"],
        apply_to_detr_encoder=c["apply_to_detr_encoder"],
        apply_to_detr_decoder=c["apply_to_detr_decoder"],
        apply_to_mask_decoder=c["apply_to_mask_decoder"])
    model=apply_lora_to_model(model,lora)
    if not Path(weights_path).exists():
        raise FileNotFoundError(weights_path)
    load_lora_weights(model,weights_path)
    model.to(device)
    return model


def make_native_sam_batch(pil_image, concept="lesion", device="cuda",resolution=1008):
    """One image, one text query; creates a SAM3 native `BatchedDatapoint`.

    `concept` is a placeholder; SEAR overwrites its embeddings downstream.
    Correct image/mask alignment must be verified if training with augmentations.
    """
    from sam3.train.data.sam3_image_dataset import Datapoint,Image as SAMImage,FindQueryLoaded,InferenceMetadata
    from sam3.train.data.collator import collate_fn_api
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.transforms.basic_for_api import ComposeAPI,RandomResizeAPI,ToTensorAPI,NormalizeAPI
    image=pil_image.convert("RGB")
    w,h=image.size
    query=FindQueryLoaded(query_text=concept,image_id=0,object_ids_output=[],is_exhaustive=True,
                          query_processing_order=0,
                          inference_metadata=InferenceMetadata(
                              coco_image_id=0,original_image_id=0,original_category_id=1,
                              original_size=[w,h],object_id=0,frame_index=0))
    datapoint=Datapoint(find_queries=[query],images=[SAMImage(data=image,objects=[],size=[h,w])])
    transform=ComposeAPI(transforms=[
        RandomResizeAPI(sizes=resolution,max_size=resolution,square=True,consistent_transform=False),
        ToTensorAPI(),NormalizeAPI(mean=[.5]*3,std=[.5]*3)])
    batch=collate_fn_api([transform(datapoint)],dict_key="input")["input"]
    return copy_data_to_device(batch,device,non_blocking=False)
