"""Experimental integration check on ONE image (NOT a validated inference system).

Requires Qwen3-VL, SAM3 and MedSAM3 v1 weights available under their licenses.
Expect to revise the SAM3 bridge for the pinned SAM3 commit if API changed.
"""
import argparse
import torch
from PIL import Image
from medqwenv2_sear.qwen import load_qwen3_vl_sear
from medqwenv2_sear.medsam3 import load_medsam3,make_native_sam_batch
from medqwenv2_sear.sam3_bridge import NativeSAM3Bridge


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--image",required=True)
    p.add_argument("--sam3-config",required=True)
    p.add_argument("--sam3-weights",required=True)
    p.add_argument("--question",default="Segment the lesion in the image.")
    p.add_argument("--resolution",type=int,default=1008)
    p.add_argument("--random-init",action="store_true")
    p.add_argument("--predicted-box",action="store_true")
    args=p.parse_args()
    pil=Image.open(args.image).convert("RGB")
    qwen, processor, n=load_qwen3_vl_sear(svd_init=not args.random_init)
    sam=load_medsam3(args.sam3_config,args.sam3_weights)
    bridge=NativeSAM3Bridge(sam)
    qwen.sam3_bridge=bridge
    messages=[{"role":"user","content":[{"type":"image","image":pil},
                {"type":"text","text":args.question}]}]
    inputs=processor.apply_chat_template(messages,tokenize=True,return_dict=True,
                                         add_generation_prompt=False,return_tensors="pt")
    inputs.pop("token_type_ids",None)
    device=next(qwen.vlm.parameters()).device
    inputs={k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in inputs.items()}
    native=make_native_sam_batch(pil,"lesion",resolution=args.resolution)
    qwen.eval();sam.eval()
    with torch.no_grad():
        output=qwen.run_vlm(inputs)
        prompts=qwen.region_head(output.hidden_states[-1])
        preds=bridge(native,prompts,use_predicted_box=args.predicted_box)
    print("SAM3 mask shape:",tuple(preds["pred_masks"].shape))
    print("SAM3 scores shape:",tuple(preds["pred_logits"].shape))
    print("WARNING: these are random, untrained SEAR prompts, NOT usable segmentation predictions")


if __name__=="__main__":main()
