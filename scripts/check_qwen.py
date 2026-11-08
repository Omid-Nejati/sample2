"""Load Qwen3-VL and install spectral adapters; requires real GPU and model access.

On large models start with --random-init if CPU full SVD is too slow. This
script does NOT train SAM3 or claim joint end-to-end integration.
"""
import argparse
import torch
from medqwenv2_sear.qwen import load_qwen3_vl_sear
from medqwenv2_sear.spectral import spectral_layers


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-id",default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--random-init",action="store_true")
    p.add_argument("--experts",type=int,default=4)
    p.add_argument("--rank",type=int,default=2)
    args=p.parse_args()
    joint,proc,n=load_qwen3_vl_sear(args.model_id,experts=args.experts,
                                   rank_per_expert=args.rank,svd_init=not args.random_init)
    print("wrapped_language_linears",n)
    print("trainable_parameters",sum(p.numel() for p in joint.parameters() if p.requires_grad))
    print("spectral_example",spectral_layers(joint)[0][0])
    print("SAM3 still requires native integration and MedSAM3 checkpoint verification")


if __name__=="__main__":main()
