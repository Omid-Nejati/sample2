"""End-to-end CPU synthetic training demonstration: not a medical performance result."""
import argparse
import torch
from medqwenv2_sear.toy import make_toy_sear,make_synthetic_batch
from medqwenv2_sear.trainer import train_pair_step
from medqwenv2_sear.losses import dice_score


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--steps",type=int,default=8)
    parser.add_argument("--batch",type=int,default=2)
    parser.add_argument("--lr",type=float,default=1e-3)
    args=parser.parse_args()
    torch.manual_seed(7)
    model=make_toy_sear()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    for i in range(args.steps):
        vqa,_=make_synthetic_batch(args.batch)
        seg,mask=make_synthetic_batch(args.batch)
        # Compute FPN features (spatial information, not ground truth masks).
        # The toy model's vision encoder is frozen; no extra gradient graph is needed.
        with torch.no_grad():
            fpn=model.vlm.vision(seg["images"])
        info=train_pair_step(model,optimizer,vqa,seg,mask,spatial_features=fpn)
        if i==0 or i==args.steps-1:
            print(f"step={i+1} vqa_loss={info['vqa_loss']:.4f} "
                  f"seg_loss={info['seg_loss']:.4f} dice={dice_score(info['mask_logits'],mask).mean():.3f} "
                  f"cos={info['conflict_cosine']}")
    torch.save({"trainable":{k:v.cpu() for k,v in model.state_dict().items()}},"toy_sear_checkpoint.pt")
    print("Saved toy_sear_checkpoint.pt (toy demo only)")


if __name__=="__main__":main()
