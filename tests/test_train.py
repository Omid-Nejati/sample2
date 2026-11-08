import torch
from medqwenv2_sear.toy import make_toy_sear,make_synthetic_batch
from medqwenv2_sear.trainer import train_pair_step
from medqwenv2_sear.spectral import spectral_layers


def test_single_train_step():
    torch.manual_seed(42)
    model=make_toy_sear(hidden=16,experts=2,rank=2)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    vqa,_=make_synthetic_batch(2,size=16)
    seg,gt=make_synthetic_batch(2,size=16)
    with torch.no_grad():
        features=model.vlm.vision(seg["images"])
    result=train_pair_step(model,optimizer,vqa,seg,gt,spatial_features=features)
    assert result["mask_logits"].shape[-2:]==features.shape[-2:]
    assert torch.isfinite(result["mask_logits"]).all()
    assert result["conflict_cosine"] is not None
    assert all(m.stats_steps.item()==1 for _,m in spectral_layers(model))
