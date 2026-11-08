import torch
from medqwenv2_sear.prompts import RegionPromptHead, gt_boxes_from_masks
from medqwenv2_sear.losses import segmentation_loss,dice_score,boundary_f1


def test_prompt_dimensions_and_grad():
    p=RegionPromptHead(hidden_dim=16,prompt_dim=8,num_queries=3,spatial_dim=5)
    lang=torch.randn(2,7,16,requires_grad=True)
    spatial=torch.randn(2,5,12,13,requires_grad=True)
    out=p(lang,spatial)
    assert out.tokens.shape==(2,3,8)
    assert out.boxes.shape==(2,3,4)
    assert out.coarse_mask_logits.shape==(2,3,12,13)
    assert out.sam3_language_features()["language_features"].shape==(3,2,8)
    (out.coarse_mask_logits.square().mean()+out.boxes.mean()).backward()
    assert lang.grad is not None and spatial.grad is not None


def test_masks_and_gt_boxes():
    gt=torch.zeros(2,1,10,10)
    gt[0,0,2:5,3:8]=1
    boxes=gt_boxes_from_masks(gt)
    torch.testing.assert_close(boxes[0],torch.tensor([.55,.35,.5,.3]))
    assert boxes[1].abs().sum()==0
    logits=torch.where(gt>0,torch.tensor(10.),torch.tensor(-10.))
    assert dice_score(logits,gt).min()>.99
    assert boundary_f1(logits,gt).min()>.99
    d=segmentation_loss(logits,gt)
    assert torch.isfinite(d["loss"])


def test_sam3_topk_semantic_union_grad():
    from medqwenv2_sear.sam3_bridge import sam3_semantic_union
    logits=torch.randn(2,200,8,8,requires_grad=True)
    scores=torch.randn(2,200,requires_grad=True)
    prediction=sam3_semantic_union(logits,scores,max_instances=3)
    assert prediction.shape==(2,1,8,8)
    prediction.mean().backward()
    assert scores.grad is not None and logits.grad is not None
