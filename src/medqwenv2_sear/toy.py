"""Tiny test model. Do NOT treat synthetic outcomes as medical benchmark results."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .spectral import install_spectral_experts
from .model import SEARModel


class TinyVisionLanguage(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(1,hidden,3,padding=1),nn.GELU(),
                                    nn.Conv2d(hidden,hidden,3,stride=2,padding=1),nn.GELU())
        self.word = nn.Embedding(8,hidden)
        self.q_proj = nn.Linear(hidden,hidden)
        self.v_proj = nn.Linear(hidden,hidden)
        self.out = nn.Linear(hidden,hidden)
        self.classifier = nn.Linear(hidden,2)

    def forward(self,images,question_ids,labels=None,output_hidden_states=True,return_dict=True):
        spatial = self.vision(images)
        b,c,h,w = spatial.shape
        pooled = F.adaptive_avg_pool2d(spatial,(4,4)).flatten(2).transpose(1,2)
        qs = self.word(question_ids).unsqueeze(1)
        x = torch.cat((qs,pooled),dim=1)
        x = x + self.out((self.q_proj(x).tanh() * self.v_proj(x)).tanh())
        logit = self.classifier(x[:,0])
        loss = F.cross_entropy(logit,labels) if labels is not None else None
        return SimpleNamespace(loss=loss,logits=logit,hidden_states=[x],spatial_features=spatial)


def make_toy_sear(hidden=32, experts=4, rank=2, top_k=2):
    net=TinyVisionLanguage(hidden)
    net.vision.requires_grad_(False)  # analogous to frozen vision backbone
    install_spectral_experts(net,only_language=False,experts=experts,
                             rank_per_expert=rank,top_k=top_k,svd_init=True)
    return SEARModel(net,hidden_dim=hidden,spatial_dim=hidden,num_region_queries=2,prompt_dim=32)


def make_synthetic_batch(batch=2,size=32,device="cpu"):
    # Foreground is a randomly positioned disk; question is "Is a lesion present?"
    yy,xx=torch.meshgrid(torch.arange(size,device=device),torch.arange(size,device=device),indexing="ij")
    masks=[]
    for i in range(batch):
        cx,cy = torch.randint(size//4,3*size//4,(2,),device=device)
        radius = torch.randint(max(2,size//10),max(3,size//5),(1,),device=device).item()
        mask = (((xx-cx)**2+(yy-cy)**2) <= radius**2).float()
        if i%3==2:
            mask*=0
        masks.append(mask)
    target=torch.stack(masks)[:,None]
    images = target * 0.85 + torch.randn_like(target)*0.07
    labels=(target.flatten(1).sum(1)>0).long()
    vlm_input={"images":images,"question_ids":torch.zeros(batch,device=device,dtype=torch.long),"labels":labels}
    return vlm_input,target
