# MedQwenV2-SEAR — executable method prototype (2026-11-06)

**Purpose:** research starter code for the proposed *Spatial Evidence-Aligned Spectral Expert Routing* architecture for medical VQA + 2D medical segmentation. It is **not** a trained system, tested clinical model, or claimed CVPR result.

## Verified vs. experimental

| Part | Status |
|---|---|
| Standalone SVD-initialized spectral MoE layers | CPU-tested |
| Exact pretrained-output centering at adapter initialization (top-k included) | CPU-tested |
| Projected, factorization-independent spectral gradient conflict | CPU-tested |
| Task-agnostic inference router, EMA role specialization, sequential two-task training | CPU-tested |
| Region-query prompt projector + predicted cxcywh boxes + coarse localization | CPU-tested |
| Semantic Dice/BCE/boundary losses, presence and grounding consistency | CPU-tested |
| End-to-end toy VQA + segmentation training | CPU-tested (synthetic circles; not a medical benchmark) |
| Actual Qwen3-VL-8B weights, environment/quantization/FSDP | **Unverified on GPU** |
| Native SAM3 injected prompt forward and MedSAM3-v1 LoRA | **Unverified on real weights** |
| Medical dataset loaders, native SAM3 instance Hungarian loss, full multi-GPU training | **Not implemented** |
| Segmentation-error-supervised routing difficulty predictor | CPU-tested (toy) |
| Mask-conditioned evidence pooled into an answer-verification head | CPU-tested (toy) |
| Dynamic per-direction SVD rank reallocation; full mask feedback into autoregressive generation | **Not implemented** |

The current code implements **soft adaptive expert specialization via routing biases**, NOT actual physical SVD rank reallocation. A mask-conditioned region feature does feed an answer-presence/verification head; it **does not** yet alter the autoregressive Qwen token-generation path. These are explicit limitations of v0.1.

## Install on a machine with PyTorch

```bash
cd MedQwenV2_SEAR
python -m pip install -e '.[test]'
python -m pytest -q tests/
python scripts/train_toy.py --steps 8
```

The toy script prints VQA loss, segmentation loss, training-batch Dice, and gradient conflict. Synthetic Dice is **not** a clinical result and should not be cited in a paper. A toy checkpoint is saved in the working directory.

## Qwen3-VL GPU smoke test (weights are downloaded or cached)

```bash
pip install -e '.[qwen,test]'
python scripts/check_qwen.py --random-init
```

`--random-init` only skips **spectral SVD initialization** of adapters. It does **not** initialize the pretrained Qwen from random weights. Remove it to initialize adapters from SVD chunks (costly unless SVD caching is added). Freeze Qwen first, inject experts into language `q_proj,v_proj`, and use Qwen's native image processor (including `mm_token_type_ids` where required).

## SAM3 / MedSAM3

Install SAM3 from its official repository and obtain authorization to the model checkpoint. Install MedSAM3 separately and obtain its LoRA weights; verify their config matches `configs/full_lora_config.yaml`. The native integration is in:

- `src/medqwenv2_sear/medsam3.py`: load base SAM3 and MedSAM3 LoRA; transform a PIL image into a native SAM3 batch.
- `src/medqwenv2_sear/sam3_bridge.py`: swap in learned prompt features for `backbone.forward_text` during **model.forward**, not inference-only processor wrappers; optionally inject predicted boxes.
- `scripts/real_smoke.py`: forward once through real Qwen3-VL + SAM3. **Untrained learned prompts will give meaningless masks**; use this script to check shapes/gradient graph, not accuracy.

```bash
python scripts/real_smoke.py \
 --image /path/to/medical_image.png \
 --sam3-config /path/to/MedSAM3/configs/full_lora_config.yaml \
 --sam3-weights /path/to/MedSAM3_v1.pt \
 --random-init
```

The prompt injection hook depends on SAM3 internals and must be validated against a pinned commit. SAM3's `Sam3Processor` inference utilities contain `torch.inference_mode()` decorators and must **not** be used for differentiable joint training. For native training, you must also supply masks in the same transformed coordinate frame and implement correct target/mask matching.

## Proposed real training stages (not one-command complete)

1. **Bridge alignment:** freeze Qwen's experts and all SAM3 parameters. Train `RegionPromptHead` on matched image/mask pairs with Qwen hidden states and predicted mask losses. Verify that the losses backpropagate to region queries and that boxes are normalized to the SAM3 input size.
2. **Joint spectral adaptation:** unfreeze MoE A/B + routers; alternate VQA and seg minibatches with `train_pair_step`. Track projected gradient cosine, negative-transfer gap, expert usage and forgetting.
3. **Full research extension:** the included difficulty predictor is supervised by boundary/foreground error, but requires calibration on real data. Add *budgeted* rank-capacity reallocation, and extend mask-feature feedback from the auxiliary verification head into language token generation. These are required for the full final-method claim and remain future work.
4. **Ablations:** matched-parameter joint LoRA, GOAT, fixed MoE, PCGrad, conflict-only, evidence-router; test with both SAM3-base and MedSAM3 weights and separate predicted-box vs oracle-box evaluation.

## Key file map

- `spectral.py`: spectral experts, exact-anchor initialization, projected-gradient conflict, EMA specialist roles, token routing.
- `prompts.py`: learned cross-attention region queries, normalized boxes, presence, coarse masks.
- `sam3_bridge.py`: native prompt substitution / predicted geometry + semantic mask union.
- `model.py`: VQA+seg task wrapper; coarse-mask and SAM3 modes; mask-conditioned answer-verification head.
- `losses.py`: Dice, BCE, boundary, presence consistency.
- `trainer.py`: memory-conscious sequential two-task optimization, lagged conflict regularization, segmentation-error-supervised difficulty.
- `qwen.py` / `medsam3.py`: checkpoint loaders, optional dependencies.
- `tests/`: CPU functionality checks.

## Evaluation requirements

Do not report MedSAM3's text+ground-truth-box oracle results alongside automatic models without clearly labeling them. Report mean Dice, gIoU/cIoU, boundary F1 and HD95, small-lesion Dice, absent-target performance, VQA and forgetting, FPS/VRAM and trainable parameters. Audit possible overlap with MedSAM3 pretraining and compute confidence intervals across patient-separated splits.

## References and attribution

- MedQwen (arXiv:2604.01310): user-provided original work.
- GOAT (arXiv:2502.16894): SVD expert initialization ideas; cite when evaluating.
- Qwen3-VL: https://huggingface.co/docs/transformers/main/model_doc/qwen3_vl
- SAM3: https://github.com/facebookresearch/sam3
- MedSAM3: https://github.com/Joey-S-Liu/MedSAM3
- MedSIGHT: https://arxiv.org/abs/2606.06760

Code here is an original prototype, not a copy of GOAT's or MedSAM3's implementation. Respect the licenses of Qwen, SAM3, and MedSAM3 checkpoints and source code.
