"""MedQwenV2-SEAR: research prototype, not a validated clinical model."""
from .spectral import SpectralMoELinear, install_spectral_experts, spectral_layers
from .prompts import RegionPromptHead
from .losses import segmentation_loss, dice_score, boundary_f1, grounding_consistency_loss
from .trainer import train_pair_step
__all__ = ["SpectralMoELinear", "install_spectral_experts", "spectral_layers", "RegionPromptHead",
           "segmentation_loss", "dice_score", "boundary_f1", "grounding_consistency_loss", "train_pair_step"]
