import torch
import wandb

from typing import Sequence
from omegaconf import OmegaConf
from config import RunConfig


def move_to_device(
    data: torch.Tensor | Sequence[torch.Tensor],
    device: torch.device,
) -> torch.Tensor | Sequence[torch.Tensor]:
    if isinstance(data, Sequence):
        return (tensor.to(device) for tensor in data)
    return data.to(device)


def init_wandb(config: RunConfig) -> None:
    wandb.init(project="indexing", entity="physicsofintelligence")
    wandb.config.update(
        OmegaConf.to_container(
            config,
            resolve=True,
            throw_on_missing=True,
        )
    )
