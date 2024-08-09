import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from typing import Literal, Sequence
from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

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


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    file: str | Path,
) -> None:
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, file)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: Literal["cpu", "cuda"],
) -> None:
    # set model to evaluation mode
    model.eval()

    # initialize metrics
    loss: float = 0.0  # average loss over the dataset
    correct: int = 0  # number of samples correctly predicted
    total: int = 0  # total number of sample

    # evaluation loop
    with torch.no_grad():
        for batch in dataloader:
            # move batch to device
            array, index, target = move_to_device(batch, device)

            # forward pass
            output = model(array, index)

            # compute accuracy
            correct += (output.argmax(dim=-1) == target).sum().item()

            # compute loss
            loss += F.cross_entropy(output, target).item()

            # update total number of samples
            batch_size, *_ = array.shape
            total += batch_size

    # set model back to training mode
    model.train()

    # return metrics
    return {"loss": loss / total, "accuracy": correct / total}
