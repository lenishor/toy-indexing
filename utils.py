import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from itertools import islice
from typing import Any, Literal, Sequence
from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from config import RunConfig
from models import ToyTransformer


def move_to_device(
    data: torch.Tensor | Sequence[torch.Tensor],
    device: torch.device,
) -> torch.Tensor | Sequence[torch.Tensor]:
    if isinstance(data, Sequence):
        return (tensor.to(device) for tensor in data)
    return data.to(device)


def init_wandb(config: RunConfig) -> None:
    wandb.init(project="indexing-test", entity="physicsofintelligence")
    wandb.config.update(
        OmegaConf.to_container(
            config,
            resolve=True,
            throw_on_missing=True,
        )
    )
    if config.run_name:
        wandb.run.name = config.run_name  # Set run name if provided
    else:
        wandb.run.name = wandb.run.id


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
    num_samples: int,
) -> dict[str, float]:
    model.eval()
    loss: float = 0.0
    correct: int = 0
    total: int = 0

    with torch.no_grad():
        for batch in islice(dataloader, num_samples // dataloader.dataset.batch_size):
            array, index, target = move_to_device(batch, device)
            output = model(array, index)
            correct += (output.argmax(dim=-1) == target).sum().item()
            loss += F.cross_entropy(output, target).item()
            total += array.shape[0]

            if total >= num_samples:
                break

    model.train()
    return {"loss": loss / total, "accuracy": correct / total}


def get_model_metrics(model: ToyTransformer) -> dict[str, Any]:
    qk_circuit = model.qk_circuit()
    ov_circuit = model.ov_circuit()

    return {
        "qk_circuit": wandb.Image(qk_circuit.cpu()),
        "ov_circuit": wandb.Image(ov_circuit.cpu()),
    }
