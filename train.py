import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from pathlib import Path

from torch.optim import Optimizer
from torch.utils.data import DataLoader

from data import SelectApplyDataset
from models import ToyTransformer
from utils import weight_norm, qk_weight_norm, ov_weight_norm


SEED: int = 72

# data parameters
N: int = 22
K: int = 4
R: int = 2
DOMAIN_SIZE: int = N * K
RANGE_SIZE: int = N
ARRAY_LEN: int = N * R
BATCH_SIZE: int = 200

# model hyperparameters
EMBEDDING_DIM: int = 64
PROJECTION_DIM: int = 32

# optimizer hyperparameters
NUM_STEPS: int = 100_000
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-2
BETA_1: float = 0.90
BETA_2: float = 0.99
GRAD_CLIP: float = 1.0

# logging parameters
LOG_EVERY: int = 1
SAVE_EVERY: int = 1_000_000


def train(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, device: torch.device):
    # set model to training mode
    model.train()
    # move model to device
    model.to(device)

    # main training loop
    step = 0
    for sequence, value in dataloader:
        # break if we reach the maximum number of steps
        if step > NUM_STEPS:
            break

        # move data to device
        sequence, value = sequence.to(device), value.to(device)

        # zero out gradients
        optimizer.zero_grad(set_to_none=True)

        # forward pass
        logits = model(sequence)
        loss = F.cross_entropy(logits, value)

        # backward pass
        loss.backward()

        # clip gradients
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # optimizer step
        optimizer.step()

        # compute and log metrics
        if step % LOG_EVERY == 0:
            # compute metrics
            with torch.no_grad():
                # compute accuracy
                accuracy = (logits.argmax(dim=-1) == value).float().mean()

                # compute weight norms
                norm = weight_norm(model)
                qk_norm = qk_weight_norm(model)
                ov_norm = ov_weight_norm(model)

            # log metrics to wandb
            wandb.log(
                {
                    "step": step,
                    "loss": loss.item(),
                    "accuracy": accuracy.item(),
                    "weight_norm": norm,
                    "qk_weight_norm": qk_norm,
                    "ov_weight_norm": ov_norm,
                }
            )

        # save model checkpoint
        if step % SAVE_EVERY == 0:
            torch.save(model.state_dict(), f"artifacts/{wandb.run.name}/{step}.pt")

        # increment step
        step += 1


if __name__ == "__main__":
    # initialize wandb
    wandb.init(
        project="select-apply",
        name=f"n={N}, seed={SEED}",
        config={
            "seed": SEED,
            "dataset": {
                "domain_size": DOMAIN_SIZE,
                "range_size": RANGE_SIZE,
                "array_len": ARRAY_LEN,
                "batch_size": BATCH_SIZE,
            },
            "model": {
                "embedding_dim": EMBEDDING_DIM,
                "projection_dim": PROJECTION_DIM,
            },
            "optimizer": {
                "num_steps": NUM_STEPS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "beta_1": BETA_1,
                "beta_2": BETA_2,
                "grad_clip": GRAD_CLIP,
            },
        },
    )

    # set seed
    torch.manual_seed(SEED)

    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize dataset and dataloader
    dataset = SelectApplyDataset(
        domain_size=DOMAIN_SIZE,
        range_size=RANGE_SIZE,
        array_len=ARRAY_LEN,
        device=device,
        seed=SEED,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # initialize model
    model = ToyTransformer(
        domain_size=DOMAIN_SIZE,
        range_size=RANGE_SIZE,
        seq_len=(ARRAY_LEN + 1),
        embedding_dim=EMBEDDING_DIM,
        projection_dim=PROJECTION_DIM,
    ).to(device)

    # initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(BETA_1, BETA_2),
    )

    # create artifacts directory
    Path(f"artifacts/{wandb.run.name}").mkdir(parents=True, exist_ok=True)

    # train model
    train(model, dataloader, optimizer, device)

    # finish wandb
    wandb.finish()
