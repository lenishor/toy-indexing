import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from pathlib import Path

from torch.optim import Optimizer
from torch.utils.data import DataLoader

from data import IndexingDataset
from models import ToyTransformer


SEED: int = 42

# dataset parameters
NUM_SYMBOLS: int = 128
ARRAY_LEN: int = 64
BATCH_SIZE: int = 1_024

# model hyperparameters
EMBEDDING_DIM: int = 64
PROJECTION_DIM: int = 32

# optimizer hyperparameters
NUM_STEPS: int = 1_000
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
BETA_1: float = 0.90
BETA_2: float = 0.99
GRAD_CLIP: float = 1.0

# logging parameters
SAVE_INTERVAL: int = 100


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

        # increment step
        step += 1

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

        # log metrics to wandb
        wandb.log({"step": step, "loss": loss.item()})

        # save model checkpoint
        if step % SAVE_INTERVAL == 1:
            torch.save(model.state_dict(), f"artifacts/{wandb.run.name}/{step}.pt")


if __name__ == "__main__":
    # initialize wandb
    wandb.init(
        project="indexing",
        name="baseline",
        config={
            "seed": SEED,
            "dataset": {
                "num_symbols": NUM_SYMBOLS,
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
    dataset = IndexingDataset(num_symbols=NUM_SYMBOLS, array_len=ARRAY_LEN, batch_size=BATCH_SIZE)
    dataloader = DataLoader(dataset, batch_size=None)

    # initialize model
    model = ToyTransformer(
        num_symbols=NUM_SYMBOLS,
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
