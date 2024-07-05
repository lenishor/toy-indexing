import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from torch.optim import Optimizer
from torch.utils.data import DataLoader

from data import IndexingDataset
from models import ToyTransformer


# dataset parameters
NUM_SYMBOLS = 128
ARRAY_LEN = 64
BATCH_SIZE = 1_024

# model hyperparameters
EMBEDDING_DIM = 64
PROJECTION_DIM = 32

# training hyperparameters
LEARNING_RATE = 0.001
NUM_STEPS = 1_000


def train(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, device: torch.device):
    # set model to training mode
    model.train()
    # move model to device
    model.to(device)

    # main training loop
    step = 0
    for sequence, value in dataloader:
        # break if we reach the maximum number of steps
        if step >= NUM_STEPS:
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

        # optimizer step
        optimizer.step()

        # log metrics to wandb
        wandb.log({"step": step, "loss": loss.item()})


if __name__ == "__main__":
    # initialize wandb
    wandb.init(
        project="toy-transformer",
        config={
            "dataset": {
                "num_symbols": NUM_SYMBOLS,
                "array_len": ARRAY_LEN,
                "batch_size": BATCH_SIZE,
            },
            "model": {
                "embedding_dim": EMBEDDING_DIM,
                "projection_dim": PROJECTION_DIM,
            },
            "training": {
                "learning_rate": LEARNING_RATE,
                "num_steps": NUM_STEPS,
            },
        },
    )

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
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train model
    train(model, dataloader, optimizer, device)

    # finish wandb
    wandb.finish()
