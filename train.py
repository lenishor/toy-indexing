import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from itertools import islice
from pathlib import Path
from typing import Literal

from tqdm import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config import RunConfig, setup_config_store
from data import get_dataloader
from models import ToyTransformer
from utils import init_wandb, move_to_device, save_checkpoint, evaluate


def train(
    config: RunConfig,
    model: ToyTransformer,
    train_dataloader: DataLoader,
    ov_eval_dataloader: DataLoader,
    optimizer: Optimizer,
    device: Literal["cpu", "cuda"],
) -> None:
    # get run name
    run_name = config.run_name or wandb.run.id

    # create run directory
    run_dir = Path("runs") / run_name
    run_dir.mkdir()

    # set model to training mode
    model.train()

    # save initial model checkpoint
    save_checkpoint(model, optimizer, run_dir / "start.pth")

    # slice training dataloader to contain the desired number of steps
    train_dataloader = islice(train_dataloader, config.data.num_steps)

    # initialize learning rate scheduler
    if config.scheduler.type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config.data.num_steps, eta_min=config.scheduler.min_learning_rate)
    else:
        scheduler = None

    # training loop
    for step, batch in enumerate(tqdm(train_dataloader, total=config.data.num_steps)):
        # move batch to device
        array, index, target = move_to_device(batch, device)

        # zero out gradients
        optimizer.zero_grad(set_to_none=True)

        # forward pass
        output = model(array, index)

        # compute loss
        loss = F.cross_entropy(output, target)

        # backward pass
        loss.backward()

        # clip gradients
        nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip)

        # take an optimization step
        optimizer.step()

        # step the scheduler
        if scheduler is not None:
            scheduler.step()

        # log training metrics
        if step % config.log.log_every == 0:
            # compute accuracy
            accuracy = (output.argmax(dim=-1) == target).float().mean()

            # compute the average attention paid to the correct position
            batch_size, *_ = model.last_attention.shape
            batch_indices = torch.arange(batch_size, device=index.device)
            correct_attention = model.last_attention[batch_indices, index].mean()

            # log metrics
            data = {
                "train/loss": loss.item(),
                "train/accuracy": accuracy.item(),
                "train/correct_attention": correct_attention.item(),
            }

            # log learning rate
            if scheduler is not None:
                data["learning_rate"] = scheduler.get_last_lr()[0]

            # log data to wandb
            wandb.log(data, step=step)

        # evaluate on OV dataset and save model checkpoint
        if step % config.log.eval_every == 0:
            ov_metrics = evaluate(model, ov_eval_dataloader, num_samples=config.eval.num_samples, device=device)
            wandb.log({
                "ov_eval/loss": ov_metrics["loss"],
                "ov_eval/accuracy": ov_metrics["accuracy"]
            }, step=step)
            save_checkpoint(model, optimizer, run_dir / f"step_{step}.pth")

    # save final model checkpoint
    save_checkpoint(model, optimizer, run_dir / "end.pth")


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: RunConfig) -> None:
    # initialize wandb
    init_wandb(config)

    # set random seed
    torch.manual_seed(config.seed)

    # initialize training dataloader
    train_dataloader = get_dataloader(config, dataset_type="indexing")

    # initialize OV evaluation dataloader
    ov_eval_dataloader = get_dataloader(config, dataset_type="ov_evaluation")

    # initialize model
    model = ToyTransformer(
        vocab_size=config.data.num_symbols,
        sequence_len=config.data.array_len,
        embedding_dim=config.model.embedding_dim,
        save_attention=True,
    ).to(config.device)

    # initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        betas=(config.optimizer.beta_1, config.optimizer.beta_2),
    )

    # train the model
    train(config, model, train_dataloader, ov_eval_dataloader, optimizer, config.device)


if __name__ == "__main__":
    setup_config_store()
    main()
