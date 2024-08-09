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
from torch.utils.data import DataLoader

from config import RunConfig
from data import get_dataloader
from models import ToyTransformer
from utils import evaluate, init_wandb, move_to_device, save_checkpoint


def train(
    config: RunConfig,
    model: ToyTransformer,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: Literal["cpu", "cuda"],
) -> None:
    # create run directory
    # run_dir = Path("runs") / wandb.run.id
    run_dir = Path("runs") / str(config.data.num_symbols)
    run_dir.mkdir()

    # set model to training mode
    model.train()

    # save initial model checkpoint
    save_checkpoint(model, optimizer, run_dir / "start.pth")

    # slice training dataloader to contain the desired number of steps
    dataloader = islice(dataloader, config.data.num_steps)

    # training loop
    for step, batch in enumerate(tqdm(dataloader, total=config.data.num_steps)):
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

        # log training metrics
        if step % config.log.log_every == 0:
            # compute accuracy
            accuracy = (output.argmax(dim=-1) == target).float().mean()

            # compute weight norms
            query_norm = model.query_map.weight.norm()
            key_norm = model.key_map.weight.norm()
            value_norm = model.value_map.weight.norm()

            # log metrics
            wandb.log(
                {
                    "loss": loss.item(),
                    "accuracy": accuracy.item(),
                    "query_norm": query_norm.item(),
                    "key_norm": key_norm.item(),
                    "value_norm": value_norm.item(),
                    # "qk_circuit": wandb.Image(model.qk_circuit().cpu()),
                    # "value_circuit": wandb.Image(model.value_circuit().cpu()),
                },
                step=step,
            )

        # TODO: log evaluation metrics
        # right now we're doing online learning and there's no distribution shift, so w/e
        # if step % config.log.eval_every == 0:
        #     eval_metrics = evaluate(model, dataloader, device)
        #     wandb.log(eval_metrics, step=step)

        # log model metrics and save model checkpoint
        if step % config.log.save_every == 0:
            wandb.log(
                {
                    "qk_circuit": wandb.Image(model.qk_circuit().cpu()),
                    "value_circuit": wandb.Image(model.value_circuit().cpu()),
                },
                step=step,
            )
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
    dataloader = get_dataloader(config)

    # initialize model
    model = ToyTransformer(
        num_symbols=config.data.num_symbols,
        max_array_len=config.data.array_len,
        embedding_dim=config.model.embedding_dim,
    ).to(config.device)

    # initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        betas=(config.optimizer.beta_1, config.optimizer.beta_2),
    )

    # set wandb to watch model
    wandb.watch(model, log="gradients", log_freq=config.log.save_every)

    # train the model
    train(config, model, dataloader, optimizer, config.device)


if __name__ == "__main__":
    main()
