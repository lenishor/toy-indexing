import torch
from typing import Iterator, Optional
from torch.utils.data import DataLoader, IterableDataset
from config import RunConfig


class IndexingDataset(IterableDataset):
    """A dataset for an indexing task. Samples are drawn uniformly."""

    def __init__(
        self,
        num_symbols: int,
        array_len: int,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None,
    ) -> None:
        self.num_symbols = num_symbols
        self.array_len = array_len

        self.batch_size = batch_size
        self.device = device
        self.seed = seed

        self.gen = torch.Generator(device=device)
        if self.seed is not None:
            self.gen.manual_seed(seed)

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Yields an instance of the indexing task.

        Shape:
            - array: (batch_size, array_len)
            - index: (batch_size,)
            - target: (batch_size,)
        """
        array = torch.randint(
            low=0,
            high=self.num_symbols,
            size=(self.batch_size, self.array_len),
            device=self.device,
            generator=self.gen,
        )
        index = torch.randint(
            low=0,
            high=self.array_len,
            size=(self.batch_size,),
            device=self.device,
            generator=self.gen,
        )
        target = array[torch.arange(self.batch_size), index]
        return array, index, target


def get_dataloader(config: RunConfig) -> DataLoader:
    dataset = IndexingDataset(
        num_symbols=config.data.num_symbols,
        array_len=config.data.array_len,
        batch_size=config.data.batch_size,
    )
    return DataLoader(dataset, batch_size=None)
