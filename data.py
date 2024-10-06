import torch
from typing import Iterator, Literal, Optional
from einops import repeat
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
        array = self.generate_array()
        index = self.generate_index()
        target = array[torch.arange(self.batch_size), index]
        return array, index, target

    def generate_array(self) -> torch.Tensor:
        return torch.randint(
            low=0,
            high=self.num_symbols,
            size=(self.batch_size, self.array_len),
            device=self.device,
            generator=self.gen,
        )

    def generate_index(self) -> torch.Tensor:
        return torch.randint(
            low=0,
            high=self.array_len,
            size=(self.batch_size,),
            device=self.device,
            generator=self.gen,
        )


class OVEvaluationDataset(IndexingDataset):
    """A dataset for evaluating OV performance with uniform sequences."""

    def generate_array(self) -> torch.Tensor:
        symbols = torch.randint(
            low=0,
            high=self.num_symbols,
            size=(self.batch_size,),
            device=self.device,
            generator=self.gen,
        )
        return repeat(symbols, 'b -> b l', l=self.array_len)

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        array = self.generate_array()
        index = self.generate_index()
        target = array[:, 0]  # target is the same as the symbol for each sample
        return array, index, target


def get_dataloader(config: RunConfig, dataset_type: Literal["indexing", "ov_evaluation"] = "indexing") -> DataLoader:
    dataset_class = IndexingDataset if dataset_type == "indexing" else OVEvaluationDataset
    dataset = dataset_class(
        num_symbols=config.data.num_symbols,
        array_len=config.data.array_len,
        batch_size=config.data.batch_size,
        device=config.device,
        seed=config.seed,
    )
    return DataLoader(dataset, batch_size=None, num_workers=0, batch_sampler=None)
