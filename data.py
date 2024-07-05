import torch

from typing import Optional

from einops import rearrange, repeat
from torch import Size, Tensor
from torch.distributions import Categorical, Distribution
from torch.utils.data import IterableDataset


class IndexingDataset(IterableDataset):
    def __init__(
        self,
        num_symbols: int,
        array_len: Optional[int] = None,
        alpha: float = 0.0,
        batch_size: int = 1,
    ):
        super().__init__()
        self.num_symbols: int = num_symbols
        self.array_len: int = num_symbols if array_len is None else array_len
        self.seq_len: int = self.array_len + 1
        self.alpha: float = alpha
        self.batch_size: int = batch_size

        # symbol distribution is power-law with parameter `alpha`
        logits = -alpha * torch.arange(start=1, end=(num_symbols + 1), dtype=torch.float32).log()
        self.symb_dist: Distribution = Categorical(logits=logits)

        self.seq_shape: Size = Size([batch_size, self.seq_len])

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        """
        Returns:
            sequence: (batch_size, seq_len)
            value: (batch_size,)
        """
        sequence = self.symb_dist.sample(self.seq_shape)
        index = repeat(sequence[:, -1] % self.array_len, "b -> b 1")
        value = rearrange(sequence.gather(dim=-1, index=index), "b 1 -> b")
        return sequence, value
