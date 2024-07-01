import torch

from typing import Literal, Optional

from einops import repeat
from torch import Size, Tensor
from torch.distributions import Categorical, Distribution, Uniform
from torch.utils.data import IterableDataset


class RecursiveIndexingTask:
    def __init__(
        self,
        num_symbols: int,
        max_depth: int = 1,
        array_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert num_symbols > 0, "number of symbols must be positive"
        self.num_tokens: int = num_symbols
        assert max_depth >= 0, "maximum recursion depth must be nonnegative"
        self.max_depth: int = max_depth
        if array_len is None:
            array_len = num_symbols
        assert num_symbols % array_len == 0, "number of symbols must be divisible by array length"
        self.array_len: int = array_len

    def __call__(self, array: Tensor, index: Tensor, depth: Tensor) -> Tensor:
        """
        Args:
            array: (batch_size, array_len)
            index: (batch_size,)
            num_iters: (batch_size,)

        Returns:
            index: (batch_size,)
        """
        # repeat `index` to match the number of dimensions of `array`
        index = repeat(index, "b -> b 1")
        # clone `depth` to avoid modifying the original tensor
        depth = depth.clone()
        for _ in range(self.max_depth):
            # get `array` element corresponding to `index`
            element = torch.gather(input=array, dim=-1, index=index)
            # update `index` using `element`
            index = torch.where(depth >= 0, element % self.array_len, index)
            # decrement `depth` in-place
            depth -= 1  # we don't care if `depth` becomes negative, so no need to check
        return index


class IterativeIndexingDataset(IterableDataset):
    def __init__(
        self,
        num_tokens: int,
        max_depth: int = 1,
        array_len: Optional[int] = None,
        alpha: float = 0.0,
        batch_size: int = 1,
    ):
        super().__init__()
        self.task = RecursiveIndexingTask(num_tokens, max_depth, array_len)

        logits = -alpha * torch.arange(start=1, end=(num_tokens + 1), dtype=torch.float32).log()
        self.alpha: float = alpha
        self.token_dist: Distribution = Categorical(logits=logits)

        self.batch_size: int = batch_size
        self.array_len: int = self.task.array_len
        self.array_shape: Size = Size([self.batch_size, self.array_len])
        self.index_shape: Size = Size([self.batch_size])
        self.depth_shape: Size = Size([self.batch_size])

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns:
            array: (batch_size, array_len)
            index: (batch_size,)
            depth: (batch_size,)
            value: (batch_size,)
        """
        # sample `array` and `index` according to `self.token_dist`
        array = self.token_dist.sample(sample_shape=self.array_shape)
        index = self.token_dist.sample(sample_shape=self.index_shape)
        # sample `depth` uniformly from [0, self.task.max_iters)
        depth = torch.randint(low=0, high=self.task.max_depth, size=self.depth_shape)
        value = self.task(array, index, depth)
        return array, index, depth, value
