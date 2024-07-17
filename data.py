import torch

from typing import Iterator, Optional

from torch.utils.data import IterableDataset


class SelectApplyDataset(IterableDataset):
    """An iterable dataset for a select-apply task.

    The input consists of an array `xs` and an index `i`.
    The output is the value `y` obtained by applying a fixed map `f` to the selected element `x = xs[i]` of the array `xs`.
    The task, therefore, requires the composition of two distinct capabilities:
    the *selection* of the array element corresponding to the given index and the *application* of a function to the selected element.

    Currently, only supports the special case in which learning a capability in isolation does not lead to a reduction in loss.
    """

    def __init__(
        self,
        domain_size: int,
        range_size: int,
        array_len: int,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None,
    ) -> None:
        if array_len % range_size != 0:
            raise ValueError(f"range size {range_size} must divide array length {array_len}")

        if domain_size % range_size != 0:
            raise ValueError(f"range size {range_size} must divide domain size {domain_size}")

        self.domain_size: int = domain_size
        self.range_size: int = range_size
        self.kernel_size: int = domain_size // range_size
        self.array_len: int = array_len
        self.seq_len: int = array_len + 1

        self.device: torch.device = device
        self.seed: Optional[int] = seed

        self.generator: torch.Generator = torch.Generator(device=device)
        if self.seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        permutation = torch.randperm(n=self.array_len, generator=self.generator, device=self.device)
        ys = permutation % self.range_size
        xs = ys + self.range_size * torch.randint(
            low=0,
            high=self.kernel_size,
            size=(self.array_len,),
            generator=self.generator,
            device=self.device,
        )
        i = torch.randint(low=0, high=self.array_len, size=(1,), generator=self.generator, device=self.device)
        y = ys[i].squeeze()
        seq = torch.concatenate([xs, i])
        return seq, y
