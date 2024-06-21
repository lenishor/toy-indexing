import torch

from typing import Optional, Iterator

from jaxtyping import Integer
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset


def primitive_nth_root_of_unity(n: int, p: int) -> int:
    """
    Return the smallest primitive `n`-th root of unity modulo `p`.
    """
    for a in range(1, p):
        for k in range(1, n):
            if pow(a, k, mod=p) == 1:
                break
            return a
    raise ValueError(f"no primitive {n}-th root of unity modulo {p}")


def ntt_matrix(n: int, p: int, device: Optional[torch.device] = None) -> Integer[Tensor, "{n} {n}"]:
    """
    Return the number-theoretic transform (NTT) matrix of size `n` modulo `p`.
    """
    alpha = primitive_nth_root_of_unity(n, p)
    return torch.vander(alpha ** torch.arange(n, device=device), n, increasing=True) % p


class NTTDataset(IterableDataset):
    """
    Number-theoretic transform (NTT) for sequences of length `n` of integers modulo a prime `p`.
    """

    def __init__(
        self,
        n: int,
        p: int,
        batch_size: int = 1,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.n: int = n
        self.p: int = p
        self.batch_size: int = batch_size
        self.seed: Optional[int] = seed
        self.device: torch.device = device or torch.device("cpu")

        self.generator: torch.Generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(seed)

        self.alpha: int = primitive_nth_root_of_unity(n, p)
        self.ntt_matrix: Integer[Tensor, "{n} {n}"] = ntt_matrix(n, p, device=device)

    def __iter__(self) -> Iterator:
        return self

    def __next__(
        self,
    ) -> tuple[Integer[Tensor, "{self.batch_size} {self.n}"], Integer[Tensor, "{self.batch_size} {self.n}"]]:
        x = torch.randint(self.p, (self.batch_size, self.n), generator=self.generator, device=self.device)
        print(self.ntt_matrix.shape)
        print(x.shape)
        y = (x @ self.ntt_matrix.T) % self.p
        return x, y


def get_dataloader(n: int, p: int, batch_size: int, seed: int, device: Optional[torch.device] = None):
    dataset = NTTDataset(n, p, batch_size, seed, device)
    return DataLoader(dataset, batch_size=None)
