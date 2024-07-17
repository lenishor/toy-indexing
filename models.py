import torch
import torch.nn as nn

from math import sqrt


class ToyTransformer(nn.Module):
    """A simplified transformer model for indexing tasks."""

    def __init__(self, num_symbols: int, max_array_len: int, embedding_dim: int) -> None:
        """Initializes the toy transformer."""
        super().__init__()

        self.num_symbols = num_symbols
        self.max_array_len = max_array_len
        self.embedding_dim = embedding_dim

        self.query_map = nn.Embedding(max_array_len, embedding_dim)
        self.key_map = nn.Embedding(num_symbols, embedding_dim)
        self.value_map = nn.Embedding(num_symbols, num_symbols)

    def forward(self, array: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the toy transformer.

        Shape:
            - array: (batch_size, array_len)
            - index: (batch_size,)
            - output: (batch_size, num_symbols)
        """
        queries = self.query_map(index)
        keys = self.key_map(array)
        values = self.value_map(array)
        preattention = torch.einsum("bd, bjd -> bj", queries, keys) / sqrt(self.embedding_dim)
        attention = preattention.softmax(dim=-1)
        return torch.einsum("bj, bjv -> bv", attention, values)
