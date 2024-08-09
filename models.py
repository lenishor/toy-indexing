import torch
import torch.nn as nn
from math import sqrt
from config import RunConfig


class ToyTransformer(nn.Module):
    """A simplified transformer model for indexing tasks."""

    def __init__(self, num_symbols: int, max_array_len: int, embedding_dim: int) -> None:
        """Initializes the toy transformer."""
        super().__init__()

        self.num_symbols = num_symbols
        self.max_array_len = max_array_len
        self.embedding_dim = embedding_dim

        self.query_map = nn.Embedding(max_array_len, embedding_dim)
        self.key_map = nn.Embedding(max_array_len, embedding_dim)
        self.value_map = nn.Embedding(num_symbols, num_symbols)

        # normalize weights
        with torch.no_grad():
            self.query_map.weight.div_(sqrt(embedding_dim))
            self.key_map.weight.div_(sqrt(embedding_dim))
            self.value_map.weight.div_(sqrt(num_symbols))

    def forward(self, array: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the toy transformer.

        Shape:
            - array: (batch_size, array_len)
            - index: (batch_size,)
            - output: (batch_size, num_symbols)
        """
        # compute query
        query = self.query_map(index)

        # compute keys
        positions = torch.arange(self.max_array_len, device=array.device)
        keys = self.key_map(positions)

        # compute values
        values = self.value_map(array)

        # compute attention
        preattention = torch.einsum("bd, jd -> bj", query, keys)
        attention = preattention.softmax(dim=-1)
        return torch.einsum("bj, bjv -> bv", attention, values)

    def qk_circuit(self) -> torch.Tensor:
        """Returns a matrix of query-key dot products.

        Shape:
            - output: (max_array_len, max_array_len)
        """
        queries = self.query_map.weight
        keys = self.key_map.weight
        return torch.einsum("qd, kd -> qk", queries, keys)

    def value_circuit(self) -> torch.Tensor:
        """Returns a matrix of symbol logits.

        Shape:
            - output: (num_symbols, num_symbols)
        """
        return self.value_map.weight


def get_model(config: RunConfig) -> ToyTransformer:
    return ToyTransformer(
        num_symbols=config.data.num_symbols,
        max_array_len=config.data.array_len,
        embedding_dim=config.model.embedding_dim,
    )
