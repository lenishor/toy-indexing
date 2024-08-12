import torch
import torch.nn as nn

from math import sqrt


class ToyTransformer(nn.Module):
    """A simplified transformer model for indexing tasks."""

    def __init__(self, vocab_size: int, sequence_len: int, embedding_dim: int) -> None:
        """Initializes the toy transformer."""
        super().__init__()

        self.vocab_size = vocab_size
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim

        # TODO: figure out if there's a nicer way of initializing these that's as convenient for downstream use
        self.query_map = nn.Embedding(sequence_len, embedding_dim)
        self.keys = nn.Parameter(torch.randn(sequence_len, embedding_dim))
        self.value_map = nn.Embedding(vocab_size, embedding_dim)
        self.output_matrix = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, sequence: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Does a forward pass through the toy transformer.

        Shape:
            - sequence: (batch_size, sequence_len)
            - index: (batch_size,)
            - output: (batch_size, num_symbols)
        """
        # compute query
        query = self.query_map(index)

        # compute preattention
        preattention = (query @ self.keys) / sqrt(self.embedding_dim)

        # compute attention
        attention = preattention.softmax(dim=-1)

        # compute values
        values = self.value_map(sequence)

        # compute results
        results = (attention @ values).sum(dim=-2)

        # compute output
        output = self.output_matrix(results)

        return output
