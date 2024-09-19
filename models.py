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

        self.query_map = nn.Embedding(sequence_len, embedding_dim)
        self.key_map = nn.Embedding(sequence_len, embedding_dim)
        self.value_map = nn.Embedding(vocab_size, embedding_dim)
        self.output_map = nn.Linear(embedding_dim, vocab_size, bias=False)

        self.last_attention = None

    def forward(self, sequence: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Does a forward pass through the toy transformer.

        Shape:
            - sequence: (batch_size, sequence_len)
            - index: (batch_size,)
            - output: (batch_size, num_symbols)
        """
        # compute query
        query = self.query_map(index)

        # compute keys
        positions = torch.arange(self.sequence_len, device=index.device)
        keys = self.key_map(positions)

        # compute preattention
        preattention = torch.einsum("bd, kd -> bk", query, keys) / sqrt(self.embedding_dim)

        # compute attention
        attention = preattention.softmax(dim=-1)
        self.last_attention = attention  # store the attention

        # compute values
        values = self.value_map(sequence)

        # compute results
        results = torch.einsum("bk, bkd -> bd", attention, values)

        # compute output
        output = self.output_map(results)

        return output

    def qk_circuit(self) -> torch.Tensor:
        """Returns the QK circuit matrix.

        Shape:
            - qk_circuit: (sequence_len, sequence_len)
        """
        device = self.key_map.weight.device
        positions = torch.arange(self.sequence_len, device=device)
        queries = self.query_map(positions)
        keys = self.key_map(positions)
        qk_circuit = torch.einsum("qd, kd -> qk", queries, keys) / sqrt(self.embedding_dim)
        return qk_circuit

    def ov_circuit(self) -> torch.Tensor:
        """Returns the OV circuit matrix.

        Shape:
            - ov_circuit: (vocab_size, vocab_size)
        """
        device = self.value_map.weight.device
        tokens = torch.arange(self.vocab_size, device=device)
        values = self.value_map(tokens)
        outputs = self.output_map.weight
        ov_circuit = torch.einsum("vd, od -> vo", values, outputs)
        return ov_circuit
