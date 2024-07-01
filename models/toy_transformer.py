import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

from einops import repeat
from torch import Tensor


class ToyAttentionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.projection_dim: int = projection_dim
        self.query_projection = nn.Linear(in_features=embedding_dim, out_features=projection_dim)
        self.key_projection = nn.Linear(in_features=embedding_dim, out_features=projection_dim)
        self.ov_circuit = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, seq_len, embedding_dim)
        """
        queries = self.query_projection(x)  # (batch_size, seq_len, projection_dim)
        keys = self.key_projection(x)  # (batch_size, seq_len, projection_dim)
        attention_scores = torch.einsum("b q p, b k p -> b q k", queries, keys) / sqrt(self.projection_dim)
        attention_pattern = F.softmax(attention_scores, dim=-1)
        ov_circuit_output = self.ov_circuit(x)  # (batch_size, seq_len, embedding_dim)
        return torch.einsum("b q k, b k p -> b q p", attention_pattern, ov_circuit_output)


class ToyTransformer(nn.Module):
    def __init__(
        self,
        num_symbols: int,
        max_depth: int,
        seq_len: int,
        embedding_dim: int,
        projection_dim: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_symbols: int = num_symbols
        self.max_depth: int = max_depth
        self.seq_len: int = seq_len

        self.embedding_dim: int = embedding_dim
        self.projection_dim: int = projection_dim
        self.num_layers: int = num_layers

        self.symbol_embed = nn.Embedding(num_embeddings=num_symbols, embedding_dim=embedding_dim)
        self.depth_embed = nn.Embedding(num_embeddings=max_depth, embedding_dim=embedding_dim)
        self.position_embed = nn.Embedding(num_embeddings=seq_len, embedding_dim=embedding_dim)
        self.layers = nn.ModuleList([ToyAttentionHead(embedding_dim, projection_dim) for _ in range(num_layers)])
        self.unembed = nn.Linear(in_features=embedding_dim, out_features=num_symbols)

    def forward(self, sequence: Tensor, depth: Tensor) -> Tensor:
        """
        Args:
            seq: (batch_size, seq_len)
            depth: (batch_size,)
        """
        batch_size, _ = sequence.size()
        positions = repeat(torch.arange(self.seq_len, device=sequence.device), "s -> b s", b=batch_size)
        x = self.symbol_embed(sequence) + self.position_embed(positions)
        x[:, -1] += self.depth_embed(depth)
        for layer in self.layers:
            x += layer(x)
        return self.unembed(x)
