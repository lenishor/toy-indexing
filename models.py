import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

from einops import repeat
from torch import Tensor


class ToyAttentionHead(nn.Module):
    """An attention head that only has an output at the last position of the input sequence.

    This is intended to be used as the final layer of a transformer model for use with algorithmic tasks.
    Note that this implementation merges the output and the value projections into a single linear layer.

    Attributes:
        embedding_dim: The dimension of the input embeddings.
        projection_dim: The dimension of the query and key spaces.
        query_projection: A linear layer that projects the input to the query space.
        key_projection: A linear layer that projects the input to the key space.
        output_projection: A linear layer that projects the input to the output space.
    """

    def __init__(self, embedding_dim: int, projection_dim: int) -> None:
        """Initializes the FinalPositionAttentionHead.

        Args:
            embedding_dim: The dimension of the input embeddings.
            projection_dim: The dimension of the query and key spaces.
        """
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.projection_dim: int = projection_dim
        self.query_projection = nn.Linear(in_features=embedding_dim, out_features=projection_dim)
        self.key_projection = nn.Linear(in_features=embedding_dim, out_features=projection_dim)
        self.output_projection = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def forward(self, inp: Tensor) -> Tensor:
        """Does a forward pass through the attention head.

        Args:
            inp: The input; that is, the residual stream. Shape: (batch_size, seq_len, embedding_dim)

        Returns:
            The output of the attention head. Shape: (batch_size, embedding_dim)
        """
        query = self.query_projection(inp)[:, -1]  # shape: (batch_size, projection_dim)
        keys = self.key_projection(inp)  # shape: (batch_size, seq_len, projection_dim)
        attention_scores = torch.einsum("b p, b k p -> b k", query, keys)  # shape: (batch_size, seq_len)
        attention_scores /= sqrt(self.projection_dim)  # normalization from ``Attention is All You Need''
        attention_pattern = F.softmax(attention_scores, dim=-1)  # shape: (batch_size, seq_len)
        tokenwise_out = self.output_projection(inp)  # shape: (batch_size, seq_len, embedding_dim)
        out = torch.einsum("b k, b k d -> b d", attention_pattern, tokenwise_out)  # shape: (batch_size, embedding_dim)
        return out


class ToyTransformer(nn.Module):
    """An attention-only transformer model for use in an indexing task.

    The model outputs logits only for the last position of the input sequence.
    Currently only supports a single layer with a single attention head, as this suffices for a (nonrecursive) indexing task.

    Attributes:
        num_symbols: The number of symbols in the input sequence.
        seq_len: The length of the input sequence.
        embedding_dim: The dimension of the input embeddings.
        projection_dim: The dimension of the query and key spaces.
        symbol_embed: An embedding layer for the input symbols.
        position_embed: An embedding layer for the positions in the input sequence.
        attention_head: An attention head that outputs only at the last position of the input sequence.
        unembed: A linear layer that projects the output to logits.
    """

    def __init__(
        self,
        num_symbols: int,
        seq_len: int,
        embedding_dim: int,
        projection_dim: int,
    ) -> None:
        """Initializes the ToyTransformer.

        Args:
            num_symbols: The number of symbols in the input sequence.
            seq_len: The length of the input sequence.
            embedding_dim: The dimension of the input embeddings.
            projection_dim: The dimension of the query and key spaces.
        """
        super().__init__()

        self.num_symbols: int = num_symbols
        self.seq_len: int = seq_len
        self.embedding_dim: int = embedding_dim
        self.projection_dim: int = projection_dim

        self.symbol_embed = nn.Embedding(num_embeddings=num_symbols, embedding_dim=embedding_dim)
        self.position_embed = nn.Embedding(num_embeddings=seq_len, embedding_dim=embedding_dim)
        self.attention_head = ToyAttentionHead(embedding_dim=embedding_dim, projection_dim=projection_dim)
        self.unembed = nn.Linear(in_features=embedding_dim, out_features=num_symbols)

    def forward(self, sequence: Tensor) -> Tensor:
        """Does a forward pass through the transformer model.

        Args:
            sequence: The input sequence. Shape: (batch_size, seq_len)

        Returns:
            Logits computed at the last position of the input sequence. Shape: (batch_size, num_symbols)
        """
        batch_size, _ = sequence.size()
        positions = torch.arange(self.seq_len, device=sequence.device)  # shape: (seq_len,)
        positions = repeat(positions, "s -> b s", b=batch_size)  # shape: (batch_size, seq_len)
        symbol_embeds = self.symbol_embed(sequence)  # shape: (batch_size, seq_len, embedding_dim)
        position_embeds = self.position_embed(positions)  # shape: (batch_size, seq_len, embedding_dim)
        embeds = symbol_embeds + position_embeds  # shape: (batch_size, seq_len, embedding_dim)
        embeds = self.attention_head(embeds)  # shape: (batch_size, embedding_dim)
        out = self.unembed(embeds)  # shape: (batch_size, num_symbols)
        return out
