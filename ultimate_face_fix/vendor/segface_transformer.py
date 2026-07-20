# Adapted from Meta's Segment Anything transformer implementation as used by
# Kartik Narayan et al.'s MIT-licensed SegFace repository.

from __future__ import annotations

import math
from typing import Type

import torch
from torch import Tensor, nn


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * normalized + self.bias[:, None, None]


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList(
            TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                activation=activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=index == 0,
            )
            for index in range(depth)
        )
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding: Tensor, image_pe: Tensor, point_embedding: Tensor) -> tuple[Tensor, Tensor]:
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe)
        query = queries + point_embedding
        key = keys + image_pe
        queries = self.norm_final_attn(queries + self.final_attn_token_to_image(q=query, k=key, v=keys))
        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor) -> tuple[Tensor, Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            query = queries + query_pe
            queries = queries + self.self_attn(q=query, k=query, v=queries)
        queries = self.norm1(queries)

        query = queries + query_pe
        key = keys + key_pe
        queries = self.norm2(queries + self.cross_attn_token_to_image(q=query, k=key, v=keys))
        queries = self.norm3(queries + self.mlp(queries))

        query = queries + query_pe
        key = keys + key_pe
        keys = self.norm4(keys + self.cross_attn_image_to_token(q=key, k=query, v=queries))
        return queries, keys


class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        if self.internal_dim % num_heads != 0:
            raise ValueError("num_heads must divide the internal embedding dimension")
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    @staticmethod
    def _separate_heads(x: Tensor, num_heads: int) -> Tensor:
        batch, tokens, channels = x.shape
        return x.reshape(batch, tokens, num_heads, channels // num_heads).transpose(1, 2)

    @staticmethod
    def _recombine_heads(x: Tensor) -> Tensor:
        batch, heads, tokens, channels = x.shape
        return x.transpose(1, 2).reshape(batch, tokens, heads * channels)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        query = self._separate_heads(self.q_proj(q), self.num_heads)
        key = self._separate_heads(self.k_proj(k), self.num_heads)
        value = self._separate_heads(self.v_proj(v), self.num_heads)
        attention = query @ key.transpose(-2, -1)
        attention = torch.softmax(attention / math.sqrt(query.shape[-1]), dim=-1)
        return self.out_proj(self._recombine_heads(attention @ value))
