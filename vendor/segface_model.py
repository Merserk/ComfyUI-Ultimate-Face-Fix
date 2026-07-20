from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import convnext_base

from .segface_transformer import LayerNorm2d, TwoWayTransformer


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        hidden = [hidden_dim] * (num_layers - 1)
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            nn.Linear(source, target)
            for source, target in zip([input_dim] + hidden, hidden + [output_dim])
        )
        self.sigmoid_output = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < self.num_layers - 1 else layer(x)
        return x


class FaceDecoder(nn.Module):
    def __init__(self, transformer_dim: int, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.background_token = nn.Embedding(1, transformer_dim)
        self.neck_token = nn.Embedding(1, transformer_dim)
        self.face_token = nn.Embedding(1, transformer_dim)
        self.cloth_token = nn.Embedding(1, transformer_dim)
        self.rightear_token = nn.Embedding(1, transformer_dim)
        self.leftear_token = nn.Embedding(1, transformer_dim)
        self.rightbro_token = nn.Embedding(1, transformer_dim)
        self.leftbro_token = nn.Embedding(1, transformer_dim)
        self.righteye_token = nn.Embedding(1, transformer_dim)
        self.lefteye_token = nn.Embedding(1, transformer_dim)
        self.nose_token = nn.Embedding(1, transformer_dim)
        self.innermouth_token = nn.Embedding(1, transformer_dim)
        self.lowerlip_token = nn.Embedding(1, transformer_dim)
        self.upperlip_token = nn.Embedding(1, transformer_dim)
        self.hair_token = nn.Embedding(1, transformer_dim)
        self.glass_token = nn.Embedding(1, transformer_dim)
        self.hat_token = nn.Embedding(1, transformer_dim)
        self.earring_token = nn.Embedding(1, transformer_dim)
        self.necklace_token = nn.Embedding(1, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.output_hypernetwork_mlps = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

    def forward(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor) -> torch.Tensor:
        output_tokens = torch.cat(
            [
                self.background_token.weight,
                self.neck_token.weight,
                self.face_token.weight,
                self.cloth_token.weight,
                self.rightear_token.weight,
                self.leftear_token.weight,
                self.rightbro_token.weight,
                self.leftbro_token.weight,
                self.righteye_token.weight,
                self.lefteye_token.weight,
                self.nose_token.weight,
                self.innermouth_token.weight,
                self.lowerlip_token.weight,
                self.upperlip_token.weight,
                self.hair_token.weight,
                self.glass_token.weight,
                self.hat_token.weight,
                self.earring_token.weight,
                self.necklace_token.weight,
            ],
            dim=0,
        )
        tokens = output_tokens.unsqueeze(0).expand(image_embeddings.shape[0], -1, -1)
        batch, channels, height, width = image_embeddings.shape
        token_output, source = self.transformer(image_embeddings, image_pe.expand(batch, -1, -1, -1), tokens)
        source = source.transpose(1, 2).reshape(batch, channels, height, width)
        upscaled = self.output_upscaling(source)
        hyper = self.output_hypernetwork_mlps(token_output)
        return (hyper @ upscaled.flatten(2)).reshape(batch, -1, upscaled.shape[2], upscaled.shape[3])


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("positional_encoding_gaussian_matrix", scale * torch.randn((2, num_pos_feats)))

    def _encode(self, coords: torch.Tensor) -> torch.Tensor:
        coords = (2 * coords - 1) @ self.positional_encoding_gaussian_matrix
        coords = 2 * math.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: tuple[int, int]) -> torch.Tensor:
        height, width = size
        grid = torch.ones(
            (height, width),
            device=self.positional_encoding_gaussian_matrix.device,
            dtype=self.positional_encoding_gaussian_matrix.dtype,
        )
        y = (grid.cumsum(dim=0) - 0.5) / height
        x = (grid.cumsum(dim=1) - 0.5) / width
        return self._encode(torch.stack([x, y], dim=-1)).permute(2, 0, 1)


class SegfaceMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 256)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states.flatten(2).transpose(1, 2))


class SegFaceCeleb(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = convnext_base(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.target_layer_names = {"0.1", "0.3", "0.5", "0.7"}
        self.multi_scale_features: list[torch.Tensor] = []
        for name, module in self.backbone.named_modules():
            if name in self.target_layer_names:
                module.register_forward_hook(self._save_features)
        self.pe_layer = PositionEmbeddingRandom(128)
        self.face_decoder = FaceDecoder(
            transformer_dim=256,
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
        )
        self.linear_c = nn.ModuleList(SegfaceMLP(size) for size in (128, 256, 512, 1024))
        self.linear_fuse = nn.Conv2d(1024, 256, kernel_size=1, bias=False)

    def _save_features(self, _module, _inputs, output) -> None:
        self.multi_scale_features.append(output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.multi_scale_features.clear()
        self.backbone(x)
        if len(self.multi_scale_features) != 4:
            raise RuntimeError(f"SegFace expected four ConvNeXt feature maps, got {len(self.multi_scale_features)}")
        batch = self.multi_scale_features[-1].shape[0]
        states = []
        output_size = self.multi_scale_features[0].shape[2:]
        for feature, projection in zip(self.multi_scale_features, self.linear_c):
            height, width = feature.shape[2:]
            projected = projection(feature).permute(0, 2, 1).reshape(batch, 256, height, width)
            states.append(F.interpolate(projected, size=output_size, mode="bilinear", align_corners=False))
        fused = self.linear_fuse(torch.cat(states[::-1], dim=1))
        positional = self.pe_layer(fused.shape[2:]).unsqueeze(0)
        return self.face_decoder(fused, positional)
