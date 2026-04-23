import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjector(nn.Module):
    # Shared 2-layer MLP used by the spatial/temporal/context projection blocks.
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SpatialProjector(nn.Module):
    # Merge each 2x2 spatial neighborhood into one token so the video token count
    # matches the planned [B, 21, 15, 26, 2048] intermediate shape.
    def __init__(self, in_dim=1536, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.projector = MLPProjector(in_dim * 4, hidden_dim, out_dim)

    def forward(self, x):
        # x: [B, T, H, W, C]
        batch_size, frames, height, width, channels = x.shape
        if height % 2 != 0 or width % 2 != 0:
            # Pad odd grids so the 2x2 merge is always valid.
            x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2, 0, 0))
            height = x.size(2)
            width = x.size(3)
        # Repack neighboring spatial cells into the channel dimension, then
        # learn the actual projection with the shared MLP.
        x = x.view(batch_size, frames, height // 2, 2, width // 2, 2, channels)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(
            batch_size,
            frames,
            height // 2,
            width // 2,
            channels * 4,
        )
        return self.projector(x)


class TemporalProjector(nn.Module):
    # Merge neighboring frames in time so the final compressed latent uses
    # 11 temporal positions instead of 21.
    def __init__(self, dim=2048, hidden_dim=2048):
        super().__init__()
        self.projector = MLPProjector(dim * 2, hidden_dim, dim)

    def forward(self, x):
        # x: [B, T, H, W, C]
        if x.size(1) % 2 != 0:
            # The plan pads the last frame once when the frame count is odd.
            x = torch.cat([x, x[:, -1:, ...]], dim=1)
        batch_size, frames, height, width, channels = x.shape
        x = x.view(batch_size, frames // 2, 2, height, width, channels)
        x = x.permute(0, 1, 3, 4, 2, 5).reshape(
            batch_size,
            frames // 2,
            height,
            width,
            channels * 2,
        )
        return self.projector(x)


class VideoProjector(nn.Module):
    # End-to-end latent compressor:
    # 1) Wan-style patch embedding
    # 2) spatial 2x2 merge
    # 3) temporal pair merge
    def __init__(
        self,
        latent_dim=16,
        patch_dim=1536,
        hidden_dim=2048,
        patch_kernel=(1, 2, 2),
        patch_stride=(1, 2, 2),
    ):
        super().__init__()
        self.patch_embed = nn.Conv3d(
            latent_dim,
            patch_dim,
            kernel_size=patch_kernel,
            stride=patch_stride,
        )
        self.spatial_projector = SpatialProjector(
            in_dim=patch_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
        )
        self.temporal_projector = TemporalProjector(
            dim=hidden_dim,
            hidden_dim=hidden_dim,
        )
        self.output_grid = None

    def forward(self, latent):
        # latent: [B, C, T, H, W]
        # Conv3d keeps time unchanged and halves the spatial resolution once.
        tokens = self.patch_embed(latent)
        tokens = tokens.permute(0, 2, 3, 4, 1).contiguous()
        tokens = self.spatial_projector(tokens)
        tokens = self.temporal_projector(tokens)
        # Keep the compressed 3D grid for debugging and shape validation.
        self.output_grid = tuple(tokens.shape[1:4])
        return tokens

    def output_tokens(self, latent_frames=21, latent_height=60, latent_width=104):
        # Cheap shape helper for tests and config validation without a forward pass.
        height = math.ceil(latent_height / 2)
        width = math.ceil(latent_width / 2)
        height = math.ceil(height / 2)
        width = math.ceil(width / 2)
        frames = latent_frames
        if frames % 2 != 0:
            frames += 1
        frames //= 2
        return frames, height, width


class ContextProjector(nn.Module):
    # Project text/image condition tokens into the same 2048-d hidden space as
    # the compressed video tokens before feeding them to the Qwen comparator.
    def __init__(
        self,
        text_dim=4096,
        image_dim=1280,
        hidden_dim=2048,
        max_text_tokens=512,
        max_image_tokens=64,
    ):
        super().__init__()
        self.max_text_tokens = max_text_tokens
        self.max_image_tokens = max_image_tokens
        self.text_projector = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.image_projector = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def project_text(self, text_emb):
        # Keep only the first 512 text tokens, matching the design note.
        if text_emb.size(1) > self.max_text_tokens:
            text_emb = text_emb[:, : self.max_text_tokens]
        return self.text_projector(text_emb)

    def project_image(self, image_emb):
        # Pool visual tokens down to at most 64 to control comparator context length.
        if image_emb.size(1) > self.max_image_tokens:
            image_emb = image_emb.transpose(1, 2)
            image_emb = F.adaptive_avg_pool1d(image_emb, self.max_image_tokens)
            image_emb = image_emb.transpose(1, 2)
        return self.image_projector(image_emb)

    def forward(self, text_emb, image_emb):
        # Context order follows the plan: image tokens first, then text tokens.
        image_tokens = self.project_image(image_emb)
        text_tokens = self.project_text(text_emb)
        return torch.cat([image_tokens, text_tokens], dim=1)
