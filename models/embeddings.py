import torch
import torch.nn as nn


class Learned3DPositionEmbedding(nn.Module):
    # Learned 3D absolute position embeddings added after latent compression.
    # This explicitly restores spatiotemporal position cues lost by the shared
    # projector pipeline.
    def __init__(self, frames=11, height=15, width=26, dim=2048):
        super().__init__()
        self.pos_t = nn.Parameter(torch.zeros(frames, dim))
        self.pos_h = nn.Parameter(torch.zeros(height, dim))
        self.pos_w = nn.Parameter(torch.zeros(width, dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.pos_t, std=0.02)
        nn.init.normal_(self.pos_h, std=0.02)
        nn.init.normal_(self.pos_w, std=0.02)

    def forward(self, x):
        # x: [B, T, H, W, C]
        # Add independent learned offsets along time, height, and width, then
        # flatten into the sequence form expected by the comparator.
        time_steps, height, width = x.size(1), x.size(2), x.size(3)
        if time_steps > self.pos_t.size(0) or height > self.pos_h.size(0) or width > self.pos_w.size(0):
            raise ValueError(
                "Input grid is larger than the configured learned position embedding table: "
                f"input={(time_steps, height, width)} vs table="
                f"{(self.pos_t.size(0), self.pos_h.size(0), self.pos_w.size(0))}"
            )
        x = x + self.pos_t[None, :time_steps, None, None, :]
        x = x + self.pos_h[None, None, :height, None, :]
        x = x + self.pos_w[None, None, None, :width, :]
        return x.reshape(x.size(0), -1, x.size(-1))


class SegmentEmbedding(nn.Module):
    # Segment embeddings mark token provenance so the comparator can tell apart
    # query tokens, candidate-1 tokens, candidate-2 tokens, context, task prompt, and stage.
    SEGMENTS = ("query", "h1", "h2", "context", "prompt", "stage")

    def __init__(self, dim=2048):
        super().__init__()
        self.segment_to_index = {name: idx for idx, name in enumerate(self.SEGMENTS)}
        self.embedding = nn.Parameter(torch.zeros(len(self.SEGMENTS), 1, dim))
        nn.init.normal_(self.embedding, std=0.02)

    def add(self, x, segment_name):
        index = self.segment_to_index[segment_name]
        return x + self.embedding[index]


class StageEmbedding(nn.Module):
    # Stage token for early / middle / late denoising regions.
    def __init__(self, num_stages=3, dim=2048):
        super().__init__()
        self.embedding = nn.Embedding(num_stages, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, stage_id):
        return self.embedding(stage_id).unsqueeze(1)


class QueryEmbedding(nn.Module):
    # Learned query probes that summarize the pairwise decision after the Qwen
    # comparator updates them.
    def __init__(self, num_queries=4, dim=2048):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, num_queries, dim))
        nn.init.normal_(self.query, mean=0.0, std=dim ** -0.5)

    def forward(self, batch_size):
        return self.query.expand(batch_size, -1, -1)
