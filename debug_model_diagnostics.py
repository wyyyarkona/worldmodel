from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from score_model_v2.eval_manifest import build_model_from_config
from score_model_v2.train_v2 import (
    PairwiseLatentDatasetV2,
    build_dataloader,
    load_config,
    move_batch,
    resolve_repo_relative_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run structural diagnostics for score_model_v2 on a small batch of samples."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=32)
    return parser.parse_args()


def tensor_summary(tensor: torch.Tensor):
    tensor = tensor.detach().to(dtype=torch.float32)
    flat = tensor.reshape(-1)
    return {
        "shape": list(tensor.shape),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def sample_variation_summary(tensor: torch.Tensor):
    # Measures how much this representation varies across samples in the batch.
    # If mean_per_feature_std_across_batch is ~0, the model is effectively
    # producing almost the same feature vector for every sample.
    tensor = tensor.detach().to(dtype=torch.float32)
    if tensor.size(0) <= 1:
        return {
            "batch_size": int(tensor.size(0)),
            "mean_per_feature_std_across_batch": 0.0,
            "max_per_feature_std_across_batch": 0.0,
        }
    flattened = tensor.reshape(tensor.size(0), -1)
    per_feature_std = flattened.std(dim=0, unbiased=False)
    return {
        "batch_size": int(tensor.size(0)),
        "mean_per_feature_std_across_batch": float(per_feature_std.mean().item()),
        "max_per_feature_std_across_batch": float(per_feature_std.max().item()),
    }


def pairwise_difference_summary(a: torch.Tensor, b: torch.Tensor):
    diff = (a.detach().to(dtype=torch.float32) - b.detach().to(dtype=torch.float32)).abs()
    flat = diff.reshape(diff.size(0), -1)
    per_sample_mean = flat.mean(dim=1)
    per_sample_max = flat.max(dim=1).values
    return {
        "global_mean_abs_diff": float(flat.mean().item()),
        "global_max_abs_diff": float(flat.max().item()),
        "per_sample_mean_abs_diff": [float(x.item()) for x in per_sample_mean],
        "per_sample_max_abs_diff": [float(x.item()) for x in per_sample_max],
    }


@torch.no_grad()
def run_diagnostics(model, batch):
    f1 = batch["f1"]
    f2 = batch["f2"]
    text_emb = batch["text_emb"]
    image_emb = batch["image_emb"]
    stage_id = batch["stage_id"]

    h1 = model.encode_video(f1)
    h2 = model.encode_video(f2)
    context = model.context_projector(text_emb, image_emb)
    sequence, attention_mask = model.build_sequence(h1, h2, context, stage_id)
    query_len = model.query_embed.query.size(1)
    h1_len = h1.size(1)
    h2_len = h2.size(1)
    context_len = context.size(1)
    query_input = sequence[:, -query_len:]

    comparator_param = next(model.comparator.parameters())
    sequence = sequence.to(device=comparator_param.device, dtype=comparator_param.dtype)
    attention_mask = attention_mask.to(device=comparator_param.device)
    hidden_states = model.comparator(sequence, attention_mask=attention_mask)
    hidden_parts = model.split_hidden_states(hidden_states, h1_len=h1_len, h2_len=h2_len, context_len=context_len)
    query_hidden = hidden_parts["query"]
    h1_hidden = hidden_parts["h1"]
    h2_hidden = hidden_parts["h2"]
    context_hidden = hidden_parts["context"]
    readout_features = model.build_readout_features(hidden_parts)

    head_param = next(model.score_head.parameters())
    readout_features = readout_features.to(device=head_param.device, dtype=head_param.dtype)
    logits = model.score_head(readout_features).squeeze(-1)
    scores = torch.sigmoid(logits)

    swapped_outputs = model(f2, f1, text_emb, image_emb, stage_id, return_aux_stats=False)
    swapped_logits = swapped_outputs["logit"].detach().to(dtype=torch.float32)
    swapped_scores = swapped_outputs["score"].detach().to(dtype=torch.float32)

    swapped_h1 = model.encode_video(f2)
    swapped_h2 = model.encode_video(f1)
    swapped_context = model.context_projector(text_emb, image_emb)
    swapped_sequence, swapped_attention_mask = model.build_sequence(swapped_h1, swapped_h2, swapped_context, stage_id)
    swapped_query_input = swapped_sequence[:, -query_len:]
    swapped_sequence = swapped_sequence.to(device=comparator_param.device, dtype=comparator_param.dtype)
    swapped_attention_mask = swapped_attention_mask.to(device=comparator_param.device)
    swapped_hidden_states = model.comparator(swapped_sequence, attention_mask=swapped_attention_mask)
    swapped_hidden_parts = model.split_hidden_states(
        swapped_hidden_states,
        h1_len=h1_len,
        h2_len=h2_len,
        context_len=context_len,
    )
    swapped_query_hidden = swapped_hidden_parts["query"]
    swapped_h1_hidden = swapped_hidden_parts["h1"]
    swapped_h2_hidden = swapped_hidden_parts["h2"]
    swapped_context_hidden = swapped_hidden_parts["context"]
    swapped_readout = model.build_readout_features(swapped_hidden_parts)

    score_head_weight = next(model.score_head.parameters()).detach().to(dtype=torch.float32)
    last_linear = None
    for module in reversed(model.score_head):
        if isinstance(module, torch.nn.Linear):
            last_linear = module
            break
    score_head_bias = (
        last_linear.bias.detach().to(dtype=torch.float32)
        if last_linear is not None and last_linear.bias is not None
        else torch.zeros(1, dtype=torch.float32)
    )

    result = {
        "score_head_weight": tensor_summary(score_head_weight),
        "score_head_bias": tensor_summary(score_head_bias),
        "h1": {
            "tensor": tensor_summary(h1),
            "variation_across_samples": sample_variation_summary(h1),
        },
        "h2": {
            "tensor": tensor_summary(h2),
            "variation_across_samples": sample_variation_summary(h2),
        },
        "h1_vs_h2": pairwise_difference_summary(h1, h2),
        "context": {
            "tensor": tensor_summary(context),
            "variation_across_samples": sample_variation_summary(context),
        },
        "query_input": {
            "tensor": tensor_summary(query_input),
            "variation_across_samples": sample_variation_summary(query_input),
        },
        "query_hidden": {
            "tensor": tensor_summary(query_hidden),
            "variation_across_samples": sample_variation_summary(query_hidden),
        },
        "h1_hidden": {
            "tensor": tensor_summary(h1_hidden),
            "variation_across_samples": sample_variation_summary(h1_hidden),
        },
        "h2_hidden": {
            "tensor": tensor_summary(h2_hidden),
            "variation_across_samples": sample_variation_summary(h2_hidden),
        },
        "context_hidden": {
            "tensor": tensor_summary(context_hidden),
            "variation_across_samples": sample_variation_summary(context_hidden),
        },
        "query_input_vs_query_hidden": pairwise_difference_summary(query_input, query_hidden),
        "h1_vs_h1_hidden": pairwise_difference_summary(sequence[:, :h1_len], h1_hidden),
        "h2_vs_h2_hidden": pairwise_difference_summary(
            sequence[:, h1_len:h1_len + h2_len],
            h2_hidden,
        ),
        "readout_features": {
            "tensor": tensor_summary(readout_features),
            "variation_across_samples": sample_variation_summary(readout_features),
        },
        "logits": tensor_summary(logits),
        "scores": tensor_summary(scores),
        "swap_test": {
            "logit_abs_diff_mean": float((logits.detach().to(dtype=torch.float32) - swapped_logits).abs().mean().item()),
            "logit_abs_diff_max": float((logits.detach().to(dtype=torch.float32) - swapped_logits).abs().max().item()),
            "score_abs_diff_mean": float((scores.detach().to(dtype=torch.float32) - swapped_scores).abs().mean().item()),
            "score_abs_diff_max": float((scores.detach().to(dtype=torch.float32) - swapped_scores).abs().max().item()),
            "query_input_abs_diff": pairwise_difference_summary(query_input, swapped_query_input),
            "query_hidden_abs_diff": pairwise_difference_summary(query_hidden, swapped_query_hidden),
            "h1_hidden_abs_diff": pairwise_difference_summary(h1_hidden, swapped_h1_hidden),
            "h2_hidden_abs_diff": pairwise_difference_summary(h2_hidden, swapped_h2_hidden),
            "context_hidden_abs_diff": pairwise_difference_summary(context_hidden, swapped_context_hidden),
            "readout_abs_diff": pairwise_difference_summary(readout_features, swapped_readout),
        },
    }
    return result


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)

    model = build_model_from_config(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    dataset = PairwiseLatentDatasetV2(
        resolve_repo_relative_path(args.manifest),
        curriculum_stages=None,
        max_samples=args.max_samples,
    )
    dataloader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=None,
        shuffle=False,
        pin_memory=False,
        persistent_workers=False,
    )
    batch = next(iter(dataloader))
    batch = move_batch(batch, device)

    diagnostics = run_diagnostics(model, batch)
    text = json.dumps(diagnostics, indent=2, ensure_ascii=False)
    print(text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"[done] diagnostics saved to {output_path}")


if __name__ == "__main__":
    main()
