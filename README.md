# Score Model V2

`score_model_v2/` is a fresh Qwen2.5-VL-based pairwise scorer implementation
built from `score_model_v2_plan(1).md`.

Core design:

- shared Wan-latent video compressor
- learned 3D position embeddings after compression
- projected text/image condition tokens in the same 2048-d space
- segment embeddings for `query / h1 / h2 / context / stage`
- stage embedding for `early / middle / late`
- first 6 Qwen2.5-VL language-model blocks as a bidirectional comparator
- LoRA on Qwen attention + FFN projections

Files:

- `models/projectors.py`: video/text/image projection blocks
- `models/embeddings.py`: position, segment, stage, query embeddings
- `models/qwen_comparator.py`: Qwen LM loader + first-6-layer comparator wrapper
- `models/score_model_v2.py`: end-to-end scorer
- `losses.py`: weighted soft-label BCE + warmup distribution-alignment loss
- `train_v2.py`: three-stage training entrypoint
- `configs/v2_default.yaml`: server-oriented default config
- `train_v2_server.sh`: Linux launcher
- `当前模型结构说明_CN.md`: current Chinese architecture summary aligned with the latest code

Notes:

- This implementation assumes pairwise manifests already provide latent paths
  plus precomputed text/image condition tensors.
- Repo configs can now use repo-relative paths such as `../models/...` and
  `../data/...`; `train_v2.py` resolves them against the repository root.
- `context_path` is treated as text embeddings and `clip_fea_path` as image
  embeddings for backward compatibility with the current data pipeline.
- Pair manifests may provide either `f1_path` / `f2_path` or the current
  `z_t_a_path` / `z_t_b_path` field names.
- Stage bucketing in training is based on denoising step index fields such as
  `t_idx` / `step_idx` / `step` / `step_id`, not on raw timestep values.
- Training requires `peft` for LoRA and `PyYAML` for YAML config parsing.
- Qwen internals are handled with attribute probing because exact module paths
  can differ across transformers versions.
- `train_v2.py --smoke_test` limits the run to a tiny number of samples/steps so
  you can validate the full pipeline before launching the full dataset job.
- Training automatically resumes from `output_dir/latest_checkpoint.pt` unless
  `--no_auto_resume` is set. You can also pass `--resume_from /path/to/checkpoint.pt`.

Quick usage:

- Single-process smoke test:
  `SMOKE_TEST=1 USE_TORCHRUN=0 bash score_model_v2/train_v2_server.sh`
- Multi-GPU smoke test:
  `SMOKE_TEST=1 USE_TORCHRUN=1 NPROC_PER_NODE=2 bash score_model_v2/train_v2_server.sh`
- Full multi-GPU training:
  `USE_TORCHRUN=1 NPROC_PER_NODE=4 bash score_model_v2/train_v2_server.sh`
- Resume from the latest checkpoint automatically:
  rerun the same command in the same `output_dir`
- Resume from an explicit checkpoint:
  `RESUME_FROM=/path/to/checkpoint.pt USE_TORCHRUN=1 NPROC_PER_NODE=4 bash score_model_v2/train_v2_server.sh`
- Plot training metrics from `metrics.json`:
  `python -m score_model_v2.plot_training_metrics --run_dir /path/to/output_dir`
  This now plots loss curves, ranking metrics, probability metrics, stage/step
  accuracy curves, and a separate margin-bucket accuracy figure.

Important runtime expectations:

- `f1_path` / `f2_path` or `z_t_a_path` / `z_t_b_path` should point to latent
  tensors shaped like `[16, 21, 60, 104]` or batched into
  `[B, 16, 21, 60, 104]` after collation.
- `context_path` should contain text embeddings with feature dim `4096`.
- `clip_fea_path` should contain image embeddings with feature dim `1280`.
- Qwen model loading, FlashAttention, PEFT, and bf16 behavior must be validated
  in the target Linux training environment.
- LoRA now attaches only to the front-6-layer comparator backbone that actually
  participates in forward, not to the entire loaded Qwen model.
