# Score Model V2 Baseline 操作手册

本文档整理当前 `score_model_v2` baseline 的结构、数据抽样方式、训练命令、可视化命令，以及结果查看方式。

## 1. 目的

baseline 的目标不是替代正式模型，而是做一个更简单、更稳定的对照实验，用来回答：

- 数据本身是不是可分
- `VideoProjector` 提取的视频特征有没有信号
- 正式模型效果差，是不是因为 comparator / query readout 设计出了问题

## 2. 当前 baseline 支持的两种结构

当前 baseline 训练脚本支持两种结构，通过 `MODEL_VARIANT` 切换。

### 2.1 结构 A

```text
f1 -> VideoProjector -> mean pool -> h1_pool
f2 -> VideoProjector -> mean pool -> h2_pool

feature = [h1_pool ; h2_pool ; h1_pool - h2_pool ; h1_pool * h2_pool]
feature -> MLP head -> logit -> sigmoid
```

解释：

- `VideoProjector` 负责把 latent 压缩成视频特征
- `mean pool` 把 token 序列压成固定长度向量
- 再构造四类比较特征：
  - `h1_pool`
  - `h2_pool`
  - `h1_pool - h2_pool`
  - `h1_pool * h2_pool`
- 最后送进一个小 MLP 输出 `P(f1 > f2)`

特点：

- 信息最完整
- 表达能力更强
- 更适合作为正式 baseline

对应参数：

- `MODEL_VARIANT=a`

### 2.2 结构 B

```text
f1 -> VideoProjector -> mean pool -> h1_pool
f2 -> VideoProjector -> mean pool -> h2_pool

(h1_pool - h2_pool) -> LayerNorm -> Linear -> logit -> sigmoid
```

解释：

- 只使用最直接的差值特征 `h1_pool - h2_pool`
- 用一个简单线性头输出结果

特点：

- 结构最简单
- 更接近最小 sanity-check probe
- 很适合验证“仅靠差值特征是否已经可分”

对应参数：

- `MODEL_VARIANT=b`

## 3. baseline 使用的数据

当前 baseline 不再从一个训练 manifest 内部随机切 train/val，而是：

- 从全量 `pairs_train.jsonl` 里抽样 train
- 从全量 `pairs_val.jsonl` 里抽样 val

路径是：

- `/data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/pairs_train.jsonl`
- `/data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/pairs_val.jsonl`

## 4. 抽样方式

抽样脚本：

- `score_model_v2/sample_baseline_manifests.py`

当前抽样不是按单条记录随机抽，而是按**对称 pair 成组抽样**。

也就是说，如果原始数据里有：

- `(f1, f2, y)`
- `(f2, f1, 1-y)`

那么抽样时会尽量保证：

- 两条一起保留
- 不会只保留其中一条

这样做的好处是：

- 保持 pairwise 对称增强设计
- 避免顺序偏置
- 让 baseline 和正式模型的对比更干净

## 5. 抽样命令

先从全量数据里构建一个 5k baseline 数据集：

```bash
cd /data/zhulanyun/lihailong/repo
source /data/zhulanyun/lihailong/venv_unified310/bin/activate

python -m score_model_v2.sample_baseline_manifests \
  --train_manifest /data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/pairs_train.jsonl \
  --val_manifest /data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/pairs_val.jsonl \
  --output_dir /data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/baseline_5k_split_v2 \
  --train_samples 4500 \
  --val_samples 500 \
  --seed 42
```

抽样完成后会生成：

- `pairs_train.baseline5k.jsonl`
- `pairs_val.baseline5k.jsonl`
- `sampling_summary.json`

## 6. 训练脚本

当前 baseline 训练主脚本：

- `score_model_v2/baseline_train_ddp.py`

服务器启动脚本：

- `score_model_v2/baseline_train_server.sh`

支持能力：

- 独立 train/val manifest
- 8 卡 DDP
- checkpoint / resume
- `metrics.json`
- `predictions/*.jsonl`
- 和正式模型同口径的验证指标

## 7. 结构 A 训练命令

结构 A 是当前默认 baseline。

```bash
cd /data/zhulanyun/lihailong/repo
source /data/zhulanyun/lihailong/venv_unified310/bin/activate

mkdir -p /data/zhulanyun/lihailong/outputs/score_model_v2_baseline_5k_ddp_a_v1

MASTER_PORT=29542 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TRAIN_MANIFEST=/data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/baseline_5k_split_v2/pairs_train.baseline5k.jsonl \
VAL_MANIFEST=/data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/baseline_5k_split_v2/pairs_val.baseline5k.jsonl \
OUTPUT_DIR=/data/zhulanyun/lihailong/outputs/score_model_v2_baseline_5k_ddp_a_v1 \
USE_TORCHRUN=1 \
NPROC_PER_NODE=8 \
NO_AUTO_RESUME=1 \
EPOCHS=20 \
LR=1e-4 \
MODEL_VARIANT=a \
bash score_model_v2/baseline_train_server.sh
```

### 带日志版本

```bash
cd /data/zhulanyun/lihailong/repo
source /data/zhulanyun/lihailong/venv_unified310/bin/activate

mkdir -p /data/zhulanyun/lihailong/outputs/score_model_v2_baseline_5k_ddp_a_v1

MASTER_PORT=29542 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TRAIN_MANIFEST=/data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/baseline_5k_split_v2/pairs_train.baseline5k.jsonl \
VAL_MANIFEST=/data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/baseline_5k_split_v2/pairs_val.baseline5k.jsonl \
OUTPUT_DIR=/data/zhulanyun/lihailong/outputs/score_model_v2_baseline_5k_ddp_a_v1 \
USE_TORCHRUN=1 \
NPROC_PER_NODE=8 \
NO_AUTO_RESUME=1 \
EPOCHS=20 \
LR=1e-4 \
MODEL_VARIANT=a \
bash score_model_v2/baseline_train_server.sh 2>&1 | tee /data/zhulanyun/lihailong/outputs/score_model_v2_baseline_5k_ddp_a_v1/train.log
```

## 8. 结构 B 训练命令

结构 B 是更简单的 baseline。

```bash
cd /data/zhulanyun/lihailong/repo
source /data/zhulanyun/lihailong/venv_unified310/bin/activate

mkdir -p /data/zhulanyun/lihailong/outputs/score_model_v2_baseline_5k_ddp_b_v1

MASTER_PORT=29543 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TRAIN_MANIFEST=/data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/baseline_5k_split_v2/pairs_train.baseline5k.jsonl \
VAL_MANIFEST=/data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/baseline_5k_split_v2/pairs_val.baseline5k.jsonl \
OUTPUT_DIR=/data/zhulanyun/lihailong/outputs/score_model_v2_baseline_5k_ddp_b_v1 \
USE_TORCHRUN=1 \
NPROC_PER_NODE=8 \
NO_AUTO_RESUME=1 \
EPOCHS=20 \
LR=1e-4 \
MODEL_VARIANT=b \
bash score_model_v2/baseline_train_server.sh
```

### 带日志版本

```bash
cd /data/zhulanyun/lihailong/repo
source /data/zhulanyun/lihailong/venv_unified310/bin/activate

mkdir -p /data/zhulanyun/lihailong/outputs/score_model_v2_baseline_5k_ddp_b_v1

MASTER_PORT=29543 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TRAIN_MANIFEST=/data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/baseline_5k_split_v2/pairs_train.baseline5k.jsonl \
VAL_MANIFEST=/data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/baseline_5k_split_v2/pairs_val.baseline5k.jsonl \
OUTPUT_DIR=/data/zhulanyun/lihailong/outputs/score_model_v2_baseline_5k_ddp_b_v1 \
USE_TORCHRUN=1 \
NPROC_PER_NODE=8 \
NO_AUTO_RESUME=1 \
EPOCHS=20 \
LR=1e-4 \
MODEL_VARIANT=b \
bash score_model_v2/baseline_train_server.sh 2>&1 | tee /data/zhulanyun/lihailong/outputs/score_model_v2_baseline_5k_ddp_b_v1/train.log
```

## 9. 训练输出内容

baseline 训练完成后，输出目录里通常会有：

- `metrics.json`
- `latest_checkpoint.pt`
- `train.log`
- `predictions/*.jsonl`
- 按 epoch 保存的 checkpoint 目录（如果到达保存间隔）

其中：

- `metrics.json` 是 epoch 级汇总指标
- `predictions/*.jsonl` 是样本级验证结果

## 10. baseline 可视化命令

baseline 的可视化和正式模型共用同一个绘图脚本：

- `score_model_v2/plot_training_metrics.py`

### 结构 A 可视化

```bash
cd /data/zhulanyun/lihailong/repo
source /data/zhulanyun/lihailong/venv_unified310/bin/activate

python -m score_model_v2.plot_training_metrics \
  --run_dir /data/zhulanyun/lihailong/outputs/score_model_v2_baseline_5k_ddp_a_v1
```

### 结构 B 可视化

```bash
python -m score_model_v2.plot_training_metrics \
  --run_dir /data/zhulanyun/lihailong/outputs/score_model_v2_baseline_5k_ddp_b_v1
```

运行完成后，会在对应输出目录里生成：

- `training_metrics_v2.png`
- `training_metrics_margin_buckets.png`
- `metrics_summary.json`

## 11. baseline 最值得关注的指标

优先看：

1. `val_pairwise_accuracy`
2. `val_weighted_pairwise_accuracy`
3. `val_auc`
4. `val_accuracy_by_step`
5. `val_accuracy_by_margin_bucket`

辅助再看：

- `train_loss`
- `val_loss`
- `val_brier_score`

## 12. 查看样本级预测

如果你想看 baseline 有没有塌成常数输出，或者想分析某一轮的样本级行为，可以用：

```bash
python -m score_model_v2.analyze_predictions \
  --predictions /path/to/output_dir/predictions/某个_val_predictions.jsonl
```

这个脚本会输出：

- 总体 accuracy
- `pred_prob` 分布
- `pred_logit` 分布
- `accuracy_by_step`
- `accuracy_by_stage`
- `accuracy_by_margin_bucket`

## 13. 推荐使用顺序

建议按下面顺序走：

1. 从全量 `pairs_train.jsonl / pairs_val.jsonl` 抽样得到 baseline 5k 数据集
2. 先跑结构 A
3. 再跑结构 B
4. 分别画图
5. 对比 A/B 的：
   - `val_pairwise_accuracy`
   - `val_auc`
   - `val_accuracy_by_step`
6. 必要时再结合 `predictions/*.jsonl` 做样本级分析

## 14. 一句话总结

当前 baseline 手册的核心流程是：

- 先从全量 train/val 数据中按对称 pair 成组抽样
- 再分别训练结构 A 和结构 B 两个 baseline
- 最后用统一的 `metrics.json + plot_training_metrics.py` 做对比分析
