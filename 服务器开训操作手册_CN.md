# Score Model V2 服务器开训操作手册

本文档面向当前服务器环境，目标是把 `score_model_v2` 从“代码已落地”推进到“可以先冒烟测试，再正式多卡开训”。

当前约定路径：

- 仓库路径：`/data/zhulanyun/lihailong/repo`
- 模型路径：`/data/zhulanyun/lihailong/models`
- 数据路径：`/data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f`

相关代码目录：

- `score_model_v2/`

相关配置文件：

- `score_model_v2/configs/v2_default.yaml`

相关启动脚本：

- `score_model_v2/train_v2_server.sh`

## 1. 开训前目标

建议不要一上来就直接全量多卡训练，而是按下面顺序推进：

1. 检查 Python 与核心依赖
2. 检查数据 manifest 和关键 tensor 文件是否存在
3. 做 Python 语法编译检查
4. 做单卡小样本 smoke test
5. 做多卡小样本 smoke test
6. 正式多卡训练
7. 中断后使用自动恢复继续训练

## 2. 进入仓库

先进入仓库根目录：

```bash
cd /data/zhulanyun/lihailong/repo
```

建议后面的命令都在这个目录下执行。

## 3. 检查 Python 和依赖

先确认当前训练环境的 Python 可以使用：

```bash
python --version
which python
```

再检查关键依赖是否已经安装：

```bash
python - <<'PY'
import importlib
mods = [
    "torch",
    "yaml",
    "transformers",
    "peft",
]
for name in mods:
    try:
        importlib.import_module(name)
        print(name, "OK")
    except Exception as exc:
        print(name, "MISSING", type(exc).__name__, exc)
PY
```

如果缺包，优先补下面这些：

```bash
pip install pyyaml peft
```

如果你当前环境还没装好训练相关依赖，也要确认这些包可用：

- `torch`
- `transformers`
- `flash_attn`

## 4. 检查 Qwen 模型路径

默认配置中使用的是：

- `/data/zhulanyun/lihailong/models/Qwen2.5-VL-3B-Instruct`

先确认它确实存在：

```bash
ls -lah /data/zhulanyun/lihailong/models
ls -lah /data/zhulanyun/lihailong/models/Qwen2.5-VL-3B-Instruct
```

如果目录名不一致，需要修改：

- `score_model_v2/configs/v2_default.yaml`

修改字段：

- `model.qwen_model_path`

## 5. 检查训练数据路径

默认配置里使用的是：

- `train_manifest: /data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/pairs_train.jsonl`
- `val_manifest: /data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/pairs_val.jsonl`

先确认 manifest 是否存在：

```bash
ls -lah /data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests
```

如果没有 `pairs_train.jsonl` 或 `pairs_val.jsonl`，需要：

1. 先完成数据构建
2. 或者修改 `score_model_v2/configs/v2_default.yaml` 中的 manifest 路径

## 6. 抽查 manifest 内容

先看几行，确认字段是否正常：

```bash
head -n 3 /data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/pairs_train.jsonl
head -n 3 /data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/pairs_val.jsonl
```

当前训练脚本最希望看到这些字段：

- `f1_path`
- `f2_path`
- `context_path` 或 `text_emb_path`
- `clip_fea_path` 或 `image_emb_path`
- `score_a`
- `score_b`
- `weight`
- `t_idx` 或 `step_idx` 或 `step` 或 `step_id`

其中 stage 分桶规则已经改成：

- 优先按步索引字段分桶
- 不按 `timestep` 分桶

## 7. 抽查 tensor 文件

可以随便挑一条样本里的路径，确认文件存在：

```bash
python - <<'PY'
import json
path = "/data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/pairs_train.jsonl"
with open(path, "r", encoding="utf-8") as f:
    item = json.loads(next(line for line in f if line.strip()))
for key in ["f1_path", "f2_path", "context_path", "clip_fea_path"]:
    print(key, item.get(key))
PY
```

然后手动检查这些路径是否存在：

```bash
ls -lah <上一步打印出来的f1_path>
ls -lah <上一步打印出来的f2_path>
ls -lah <上一步打印出来的context_path>
ls -lah <上一步打印出来的clip_fea_path>
```

## 8. Python 语法编译检查

在正式运行前，先做 `py_compile`：

```bash
python -m py_compile \
  score_model_v2/__init__.py \
  score_model_v2/losses.py \
  score_model_v2/train_v2.py \
  score_model_v2/models/__init__.py \
  score_model_v2/models/projectors.py \
  score_model_v2/models/embeddings.py \
  score_model_v2/models/qwen_comparator.py \
  score_model_v2/models/score_model_v2.py
```

如果这里报错，先修语法或导入问题，不要继续往下跑。

## 9. 单卡小样本冒烟测试

第一轮建议先单卡跑通，确认：

- 数据加载没问题
- Qwen 可以加载
- LoRA 逻辑没问题
- forward/backward 没问题
- checkpoint 能写出来

命令：

```bash
cd /data/zhulanyun/lihailong/repo

SMOKE_TEST=1 USE_TORCHRUN=0 bash score_model_v2/train_v2_server.sh
```

当前 smoke test 默认会限制：

- 训练样本数
- 验证样本数
- 每 epoch 的训练 step
- 每次评估 step

这些限制定义在：

- `score_model_v2/configs/v2_default.yaml`

如果你想手动覆盖，也可以不用脚本，直接运行：

```bash
python -m score_model_v2.train_v2 \
  --config score_model_v2/configs/v2_default.yaml \
  --smoke_test \
  --max_train_samples 8 \
  --max_val_samples 4 \
  --max_train_steps_per_epoch 2 \
  --max_eval_steps 1
```

## 10. 多卡小样本冒烟测试

单卡通过后，再测多卡链路，确认：

- `torchrun` 能正常拉起
- DDP sampler 正常
- 多卡 loss 聚合正常
- 主进程写 checkpoint 正常
- 自动恢复状态文件正常

示例，两卡：

```bash
cd /data/zhulanyun/lihailong/repo

SMOKE_TEST=1 USE_TORCHRUN=1 NPROC_PER_NODE=2 bash score_model_v2/train_v2_server.sh
```

如果你有 4 卡，也可以直接这样测：

```bash
SMOKE_TEST=1 USE_TORCHRUN=1 NPROC_PER_NODE=4 bash score_model_v2/train_v2_server.sh
```

## 11. 正式多卡训练

当单卡和多卡 smoke test 都通过后，再开始正式训练。

例如 4 卡训练：

```bash
cd /data/zhulanyun/lihailong/repo

USE_TORCHRUN=1 NPROC_PER_NODE=4 bash score_model_v2/train_v2_server.sh
```

例如 8 卡训练：

```bash
USE_TORCHRUN=1 NPROC_PER_NODE=8 bash score_model_v2/train_v2_server.sh
```

如果你希望指定输出目录：

```bash
OUTPUT_DIR=/data/zhulanyun/lihailong/repo/outputs/score_model_v2_exp1 \
USE_TORCHRUN=1 NPROC_PER_NODE=4 \
bash score_model_v2/train_v2_server.sh
```

## 12. 断点恢复

当前训练框架已经支持断点恢复。

### 12.1 自动恢复

默认行为是：

- 如果 `output_dir/latest_checkpoint.pt` 存在
- 会自动从它恢复

所以很多时候你只需要重新执行同一条训练命令即可。

例如：

```bash
USE_TORCHRUN=1 NPROC_PER_NODE=4 bash score_model_v2/train_v2_server.sh
```

### 12.2 显式恢复

如果你想从某个指定 checkpoint 恢复：

```bash
RESUME_FROM=/data/zhulanyun/lihailong/repo/outputs/score_model_v2/latest_checkpoint.pt \
USE_TORCHRUN=1 NPROC_PER_NODE=4 \
bash score_model_v2/train_v2_server.sh
```

### 12.3 禁用自动恢复

如果你希望强制从头开始跑：

```bash
NO_AUTO_RESUME=1 USE_TORCHRUN=1 NPROC_PER_NODE=4 bash score_model_v2/train_v2_server.sh
```

## 13. 常见排查顺序

如果训练报错，建议按下面顺序排查。

### 13.1 找不到 manifest

优先检查：

- `score_model_v2/configs/v2_default.yaml`
- 数据目录是否真的生成了 `pairs_train.jsonl` / `pairs_val.jsonl`

### 13.2 找不到 tensor 文件

优先检查 manifest 中的：

- `f1_path`
- `f2_path`
- `context_path`
- `clip_fea_path`

是否都存在。

### 13.3 Qwen 加载报错

优先检查：

- `model.qwen_model_path` 是否写对
- 当前 `transformers` 版本是否支持 `Qwen2_5_VLForConditionalGeneration`
- 是否需要退到 `AutoModelForImageTextToText`

### 13.4 LoRA 相关报错

优先检查：

- `peft` 是否安装
- `peft` 版本是否能包装当前 Qwen 层结构

当前代码已经做了兜底：

- 如果某些 `peft` 版本不能包装当前 comparator，训练会回退到直接训练 comparator 参数，而不是直接崩溃

### 13.5 多卡报错

优先检查：

- `NPROC_PER_NODE` 是否超过可用 GPU 数
- `CUDA_VISIBLE_DEVICES` 是否设置正确
- `torchrun` 是否可用

### 13.6 FlashAttention / bf16 报错

优先检查：

- CUDA 版本
- `flash_attn` 安装是否匹配当前环境
- GPU 是否支持 bf16

必要时可以先把问题定位到：

- 单卡 smoke test
- 或改小 batch

## 14. 推荐的实际执行顺序

如果要稳妥推进，建议按这条顺序做：

1. `cd /data/zhulanyun/lihailong/repo`
2. 检查 `python --version`
3. 检查 `torch / transformers / peft / yaml`
4. 检查 Qwen 模型路径
5. 检查 manifest 路径
6. `python -m py_compile ...`
7. 单卡 `SMOKE_TEST=1`
8. 多卡 `SMOKE_TEST=1`
9. 正式多卡训练
10. 中断后直接用同一条命令恢复

## 15. 建议保留的命令模板

### 单卡 smoke test

```bash
cd /data/zhulanyun/lihailong/repo
SMOKE_TEST=1 USE_TORCHRUN=0 bash score_model_v2/train_v2_server.sh
```

### 两卡 smoke test

```bash
cd /data/zhulanyun/lihailong/repo
SMOKE_TEST=1 USE_TORCHRUN=1 NPROC_PER_NODE=2 bash score_model_v2/train_v2_server.sh
```

### 四卡正式训练

```bash
cd /data/zhulanyun/lihailong/repo
USE_TORCHRUN=1 NPROC_PER_NODE=4 bash score_model_v2/train_v2_server.sh
```

### 四卡显式恢复

```bash
cd /mnt/nvme3/zhulanyun/lihailong/repo
RESUME_FROM=/data/zhulanyun/lihailong/repo/outputs/score_model_v2/latest_checkpoint.pt \
USE_TORCHRUN=1 NPROC_PER_NODE=4 \
bash score_model_v2/train_v2_server.sh
```

### 四卡从头重新开始

```bash
cd /data/zhulanyun/lihailong/repo
NO_AUTO_RESUME=1 USE_TORCHRUN=1 NPROC_PER_NODE=4 bash score_model_v2/train_v2_server.sh
```

## 16. 一句话总结

当前最推荐的做法不是直接全量开训，而是：

- 先单卡 smoke test
- 再多卡 smoke test
- 报错尽早修
- 最后正式多卡训练

这样最省时间，也最不容易把大训练资源浪费在明显的环境问题和数据路径问题上。
