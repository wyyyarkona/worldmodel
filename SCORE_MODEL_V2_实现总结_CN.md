# Score Model V2 实现总结

本文档总结当前 `score_model_v2/` 中已经实现的评分模型与训练框架，方便后续继续开发、排查问题和正式开训。

## 1. 目标

当前实现的目标是构建一个新的 pairwise 评分模型，用来比较两个中间去噪 step 的 Wan latent，输出：

- `P(f1 > f2)`

也就是“候选 1 比候选 2 更好”的概率。

整个方案围绕以下主线展开：

1. 把 Wan latent 压缩成适合比较的高维 token 序列
2. 把文本条件和图像条件也投影到同一隐藏维度
3. 用 Qwen2.5-VL 的前 6 层 language model block 作为 comparator
4. 通过 LoRA 微调 comparator，而不是整体重训 Qwen
5. 使用 pairwise soft label loss 做训练

## 2. 当前目录结构

当前实现位于：

- [score_model_v2](/d:/worldmodel/repo/score_model_v2)

主要文件：

- [models/projectors.py](/d:/worldmodel/repo/score_model_v2/models/projectors.py:1)
  负责 latent 压缩、文本/图像投影
- [models/embeddings.py](/d:/worldmodel/repo/score_model_v2/models/embeddings.py:1)
  负责 3D 位置编码、segment/stage/query embedding
- [models/qwen_comparator.py](/d:/worldmodel/repo/score_model_v2/models/qwen_comparator.py:1)
  负责加载 Qwen2.5-VL，并截取前 6 层作为 comparator
- [models/score_model_v2.py](/d:/worldmodel/repo/score_model_v2/models/score_model_v2.py:1)
  主模型，负责把所有模块拼接起来
- [losses.py](/d:/worldmodel/repo/score_model_v2/losses.py:1)
  pairwise loss 与 warmup 对齐损失
- [train_v2.py](/d:/worldmodel/repo/score_model_v2/train_v2.py:1)
  训练入口，支持多卡、smoke test、断点恢复
- [configs/v2_default.yaml](/d:/worldmodel/repo/score_model_v2/configs/v2_default.yaml:1)
  默认训练配置
- [train_v2_server.sh](/d:/worldmodel/repo/score_model_v2/train_v2_server.sh:1)
  Linux 服务器启动脚本

## 3. 模型结构

### 3.1 视频分支

输入的两个候选 latent 是：

- `f1`
- `f2`

形状预期为：

- `[B, 16, 21, 60, 104]`

视频分支在 [projectors.py](/d:/worldmodel/repo/score_model_v2/models/projectors.py:1) 中实现，分 3 步：

1. `Conv3d(16 -> 1536, kernel=(1,2,2), stride=(1,2,2))`
   把 latent 做初始 patch embedding
2. `SpatialProjector`
   把相邻 `2x2` 空间位置合并
3. `TemporalProjector`
   把相邻 2 帧合并

这样得到压缩后的 3D token 网格，大致对应：

- 时间维：`11`
- 高度：`15`
- 宽度：`26`
- hidden dim：`2048`

随后在 [embeddings.py](/d:/worldmodel/repo/score_model_v2/models/embeddings.py:1) 中加入：

- 时间位置 embedding
- 高度位置 embedding
- 宽度位置 embedding

最后 flatten 成序列。

### 3.2 条件分支

条件分支接收两类输入：

- 文本 embedding
- 图像 embedding

当前兼容已有数据管线中的字段：

- `context_path` 视为文本 embedding
- `clip_fea_path` 视为图像 embedding

其中：

- 文本特征维度默认是 `4096`
- 图像特征维度默认是 `1280`

在 [ContextProjector](/d:/worldmodel/repo/score_model_v2/models/projectors.py:113) 中：

- 文本最多保留 512 token
- 图像最多保留 64 token
- 然后都投影到 `2048` 维

最后拼成：

- `[image_tokens ; text_tokens]`

## 4. Embedding 设计

当前实现了 4 类辅助 embedding：

### 4.1 3D 位置编码

位置编码在压缩后的 latent 上添加，用来弥补 projector 压缩后原始时空位置信息的损失。

### 4.2 Segment embedding

用于区分不同 token 来源，目前包含：

- `query`
- `h1`
- `h2`
- `context`
- `stage`

这是 comparator 能否正确理解序列结构的关键部分。

### 4.3 Stage embedding

阶段 token 表示当前样本属于：

- `early`
- `middle`
- `late`

### 4.4 Query embedding

使用 4 个可学习 query token，最终只取 query token 的输出做打分。

## 5. Qwen Comparator

Qwen comparator 在 [qwen_comparator.py](/d:/worldmodel/repo/score_model_v2/models/qwen_comparator.py:1) 中实现。

当前做法：

1. 加载 Qwen2.5-VL 模型
2. 找到 language model 主体
3. 只截取前 6 层 transformer block
4. 尝试加 LoRA
5. 改掉 causal mask，使 attention 变成双向

双向 attention 是必要的，因为这里不是生成任务，而是一个比较任务。

### 5.1 LoRA

LoRA 目标模块包括：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

当前实现已经改成更严格的 LoRA 行为：

- LoRA 优先挂到 Qwen 主模型上，而不是只挂到局部层列表
- 如果 `peft` 不可用、LoRA 挂载失败，或者最终没有生成任何 `lora_` 参数
- 训练会直接报错，而不是再退化成全参训练

### 5.2 Qwen 兼容性处理

由于不同 transformers 版本下，Qwen2.5-VL 的模块路径和 layer forward 参数可能不同，当前实现中已经加入：

- 属性探测
- 多种 forward 签名兜底
- causal mask 覆盖

这部分属于当前实现里最“易变”的部分。

## 6. 主模型前向流程

主模型在 [score_model_v2.py](/d:/worldmodel/repo/score_model_v2/models/score_model_v2.py:1) 中实现。

前向流程如下：

1. `f1 -> h1`
2. `f2 -> h2`
3. `text_emb + image_emb -> context`
4. 构造序列：
   `[query ; h1 ; h2 ; context ; stage]`
5. 送入 Qwen comparator
6. 取 query token 输出
7. mean pool
8. `LayerNorm + Linear`
9. `sigmoid`

输出：

- `score`
- `logit`

在 warmup 阶段还会返回：

- `proj_mean`
- `proj_std`

## 7. Loss 设计

### 7.1 PairwiseScoreLoss

在 [losses.py](/d:/worldmodel/repo/score_model_v2/losses.py:1) 中实现。

逻辑：

1. `delta = score_a - score_b`
2. `target = sigmoid(delta / tau)`
3. `|delta| < margin` 的样本直接忽略
4. 样本权重随 `|delta|` 增大而增加
5. 使用 `binary_cross_entropy_with_logits`

这是一个 soft label 的 pairwise 排序损失。

### 7.2 WarmupAlignmentLoss

只在 warmup 阶段使用。

它约束 projector 输出的统计量：

- `proj_mean`
- `proj_std`

与目标分布接近。当前如果没有真实的目标统计量，可以用近似占位目标。

## 8. 训练框架

训练入口是：

- [train_v2.py](/d:/worldmodel/repo/score_model_v2/train_v2.py:1)

### 8.1 三阶段训练

当前训练分 3 段：

1. `warmup`
   训练新加的 projector / embedding / score head
2. `lora`
   训练 LoRA 和新模块
3. `curriculum`
   进一步按 stage 做课程学习

### 8.2 Curriculum 规则

当前课程学习是按 stage 过滤 manifest 实现的，不是在 loss 内部实现。

### 8.3 Stage 分桶规则

这是最近刚明确修正过的部分：

- stage 分桶按“步数 / 步索引”来分
- 不按 `timestep` 数值分

优先顺序：

1. 如果 manifest 已经给了显式 `stage_id`
2. 否则看：
   - `t_idx`
   - `step_idx`
   - `step`
   - `step_id`
3. 根据数据里出现的最小/最大 step index 做三等分

即：

- 小 step index -> `early`
- 中 step index -> `middle`
- 大 step index -> `late`

## 9. 多卡训练

当前训练入口已支持 DDP 多卡训练。

核心能力包括：

- `torchrun` 启动
- `DistributedSampler`
- 只在主进程写 checkpoint / metrics
- train/eval loss 跨卡聚合
- 多卡训练时建议显式加上 `--ddp_find_unused_parameters`

启动脚本：

- [train_v2_server.sh](/d:/worldmodel/repo/score_model_v2/train_v2_server.sh:1)

示例：

```bash
USE_TORCHRUN=1 NPROC_PER_NODE=4 bash score_model_v2/train_v2_server.sh
```

## 10. 少量样本冒烟测试

为了尽快发现模型、数据、Qwen、LoRA、DDP 里的报错，当前训练脚本已经支持 smoke test：

- `--smoke_test`

同时，第三阶段 `curriculum` 现在支持命令行关闭：

- `--disable_curriculum`

训练与验证过程也已经补充了 `tqdm` 进度条，用于显示：

- 当前 step / 总 step
- 当前 step loss
- 当前 epoch 平均 loss

默认配置里也写了：

- 训练样本数上限
- 验证样本数上限
- 每个 epoch 的最大训练 step
- 最大验证 step

这样可以在数据集完全准备好之前，先拿少量 pair 跑通全流程。

示例：

```bash
SMOKE_TEST=1 USE_TORCHRUN=0 bash score_model_v2/train_v2_server.sh
```

或多卡链路测试：

```bash
SMOKE_TEST=1 USE_TORCHRUN=1 NPROC_PER_NODE=2 bash score_model_v2/train_v2_server.sh
```

## 11. 断点恢复

当前训练已经支持断点恢复。

### 11.1 自动恢复

默认会尝试从：

- `output_dir/latest_checkpoint.pt`
- 分阶段 checkpoint 目录名会带 `phase`
- 例如 `warmup_phase1_epoch_001`、`lora_phase1_epoch_002`、`curriculum_phase3_epoch_001`

自动恢复。

恢复内容包括：

- 模型参数
- 优化器状态
- 当前 stage
- 当前 phase
- 当前 epoch
- 已记录 metrics

### 11.2 显式恢复

也可以手动指定：

```bash
--resume_from /path/to/checkpoint.pt
```

### 11.3 禁止自动恢复

如果你希望强制重新开始，可以使用：

```bash
--no_auto_resume
```

### 11.4 恢复粒度

恢复不是简单地“加载模型权重”，而是会在 staged training 循环中跳过已经完成的：

- stage
- phase
- epoch

因此更适合长时间正式训练。

## 12. 当前实现与计划的一致性

已经实现的关键部分：

- 新评分模型独立目录
- latent projector
- text/image projector
- 3D 位置编码
- segment/stage/query embedding
- Qwen comparator
- LoRA 接口
- pairwise soft label loss
- warmup alignment loss
- 三阶段训练
- curriculum filtering
- 多卡 DDP
- smoke test
- 断点恢复

尚需在真实服务器环境重点验证的部分：

- Qwen2.5-VL 在目标 transformers 版本下的真实模块结构
- `peft` 是否能稳定包装当前 comparator 层结构
- FlashAttention / bf16 / Qwen / CUDA 组合是否兼容
- 实际 manifest 中 `context_path` 与 `clip_fea_path` 的张量维度是否完全符合当前假设
- 小样本 smoke test 是否能完整跑通 forward/backward/checkpoint

## 13. 推荐的实际使用顺序

建议按以下顺序推进：

1. 在服务器环境安装依赖
2. 先跑 `py_compile`
3. 再跑单卡 `SMOKE_TEST=1`
4. 再跑多卡 `SMOKE_TEST=1`
5. 再跑 `train4.5k / val0.5k` 这类中等规模预跑
6. 修完所有报错后，再切正式多卡训练
6. 训练中断时直接重启同一条命令，依靠自动恢复继续跑

## 14. 一句话总结

当前 `score_model_v2` 已经具备一个“可继续迭代的完整训练框架”：

- 模型结构已经搭好
- 数据读取已经接入
- 按步数分桶已经修正
- 多卡训练已经接入
- 小样本全流程验证已经支持
- 断点恢复已经支持

接下来最关键的工作，不再是补基础训练框架，而是在服务器上做真实 smoke test，把 Qwen/LoRA/数据格式的实际报错尽早打掉。
