# Score Model V2 当前模型结构说明

本文档描述当前 `score_model_v2` 代码中实际实现的模型结构，以当前仓库代码为准，而不是以早期方案草稿为准。

## 1. 总体目标

模型输入：

- `f1`：第一个候选中间 step 的 Wan latent
- `f2`：第二个候选中间 step 的 Wan latent
- `text_emb`：文本条件 embedding
- `image_emb`：图像条件 embedding
- `stage_id`：阶段标签，取值为 `early / middle / late` 对应的 id

模型输出：

- `score = P(f1 > f2)`
- `logit`

也就是说，这个模型做的是一个 pairwise ranking / pairwise scoring 任务，用来判断在相同条件下，`f1` 是否优于 `f2`。

## 2. 整体前向结构

当前前向主链路可以概括为：

```text
f1 -> VideoProjector -> 3D Pos Embed -> h1
f2 -> VideoProjector -> 3D Pos Embed -> h2

text_emb + image_emb -> ContextProjector -> context
stage_id -> StageEmbedding -> stage token
learnable query -> QueryEmbedding -> query tokens

[query ; h1 ; h2 ; context ; stage]
    -> SegmentEmbedding
    -> QwenComparator
    -> 取 query token 输出
    -> mean pool
    -> LayerNorm
    -> Linear
    -> sigmoid
    -> score
```

在代码里，对应主模块是 [models/score_model_v2.py](/d:/worldmodel/repo/score_model_v2/models/score_model_v2.py:1)。

## 3. 视频分支

视频分支负责把 Wan latent 压缩成适合 comparator 处理的 token 序列。

当前输入 latent 预期形状：

- 单样本：`[16, 21, 60, 104]`
- batch 后：`[B, 16, 21, 60, 104]`

视频分支包含三步：

1. `patch embedding`
2. `spatial projector`
3. `temporal projector`

对应模块在：

- [models/projectors.py](/d:/worldmodel/repo/score_model_v2/models/projectors.py:1)

压缩后得到的 3D token 网格大致为：

- 时间：`11`
- 高度：`15`
- 宽度：`26`
- hidden dim：`2048`

之后再通过 3D 位置编码加上时空位置信息，并 flatten 成序列。

## 4. 条件分支

条件分支把文本和图像条件投影到和 comparator 一致的 hidden space。

当前默认输入维度是：

- 文本 embedding：`4096`
- 图像 embedding：`1280`

当前实现会：

- 截断文本 token 到最多 `512`
- 截断图像 token 到最多 `64`
- 分别投影到 `2048`
- 按 `[image_tokens ; text_tokens]` 顺序拼接

对应模块在：

- [models/projectors.py](/d:/worldmodel/repo/score_model_v2/models/projectors.py:1)

## 5. Embedding 结构

当前模型一共显式用了四类 embedding。

### 5.1 3D Position Embedding

作用：

- 给压缩后的 latent token 补充时间、高度、宽度三个维度的位置编码

### 5.2 Segment Embedding

作用：

- 区分不同来源的 token

当前 token 类型包括：

- `query`
- `h1`
- `h2`
- `context`
- `stage`

### 5.3 Stage Embedding

作用：

- 表示当前样本属于哪个阶段 bucket

当前阶段包括：

- `early`
- `middle`
- `late`

### 5.4 Query Embedding

作用：

- 提供可学习的 query token，最终打分时只读取 query token 的输出

当前默认 query token 数量：

- `4`

## 6. Qwen Comparator

Qwen comparator 是当前模型里最核心的 backbone 部分，对应文件：

- [models/qwen_comparator.py](/d:/worldmodel/repo/score_model_v2/models/qwen_comparator.py:1)

当前实现不是直接手写一套 transformer，而是：

1. 通过 `transformers` 加载完整的 Qwen2.5-VL checkpoint
2. 定位到其中的 language model 主体
3. 截取前 `6` 层 transformer block
4. 取出对应的 `final_norm` 和 `rotary_emb`
5. 组装成一个独立的 `FrontQwenBackbone`
6. 用这个前 6 层 backbone 作为 comparator

也就是说，当前真正参与前向的是：

- Qwen2.5-VL language model 的前 `6` 层

而不是整个 Qwen 模型。

## 7. 当前 LoRA 结构

这是近期更新过的重点。

当前 LoRA 的挂载方式是：

- **只挂在前 6 层 comparator backbone 上**
- **不再挂在完整 Qwen 主模型上**

这意味着：

- LoRA 的 trainable 参数和真实前向路径一致
- optimizer 不会再混入大量根本没参与前向的 LoRA 参数
- 多卡 DDP 下 unused parameter 风险更低

LoRA target modules 包括：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

所以当前第二阶段训练时，训练的是：

- 新增模块
- comparator 前 6 层上的 LoRA 参数

不会训练的部分是：

- Qwen backbone 的原始主权重

## 8. Attention 方式

Qwen 原始 language model 是 decoder-only 结构，自带 causal mask。

但当前任务不是自回归生成，而是比较任务，所以当前实现会：

- patch 掉 `_update_causal_mask`
- 让 comparator 变成双向 attention

这样整个拼接序列里的 token 都可以互相看到。

## 9. 打分头

当前打分头非常直接：

1. 取 comparator 输出里的 query token 切片
2. 对 query token 做 mean pool
3. `LayerNorm`
4. `Linear(hidden_dim -> 1)`
5. `sigmoid`

最终得到：

- `score = P(f1 > f2)`

## 10. 当前训练结构

代码层面依然支持三个阶段：

1. `warmup`
2. `lora`
3. `curriculum`

但当前实际常用方式通常是：

- 只训练前两个阶段
- 通过 `--disable_curriculum` 关闭第三阶段

也就是说，当前最常见的实际训练流程是：

```text
warmup
↓
只训练新模块
↓
lora
↓
继续训练新模块 + 训练 comparator 前 6 层上的 LoRA
```

### 10.1 warmup 阶段训练什么

当前会训练：

- `video_projector`
- `video_pos_embed`
- `context_projector`
- `segment_embed`
- `stage_embed`
- `query_embed`
- `score_norm`
- `score_head`

### 10.2 lora 阶段训练什么

当前会训练：

- warmup 阶段这些新模块继续训练
- comparator backbone 上的 `lora_` 参数

当前不会训练：

- Qwen 原始 backbone 主权重

## 11. 当前模型结构和早期方案的关系

当前实现和早期方案主干是一致的：

- latent 压缩
- 条件投影
- query/context/stage 组序列
- Qwen comparator
- pairwise 打分

但也有两个当前已经明确的实现结论：

1. comparator backbone 实际是“Qwen 前 6 层 + LoRA 微调”，不是整体 Qwen 微调
2. 实际常用训练方式是“两阶段训练”，第三阶段 curriculum 常常显式关闭

## 12. 一句话总结

当前 `score_model_v2` 的实际结构可以概括为：

- 用视频压缩器把两个 Wan latent 压成高维 token 序列
- 把文本和图像条件投影到同一个 hidden space
- 用 query/context/stage 组织统一比较序列
- 用 Qwen2.5-VL language model 前 6 层作为 comparator backbone
- 只对这前 6 层 backbone 挂 LoRA 并在第二阶段训练
- 最终输出 `P(f1 > f2)` 作为评分结果
