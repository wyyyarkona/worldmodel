# Qwen 条件上下文改造方案

本文档整理一套面向 `score_model_v2` 的详细改造思路，目标是把当前“数值特征直接送入 comparator”的条件建模方式，逐步升级为“更接近 Qwen 原生理解方式”的多模态比较器。

核心改造目标有三点：

1. 将视频 prompt 和参考首帧图片的条件信息，改造成更适合 Qwen 理解的输入形式。
2. 在输入 comparator 的各段特征上增加更明确的“身份标记”，让模型知道哪一段是 Video A、哪一段是 Video B、哪一段是文本 prompt、哪一段是参考图、哪一段是任务说明。
3. 给 Qwen 一段明确的英文任务说明 prompt，使其在比较时更清楚当前任务目标。

当前希望达到的目标输入形式是：

```text
[h1, h2, context_text, reference_image, task_prompt, query]
```

其中：

- `h1`：视频 A 的特征 token
- `h2`：视频 B 的特征 token
- `context_text`：视频 prompt 对应的文本 token
- `reference_image`：首帧参考图对应的图像 token
- `task_prompt`：英文任务说明 token
- `query`：最终读出 token

最终输出仍然可以配置为：

- `query` 读出
- `h1_h2` 读出
- `hybrid` 混合读出

本文档会详细说明：

- 为什么要这么改
- 哪些模块需要改
- 如何分阶段落地
- 有哪些风险
- 推荐的实施顺序

---

## 1. 当前问题背景

当前 `score_model_v2` 的整体输入链路大致是：

```text
f1 -> VideoProjector -> h1
f2 -> VideoProjector -> h2
text_emb + image_emb -> ContextProjector -> context
[h1, h2, context, stage, query] -> QwenComparator
-> readout -> score
```

已经观察到的关键问题包括：

1. `h1_hidden / h2_hidden` 往往是有变化的。
2. 交换 `(f1, f2)` 后，`h1_hidden / h2_hidden` 往往也会变化。
3. 但 `query_hidden`、`pooled`、最终 `logit` 很长时间内会退化成近似常数或单类预测。

这说明：

- comparator 内部不是完全没工作
- 但 query-based readout 没有稳定读出有效比较信息

在这种情况下，除了改 readout 方式，一个非常自然的方向就是：

**让 Qwen 更清楚地理解“输入里每一段是什么、任务到底是什么”。**

这就是本改造方案的出发点。

---

## 2. 这次改造的核心思想

### 2.1 从“数值对齐”走向“语义对齐”

当前的 `ContextProjector` 更像一个“维度对齐器”：

- 文本 embedding 从 `text_dim` 投到 `hidden_dim`
- 图像 embedding 从 `image_dim` 投到 `hidden_dim`
- 然后直接拼到 comparator

这种做法的优点是简单，缺点是：

- Qwen 看到的是一堆“已经投影好的向量”
- 但并不知道这些向量背后的语义身份
- 也没有显式任务提示来告诉它“现在是在比较视频 A 和视频 B”

所以更理想的方向是：

- 文本 prompt 尽量以 Qwen 熟悉的文本 token 形式出现
- 参考图尽量以更接近 Qwen 视觉输入的 token 形式出现
- 任务说明 prompt 显式告诉模型要做什么
- 再辅以 segment/type embedding 做结构标注

这样 comparator 输入不仅“数值可用”，而且“语义更清楚”。

---

## 3. 总体改造结构

推荐的目标结构如下：

```text
video A latent -> VideoProjector -> h1
video B latent -> VideoProjector -> h2

raw text prompt
  -> Qwen tokenizer
  -> Qwen token embedding
  -> context_text_tokens

reference image
  -> image encoder / image projector
  -> reference_image_tokens

task instruction prompt
  -> Qwen tokenizer
  -> Qwen token embedding
  -> task_prompt_tokens

[h1, h2, context_text_tokens, reference_image_tokens, task_prompt_tokens, query]
  + segment/type embeddings
  -> QwenComparator
  -> configurable readout
  -> score
```

这里推荐保留：

- 视频仍然走现有 `VideoProjector`
- 文本 prompt 改成 Qwen 原生 token embedding
- 首帧图像逐步改造成更接近 Qwen 视觉理解的 token
- 任务说明 prompt 作为额外 token 组加入

---

## 4. 各输入部分如何处理

### 4.1 视频 A / 视频 B

建议第一阶段不要动视频输入路线。

仍然保持：

```text
f1 -> VideoProjector -> h1
f2 -> VideoProjector -> h2
```

原因：

1. 现有训练管线已经围绕 latent 输入构建。
2. 当前主要问题不在视频 token 本身完全没信号，而在读出和条件理解。
3. 如果一开始连视频侧也一起重构，变量会太多，不利于定位问题。

所以视频 A/B 先保持不变，只增强“条件信息”和“任务说明”。

---

### 4.2 视频 prompt

这是最值得优先改造的一部分。

当前情况：

- prompt 很可能已经被离线处理成 `text_emb`
- 然后进 `ContextProjector`

建议改成：

```text
raw prompt text
  -> Qwen tokenizer
  -> Qwen embed_tokens
  -> prompt text tokens
```

优点：

1. Qwen 对文本 token 的理解是它最擅长的输入形式之一。
2. 不再只是“一个被投影后的条件向量”，而是显式的文本序列。
3. 更利于任务 prompt、标签提示和条件文本之间形成自然语言层面的关联。

实现上可以：

- 在 dataset 中提供原始 prompt 文本字段
- 在模型初始化时拿到 Qwen tokenizer
- 调 `embed_tokens` 把文本 prompt 转成 hidden tokens

如果担心序列太长，可限制：

- 最大 token 数，例如 64 或 128

---

### 4.3 参考首帧图片

这一部分有两种落地路径。

#### 路径 A：过渡方案，先保留现有图像特征

如果当前 manifest 已经有：

- `clip_fea_path`
- 或 `image_emb_path`

那第一阶段可以先保留这套输入来源，只是：

1. 单独拆成 `reference_image_tokens`
2. 不再把它笼统地归到 `context`
3. 给它单独的 segment/type embedding：
   - `reference_image`

这样虽然它还不是 Qwen 原生视觉 token，但至少身份清楚了。

#### 路径 B：目标方案，走 Qwen-VL 视觉编码路径

理想上可以：

```text
reference image
  -> Qwen-VL image preprocessing
  -> visual encoder
  -> image tokens aligned to language hidden size
```

优点：

- 输入形式更接近 Qwen2.5-VL 原生多模态建模方式
- 文本 prompt 和参考图可以处在更一致的语义空间里

缺点：

- 改造成本更高
- 需要认真处理视觉塔接口
- 当前 comparator 只保留了语言模型前几层，可能需要额外接视觉前处理逻辑

因此建议：

- 第一阶段先走路径 A
- 第二阶段再考虑切到 Qwen-VL 原生视觉 token

---

### 4.4 任务说明 prompt

这一部分建议一定要做，而且实现成本最低。

示例英文 prompt：

```text
You are comparing two candidate video representations, Video A and Video B.
You are also given the original text prompt and the reference image.
Decide whether Video A is better than Video B based on overall visual quality,
alignment with the prompt, and physical consistency.
```

还可以进一步加强结构标签：

```text
[Video A]
[Video B]
[Text Prompt]
[Reference Image]
[Task]
Decide whether Video A is better than Video B.
```

建议作为单独一段 token：

```text
task_prompt_tokens
```

并通过 Qwen tokenizer + `embed_tokens` 编码。

作用：

1. 明确告诉模型当前任务是什么
2. 明确比较目标是 A 相对 B，而不是无条件评分
3. 为 query 读出提供更强的任务约束

---

## 5. 身份标记：不仅要有 token，还要有“谁是谁”

如果只把各类 token 拼进去，但不显式标身份，Qwen 仍然要自己猜：

- 哪些 token 属于视频 A
- 哪些属于视频 B
- 哪些属于文本 prompt
- 哪些属于参考图
- 哪些属于任务说明

这会增加学习难度。

因此建议同时使用两种身份标记方式。

### 5.1 Segment / Type Embedding

扩展当前 `SegmentEmbedding`。

当前建议的段类型：

- `video_a`
- `video_b`
- `text_prompt`
- `reference_image`
- `task_prompt`
- `query`
- `stage`

如果为了兼容现有代码，也可以继续沿用：

- `h1`
- `h2`

但从语义上更推荐改成：

- `video_a`
- `video_b`

### 5.2 显式自然语言标签

除了 embedding 标签，再用文字提示强化身份：

例如 task prompt 中包含：

```text
Video A tokens describe candidate A.
Video B tokens describe candidate B.
Text Prompt tokens describe the generation instruction.
Reference Image tokens describe the conditioning image.
```

这样模型不仅从 embedding 上知道身份，也从自然语言说明上知道身份。

---

## 6. 推荐的 sequence 顺序

在当前 causal attention + query 在末尾的设计下，推荐顺序是：

```text
[h1, h2, text_prompt, reference_image, task_prompt, query]
```

原因：

1. query 在末尾，可以在 causal attention 下看到全部前文。
2. 先放视频 A/B，有利于后续条件信息作为“判断依据”补充到后文。
3. 文本 prompt 和参考图在 query 之前，方便 query 汇总这些条件。
4. task prompt 靠近 query，有利于任务指令直接影响最后读出。

当然也可以尝试：

```text
[task_prompt, text_prompt, reference_image, h1, h2, query]
```

但建议先从前一种更接近“比较对象 -> 条件 -> 任务 -> 读出”的顺序开始。

---

## 7. 读出方式仍然要保留可配置

即使加入任务 prompt 和更清晰的条件输入，也不建议只保留 `query` 一种读出。

推荐继续保留：

### 7.1 `query`

```text
query_hidden.mean(dim=1) -> score_head
```

适合验证：

- 任务 prompt + 条件 token 是否能真正让 query 学会读出比较信息

### 7.2 `h1_h2`

```text
h1_pool = mean(h1_hidden)
h2_pool = mean(h2_hidden)
features = [h1_pool, h2_pool, h1_pool - h2_pool]
```

适合验证：

- comparator 后的候选表示本身是否更可分

### 7.3 `hybrid`

```text
[query_pool, h1_pool, h2_pool, h1_pool - h2_pool]
```

适合验证：

- 任务 prompt 是否增强了 query
- 同时保留候选直读信息，降低 query-only 失效风险

结论：

**即使加了 prompt，也不要只押宝 `query-only`。**

---

## 8. 需要修改的模块

### 8.1 `ScoreModelV2`

文件：

- `score_model_v2/models/score_model_v2.py`

建议改动：

1. 新增可配置的 `task_prompt`
2. 增加 `text_prompt_tokens`、`reference_image_tokens` 的拼接逻辑
3. 更新 `build_sequence()`
4. 更新 `split_hidden_states()`
5. 保持 `readout_mode` 可配置

---

### 8.2 `QwenComparator`

文件：

- `score_model_v2/models/qwen_comparator.py`

建议改动：

1. 持有 Qwen tokenizer
2. 暴露 `embed_tokens`
3. 提供文本 prompt 编码接口
4. 如果未来接入 Qwen-VL 原生图片输入，需要进一步接视觉前处理接口

---

### 8.3 `SegmentEmbedding`

文件：

- `score_model_v2/models/embeddings.py`

建议改动：

增加新的 segment/type：

- `video_a`
- `video_b`
- `text_prompt`
- `reference_image`
- `task_prompt`

或者至少补上：

- `prompt`
- `reference_image`

---

### 8.4 `ContextProjector`

文件：

- `score_model_v2/models/projectors.py`

建议不要继续把所有条件都塞进同一个 `ContextProjector`。

推荐拆分：

1. `PromptTextEncoder`
   - 原始文本 -> tokenizer -> embed_tokens

2. `ReferenceImageEncoder`
   - 现有图片特征 projector 或未来的 Qwen-VL 视觉编码器

这样结构更清晰，也便于逐步替换。

---

### 8.5 Dataset / manifest

文件主要在：

- `score_model_v2/train_v2.py`
- 上游 `openvid_i2v_tools`

需要保证 manifest 可以提供：

1. 原始视频 prompt 文本
2. 参考首帧图路径，或能加载的图片对象
3. 现有 `f1/f2` latent 路径

建议新增字段例如：

- `raw_prompt`
- `reference_image_path`

如果暂时不想改上游，也可以先在 dataset 里通过已有字段反查。

---

## 9. 推荐的英文任务 prompt

下面给一个可以直接作为默认值的版本：

```text
You are comparing two candidate video representations, Video A and Video B.
You are also given the original text prompt and the reference image.
Determine whether Video A is better than Video B.
Judge the comparison using overall visual quality, alignment with the prompt,
and physical consistency.
Return evidence through the internal comparison process so that the query tokens
can summarize the final decision.
```

更短一点的版本：

```text
Compare Video A and Video B under the same text prompt and reference image.
Decide whether Video A is better than Video B based on visual quality,
prompt alignment, and physical consistency.
```

建议先用短版，避免无意义拉长序列。

---

## 10. 推荐的实施顺序

不要一次性把所有改动都做满，建议按下面顺序渐进实施。

### 阶段 1：最低风险验证

目标：

- 先验证“任务 prompt + 更清晰身份标记”是否有帮助

做法：

1. 保留 `VideoProjector`
2. 保留现有 `image_emb` 输入方式
3. 只新增：
   - `task_prompt_tokens`
   - 更清晰的 segment embedding
4. sequence 变成：
   - `[h1, h2, context, stage, task_prompt, query]`

这个阶段实现成本低，适合先看：

- AUC 是否先有改善
- query 是否不再完全塌缩

### 阶段 2：文本 prompt 切到 Qwen token

目标：

- 让条件文本更像 Qwen 原生输入

做法：

1. 引入 `raw_prompt`
2. `raw_prompt -> tokenizer -> embed_tokens`
3. `context_text_tokens` 取代当前 text_emb projector 的部分作用

此时 sequence 变成：

```text
[h1, h2, context_text, reference_image, task_prompt, query]
```

### 阶段 3：参考图切到更强视觉表示

目标：

- 让参考图也更接近 Qwen-VL 原生多模态语义空间

做法：

1. 先尝试独立 image projector + 明确标签
2. 再考虑接入 Qwen-VL 视觉塔

### 阶段 4：系统性比较 readout

对比：

- `query`
- `h1_h2`
- `hybrid`

看加 prompt 后到底哪种 readout 最稳。

---

## 11. 训练与实验建议

### 11.1 不要一上来扫太多变量

如果你同时改：

- task prompt
- text prompt 编码
- reference image 编码
- readout
- learning rate

那结果很难解释。

建议一次只改 1 到 2 个关键变量。

### 11.2 先看 warmup 是否摆脱单类塌缩

优先看：

- `pred_label_counts`
- `pred_prob` 分布
- `val_auc`

如果 warmup 阶段就能从“全 0 / 全 1”变成真正双类输出，说明方向是对的。

### 11.3 优先看大 margin 样本

如果改造有效，通常最先改善的是：

- `large margin` bucket
- `late / step30`

这几个桶通常更容易出现正向信号。

---

## 12. 主要风险

### 12.1 序列会变长

加入：

- 文本 prompt token
- 参考图 token
- 任务 prompt token

都会增加序列长度，带来：

- 显存占用上升
- 训练速度下降

### 12.2 query 依然可能失效

即使加了任务 prompt，也不保证 `query-only` 一定恢复。

所以要保留：

- `h1_h2`
- `hybrid`

作为对照。

### 12.3 参考图如果不是真正 Qwen-VL 原生视觉 token

那“参考图走 Qwen”这件事只实现了一半。

所以要明确区分：

- “参考图更像 Qwen 语义空间”
- 和“参考图完全走 Qwen-VL 原生视觉输入”

这两者不是一回事。

---

## 13. 推荐的第一版落地方案

如果现在就要动手改，我最推荐的第一版是：

1. 保持视频 A/B 走现有 `VideoProjector`
2. 保持参考图先走现有图像特征输入
3. 新增任务说明英文 prompt
4. 新增更明确的 segment/type embedding
5. 把 sequence 改成：

```text
[h1, h2, context, stage, task_prompt, query]
```

6. 保留 `query / h1_h2 / hybrid` 三种读出

这是“改动最小、信息增量最大”的版本。

如果这一版出现明显提升，再继续做：

- 原始文本 prompt 改成 Qwen token
- 参考图改成更强视觉 token

---

## 14. 总结

这套改造的核心思想不是简单“多拼一段 token”，而是：

**让 comparator 输入从“单纯数值可拼接”升级为“结构清楚、语义清楚、任务清楚”的多模态比较输入。**

推荐路线是：

1. 保留视频 latent 路线不动
2. 先加任务 prompt
3. 再把视频 prompt 改成 Qwen 文本 token
4. 参考图逐步升级
5. 保留可配置 readout，不只押 `query-only`

如果最终有效，模型会从：

- “有信息但读不出来”

逐步变成：

- “知道自己在比较什么，也知道该按什么标准比较”

这就是这套改造最核心的目标。
