# Score Model V2 训练指标说明

本文档说明当前 `score_model_v2` 训练与验证阶段已经输出的主要指标，解释每个指标是什么意思、反映什么能力、以及训练时应该如何解读。

## 1. 指标输出位置

当前训练会把指标写到：

- `output_dir/metrics.json`

样本级验证结果会写到：

- `output_dir/predictions/*.jsonl`

其中：

- `metrics.json` 适合看 epoch 级别的整体趋势
- `predictions/*.jsonl` 适合看单条样本预测是否正确、错在什么地方

## 2. 记录级公共字段

每条 epoch 记录里都会包含一些公共字段：

- `stage`
  表示当前训练阶段。当前常见是：
  - `warmup`
  - `lora`

- `phase`
  表示当前阶段中的 phase 编号。关闭 curriculum 后通常为 `1`。

- `allowed_stages`
  表示当前数据过滤后允许使用的 stage bucket 列表。

- `epoch`
  表示当前阶段内的 epoch 编号。

- `train_loss`
  当前 epoch 的训练损失平均值。

- `val_loss`
  当前 epoch 的验证损失平均值。

## 3. Loss 类指标

### 3.1 `train_loss`

含义：

- 当前 epoch 在训练集上的平均损失

它代表什么：

- 模型在训练数据上对 pairwise soft label 的拟合程度

如何解读：

- 持续下降通常是正常现象
- 如果下降很快但验证指标不提升，可能是过拟合
- 单独看它不够，必须结合排序指标一起看

### 3.2 `val_loss`

含义：

- 当前 epoch 在验证集上的平均损失

它代表什么：

- 模型在未参与训练的数据上对 soft label 的拟合程度

如何解读：

- 如果 `train_loss` 降、`val_loss` 不降甚至升高，通常要警惕过拟合
- 它仍然只是辅助指标，不是这个任务的最终判断标准

## 4. 核心排序指标

### 4.1 `val_pairwise_accuracy`

含义：

- 模型判断 `f1` 和 `f2` 谁更好的准确率

计算方式：

- 先计算 teacher 差值：`delta = teacher_score_a - teacher_score_b`
- 若 `delta > 0`，则目标标签为 `1`，表示 `f1 > f2`
- 模型输出 `logit`
- 若 `logit > 0`，则预测标签为 `1`
- 统计预测标签与目标标签是否一致

它代表什么：

- 这是最直接的“模型有没有判对方向”的指标

如何解读：

- 这是你最应该优先关注的指标之一
- 如果它不提升，说明模型并没有真正学会比较

### 4.2 `val_weighted_pairwise_accuracy`

含义：

- 加权后的 pairwise 准确率

计算方式：

- 仍然看 `pred_label == target_label`
- 但每个样本的贡献由 `sample_weight` 加权

它代表什么：

- 模型在“更重要样本”上的判断准确性

这里的 `sample_weight` 指：

- manifest 中每条样本的 `weight` 字段
- 如果没有，则默认是 `1.0`

如何解读：

- 如果这个指标明显高于普通 accuracy，说明模型在高权重样本上更稳定
- 如果普通 accuracy 还行，但这个指标很差，说明模型可能对关键样本判断不好

### 4.3 `val_auc`

含义：

- 验证集上的加权 ROC-AUC

它代表什么：

- 模型整体排序能力，不依赖固定阈值 `0.5`

如何解读：

- AUC 越高，说明模型整体越能把正样本排在负样本前面
- 它适合判断“打分排序能力”是否在提升
- 如果验证集中某轮只有单一类别，AUC 可能为 `null`

## 5. 预测分布相关指标

### 5.1 `val_mean_pred_prob`

含义：

- 当前验证集上模型预测概率 `sigmoid(logit)` 的平均值

它代表什么：

- 模型整体输出概率分布的中心位置

如何解读：

- 如果长期非常接近 `0.5`，说明模型整体区分能力弱，输出偏保守
- 如果长期非常极端，但准确率并没有提升，说明模型可能过度自信

### 5.2 `val_mean_margin`

含义：

- 当前验证集 teacher 差值绝对值 `|teacher_score_a - teacher_score_b|` 的平均值

它代表什么：

- 当前验证集样本整体的区分难度

如何解读：

- 它更多是辅助解释指标
- 如果某次验证集 margin 整体很小，模型 accuracy 偏低可能是正常的

### 5.3 `val_brier_score`

含义：

- 预测概率与二值目标标签之间的均方误差

它代表什么：

- 概率预测质量和校准情况

如何解读：

- 越低越好
- 如果 accuracy 还行但 Brier 很差，说明模型虽然经常选对，但概率值不够可信

## 6. 分桶指标

## 6.1 `val_accuracy_by_stage`

含义：

- 按 stage bucket 分别统计准确率

当前 bucket 名称包括：

- `early`
- `middle`
- `late`

每个 bucket 下又会包含：

- `count`
- `accuracy`
- `weighted_accuracy`

它代表什么：

- 模型在不同去噪阶段上的判断能力

如何解读：

- 如果总体 accuracy 在涨，但某个 stage 很差，说明模型能力不均衡
- 对你当前任务来说，这个指标能帮助判断模型是不是只擅长某一个阶段

## 6.2 `val_accuracy_by_step`

含义：

- 按真实 step index 分别统计准确率

如果你的数据 step 是：

- `10`
- `20`
- `30`

那么这里会有类似：

- `10`
- `20`
- `30`

每个 step 下也会包含：

- `count`
- `accuracy`
- `weighted_accuracy`

它代表什么：

- 模型在每个真实中间步上的判断能力

如何解读：

- 对你现在这种离散 step 训练数据，这个指标比 `by_stage` 更直观
- 它能直接回答：
  - `acc@10` 怎么样
  - `acc@20` 怎么样
  - `acc@30` 怎么样

## 6.3 `val_accuracy_by_margin_bucket`

含义：

- 按 teacher margin 大小分桶统计准确率

当前桶定义是：

- `small`：`0.0 <= margin < 0.1`
- `medium`：`0.1 <= margin < 0.3`
- `large`：`margin >= 0.3`

每个 bucket 下也会包含：

- `count`
- `accuracy`
- `weighted_accuracy`

它代表什么：

- 模型在“容易样本”和“困难样本”上的表现差异

如何解读：

- 如果 `large` 很高、`small` 很低，说明模型只会判断明显样本
- 如果连 `small` 都开始提升，通常说明模型边界判断能力在增强

## 7. 样本级预测字段

每轮验证后，会导出：

- `predictions/{stage}_phase{phase}_epoch_{epoch}_val_predictions.jsonl`

每条样本级记录里包含：

- `sample_index`
  当前样本在数据集中的索引

- `step_index`
  当前样本对应的真实 step

- `stage_id`
  当前样本对应的 stage bucket id

- `stage_name`
  当前样本对应的 stage bucket 名称

- `teacher_score_a`
  第一个候选的 teacher 分数

- `teacher_score_b`
  第二个候选的 teacher 分数

- `teacher_delta`
  `teacher_score_a - teacher_score_b`

- `margin`
  `|teacher_delta|`

- `sample_weight`
  当前样本的训练/评估权重

- `pred_logit`
  模型输出的原始 logit

- `pred_prob`
  模型输出的概率 `P(f1 > f2)`

- `pred_label`
  模型二值判断结果

- `target_label`
  teacher 导出的目标二值标签

- `correct`
  当前样本是否判断正确

这些字段主要用来：

- 排查错误样本
- 按 step 看模型错在哪里
- 按 margin 看模型在难样本上的行为
- 做样本级可视化

## 8. 当前最建议关注的指标

如果只训练前两个阶段：

- `warmup`
- `lora`

那么最推荐重点关注下面这些：

1. `val_pairwise_accuracy`
2. `val_weighted_pairwise_accuracy`
3. `val_auc`
4. `val_accuracy_by_step`
5. `val_accuracy_by_margin_bucket`

辅助再看：

- `train_loss`
- `val_loss`
- `val_brier_score`

## 9. 如何解读常见现象

### 9.1 `train_loss` 降，但 `val_pairwise_accuracy` 不涨

说明：

- 模型在拟合训练目标，但没有真正学会更好的排序判断

### 9.2 `val_pairwise_accuracy` 涨，但 `val_auc` 不明显涨

说明：

- 模型在固定阈值附近可能更稳定了
- 但整体排序能力提升有限

### 9.3 `val_weighted_pairwise_accuracy` 明显低于 `val_pairwise_accuracy`

说明：

- 模型在高权重样本上表现不够好

### 9.4 `val_accuracy_by_step` 中某个 step 明显偏低

说明：

- 模型在该 step 对应的中间状态判断更弱

### 9.5 `val_accuracy_by_margin_bucket.large` 很高，但 `small` 很低

说明：

- 模型能分清明显样本，但对边界样本区分不足

### 9.6 `val_mean_pred_prob` 长期非常接近 `0.5`

说明：

- 模型整体输出偏保守，判别性可能不够

### 9.7 `val_brier_score` 很差但 accuracy 不低

说明：

- 模型可能经常选对，但概率不够可信，存在校准问题

## 10. 一句话总结

当前这套指标的核心目标是：

- 用 `loss` 看拟合
- 用 `accuracy / weighted_accuracy / auc` 看排序是否真的变好
- 用 `by_step / by_stage / by_margin` 看模型具体强在哪、弱在哪
- 用样本级预测文件看模型到底错了哪些样本
