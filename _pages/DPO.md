---
layout: article
title: dpo
permalink: /dpo/
---


咱们从上层思维出发，用类比和例子来理解 **DPO（Direct Preference Optimization）**。

### **1. 类比理解：让模型“学会被喜欢”**

假设你是一位老师，要教学生写作文。你不直接告诉他“这就是一篇完美的作文”，而是给他两个版本，让他知道：“这篇比那篇更好”。

**DPO 的做法就像这样：**

- 不告诉模型“正确答案是什么”；
- 而是说：“在这两个回答里，人类更喜欢这个”；
- 模型要学会“为什么人类会喜欢这个”，并逐渐写出更受欢迎的回答。

### **2. 实际用例举例：**

#### **假设任务：回答问题“什么是地球变暖？”**

你给模型两个回答：

- 回答 A：地球变暖是指地球的平均气温在逐渐上升，这主要是由于人类活动造成的二氧化碳排放。
- 回答 B：地球变暖是因为太阳更热了，跟人类关系不大。

人类偏好：**更喜欢 A**

**DPO 做法：**

- 模型不会只学“回答 A 是对的”；
- 而是学：“在这两者中，人类为什么偏好 A”；
- 通过统计大量这样的偏好，优化自己的输出方向。

### **3. 和传统监督学习的对比**

| **方法**       | **教学方式**            | **数据需求**            | **本质**               |
| -------------- | ----------------------- | ----------------------- | ---------------------- |
| 监督微调 (SFT) | 给出标准答案            | 输入 → 期望输出         | 学“怎么写”             |
| DPO            | 给出两个选项 + 哪个更好 | 输入 → （更好 vs 更差） | 学“哪个更好、为什么好” |

所以 **DPO 的重点是偏好（Preference）**，而不是标准答案。

### **4. 类比 RLHF 但更直接**

- RLHF 里，你要先训练一个“奖励模型”，再用 PPO（强化学习）去最大化这个奖励；
- DPO 是：“我不训练奖励模型了，我直接用偏好数据来优化模型”，所以叫“Direct” Preference Optimization。

这样就省去了中间的奖励模型和复杂的强化学习步骤。

你可以把 DPO 想成：

> “用人类的点赞数据，直接告诉模型往受欢迎的方向调”。



------



我们现在来详细讲解 DPO 的原理和优化过程：



## **一、DPO 的任务设定**

DPO 假设你有一批这样的数据：

- 输入：prompt（问题、指令）
- 两个输出：chosen（人类更喜欢的回答），rejected（人类不喜欢的回答）

例如：

```
prompt: "如何与同事有效沟通？"
chosen: "试着倾听同事的观点，并保持开放的态度……"
rejected: "只要别理他们就好了，省得麻烦。"
```

目标就是让模型在看到 prompt 时，更倾向输出 chosen 的风格。

## **二、DPO 的思路（对比 RLHF）**

| **阶段** | **RLHF**                     | **DPO**              |
| -------- | ---------------------------- | -------------------- |
| 偏好建模 | 训练奖励模型（Reward Model） | 不训练奖励模型       |
| 策略优化 | 用 PPO 优化输出策略          | 直接优化语言模型本身 |
| 目标     | 最大化奖励                   | 最大化偏好概率       |

DPO 的关键思想是：

> **直接最大化模型选择“被偏好答案”的概率，相对“被拒绝答案”的概率。**

## **三、DPO 的损失函数**

损失函数来自一个 **pairwise 的 log-likelihood 比较**：

### **数学形式（简化版）：**

给定语言模型 π（就是我们要微调的 LLM），输入 prompt x、chosen y_c 和 rejected y_r，定义：

$L_{\text{DPO}} = -\log \left( \frac{\exp(\beta \cdot \log \pi(y_c|x))}{\exp(\beta \cdot \log \pi(y_c|x)) + \exp(\beta \cdot \log \pi(y_r|x))} \right)$

其中：

- π(y|x) 是模型输出某个回答 y 的概率（对 prompt x）；
- β 是一个超参数，调节对概率差异的敏感程度；
- 整个表达式类似一个 softmax 比较，被称为 **preference likelihood**。

这个损失的直观含义是：

> 如果模型更喜欢 chosen，π(y_c|x) 就会大于 π(y_r|x)，那损失就小；反之就会惩罚。

## **四、怎么实现这个优化？**

每一个训练 step：

1. 把 prompt + chosen 和 prompt + rejected 分别送入模型；
2. 得到它们各自的 token-level log-prob；
3. 把它们的总 log-likelihood 做 softmax 比较，得到偏好概率；
4. 用上面那个 loss 函数做反向传播，更新模型参数。

## **五、DPO 的优点**

- **不需要奖励模型**：简化训练流程，避免 reward model 的偏差；
- **稳定**：不像 RLHF 那样需要复杂的 PPO；
- **端到端可微**：直接用现有的语言模型训练接口；
- **更少“意外行为”**：RLHF 会出现 reward hacking、mode collapse，DPO 风险更低。

## **六、小结**

你可以把 DPO 想成这样一句话：

> “我不给你奖励分数，但我告诉你哪个更好。你得学着把‘更好’的回答概率提上去，‘更差’的降下来。”





------





那我们就来写一个简化版的 **DPO PyTorch 实现伪代码**，方便你理解训练的流程。假设我们已经有了一个预训练的语言模型（比如用 Hugging Face 的 AutoModelForCausalLM），以及配套的 tokenizer。

## **一、数据结构：准备偏好数据**

每条样本是这样的格式：

```
{
    "prompt": "如何与同事有效沟通？",
    "chosen": "你可以主动倾听对方的意见，并尊重他们的看法……",
    "rejected": "不想理就不理，反正他们也不懂……"
}
```

## **二、伪代码结构**

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 超参数
beta = 0.1

def get_log_probs(model, input_ids, attention_mask):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    # 每个token的log概率：取 labels 的logits位置
    log_probs = -outputs.loss * input_ids.size(1)  # 总 log likelihood
    return log_probs

def dpo_loss(logp_chosen, logp_rejected, beta):
    """DPO 损失函数"""
    diff = beta * (logp_chosen - logp_rejected)
    return -torch.nn.functional.logsigmoid(diff)  # log(1 / (1 + exp(-diff)))

# 一个训练 step 示例
def train_step(prompt, chosen, rejected):
    # 拼接输入：prompt + response
    input_chosen = tokenizer(prompt + chosen, return_tensors="pt", padding=True)
    input_rejected = tokenizer(prompt + rejected, return_tensors="pt", padding=True)

    # 得到 log-likelihood
    logp_chosen = get_log_probs(model, **input_chosen)
    logp_rejected = get_log_probs(model, **input_rejected)

    # 计算 DPO 损失
    loss = dpo_loss(logp_chosen, logp_rejected, beta)

    # 反向传播更新
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()
```

## **三、注意点**

1. **log likelihood** 通常是对 response 部分（chosen/rejected）而不是 prompt 计算；
2. 实际训练中建议用 model.forward(..., labels=...) 自动算 loss；
3. 批量训练时你需要对 prompt-chosen、prompt-rejected 分别 tokenization；
4. 在 Hugging Face 上已经有一些开源的 DPO 实现，比如 [trl](https://github.com/huggingface/trl)。

## **四、可选增强点**

- 加入 KL regularization（可选项，让模型输出不要偏离原始模型太远）；
- 用 gradient_accumulation 加速训练；
- 批量处理用 DataLoader 统一 batching。



------



在实际的 DPO 实现中，**加入 KL 散度** 是为了：

> **约束新模型不要偏离旧模型太远，保持语言风格和稳定性。**

## **一、为什么加入 KL 散度？**

虽然我们希望模型更偏向人类偏好（chosen > rejected），但如果模型改得太“激进”，可能导致语法错误、输出崩溃等问题。

所以我们让微调后的模型 π 不要偏离原始的模型 π₀ 太远，加入一个约束项：

$\text{KL}(\pi \,\|\, \pi_0)$

加入 KL 项的目标是：

$\min_\pi \;\; \text{DPO Loss} + \lambda \cdot \text{KL}(\pi \,\|\, \pi_0)$

其中 λ 是一个超参数，控制平衡。



## **二、DPO 中 KL 的具体形式（Token 级）**

我们要计算：

$\text{KL}(\pi(y \mid x) \parallel \pi_0(y \mid x))$

也就是：

$\text{KL} = \sum_{t=1}^{T} \pi(y_t \mid x) \cdot \log\left(\frac{\pi(y_t \mid x)}{\pi_0(y_t \mid x)}\right)$

但在实际实现中，我们用 **response 的 log-likelihood 差值**近似 KL：

### **伪代码片段（加入 KL）：**

```
# 加载原始（旧的）模型 π₀，保持 frozen，不训练
reference_model = AutoModelForCausalLM.from_pretrained("gpt2")
reference_model.eval()

# 计算 log-likelihood for current and reference model
logp_chosen_new = get_log_probs(model, **input_chosen)
logp_chosen_ref = get_log_probs(reference_model, **input_chosen)

# KL divergence = logp_new - logp_ref（越大表示越偏离）
kl_div = logp_chosen_new - logp_chosen_ref

# 最终损失
loss_dpo = dpo_loss(logp_chosen_new, logp_rejected, beta)
loss_total = loss_dpo + lambda_kl * kl_div
```

其中：

- lambda_kl 是调节 KL 强度的权重（常用 0.1～1.0）；
- 你也可以对 rejected 答案做类似处理；
- 一般只对 chosen 加 KL 约束就足够了。

## **三、可视化直觉**

加入 KL 后，你可以想象成：

- DPO：让模型拉高“人类喜欢”的回答；
- KL：同时拉住模型别跑太远，保持语言风格稳定。

## **四、提示**

- 你可以用 torch.no_grad() 包裹 reference model；
- 训练时不要让 reference_model 更新梯度；
- 想再进一步可以加入 **token-level KL loss**（非对数似然近似），但计算代价更高。

