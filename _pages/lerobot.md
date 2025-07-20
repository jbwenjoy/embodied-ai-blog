---
layout: article
title: lerobot
permalink: /lerobot/
---


LeRobot 是一个近年来在机器人学习圈子里很受关注的项目，主打目标是：

> 🦾 **让多种机器人平台之间的模仿学习和泛化变得更简单、更标准、更可复现。**

下面我给你分点讲清楚 LeRobot 是什么、它解决了什么问题、适合谁用 👇

## **🤖 LeRobot 是什么？**

> LeRobot 是一个**用于机器人行为学习（模仿学习、强化学习）**的通用框架，由 UC Berkeley 的 RoboNet、VIL 等实验室开发。它最早用于 RT-1、RT-2 等大模型机器人项目中。

## **🧠 它解决了什么问题？**

### **✅ 1. 各种平台的差异太大**

- 不同机械臂、不同抓取器、不同控制频率都让训练过程难以复用；
- LeRobot 提供了一个标准化接口来**抽象和统一不同机器人的硬件/控制接口**。

### **✅ 2. 数据集太分散**

- 每个实验室都搞自己的 dataset（DClaw、Franka、WidowX、Bimanual…）；
- LeRobot 提供了一个**统一的 dataset 加载 & 回放接口**，可以直接读取常用模仿学习数据格式（HDF5、npz等）。

### **✅ 3. 学习模型训练流程杂乱**

- LeRobot 提供标准的数据预处理（视觉、动作）、模型定义、训练 loop；
- 你可以 plug in 自己的模型（VLA、Transformer、RNN），用标准 pipeline 训练行为策略。

## **🔧 它的功能结构（你可以想象是 PyTorch + Gym + 数据工具的组合）：**

| **模块**         | **功能**                                       |
| ---------------- | ---------------------------------------------- |
| lerobot.envs     | 模拟器或真实机器人环境封装                     |
| lerobot.data     | 数据集加载器（支持桥式数据、多相机、多机器人） |
| lerobot.models   | 视觉-动作模型结构，如 transformer policies     |
| lerobot.training | 训练器（BC、BCQ、RL）                          |
| lerobot.utils    | 视觉预处理、控制接口转换、路径采样等           |

## **🚀 和 RoboNet / RT 系列的关系？**

LeRobot 最早是为了支持 RoboNet/BridgeData 项目统一不同平台行为数据。

你可以理解为：

- RT-1 / RT-2 是上层大模型应用；
- LeRobot 是底层训练 & 数据工具基础设施。

## **🧪 适合什么人用？**

- 想在**多个机器人平台上训练通用策略**；
- 想统一处理大量 mimicking / teleop / demo 数据；
- 想搞自监督预训练 + policy learning 的机器人视觉研究；
- 想复现 RT-1 / BC-Z / BridgeData 等论文中的数据加载和行为训练方式。

## **🗂️ 官方资源 & 项目入口**

- GitHub 地址：https://github.com/rail-berkeley/lerobot
- 配套项目：
  - BridgeData V2：https://github.com/rail-berkeley/bridgedata
  - RT-X 系列：https://github.com/google-research/robotics_transformer_x

## **✅ 总结一句话**

> **LeRobot 是一个为了实现“多机器人平台共享行为数据 + 模仿学习训练”的标准框架，适合做多模态模仿学习、桥式数据预训练、多任务控制等研究。**

------



你要学 LeRobot 的代码架构是一个非常正确的切入方式 —— 它不只是工具库，更像是一个可扩展的**机器人模仿学习研究平台**。我来带你从宏观和实战两个角度，一步步了解它的整体结构。



## **🧠 先说总体印象：LeRobot 像什么？**

LeRobot 和 PyTorch Lightning 或 Stable-Baselines 类似，围绕以下几件事组织代码：

1. **环境封装（Env Wrappers）**
2. **数据加载和表示（Dataset / Episode / Sequence）**
3. **模型结构（Policy / Encoder / Decoder）**
4. **训练与评估脚本（Trainer / Runner）**
5. **配置 & 运行接口（YAML / CLI / Hydra）**



你可以想象整个框架像一台工厂流水线：数据 → 模型 → 训练器 → 策略输出。

------



## **📁 LeRobot 的代码结构解读**

我们以 GitHub 的主 repo https://github.com/rail-berkeley/lerobot 为例。

```
lerobot/
├── data/            # 数据加载器 & 轨迹预处理
├── envs/            # 环境封装（仿真 + 真实机器人）
├── models/          # 模型结构：Encoder, Policy 等
├── training/        # BC, RL, evaluation 等训练器
├── utils/           # 工具函数、camera warp、logger 等
├── configs/         # Hydra 配置文件（YAML）
├── scripts/         # 训练 / 评估脚本
└── assets/          # 默认模型权重、相机参数等
```

## **🧩 各模块功能详解**

### **🟩 1.** 

### **data/**

###  **– 数据处理模块**

- 核心文件：
  - episode_dataset.py：用于加载一个轨迹（episode）；
  - hdf5_dataset.py：HDF5 文件解析器；
  - sequence_dataset.py：将轨迹划分成模型训练需要的 windowed sequences；
- 通常你会通过：

```
from lerobot.data import SequenceDataset
```

- 得到一个标准的 PyTorch Dataset，可以直接喂给 DataLoader。

### **🟦 2.** 

### **envs/**

###  **– 环境抽象层**

- 支持多种机器人平台封装：Franka、WidowX、DClaw 等；
- 可以封装真实机器人、仿真器（PyBullet、Isaac Gym）；
- 所有 env 都会实现一个标准 API：

```
obs = env.reset()
obs, reward, done, info = env.step(action)
```

- 有时你可以用 env.wrapped_env 访问底层原始 API。

### **🟨 3.** 

### **models/**

###  **– 模型模块**

- 每个模型是一个 PyTorch Module，负责：

  - 编码视觉输入（例如 ResNet / Transformer）；
  - 输出动作或动作分布；

- 例如：

  - ImageEncoder
  - TransformerPolicy
  - MLPPolicy

  你可以继承它们快速搭出自己的策略网络。

### **🟧 4.** 

### **training/**

###  **– 训练与策略更新逻辑**

- 比如：
  - behavior_cloning_trainer.py
  - reward_weighted_regression.py
- 通常封装了整个训练 loop 和评估逻辑；
- 支持 logging、checkpoint、loss 输出等。

### **🛠️ 5.** 

### **utils/**

###  **– 工具模块**

- 相机标定、图像转换、渲染、normalization、配置解析等都在这里；

- 比如：

  - camera_utils.py
  - vis_utils.py
  - traj_utils.py

  

## **🧪 实战路径推荐：想读懂 LeRobot，可以这样入手**

| **阶段** | **重点**                                                     |
| -------- | ------------------------------------------------------------ |
| ✅ 阶段 1 | 跑通 scripts/train_behavior_cloning.py，理解 config、数据、模型、trainer 如何联动 |
| ✅ 阶段 2 | 读 models/transformer_policy.py，弄懂视觉 + transformer policy 的 forward 过程 |
| ✅ 阶段 3 | 读 data/sequence_dataset.py 和 episode_dataset.py，理解如何从原始数据构造训练输入 |
| ✅ 阶段 4 | 自己改一个模型，或者改数据预处理 pipeline，比如改输入图像尺寸或加入 attention mask |
| ✅ 阶段 5 | 尝试写自己的 env wrapper 或 dataset class，让 LeRobot 跑你自己的数据（如 VR 控制采集的 demo） |

## **✅ 总结一句话**

> LeRobot 是一个面向机器人模仿学习的大框架，结构清晰，每个模块职责单一，非常适合边学边改；你可以从数据加载、模型结构或训练脚本三个入口点入手阅读代码。

如果你告诉我你对哪块更感兴趣（比如你想看它的 transformer policy 是怎么做视觉编码的？或者它的数据预处理怎么做序列切分？），我可以直接带你一行一行读源码，也可以直接改出你自己的版本。你想从哪个模块开始？