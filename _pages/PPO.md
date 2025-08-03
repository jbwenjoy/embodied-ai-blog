---
layout: article
title: PPO
permalink: /ppo/
---


# PPO

## 前置知识 overview

### 奖励（reward）和回报（return）

t时刻的奖励是状态和动作的函数
$$r_t = r(s_t, a_t)$$
可能因状态、动作的随机而具有随机性，有随机性时使用大写$R_t$、$S_t$、$A_t$（本文档其他变量也遵循此规则）
回报是未来的累积奖励
$$U_t = R_t + R_{t+1} + ... + R_n$$

### 动作价值函数 Q

即对$U_t$关于变量 $S_{t+1}, A_{t+1}, ... , S_n, A_n$（或至无穷）求条件期望
$$Q_\pi(s_t,a_t) = \mathbb{E}_{S_{t+1},A_{t+1},\cdots,S_n,A_n}\Big[U_t\Big| S_t=s_t,A_t=a_t\Big]$$
$$Q_\pi(s_t,a_t)=\mathbb{E}_{s_{t+1},a_{t+1},...}\left[\sum_{l=0}^\infty\gamma^lr(s_{t+l})\right]$$
期望中的$S_t = s_t$和$A_t = a_t$是条件，意思是已经观测到$S_t$与$A_t$的值。条件期望的结果$Q_\pi (s_t, a_t)$即动作价值函数（action-value function）
$$Q_\star(s_t,a_t) = \max_\pi Q_\pi(s_t,a_t),\quad\forall s_t\in\mathcal{S},\quad a_t\in\mathcal{A}$$
$$\pi^\star = \underset{\pi}{\operatorname*{argmax}} Q_\pi(s_t,a_t),\quad\forall s_t\in\mathcal{S},\quad a_t\in\mathcal{A}$$

### 状态价值函数 V

$$V_{\pi}(s_{t}) = \mathbb{E}_{A_t\sim\pi(\cdot|s_t)}\Big[Q_\pi(s_t,A_t)\Big] = \sum_{a\in\mathcal{A}}\pi(a|s_t)\cdot Q_\pi(s_t,a)$$
公式里把动作$A_t$作为随机变量，然后关于$A_t$求期望，把$A_t$消掉。得到的状态价值函数$V_\pi (s_t)$只依赖于策略$\pi$与当前状态$s_t$，不依赖于动作。状态价值函数$V_\pi (s_t)$也是回报$U_t$的期望：
$$V_\pi(s_t) = \mathbb{E}_{A_t,S_{t+1},A_{t+1},\cdots,S_n,A_n} \big[ U_t | S_t=s_t \big]$$
$$V_{\pi}(s_{t})= \mathbb{E}_{a_{t},s_{t+1},...}\left[\sum_{l=0}^{\infty}\gamma^{l}r(s_{t+l})\right]$$

### 优势 Advantage

优势函数定义为
$$D_\pi(s,a)= Q_\pi(s,a)-V_\pi(s)
\\
\text{where } a_t\sim\pi(a_t|s_t), s_{t+1}\sim P(s_{t+1}|s_t,a_t)\text{ for }t\geq0$$

最优优势函数
$$D_\star(s,a)= Q_\star(s,a)-V_\star(s)$$
备注：更多资料使用$A$表示优势函数，此处为了与Action区分采用$D$

### 策略网络

策略函数的参数是状态$s$和动作$a$，输出概率值，是概率质量函数（离散的PDF）
$$\pi(a, s) = \pi(a \mid s) \triangleq \mathbb{P}(A=a \mid S=s)$$
参数$\theta$的策略网络为$\pi(a|s;\theta)$

### 价值学习

#### Temporal Difference

TODO

#### Q_Learning

TODO

#### On-Policy vs Off-Policy

**强调采样和更新的策略是否相同**

* 行为策略：控制智能体与环境交互
* 目标策略：RL训练的最终目的的策略函数

* 同策略：用相同的行为策略和目标策略（PPO、SARSA、Actor-Critic）
* 异策略：用不同的行为策略和目标策略（DQN、Q-Learning、DDPG、SAC，可经验回放）

#### Online vs Offline

**强调更新weight的时机**

* 在线学习：任务过程中实时更新policy
* 离线学习：任务完全结束后更新

#### SARSA

TODO

State-Action-Reward-State-Action

### 策略学习

#### 目标函数

如果一个策略很好，那么状态价值$V_\pi(S)$的均值应当很大，因此对状态求期望作为目标函数，并求最大化
$$\max_\theta \big\{ J(\theta) = \mathbb{E}_S \big[ V_\pi(S) \big] \big\}$$
策略梯度（Policy Gradient）是目标函数关于网络参数$\theta$的梯度，用于梯度上升
$$\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta}_{\mathrm{now}}) \triangleq \left.\frac{\partial J(\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}\right|_{\theta=\theta_{\mathrm{now}}}$$
$$\theta_\mathrm{new}\leftarrow\theta_\mathrm{now}+\beta\cdot\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta}_\mathrm{now})$$

### 策略梯度定理

假定状态$S$服从马尔科夫链的稳态分布$d(\cdot)$，取折扣率$\gamma$，一局游戏长度为$n$
$$\begin{aligned}
\frac{\partial J(\boldsymbol{\theta})}{\partial\boldsymbol{\theta}} &= \frac{1-\gamma^n}{1-\gamma} \cdot \mathbb{E}_{S\sim d(\cdot)}\Big[\mathbb{E}_{A\sim\pi(\cdot|S;\boldsymbol{\theta})} \Big[ \frac{\partial\ln\pi(A|S;\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}\cdot Q_\pi(S,A) \Big] \Big] \\
&= \frac{1-\gamma^n}{1-\gamma} \cdot \mathbb{E}_{S\sim d(\cdot)}\Big[\mathbb{E}_{A\sim\pi(\cdot|S;\boldsymbol{\theta})}\Big[ \frac{\partial\ln\pi(A|S;\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}\cdot Q_\pi(S,A) \Big] \Big]
\end{aligned}$$
系数$\frac{1-\gamma^n}{1-\gamma}$可以略去是因为其地位与学习率等价

#### 近似策略梯度

由于不知道状态$S$的PDF，无法求解出真实的$\mathbb{E}_S$，因此可以使用蒙特卡洛近似：
对action进行随机抽样：
$$a \sim \pi (\cdot \mid s; \boldsymbol{\theta})$$
计算随机梯度：
$$\boldsymbol{g}(s, a; \boldsymbol{\theta}) \triangleq Q_\pi (s, a) \cdot \nabla \ln \pi(a, \mid s; \boldsymbol{\theta})$$
随机梯度$\boldsymbol{g}(s, a; \boldsymbol{\theta})$是策略梯度$\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta})$的无偏估计：
$$\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta}) = \mathbb{E}_S\Big[ \mathbb{E}_{A\thicksim\pi(\cdot|S;\boldsymbol{\theta})}\Big[ \boldsymbol{g}(S,A;\boldsymbol{\theta}) \Big] \Big]$$
因此可以使用策略梯度的近似来做梯度上升：
$$\theta_\mathrm{new}\leftarrow\theta_\mathrm{now}+\beta\cdot\boldsymbol{g}(s, a; \boldsymbol{\theta})$$

#### Actor-Critic

但进一步，还不知道计算随机梯度$\boldsymbol{g}(s, a; \boldsymbol{\theta})$所需的动作价值函数$Q_\pi(s, a)$。Actor-Critic方法使用价值网络$q(s, a; \boldsymbol{w})$（参数$\boldsymbol{w}$）来近似$Q_\pi$。
价值网络与DQN结构相似，但有这些区别：
1. DQN是对最优动作价值函数$Q_\star(s, a)$的近似
2. 训练价值网络使用SARSA（On-Policy），不能用经验回放；而DQN训练的Q-Learning（Off-Policy）可以

Actor：策略网络$\pi(a\mid s; \boldsymbol{\theta})$
Critic：价值网络$q(s, a; \boldsymbol{w})$
$$\widehat{\boldsymbol{g}}(s,a;\boldsymbol{\theta}) \triangleq \underbrace{q(s,a;\boldsymbol{w})}_{\text{Critic's Scores}}\cdot\nabla_{\boldsymbol{\theta}}\ln\pi(a\mid s;\boldsymbol{\theta})$$

#### Training the Critic

可以用SARSA算法更新$\boldsymbol{w}$，提高评委的水平：每次从环境中观测到一个奖励$r$，把$r$看做是真相来校准评委的打分（监督学习、regression）。
在$t$时刻，价值网络输出
$$\hat{q}_t = q(s_t, a_t; \boldsymbol{w}) \approx Q_\pi(s_t, a_t)$$
在$t+1$时刻，实际观测到$r_t$、$s_{t+1}$、$a_{t+1}$，计算TD目标
$$\widehat{y}_t \triangleq r_t+\gamma\cdot q(s_{t+1},a_{t+1}; \boldsymbol{w})$$
希望$q(s_t, a_t; \boldsymbol{w})$逼近$\hat{y}_t$，于是定义MSE Loss
$$L(\boldsymbol{w})\triangleq\frac12\Big[q(s_t,a_t;\boldsymbol{w}) - \widehat{y}_t\Big]^2$$
梯度下降
$$\nabla_{\boldsymbol{w}}L(\boldsymbol{w}) = \underbrace{\left(\widehat{q}_t-\widehat{y}_t\right)}_{\text{TD 误差 }\delta_t}\cdot\nabla_{\boldsymbol{w}} q(s_t,a_t;\boldsymbol{w})$$
$$w \leftarrow w - \alpha\cdot\nabla_{\boldsymbol{w}}L(\boldsymbol{w})$$

## TRPO

### 策略目标

https://hrl.boyuai.com/chapter/2/trpo%E7%AE%97%E6%B3%95/

我们希望借助当前的策略$\pi\big(a|s; \boldsymbol{\theta}_{\mathrm{old}}\big)$来搜寻更优的策略参数，因此使用重要性采样，将前述策略学习目标函数$J(\boldsymbol{\theta})$中的状态价值函数$V_\pi(s)$变换为
$$\begin{aligned}V_{\pi}(s)&=\sum_{a\in A}\pi\big(a|s; \boldsymbol{\theta}_{\mathrm{old}}\big) \cdot \frac{\pi(a|s; \boldsymbol{\theta})}{\pi(a|s; \boldsymbol{\theta}_{\mathrm{old}})} \cdot Q_{\pi}(s,a)\\&=\mathbb{E}_{A\sim\pi(\cdot|s;\boldsymbol{\theta}_{\mathrm{old}})}\bigg[\frac{\pi(A|s;\boldsymbol{\theta})}{\pi(A|s;\boldsymbol{\theta}_{\mathrm{old}})}\cdot Q_{\pi}(s,A)\bigg]\end{aligned}$$
则目标函数$J(\theta)$等价于
$$J(\boldsymbol{\theta}) =  \mathbb{E}_S\bigg[ \mathbb{E}_{A\sim\pi(\cdot|S;\boldsymbol{\theta}_{\mathrm{old}})} \bigg[ \frac{\pi(A| S;\boldsymbol{\theta})}{\pi(A|S;\boldsymbol{\theta}_{\mathrm{old}})} \cdot Q_{\pi(A|S;\boldsymbol{\theta})} (S,A) \bigg] \bigg]$$
对$J(\theta)$的期望做蒙特卡洛近似
$$L(\boldsymbol{\theta}\mid\boldsymbol{\theta}_\mathrm{old}) = \frac1n\sum_{t=1}^n \frac{\pi(a_t \mid s_t;\boldsymbol{\theta})}{\pi(a_t \mid s_t;\boldsymbol{\theta}_{\mathrm{old}})} \cdot Q_\pi(s_t, a_t)$$
对动作价值函数做两次近似（使用旧策略采样的轨迹回报$u_t$是对$Q_{\pi_{old}}$的近似，假定新旧策略接近，则也间接是$Q_{\pi}$的近似）
$$\tilde{L}(\boldsymbol{\theta}\mid\boldsymbol{\theta}_\mathrm{old}) = \frac1n\sum_{t=1}^n \frac{\pi(a_t \mid s_t;\boldsymbol{\theta})}{\pi(a_t \mid s_t;\boldsymbol{\theta}_{\mathrm{old}})} \cdot u_t$$
使用KL散度约束来满足新旧策略接近的要求
$$\boldsymbol{\theta}_{\text{new}} = \arg\max_{\theta} \tilde{L}(\boldsymbol{\theta}\mid\boldsymbol{\theta}_\mathrm{old})
\\
\mathrm{s.t.~} \frac1t\sum_{i=1}^t\mathrm{KL}\Big[\pi( \cdot | s_i; \boldsymbol{\theta}_\mathrm{old})\Big\| \pi( \cdot | s_i;\boldsymbol{\theta}) \Big] \leq \Delta $$
备注：部分资料使用优势函数$D$（更常见地记为$A$）替代上述$Q_\pi(s, a)$，两者只相差一个常数$V_{\theta_{old}}(s)$

### 优化求解

#### 泰勒展开+KKT

TODO

#### 共轭梯度法

TODO

### 算法流程

- 初始化策略网络参数$\theta$，价值网络参数$\omega$
- For 序列$e = 1 \to E$：
  - 用当前策略$\pi_\theta$采样轨迹$\{(s_1, a_1, r_1), (s_2, a_2, r_2), \ldots\}$
  - 根据收集到的数据和价值网络估计每个状态动作对的优势$A(s_t, a_t)$
  - 计算策略目标函数的梯度$g$
  - 共轭梯度法计算$x = H^{-1}g$
  - 用线性搜索找到一个$i$值，并更新策略网络参数$\theta_{k+1} = \theta_k + \alpha^i \sqrt{\frac{2\delta}{x^T H x}} x$，其中$i \in \{1, 2, \ldots, K\}$为能提升策略并满足 KL 距离限制的最小整数
  - 更新价值网络参数（与 Actor-Critic 中的更新方法相同）
- End for

## PPO

之所以称之为近端策略优化，因为限制了策略更新幅度，避免单步更新过大使Importance Sampling产生较大偏差导致策略崩溃
主要有PPO-Penalty和PPO-Clip两种形式

### PPO-Penalty

核心思想是通过Lagrange乘数法把TRPO的KL约束以惩罚的形式放入目标函数中，求解无约束优化问题
$$\arg\max_\theta\mathbb{E}_{s\sim\nu}\cdot_{\theta_k}\mathbb{E}_{a\sim\pi_{\theta_k}(\cdot|s)}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)}A^{\pi_{\theta_k}}(s,a)-\beta D_{KL}[\pi_{\theta_k}(\cdot|s),\pi_\theta(\cdot|s)]\right]$$
其中$\frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)}$是通过重要性采样进行估计的：先通过旧策略采样状态动作对$(s_t, a_t, logprob)$，放入缓存；更新新策略后，使用新旧策略分别计算在$s_t$下做出动作$a_t$的概率（旧策略的可以直接读取缓存中的logprob）；最后计算比值
迭代中不断按照如下规则更新$\beta$（$\delta$为超参数）：
$$\beta_{k+1} = \left\{ \begin{array}{ll}
\beta_{k+1}=\beta_k/2 & D_{KL}^{\nu^{\pi_{\theta_k}}}(\pi_{\theta_k},\pi_\theta) < \delta/1.5 \\[0.5em]
\beta_{k+1}=2\beta_k & D_{KL}^{\nu^{\pi_{\theta_k}}}(\pi_{\theta_k},\pi_\theta) > 1.5\delta \\[0.5em]
\beta_{k+1}=\beta_k & \text{Otherwise}
 \end{array} \right.$$

### PPO-Clip (PPO2)

https://spinningup.openai.com/en/latest/algorithms/ppo.html

PPO-Clip直接在目标函数中进行限制，以保证新旧参数的差距不会太大，这种方式通常比PPO-Penalty更好
$$\theta_{k+1}=\arg\max_{\theta}\frac{1}{|\mathcal{D}_{k}|T}\sum_{\tau\in\mathcal{D}_{k}}\sum_{t=0}^{T}\min\left(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{k}}(a_{t}|s_{t})}A^{\pi_{\theta_{k}}}(s_{t},a_{t}), \quad g(\epsilon,A^{\pi_{\theta_{k}}}(s_{t},a_{t}))\right)$$
$$g(\epsilon,A)=\left\{\begin{array}{ll}(1+\epsilon)A&A\geq0\\(1-\epsilon)A&A<0\end{array}\right.$$
或
$$L(s,a,\theta_{k},\theta) = \left\{ \begin{array}{ll}
\min\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{k}}(a|s)},(1+\epsilon)\right)A^{\pi_{\theta_{k}}}(s,a) & \text{Positive advatange}\\
\max\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)},(1-\epsilon)\right)A^{\pi_{\theta_k}}(s,a) & \text{Negative advantage}
\end{array}\right.$$
这种方法不涉及KL散度
其中：
- $\epsilon$控制截断范围，通常位于0.1-0.3，越大越激进
- 优势函数$A_t = Q(s_t, a_t) - V(s_t)$衡量一个动作相对当前状态平均行为$V(s_t)$的好坏，相比于直接使用回报

#### 算法步骤
- 输入：初始化策略网络参数$\theta_0$和初始价值网络参数$\phi_0$
- 循环过程：对于每一轮迭代$k = 0, 1, 2, \dots$，执行：
  - 收集轨迹$\mathcal{D}_k = \{ \tau_i \}$：根据当前策略$\pi_k = \pi(\theta_k)$在环境中运行，收集一组轨迹$\mathcal{D}_k$四元组$(s_t, a_t, r_t, s_{t+1})$存放在buffer中（网络权重更新后清空，也可能多次epoch后再清空，但由于有importance sampling，PPO仍是on-policy）
  - 计算回报：对每条轨迹中的每一个时间步$t$，计算累积的回报$\hat{R}_t$（TD）
  - 计算优势估计$\hat{A}_t$：使用当前的价值函数$V_{\phi_k}$估计优势函数$\hat{A}_t$，可以使用任意的优势估计方法（TD、MC、GAE）
  - 更新策略：通过最大化 PPO-Clip 的目标函数来更新策略参数$\theta$：
    - $\theta_{k+1} = \arg \max_{\theta} \frac{1}{|\mathcal{D}_k|T} \sum_{\tau \in \mathcal{D}_k} \sum_{t=0}^T \min \left( \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_k}(a_t | s_t)} \hat{A}^{\pi_{\theta_k}}(s_t, a_t), g(\epsilon, \hat{A}^{\pi_{\theta_k}}(s_t, a_t)) \right)$
  - 拟合价值函数：通过最小化均方误差来更新价值函数参数$\phi$：
    - $\phi_{k+1} = \arg \min_{\phi} \frac{1}{|\mathcal{D}_k|T} \sum_{\tau \in \mathcal{D}_k} \sum_{t=0}^T \left( V_{\phi}(s_t) - \hat{R}_t \right)^2$
- 结束：完成循环后继续迭代，直到收敛或达到预设的训练轮数

实际训练中通常会同时运行多个agent，它们使用相同参数的policy。PPO中对每条轨迹所产生的梯度贡献进行平均，作为全局的策略更新方向
