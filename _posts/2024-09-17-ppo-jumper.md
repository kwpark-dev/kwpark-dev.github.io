---
title: "[RL Exp.] PPO Jumper"
date: 2024-09-17
tags: [RL]
---

{% include mathjax.html %}

<p>{{ page.tags | join: ", #" | prepend: "#" }}</p>

## Introduction

Proximal Policy Optimization (PPO) is one of the major milestones in deep reinforcement learning and is widely used across diverse control tasks, such as robotics and game AI. Its key components—the clipped surrogate objective and Generalized Advantage Estimation (GAE)—are primarily designed to improve learning stability. The policy gradient (PG) theorem provides the theoretical motivation for these mechanisms.

In principle, the PG framework relies on several assumptions, including:

* the first-order Markov property of the (latent) state representation,
* unbiased estimation of value functions.

However, the objective derived from the PG theorem does not guarantee that the learned representations are truly Markovian, nor that the estimated value functions accurately capture full-trajectory returns. In practice, high variance in Monte Carlo–based advantage estimates often leads to unstable learning dynamics.

Leaving aside the representation learning problem, PPO focuses on mitigating this instability at the optimization level. Noting that the Monte Carlo advantage can be expressed as a sum of temporal-difference (TD) errors over the full trajectory, GAE introduces a smoothing factor $\lambda$ alongside the discount factor $\gamma$ to control the bias–variance trade-off of advantage estimates. This reduces variance while maintaining sufficient signal for policy improvement. In addition, PPO constrains policy updates through clipping, allowing only a predefined and conservative magnitude of change per update, which further stabilizes training.  


## Configurations

### Environment

Hopper-v5 task @mujoco where observation is 11 dim and action is 3 dim.

<p align="center">
  <img src="/assets/images/ppo/Hopper-v5_91.png" width="300">
  <br>
  <em> Jumping Hopper-v5   </em>
</p>

### Model

PPO agent consists of two separated actor and critic networks to avoid conflict in gradient pathology. Critic is slightly deeper than actor and none of them contains sequential units to build belief; those are nothing but a few stacks of FC layers with layer normalization.


### Framework

As explained earlier, GAE and surrogate objective (or clipped objective) are main ingredients of PPO. Comparing with discounted sum of reward, GAE introduces bias but mitigates high variance problem. 

$$
A_t^\text{GAE} = \sum_{k=0}^{T−t−1} (\gamma \lambda)^k \delta_{t+k}
$$

where $\delta_t = r_t + \gamma V(s_{t+1}) − V(s_t)$


## Results

PPO should figure out jumping strategy without unhealthy position. Following illustrates the final result that is alive until timeout. 

<p align="center">
  <img src="/assets/images/ppo/Hopper-v5_ppo.gif" width="300">
  <br>
  <em> PPO agent works well @Mujoco, Hopper-v5 task  </em>
</p>

The figures below describes learning curves of PPO agent. During the training, mean of value/return difference is misaligned with the crtic loss because even small changes of policy can lead critic's value mismatch.  

<p align="center">
  <img src="/assets/images/ppo/Hopper-v5_ppo_summary.png" width="700">
  <br>
  <em> The left panel shows discrepancy between critic loss and value/return difference. The right panel describes improving cumulative reward   </em>
</p>