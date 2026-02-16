---
title: "IMPALA Agent Playing Vizdoom"
date: 2025-11-28
tags: [RL, SF, Vision]
---

<p>{{ page.tags | join: ", #" | prepend: "#" }}</p>

## Introduction

IMPALA employs an asynchronous, distributed architecture for scalable reinforcement learning by decoupling experience collection from policy learning. By analogy to conventional supervised or unsupervised learning workflows, where data are first collected or prepared and then consumed by a learner in a separate training phase, many traditional reinforcement learning implementations couple environment interaction and learning in a staged, synchronous manner, leading to idle time between phases. IMPALA mitigates this inefficiency by running multiple actor processes asynchronously alongside a centralized learner. Actors continuously interact with the environment and stream trajectories to the learner, while the learner consumes available data immediately and updates the policy without waiting for all actors to complete rollouts. Updated policy parameters are periodically broadcast back to actors, enabling uninterrupted data collection and efficient utilization of computational resources, with stability maintained through off-policy correction mechanisms such as V-trace.


## Configurations

### Environment

Vizdoom provides the ego-centric game environment with multi-modalities. In this project, RGB, depth and audio profile are utilized to tackle deadly-corridor task; the agent figures out survival strategy to get closer to the vest located at the end of corridor. Followings are partially observed states from the player agent.

<p align="center">
  <img src="/assets/images/impala/vision.png" width="500">
  <br> 
  <em> RGB and depth information. Both are resized by 64x64. </em>
</p>

<p align="center">
  <img src="/assets/images/impala/audio.png" width="500">
  <br>
  <em> 2-channel audio profiles. The last 1024 elements are sampled for the training. </em>
</p>


### Model 

The implemented model has two heads comprising shallow policy and critic networks. The model backbone should enable the consideration of sequential correlation crossing over trajectories. This multi-modal sequential model, ViciousTwinMAMBA, employs a state space model block called MAMBA so that the multimodal agent learns a sequence of optimal behaviors. The figure below demonstrates the model scheme. 

<p align="center">
  <img src="/assets/images/impala/vicious_twin_mamba.png" width="300">
  <br>
  <em> Block diagram for ViciousTwinMAMBA </em>
</p>

Three organ networks represent feature extraction component of each domain. Convolution applied to vision and audio to build compressed features whereas discrete actions are embedded in larger dimension. Features from vision organ are gated by semantics, enemy or not, to provide recognizing capability. Final outputs from each organ network are concatenated so that MAMBA block figures out response feature, $y_0, ..., y_N$, where $N$ is length of a rollout fragment (in IMPALA perspective).   


### Framework

IMPALA is designed to achieve scalability of RL via multiple actors and a (or multiple) learner(s). Key strategy is asynchronous parallelization; do what they have to do independently. Actors keep sending data and the learner(s) learn policy continuously. Typical strategy of deep RL has a procedure of learning, the learner always wait until new rollouts completes or sample the data from large data buffer. But IMPALA separates them and running the entire process concurrently.   

<p align="center">
  <img src="/assets/images/impala/impala_core.png" width="300">
  <br>
  <em> Asynchronous actors and policy update </em>
</p>

To train the model with stability, IMPALA extends idea of GAE to off-policy version, which is called V-trace. In principle, full rollout advantage $A_{\text{MC}}$ is discounted sum of TD errors.

$$
A_t^{\text{MC}}
= \sum_{l=0}^{\infty} (\gamma)^l \delta_{t+l}
$$

GAE multiplies $\lambda$ to discount factor for smoothing values (so, $(\gamma \lambda)^l$). Conceptually, n-step TD is interpolation of MC and **GAE make n-step TD smoother.** On the other hand, V-trace takes care of policy lag via clipped importance sampling and redefines smooth n-step TD estimator in off-policy. 

$$
A_t^{\text{V-trace}}
= r_t + \gamma v_{t+1} - V(s_t).
$$


## Insights

### Recognition vs Optimization

The crucial role of reinforcement learning (RL) is to optimize action selection so that agents achieve specified goals. When local actions are densely aligned with long-horizon objectives, RL is highly effective. However, when agents must infer context-dependent sequential decisions from high-dimensional observations such as frame images, RL alone becomes extremely challenging.

Importantly, a well-behaved RL agent does not imply an understanding of contextual circumstances. In image-based environments, context typically requires identifying entities, such as objects in vision tasks or enemies in game environments. RL, however, primarily optimizes control: it learns how to move and act without explicitly recognizing entities. The agent adapts its behavior through action–reward correlations, but it does not inherently represent or understand what the targets are.

This limitation becomes more severe when optimal behavior requires a conjunction of multiple actions in rare or special situations. Such situations are seldom observed during training, causing the associated action sequences to be statistically underestimated. As a result, the agent tends to replay previously confident and frequent actions, even when they are irrelevant or insufficient for handling the required situation.


### Quick Diagnosis

In this project setup, two actors and one learner are working. The observed throughputs are followings. The table shows learner is bottleneck of entire training process. Note, it means learner hardly consumes collected data from the actor not actors are idle until the learner digests rest of the data. The learner sample the rollout fragments temporal deque, so while new policy is being updated old trajecroy fragments are removed.  

<div align="center">

| Worker | Throughput  | Resource |
|:------:|:-----------:|:--------:|
| Actor  | 200k ~ 400k | 2 CPU    | 
| Learner| 7k ~ 9k     | 1 GPU    |

</div>

A figure below also illustrates bottlneck of learner when batch size (b) and length size (t) are changed. 

<p align="center">
  <img src="/assets/images/impala/throughput.png" width="300">
  <br>
  <em> Measured throughput ratio of actors and a learner. Environment steps are normalized by batch and length size </em>
</p>

The observed gradient norm of a few elemental blocks of ViciousTwinMAMBA delivers shock at a certain point. It can drive instability of the learning process, so gradient clipping might be necessary. Furthermore, it is essential to check the gradient contribution of three losses, actor loss, critic loss, and entropy loss, in order to choose coefficients . If the gradients of each source are unbalanced, it leads to one loss dominating the others, whereby the agent has an ill-posed policy. 


<p align="center">
  <img src="/assets/images/impala/quick_obs.png" width="500">
  <br>
  <em> Gradient norm of element blocks and policy lag </em>
</p>


## Results

### Reward Shaping
The old reward is progress of position aligned with x-direction, $x_{t+1} - x_{t}$, which lead reward hacking. In deadly corridor task, there are 7 actions and moving forward has the strongest correlation with positive reward signal. As a result, estimated value of the forward is always beneficial even in irrelevant situation. New reward shape is designed to avoid such underestimation of the actions by adding travel distance and combat sign. It has transition term $r(s_t, a_t, s_{t+1})$ and factored potential of future states $\alpha \Phi (s_{t+1})$.

$$
r_d = r(s_t, a_t, s_{t+1}) + \alpha \Phi (s_{t+1})
$$

where $r = 0.5 \text{ } \Delta d -3 \text{ } \Delta ht + 10 \text{ } \Delta h + 1 \text{ } \Delta k$ and $\Phi = (x_{t+1} - x_{t})/z + D$. Here $d, ht, h, k, z, D$ indicate travel distance, hit taken count, hit count, kill count, zone scale and damage taken, respectively. 

### Training Techniques

Severe problem is pessimistic critic, it evaluates all actions negative. Followings are tried to resolve the issue. However, they just delay the when the problem occurs not fully remove root cause.

1. Critic warm-up: freeze actor for the first a few hundreds normalized environment step. 

2. Random perturbation: During critic warm-up, perturb logits before action sampling.

3. Logits lower bound clipping


### Final outputs & Indicators

Unfortunately, the IMPALA agent rarely identifies the optimal actions. The critic's pessimism weighs down all actions, and the actor must pick the best of a bad bunch. The challenge, as mentioned earlier, is the combination of actions. While the agent is travelling, it should keep watch for enemies. Once enemies are 'detected' via the semantic gate, the agent aligns with the enemy and fires. If the agent frequently observes alignment, shooting and a successful hit, these are replayed in the test stage, but that is a rare case. Thus, the critic evaluates that turning or shooting is totally useless. 

<p align="center">
  <img src="/assets/images/impala/deadly_corridor.gif" width="300">
  <br>
  <em> Failure mode of IMPALA agent. It hardly deals with situation that requires conjunction of alignment (looking at where enemy is located) & gun firing, which is rarely observed from the actors.  </em>
</p>

The agent doesn't care about enemies; no matter how the context is risky, it just passes thorough it. 

<p align="center">
  <img src="/assets/images/impala/log_p.png" width="300">
  <br>
  <em> Mean value of target & behavior log probability.  </em>
</p>

The figure above implies that entire action probability are pressed down. Even though specific actions are necessary for certain situations, if critic rarly sees it, the evaluation never be recovered. 

<p align="center">
  <img src="/assets/images/impala/grad.png" width="500">
  <br>
  <em> Gradient norm of actor head, critic head and MAMBA block.  </em>
</p>

Estimated gradient norms also support the same symptom. Before around 1000 of normalized environment step, actor is died; no learning signal from the actor loss. Again, if some actions (or conjunction of actions) require very special circumstances, critic would underestimates. Thus it needs additional audxiliaries such as skill learning. 

## Discussion