---
title: "World Model for Navigation Task"
date: 2025-07-30
tags: [RL, Vision]
---

{% include mathjax.html %}

<p>{{ page.tags | join: ", #" | prepend: "#" }}</p>

## Introduction

The World Models framework is a model-based deep reinforcement learning approach that factorizes the agent into three components: a perceptual model V (Vision) that encodes high-dimensional observations into a latent representation, a dynamics model M (Memory) that predicts the evolution of this latent state over time, and a controller C (Control) that operates entirely within the learned latent space.
Crucially, the world model components are trained largely independently from the control policy, typically using data collected from the environment and optimized with unsupervised or self-supervised objectives, while the controller is trained on the learned latent dynamics without direct interaction with the real environment during policy optimization. In contrast, modern model-free reinforcement learning implicitly induces a latent state through function approximation and policy optimization, where the representation is shaped primarily by the policy gradient objective, $\nabla J = \mathbf{E} \left[ \nabla \log \pi \text{ } Q \right]$. Under this formulation, the optimal return objective places no explicit constraint on the quality or semantic structure of the learned representation, requiring only that it supports accurate value estimation and action selection in expectation over full trajectories. This raises a fundamental question: whether learning higher-quality latent representations—optimized for predictive structure and long-horizon dynamics rather than immediate policy gradients—can more effectively capture the geometry of optimal trajectories and thereby simplify or improve downstream control.


## Configurations

### Environment
The environment utilized by this model is [Miniworld](https://miniworld.farama.org/), specialized for simple navigation tasks. All environments offered from the miniworld are ego-centric, it might be closer to practical robot application. Following is summary of task to train an agent.

| Task | Goal | Modality |
|:------:|:--------------:|:---------:|
| sidewalk | navigation to a red box | vision |



### Model and Framework

Deep RL consists of deep learning models (policy, critic, depends on the approach) and RL framework. Frawork defines how the agent explores action trajectories in order to reach out goals. In case of WM, only controller model deals with optimal solution. Note that latent vector or trajectory modeling don't care of the optimal path; they are trained independently.

1. V model: Variational Auto-Encoder, U-net like skip connection
2. M model: MDN-RNN, with repulsion loss
3. C model: CMA-ES, with heuristic reward shaping

| Model | Parameters | Remark |
|:------:|:--------------:|:---------:|
| VAE    | 3,679,725      | U-net connection |
| MDN-RNN| 1,520,782      | repulsion loss   |
| CMA-ES | 1731           | reward shaping  |


**NOTE**: details of model architectures are different from the paper.

## Insights
1. Image Reconstruction vs Building Latent Space

The V model consists of a variational auto-encoder (VAE), which is optimized by two loss functions, reconstruction loss (MSE) and KL loss. In principle, VAE is a generative model that can sample new images. It figures out the core features of the environment (or training rollout) and learns how to reconstruct the seed image from the latent space. Balancing between reconstruction loss and KL loss is very significant to train VAE properly, because reconstructed images are easily collapsed when one dominates the others. Although the scale of MSE loss is larger than KL loss, KL often has a larger contribution to the gradient (well, it depends). Thus, a well-tuned warm-up or choosing $\beta < 1$ would be helpful for adjustment. If training fails, all reconstructed images show gray (global mean value of the dataset). Test shows that U-net connection improve the reconstruction capability dramatically by preserving high frequency texture.

<p align="center">
  <img src="/assets/images/world_model/ordinary.png" width="500">
  <br>
  <em> Output from ordinary VAE, in miniworld sidewalk task </em>
</p>

<p align="center">
  <img src="/assets/images/world_model/unet_based.png" width="500">
  <br>
  <em> Output from VAE adding U-net connection, in miniworld sidewalk task </em>
</p>


2. Mixture of Gaussian for Trajectories

Core idea beneath MDN-RNN is to model a sequence in terms of multiple Gaussian distribution by employing extracted features from RNN. It might need criteria to choose the number of mixtures and the authors of world model select 5 clusters. Following is an illustration for how mixture of Gaussian depicts trajectories. 

<p align="center">
  <img src="/assets/images/world_model/seven_cluster.png" width="500">
  <br>
  <em> Mixture of Gaussian models trajectories, N=7 </em>
</p>


3. Training Controller in Dream

The original reward shape of the sidewalk task was too sparse and had high variance. To force shorter steps to arrive at the goal, every action gets a penalty but $\gamma$ discounted. Only the initial sample $z_0$ and reward signals are collected from the real environment. From $z_1$ to $z_\text{done}$, the M model predicts the next, and the C model tries to figure out the optimal policy in the dream, like **imagination training**! In the sidewalk task, the agent should reach the goal position where the red voxel is located, but with as few steps as possible. 

<p align="center">
  <img src="/assets/images/world_model/sidewalk_task_wm.gif" width="500">
  <br>
  <em> Trained agent is heading for the red voxel. Only latent vectors are employed to train the policy, not real environment. The left panel is true miniworld whereas the right is dream. </em>
</p>

It is necessary to point out that the success of the task is highly dependent on the initial states; if the agent has never seen the target for the first time, it hardly does searching behavior. Besides, in case the agent gets lost the target while turning around, it cannot remember that there was a target before the action performed. 

<p align="center">
  <img src="/assets/images/world_model/sidewalk_failure.gif" width="500">
  <br>
  <em> Missing targets </em>
</p>

<p align="center">
  <img src="/assets/images/world_model/sidewalk_stuck.gif" width="500">
  <br>
  <em> No idea where to search </em>
</p>

Why this happens? At least there are two issues; poor credit assignment and lack of experience. This task is goal navigation, the agent should have noticed certain actions can cause going far away from the goal. No penaly, however, is applied on that behavior. Secondly, it never explore turning back solution when it stucks. The agent tries to exploit going-forward strategy in case the goal is fortunately located in front of itself.


## Results

### V model
1. Rank, Effective Rank & HSIC

It tells us linear, non-linear correlation of features. Though rank is nearly full, effective rank shows there are linearly redundant features. On the other hand, HSIC is acceptably low.  

<p align="center">
  <img src="/assets/images/world_model/rank.png" width="500">
  <br>
  <em> Comparison of rank & effective rank </em>
</p>

2. Loss Functions

It can indicate reconstruction collapse due to evolving two opposite direction, reconstruction and KL divergence. Should look at the gradient magnitude, not its scale alone in order to fine tuning. 


3. Gradients

Reading its norm and cosine similarity is quite tricky because distinctive learning paradigm prefers different evolution path. If there are multiple sources of the gradient like VAE, it would be helpful invetigating their contribution on the gradient, respectively.


4. Effective Sample Size (ESS)

Usefulness of batch samples.

### M model
1. Mixture Parameters

A plot shows mixture entropy and its values to elaborate if the distributions are not generated.
<p align="center">
  <img src="/assets/images/world_model/mixture.png" width="500">
  <br>
  <em> Estimated parameters for mixtures of Gaussian </em>
</p>

2. Loss Functions

Repulsion loss is added to mdn loss. 
<p align="center">
  <img src="/assets/images/world_model/loss.png" width="500">
  <br>
  <em> MDN loss and repulsion loss </em>
</p>

3. Gradients

It clearly shows that repulsion loss hinders degenerated states. Please compare this with loss function.
<p align="center">
  <img src="/assets/images/world_model/grad.png" width="500">
  <br>
  <em> Gradient norm and cosine similarity </em>
</p>


### C Model

1. Cumulative Reward & $\sigma ^2$

This figure shows that cumulative reward successfully arrives at (sub) optimum. As the vanila world model, it is a single layer controller without gradient tracking. But this result doesn't imply that the agent always notice where to go. Definitely it needs more indicators to evaluate the peroformance accordingly. 

<p align="center">
  <img src="/assets/images/world_model/train.png" width="500">
  <br>
  <em> Cumulative reward and variance evolution using CMA-ES </em>
</p>


## Discussion