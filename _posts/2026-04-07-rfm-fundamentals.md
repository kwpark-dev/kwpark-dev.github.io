---
title: "[RFM & RL Int.] Robot Foundation Model Fundamental"
date: 2026-04-07
tags: [RFM]
---

{% include mathjax.html %}

<p>{{ page.tags | join: ", #" | prepend: "#" }}</p>

## Introduction
Robot Foundation Model (RFM) aim to integrate human-like intelligence into robotic systems so that robots can accomplish complex tasks through high-level instructions. Humans primarily rely on language when interacting with one another, as language serves as a major abstraction for conveying information. Likewise, if a machine can capture the context of conversations or textual instructions, it could follow commands expressed in natural language. Therefore, an RFM must holds at least two key capabilities: a language model to interact with humans and additional modalities that enable interaction with the physical task space.

The structure of a RFM typically consists of a vision encoder, a large language model (LLM), and additional components responsible for action policies. It naturally aligns with perception - decision - action loop with system-level integration, which is a well-established paradigm for robot AI embodiment. We analyze where components in RFM are located in which layer of embodiment accordingly. 


## [Transformer](https://arxiv.org/abs/1706.03762)
Basically Transformer is sequence-specialized model *understanding context* of information. Then, what does context mean? Original Transformer defines context as attention mechanism that builds *dependency strucuture among the words*.

### Tokenization, Positional Encoding & Embedding
All languages have their own alphabets, words, and grammar structure. As a part of processiong, given sequences of words are tokenized and are embedded to $D$ dimensional space. Let's look at one simple example: 

```My old car kicked the bucket on the highway```

Here, define a token as a word separated by space (it is design choice). Then, tokens are following.

```My, old, car, kicked, the, bucket, on, the, highway```

To make them machine-readable, let's say they are embedded to $D=16$. Furthermore, order constraint should be given via positional encoding. Finally, each token is described in $D=16$ dimensional vector $e$ and the sentence $x$ now becomes $N \times D$ matrix. During the training, embedding vector learns semantics corresponding words. 

### Attention Mechanism
By tokenization & embedding, word vector can own semantics but we know that words and its meanings are not one-to-one mapping. Combination, position, relation with other words differentiate what sentences want to deliver. Above example doesn't imply the car crashes the bucket, but it is out of order. The separted semantics from ```kicked, the, bucket``` hardly captures the context of the sentence. Transformer tackles this via attention mechanism defined as 

$$
\text{attn} (q, k, v) = \text{softmax} \left( \frac{qk^T}{\sqrt{d}} \right) v
$$

where $q, k, v$ are linearly mapping of the embedding $e$ through $eW_q, eW_k, eW_v$, respectively (note that e is matrix). Why? What is intuition or principle behind it? Replace $q, k$ to linear mapping of $e$ and let $v$ be the arbitrary function of $e$. Now the formulation above becomes

$$
\text{attn} (e) = \phi (e M e^T) f(e)
$$

where $M = W_q W_k ^T$. Inside function $\phi$, let $M$ be identity matrix. Then, elements of $E = ee^T$ are inner product of each row, which corresponds to token. In other word, the matrix measures similarity! Now considering matrix $M = W_q W_k ^T$, it adjusts similarity from learning signals. In short, a role of $M$ is similarity metric of the embedding and function $\phi$ provides complexity for better context capacity. With non-linear similarity, multiplying embedding alone implies re-mix semantics-nothing smoothing operator. If similarity relevant information can be extracted from the embedding, attention would become more reliable. Thus, $f(e)$ should determine message representation.          


### Multi-head Attention
Transformer defines context as attention mechanism that builds dependency structure of the embedding. But there are many types of dependency structure such as syntactic relation, long-distance refence, and subject-verb agreement. Idea of multi-head attetion (MHA) is building finer dependency structures in parallel thereby the model can capture more diverse token relationships. From the single attention, MHA with $h$ heads is defined as

$$
\text{mha} (e) = \psi \left( \left[ \text{attn}_1 (e), ..., \text{attn}_h (e) \right] \right)
$$

where $\psi(\cdot)$ is linear operation.


### Cross Attention
Decoding layer in Transformer includes cross attention that entangles projections of input and output sequences. Transformer predicts one-step forward sequence so, it tries to extract selective information according to query. Such information matching capability is trained by cross attention where it comprises query ($q_{\text{output}}$) from the output, one-step shifted masked input, and sources ($k_{\text{input}}, v_{\text{input}}$).


## Embodiment Layers
Robotic agent performing manufacturing task should comprise perception - decision - action loop with system - level integration. Core components of each layer are as follow.

**Perception**

* State estimation under partial observability
* Representation grounding
* Perception robustness

**Decision**

* Task decomposition & structure based on learned representation
* Closed-loop generalization (robustness for messy environment)
* Planning or reactive policy
* Efficient skill acquisition & reuse

**Action**

* Contact-rich, precise control under uncertainties
* Real-time responsiveness

**System Integration**

* Reliability: consistent, low variance execution
* Robustness: works across distribution shift
* Anomaly detection & discovery

Now let's look at RFM. Vision-Language-Action (VLA) models finetune the Vision-Language Model (VLM) by adding action module so that VLM produces action grounded representation as stated in perception layer. VLA heavily relies on human's demo data to learn implicit sequences in demonstration as imitation learning. In short, 

* Perception: Highly dependent on pretrained VLM performance. Semantic representation is mapped to control-relevant feature via finetuning.
* Decision & Action: Trained under the demo data. Low-level control and decision-making are tightly coupled. 


## Decision & Action Modules

VLM is typically trained under the multimodal transformer but it depends on the system or architecture that engineers prefer. Just leave behind perception layer for a while, what are additional strategies that are emerging for the action capabilities? 


### Diffusion Policy
Diffusion policy learns a distribution over short horizon action chunks instead of individual actions. While it can produce full trajectory, distribution shift may hinder task accomplishment. The output of diffusion policy might be resemble with hierarchical RL but the core idea is distinguishable. It is based on [diffusion model](https://arxiv.org/abs/1503.03585), which is unsupervised generative model inspired by statistical thermodynamics. Dropping an ink droplet on water surface, the droplet becomes faded as ink particles spread out via Brownian motion. Diffusion model mimics this physical process by adding learnable recovery procedure to generate a new distribution. Forward process gradually diffuses the original data distribution $q(x_0)$ utilizing pre-defined Markovian kernel $q(x_t \mid x_{t-1})$ to reach out identity covariance Gaussian over $T$ as $q(x_T) = \mathcal{N} (0, I)$. Reverse process, on the other hand, it learns how to recover the original data from the Gaussian with identity covariance $p_{\theta}(x_0) = q(x_0)$. Diffusion model follows maximum likelihood including forward and reverse transition as latent variable. Expected log likelihood $\log p_\theta (x_0)$ over $q(x_0)$ is  

$$
\begin{aligned}
L &= \mathbf{E}_{x_0 \sim q(x_0)} \left[ \log p_\theta (x_0) \right] \\
&= \int dx_0 q(x_0) \log p_\theta (x_0)
\end{aligned}
$$

Introducing transitions $x_1, ..., x_T$ as the latent variables, the log likelihood inside expectation becomes 

$$
\begin{aligned}
\log p_\theta (x_0) &= \log \int dx_{1, ..., T} p_\theta (x_{0, ..., T}) \\
  &= \log \int dx_{1, ..., T} p_\theta (x_{0, ..., T}) \frac{q(x_{1, ..., T} \mid x_0)}{q(x_{1, ..., T} \mid x_0)} \\
  &= \log \int dx_{1, ..., T} q(x_{1, ..., T} \mid x_0) \frac{p_\theta (x_{0, ..., T})}{q(x_{1, ..., T} \mid x_0)} \\
  &= \log \int dx_{1, ..., T} q(x_{1, ..., T} \mid x_0) p_\theta (x_{T}) \prod _1 ^T \frac{p_\theta (x_{t-1} \mid x_t)}{q (x_{t} \mid x_{t-1})} \\
  &= \log \mathbf{E}_{x_{1, ..., T} ~ q(x_{1, ..., T} \mid x_0)} \left[ p_\theta (x_{T}) \prod _1 ^T \frac{p_\theta (x_{t-1} \mid x_t)}{q (x_{t} \mid x_{t-1})} \right]  
\end{aligned}
$$

Applying Jensen's inequality and a few clean-up calcuation combining with expectation over $q(x_0)$, the lower bound of $L$ becomes

$$
L \geq \mathbf{E}_{q(x_{0, ..., T})} \left[ \log p_\theta (x_0 \mid x_1) -\sum _{t=2} ^T \text{KL} (q(x_{t-1} \mid x_t, x_0) || p_\theta (x_{t-1} \mid x_t)) - \text{KL}(q(x_{T} \mid x_0) || p_\theta (x_{T})) \right]
$$

which is called ELBO (Evidence Lower Bound)

### Flow Matching


### Action Expert