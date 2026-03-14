---
title: "[RFM & RL Int.] Robot Foundation Model Fundamental"
date: 2026-03-14
tags: [RFM]
---

{% include mathjax.html %}

<p>{{ page.tags | join: ", #" | prepend: "#" }}</p>

## Introduction
Robot Foundation Model (RFM) aim to integrate human-like intelligence into robotic systems so that robots can accomplish complex tasks through high-level instructions. Humans primarily rely on language when interacting with one another, as language serves as a major abstraction for conveying information. Likewise, if a machine can capture the context of conversations or textual instructions, it could follow commands expressed in natural language. Therefore, an RFM must holds at least two key capabilities: a language model to interact with humans and additional modalities that enable interaction with the physical task space.

The basic structure of a Robot Foundation Model (RFM) typically consists of a vision encoder, a large language model (LLM), and optional components responsible for action policies. While the exact architecture depends on design choices, most recently developed RFMs heavily rely on transformer-based components. This reliance extends even to the vision encoder, which is commonly implemented using Vision Transformers (ViTs). In the future, alternative sequential models such as state-space models (SSMs) may also be considered. Before discussing RFMs in greater detail, it is therefore important to develop a precise understanding of the transformer architecture.


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
Transformer defines context as attention mechanism that builds dependency structure of the embedding. But there are many types of dependency structure such as syntactic relation, long-distance refence, and subject-verb agreement. Idea of multi-head attetion (MHA) is building finer dependency structures in parallel thereby the model can capture more diverse token relationships. From the single attention, MHA with $h$ heads is defined as follows.

$$
\text{mha} (e) = \left[ \text{attn}_1 (e), ..., \text{attn}_h (e) \right]
$$




### Experiments



## Multimodal Transformer