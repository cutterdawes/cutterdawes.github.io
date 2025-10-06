---
layout: post
title: "Recurrent Networks Add Using a Helix"
date: 2025-10-03
mathjax: true
snippet: "The first part in a series applying interpretability methods to my bachelor's thesis."
tags: [Research]
---

My bachelor's thesis at Princeton was an investigation of neural networks learning base addition---one of the simplest and most fundamental examples of symmetry.
We found that, if taught to add using the right symmetries, even simple neural networks can achieve radical generalization, and that learnability is closely correlated with the symmetry used.
I'm proud of and much enjoyed this project, enough that this past year my collaborators and I refactored it into a paper which we submitted for publication (currently under review; see [preprint](https://arxiv.org/pdf/2507.10678)).
Though our paper provides interesting insight into how symmetries enable generalization, it did not answer a fundamental question: what does it mean for a neural network to "use a symmetry"; i.e., how is the symmetry represented by the network internally?


### 1. Preliminaries

#### 1.1. Training Recurrent Networks on Base Addition

For base $b$, the problem of base addition is formulated as follows:
Two multi-digit numbers are constructed as $n = (n_k, ..., n_1)$ and $m = (m_k, ..., m_1)$, composed of lists of $b$-dimensional one-hot-encoded digits ($n_i$ and $m_i$) ordered from most ($k$) to least (1) significant.
As input, one digit from each number is presented, in this case from least to most significant, with special tokens denoting the (heldout) digits of the answer sequence $s = (s_k, ..., s_1)$.
That is, the input sequence input $X$ is

$$
X = (n_1, m_1, *, ..., n_k, m_k, *).
$$

The model is comprised of a single-layer RNN (viz., Elman Net) with input and hidden dimensions of $b$ and $4b$, respectively, followed by a linear layer with input and output dimensions of $4b$ and $b$, respectively.
For each base $b \in [3:10]$, the model is trained to add 3-digit numbers, using the cross-entropy between the output logits and the answer sequence as the loss.
Training progressed over 5,000 epochs over all 3-digit tuples ($n, m$), with a batch size of 32 and a learning rate of 0.005 using the Adam optimizer.
After training, the models were evaluated on a subset of 6-digit numbers, achieving $\sim 1.0$ accuracy for each base.


#### 1.2. Helix Fitting Procedure

Denote the RNN embeddings by $X_e \in \mathbb{R}^{b \times 4b}$ (note that these are given by the learned embedding matrix that linearly encodes the one-hot digits to the hidden space).
As helix template, construct the basis $H \in \mathbb{R}^{b \times 3}$ with rows $[\cos(t_k), \sin(t_k), t_k]$, where $t_k = 2 \pi k / b$ for $k \in [b]$.
The helical basis is fitted to the embeddings with least-squares, finding $B \in \mathbb{R}^{3 \times 4b}$ such that $X_e \approx H B$.
An identical procedure is followed for the unembeddings $X_u \in \mathbb{R}^{b \times 4b}$ (these are given by the learned unembedding matrix that linearly decodes the logits from the hidden space).


### 2. Experiments

#### 2.1. Helical Embeddings and Unembeddings

We follow the training and fitting procedure described in Section 1 for bases 3 to 10.
First, the figure below shows the helix fit for base 10---note the high quality of the fit, with with $\text{RMSE} = 0.214$ and $R^2 = 0.883$.
The other helical fits are shown below; though they vary in quality, most of them are fairly convincing.

<figure class="figure-100">
    <img src="{{ '/assets/blog/2025-10-03-addition-activations/helix_embeddings_b10.png' | relative_url }}">
    <figcaption>Best-fit helix for the embeddings of an RNN trained on addition in base 10. The helix provides a good fit, with $\text{RMSE} = 0.214$ and $R^2 = 0.883$.</figcaption>
</figure>

<figure class="figure-100">
    <img src="{{ '/assets/blog/2025-10-03-addition-activations/helix_grid_b3-10.png' | relative_url }}">
    <figcaption>Best-fit helices for the embeddings (top) and unembeddings (bottom) in bases 3-10.</figcaption>
</figure>


#### 2.2. Quality of Helix Fits

To quantitatively assess the quality of the helix fits, we compare to two baselines: (i) the first 3 principle components, which is a strong baseline given that the helix is a 1-dimensional manifold; and (ii) helix fits using random orderings (e.g., 0, 2, 1 for base 3), which probes the expressivity of the helix regardless if the ordering of digits along it is relevant to the network's computation.

<figure class="figure-100">
    <img src="{{ '/assets/blog/2025-10-03-addition-activations/r2_analysis.png' | relative_url }}">
    <figcaption>$R^2$ statistics for the best-fit helices, compared to baselines of 3D PCA and helices with random orderings (e.g., 0, 2, 1 for base 3).</figcaption>
</figure>


### 3. Discussion

#### 3.1. RNNs Add Using a Helix

From the helix figures as well as the $R^2$ analysis, we can see that---for the most part---the helices provide an impressively good fit to not only the embeddings, but also the unembeddings.
Together, this provides suggestive evidence that helical structure is important from beginning to end in the model's computation.
This mirrors recent work finding that LLMs represent the numbers 0 through 100 in a helix ([Kantamneni et al., 2025](https://arxiv.org/pdf/2502.00873)).


#### 3.2. Future Directions

There are a variety of future directions from this very preliminary first study, which I intend to follow-up on in the posts to come.

1. **Extend to other carry functions:**
In my thesis, I use group cohomology to formally construct base addition, and in the process reveal many alternative ways of carrying.
This provides the setup for the rest of the paper, which analyzes inductive biases of neural networks in terms of the complexity of the different carry functions.
In this post, I just focus on the standard carry, but I intend to replicate this for the other carry functions.
If they too display helices according to their particular systems, this provides further evidence that the helices are causally relevant to computation.

2. **Causal analysis:**
Going beyond representation to causation is an important step in interpretability, and it would be interesting to do so here as well---activation patching would be a first-step to look into.

<!-- 3. **Interpret the weights:**
sdfas

[isolating the carry function eg by solving for a linear direction that best separates activations which will have non-zero carry vs those which do not] -->