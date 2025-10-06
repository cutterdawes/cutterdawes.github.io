---
layout: post
title: "Recurrent Networks Add Using a Helix"
date: 2025-10-03
mathjax: true
snippet: "The first part in a series applying interpretability methods to my bachelor's thesis."
tags: [Research]
---

In my bachelor's thesis at Princeton, we found that even simple neural networks can achieve radical generalization if they use the right symmetries, and that learnability is closely correlated with the symmetry used.
I'm proud of and very much enjoyed this project, enough that throughout last year I extended it (with my advisor and other collaborators) and submitted it for publication (currently under review; see [preprint](https://arxiv.org/pdf/2507.10678)).
Though it provides interesting insight into how symmetries enable generalization, it did not answer a fundamental question: what does it mean for a neural network to "use" a symmetry? That is, how do their internal representations and computation reflect that symmetry?

TAKE OUT?
In my thesis, we studied the case of base addition. ...
I formally proved why we can do this---which, remarkably, had previously only been done in the two-digit case.
In particular, I used group cohomology to construct the base representation of integers, a defining feature of which is the *carry function*: the transfer of the remainder, when a sum exceeds the base modulus, to the next place.


### 1. Preliminaries

#### 1.1. Base Addition with Different Carry Functions

A paradigmatic example of radical generalization through the use of symmetry is base addition.
However, it is non-trivial that we may represent integers as sequences of digits, and add them the way that we do: digit-by-digit, and carrying to the next place as we go.
A defining feature of this is the *carry function*, which specifies the procedure of carrying to the next place.

If you unpack the formalism underlying base addition---as I did in my thesis---this reveals a variety of alternative carry functions that produce different base representations *isomorphic* to the one we are all familiar with (i.e., identical up to relabeling).
Though mathematically equivalent, these carry functions vary enormously in structure.
A subset, the *Single Value* carry functions, always carry the same integer value (aside from 0); the paradigmatic example of these is the $\mathbf{1}$ carry function given by $\mathbb{1}_{n + m \geq b}$, where $n, m \in \mathbb{Z}_b$ (and $\mathbb{1}$ is the indicator function).


#### 1.2. Teaching Small Neural Networks to Add

[introduce...]



#### 1.3. Interpreting Addition in Neural Networks

[Subhash's paper, Neel's Modular Addition paper]



### 2. Experiments

#### 2.1. Helical Embeddings

[text]

<figure class="figure-100">
    <img src="{{ '/assets/blog/2025-10-03-addition-activations/embedding_helix_b5.png' | relative_url }}">
    <figcaption>[caption]</figcaption>
</figure>

<figure class="figure-75">
    <img src="{{ '/assets/blog/2025-10-03-addition-activations/embedding_helix_b3+4.png' | relative_url }}">
    <figcaption>[caption]</figcaption>
</figure>


#### 2.2. Helical Unembeddings

[text]

<figure class="figure-100">
    <img src="{{ '/assets/blog/2025-10-03-addition-activations/unembedding_helix_orders.png' | relative_url }}">
    <figcaption>[caption]</figcaption>
</figure>


#### 2.3 Reducing the Hidden Dimension

[text]

<figure class="figure-100">
    <img src="{{ '/assets/blog/2025-10-03-addition-activations/unembedding_helix_orders_dim.png' | relative_url }}">
    <figcaption>[caption]</figcaption>
</figure>

<figure class="figure-50">
    <img src="{{ '/assets/blog/2025-10-03-addition-activations/unembedding_helix_heatmap.png' | relative_url }}">
    <figcaption>[caption]</figcaption>
</figure>


### 3. Discussion

#### 3.1. RNNs Add Using a Helix (or a Few of Them)

[text, emphasize Neel Nanda's group representations paper -- difference here is that it has to readout mid group computation; also implies that the neural network is internally computing group isomorphisms i.e. 01234 -> 02413]


#### 3.2. Future Directions

[causal analysis (PCA baseline), better baselines, looking at intermediate activations, look at weight matrices]

[isolating the carry function eg by solving for a linear direction that best separates activations which will have non-zero carry vs those which do not]