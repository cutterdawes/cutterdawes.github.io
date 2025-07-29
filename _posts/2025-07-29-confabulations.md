---
layout: post
title: "Reducing Confabulation with Semantic Entropy Steering Vectors"
date: 2025-07-28 10:00:00 -0400
categories: [general]
mathjax: true
snippet: "A brief look at how semantic entropy can be used to detect and reduce confabulations in LLMs."
---

Despite their remarkable fluency, large language models can produce confident yet incorrect outputs—so-called "confabulations".
In this post, I explore how semantic entropy, a metric that captures uncertainty over meaning rather than form, can both detect and help reduce such failures through targeted steering interventions.

### Background

The paper [*Detecting hallucinations in large language models using semantic entropy* (Nature, 2024)](https://www.nature.com/articles/s41586-024-07421-0) introduced semantic entropy as a measurement and predictor of *confabulations*—a subset of hallucinations that involve false statements which are sensitive to random seed variations.
This builds upon the use of naive *predictive entropy* ($PE$), defined as:

$$
PE = -\sum_y P(y|x) \log P(y|x),
$$

where $x$ and $y$ are language sequences. While naive entropy treats each $y$ as a distinct outcome, *semantic entropy* ($SE$) accounts for the fact that different sequences may express the same underlying meaning—for instance, “It is Paris” and “Paris” are semantically equivalent. That is, $SE$ clusters outputs ($y_n$) into semantic equivalence classes ($C_k$) and then measures entropy across these classes, summing the probabilities of sequences within each class.
Then,

$$
\begin{align*}
SE &= -\sum_{C_k} P(C_k|x) \log P(C_k|x)
\\ &= -\sum_{C_k}  \left( \sum_{y_n \in C_k} P(y_n|x) \right) \log \left( \sum_{y_n \in C_k} P(y_n|x) \right).
\end{align*}
$$

In practice, estimating semantic entropy requires generating multiple model outputs (typically 5 to 10) at higher temperature settings to capture meaningful variation in responses.

### Experiments

All experiments were conducted using LLaMA-7B on a subset of 500 examples from the TriviaQA dataset (reduced for time constraints).

To begin, I examined whether semantic equivalence classes corresponded to structure in the model’s embedding space. For each input question, I sampled 50 answers and visualized their last-layer embeddings. As shown below, sequences grouped into the same semantic class tended to cluster closely in the embedding space.

<figure class="figure-75">
    <img src="{{ '/assets/blog/2025-07-28-confabulations/semantic_embeddings.png' | relative_url }}">
    <figcaption>UMAP projection of embeddings labelled by semantic equivalence class.</figcaption>
</figure>

Intuitively, one might expect the model to be more prone to confabulation on inputs that are underrepresented in its training distribution. To test this, I estimated the log-probability of each question and examined its correlation with the semantic entropy of the corresponding answers. Surprisingly, I found little to no correlation between the two.

<figure class="figure-75">
    <img src="{{ '/assets/blog/2025-07-28-confabulations/question_entropy_correlation.png' | relative_url }}">
    <figcaption>Correlation between question log-probability and answer semantic entropy.</figcaption>
</figure>

Given that semantic entropy is expensive to compute, prior work has proposed training a linear probe to predict it directly from internal embeddings. I replicated this approach and found that while such probes outperform baseline heuristics, their predictive power still falls short of using true semantic entropy.

As a further experiment, I explored whether semantic entropy could be used to steer the model away from confabulations. I constructed a *semantic entropy steering vector* by subtracting the mean embedding of low-entropy outputs from that of high-entropy ones. The model’s final-layer activations were then adjusted via:

$$
a' = a - \lambda v_{SE},
$$

where $a$ are the original activations, $\lambda$ is a tunable scaling factor, and $v_{SE}$ is the steering vector. As $\lambda$ increased, the model began to refuse to answer (producing a stop token), and beyond a certain point, its outputs became increasingly incoherent. Measuring accuracy and refusal rate across the 500-example subset revealed a kind of “phase transition” in behavior as $\lambda$ varied.

<figure class="figure-100">
    <img src="{{ '/assets/blog/2025-07-28-confabulations/acc_refusal_with_steering.png' | relative_url }}">
    <figcaption>Accuracy and refusal rate when steered by the semantic entropy vector.</figcaption>
</figure>

### Future Directions

Several avenues remain for deeper investigation:

- Expand the experiments to the full 15,000 examples in the TriviaQA dataset.
- Compute and log semantic entropy for all examples to enable a broader statistical analysis.
- Re-estimate the steering vector using the full labeled dataset and explore more nuanced ways to group responses into semantic classes.
- Experiment with applying the steering vector at other layers (or across multiple layers) of the model.
- Conduct a more fine-grained search over the magnitude parameter $\lambda$ to better characterize the transition dynamics and optimal operating points.

### References

1. Kadavath, S., Conerly, T., Tran-Johnson, M., et al. (2024). [*Detecting hallucinations in large language models using semantic entropy.*](https://www.nature.com/articles/s41586-024-07421-0) *Nature*. https://doi.org/10.1038/s41586-024-07421-0