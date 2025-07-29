---
layout: post
title: "Reducing confabulation with semantic entropy steering vectors"
date: 2025-07-28 10:00:00 -0400
categories: [general]
mathjax: true
snippet: "A look at how semantic entropy can be used to detect and reduce confabulations in large language models."
---

### Background
“Detecting hallucinations in large language models using semantic entropy” looked at semantic entropy as a measurement/predictor of “confabulations”—a subset of hallucinations which involve incorrect claims are sensitive to random seeding.
Building upon the use of naive predictive entropy as a measurement of confabulation, where naive predictive entropy ($E$) is measured as

$$E = -\sum_y P(y|x) \log P(y|x),$$

where $x$ and $y$ are language sequences.
Semantic entropy ($SE$) notes that different $y$ may be semantically equivalent (e.g., the sequences “It is Paris” and “Paris” are equivalent despite not being identical).
Groups sequences into semantic equivalence classes, and measures the entropy over the classes (taking the class probability to be the sum of its members).
In practice, measuring semantic entropy requires sampling multiple answers (often 5 to 10) at a higher temperature.

### Experiments
For all of the below, we used llama-7b and the TriviaQA dataset (we further restricted to 500 examples in TriviaQA due to time constraints).
First, we compared the last-layer embeddings to the model to the semantic equivalence classes, as shown below for one example question (with 50 sampled answers).
Nicely, semantic classes correspond to clusters in the embedding space.
	
It is intuitive that a model would confabulate for questions which it has not seen as much in its training distribution.
To probe this relationship, I estimated the log-probability of the question and looked at its correlation to the semantic entropy of the answer
Unfortunately, there was little correlation between the two.
To reduce the computational overhead necessary for measuring semantic entropy while retaining its predictiveness for confabulations, others have tried training a linear probe on the embeddings to predict the semantic entropy.
I reproduced these results, which demonstrate limited effectiveness; they surpass other methods but do not match semantic entropy.
Finally, I experimented with steering the model using a “semantic entropy steering vector”, estimated as the difference between the embeddings corresponding to answers which exceeded a semantic entropy threshold and those that did not.
Concretely, we modified final-layer activations to be $a - \lambda v_{SE}$, where $a$ are the original activations, $\lambda$ a tunable magnitude parameter, and $v_{SE}$ the steering vector.
Interestingly, the model began to refuse to answer (simply outputting the stop token) as $\lambda$ increased.
It also seemed that answers became more nonsensical beyond a certain threshold of $\lambda$.
We measured the accuracy and refusal rate among the 500 examples, varying $\lambda \in [0:10]$; we noticed a “phase transition” in both accuracy and refusal rates, as shown below.

We introduced a third measure, which we call the “non-refused accuracy”, and measures the accuracy on questions that the model does not refuse to answer; we also plot this above.

### Future Directions
Extend to all 15k examples in the TriviaQA dataset
Measure semantic entropy for all examples
Estimate semantic entropy steering vector using all examples labelled with semantic entropy (also, decide between “classes” of semantic entropy in a more sophisticated
Extend steering vectors to other/multiple layers
Conduct a more thorough fine-tuning of the steering vector magnitude
