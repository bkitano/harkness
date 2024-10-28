## Effects of Parameter Norm Growth During Transformer Training: Inductive Bias from Gradient Descent

### William Merrill[∗†] Vivek Ramanujan[∗‡] Yoav Goldberg[∗§] Roy Schwartz[¶] Noah A. Smith[∗‡]

_∗_ Allen Institute for AI _† New York University_ _‡ University of Washington_
_§ Bar Ilan University_ _¶Hebrew University of Jerusalem_
### willm@nyu.edu ramanv@cs.washington.edu {yoavg,noah}@allenai.org roys@cs.huji.ac.il


### Abstract


The capacity of neural networks like the
widely adopted transformer is known to be
very high. Evidence is emerging that they
learn successfully due to inductive bias in the
training routine, typically a variant of gradient descent (GD). To better understand this
bias, we study the tendency for transformer parameters to grow in magnitude (ℓ2 norm) during training, and its implications for the emergent representations within self attention layers. Empirically, we document norm growth
in the training of transformer language models,
including T5 during its pretraining. As the parameters grow in magnitude, we prove that the
network approximates a discretized network
with saturated activation functions. Such “saturated” networks are known to have a reduced
capacity compared to the full network family
that can be described in terms of formal languages and automata. Our results suggest saturation is a new characterization of an inductive
bias implicit in GD of particular interest for
NLP. We leverage the emergent discrete structure in a saturated transformer to analyze the
role of different attention heads, finding that
some focus locally on a small number of positions, while other heads compute global averages, allowing counting. We believe understanding the interplay between these two capabilities may shed further light on the structure
of computation within large transformers.

### 1 Introduction


Transformer-based models (Vaswani et al., 2017)
like BERT (Devlin et al., 2019), XLNet (Yang et al.,
2019), RoBERTa (Liu et al., 2019), and T5 (Raffel
et al., 2019) have pushed the state of the art on an
impressive array of NLP tasks. Overparameterized
transformers are known to be universal approximators (Yun et al., 2020), suggesting their generalization performance ought to rely on useful biases
or constraints imposed by the learning algorithm.
Despite various attempts to study these biases in


transformers (Rogers et al., 2020; Lovering et al.,
2021), it remains an interesting open question what
they are, or even how to characterize them in a way
relevant to the domain of language.
In this work, we take the perspective that thoroughly understanding the dynamics of gradient descent (GD) might clarify the linguistic biases of
transformers, and the types of representations they
acquire. We start by making a potentially surprising
empirical observation (§3): the parameter ℓ2 norm
_√_
grows proportional to _t (where t is the timestep)_

during the training of T5 (Raffel et al., 2019) and
other transformers. We refer to the phenomenon of
growing parameter norm during training as norm
_growth. Previous work has analyzed norm growth_
in simplified classes of feedforward networks (Li
and Arora, 2019; Ji and Telgarsky, 2020), but, to
our knowledge, it has not been thoroughly demonstrated or studied in the more complicated and practical setting of transformers.
Our main contribution is analyzing the effect
of norm growth on the representations within the
transformer (§4), which control the network’s grammatical generalization. With some light assumptions, we prove that any network where the parameter norm diverges during training approaches a sat_urated network (Merrill et al., 2020): a restricted_
network variant whose discretized representations
are understandable in terms of formal languages
and automata. Empirically, we find that internal
representations of pretrained transformers approximate their saturated counterparts, but for randomly
initialized transformers, they do not. This suggests
that the norm growth implicit in training guides
transformers to approximate saturated networks,
justifying studying the latter (Merrill, 2019) as a
way to analyze the linguistic biases of NLP architectures and the structure of their representations.
Past work (Merrill, 2019; Bhattamishra et al.,

2020) reveals that saturation permits two useful
types of attention heads within a transformer: one


-----

that locally targets a small number of positions, and
one that attends uniformly over the full sequence,
enabling an “average” operation. Empirically, we
find that both of these head types emerge in trained
transformer language models. These capabilities
reveal how the transformer can process various
formal languages, and could also suggest how it
might represent the structure of natural language.
Combined, our theoretical and empirical results
shed light on the linguistic inductive biases imbued
in the transformer architecture by GD, and could
serve as a tool to analyze transformers, visualize
them, and improve their performance.
Finally, we discuss potential causes of norm
growth in §5. We prove transformers are approximately homogeneous (Ji and Telgarsky, 2020), a
property that has been extensively studied in deep
learning theory. With some simplifying assumptions, we then show how homogeneity might ex_√_
plain the _t growth observed for T5.[1]_

### 2 Background and Related Work

**2.1** **GD and Deep Learning Theory**

A simple case where deep learning theory has studied the generalization properties of GD is matrix
factorization (Gunasekar et al., 2017; Arora et al.,
2019; Razin and Cohen, 2020). It has been observed that deep matrix factorization leads to lowrank matrix solutions. Razin and Cohen (2020)
argued theoretically that this bias of GD cannot
be explained as an implicit regularizer minimizing
some norm. Rather, they construct cases where all
parameter norms diverge during GD.
Similar ideas have emerged in recent works
studying feedforward networks. Analyzing biasless ReLU networks with cross-entropy loss, Poggio et al. (2019, 2020) show that the magnitude
(ℓ2 norm) of the parameter vector continues to
grow during GD, while its direction converges.
Li and Arora (2019) present a similar argument
for scale-invariant networks, meaning that scaling
the parameters by a constant does not change the
output. Studying homogeneous networks, Ji and
Telgarsky (2020) show that the gradients become
_aligned as t_, meaning that their direction
_→∞_
converges to the parameter direction. This means
the norm will grow monotonically with t. The
perspective developed by these works challenges
the once conventional wisdom that the parameters

1Code available at [https://github.com/](https://github.com/viking-sudo-rm/norm-growth)
[viking-sudo-rm/norm-growth.](https://github.com/viking-sudo-rm/norm-growth)


converge to a finite local minimum during GD training. Rather, it suggests that GD follows a normincreasing trajectory along which network behavior
stabilizes. These analyses motivate investigation of
this trajectory-driven perspective of training.
From a statistical perspective, work in this vein
has considered the implications of these training
dynamics for margin maximization (Poggio et al.,
2019; Nacson et al., 2019; Lyu and Li, 2019).
While these works vary in the networks they consider and their assumptions, they reach similar conclusions: GD follows trajectories diverging in the
direction of a max-margin solution. As margin
maximization produces a simple decision boundary,
this property suggests better generalization than an
arbitrary solution with low training loss. This point
of view partially explains why growing norm is
associated with better generalization performance.

**2.2** **NLP and Formal Language Theory**

Norm growth has another interpretation for NLP
models. Past work characterizes the capacity of
infinite-norm networks in terms of formal languages and automata theory. Merrill (2019) and
Merrill et al. (2020) propose saturation, a framework for theoretical analysis of the capacity of NLP
architectures. A network is analyzed by assuming
it saturates its nonlinearities, which means replacing functions like σ and tanh with step functions.
This is equivalent to the following definition:

**Definition 1 (Saturation; Merrill et al., 2020) Let**
_f_ (x; θ) be a neural network with inputs x and
weights θ. The saturated network sf (x; θ) is[2]

sf (x; θ) = lim
_c→∞_ _[f]_ [(][x][;][ cθ][)][,]

where the limit exists, and undefined elsewhere.

Saturation reduces continuous neural networks
to discrete computational models resembling automata or circuits, making some kinds of formal
linguistic analysis easier. For many common architectures, the saturated capacity is known to be significantly weaker than the full capacity of the network with rational-valued weights (Merrill, 2019),
which, classically, is Turing-complete for even simple RNNs (Siegelmann and Sontag, 1992).
For example, one can hand-construct an RNN
or LSTM encoding a stack in its recurrent memory
(Kirov and Frank, 2012). Stacks are useful for processing compositional structure in linguistic data

2The limit over f is taken pointwise. The range of sf is R.


-----

(Chomsky, 1956), e.g., for semantic parsing. However, a saturated LSTM does not have enough memory to simulate a stack (Merrill, 2019). Rather, saturated LSTMs resemble classical counter machines
(Merrill, 2019): automata limited in their ability to
model hierarchical structure (Merrill, 2020). Experiments suggest that LSTMs trained on synthetic
tasks learn to implement counter memory (Weiss
et al., 2018; Suzgun et al., 2019a), and that they
fail on tasks requiring stacks and other deeper models of structure (Suzgun et al., 2019b). Similarly,
Shibata et al. (2020) found that LSTM language
models trained on natural language data acquire
saturated representations approximating counters.
Recent work extends saturation analysis to transformers (Merrill, 2019; Merrill et al., 2020). Saturated attention heads reduce to generalized hard
attention, where the attention scores can tie. In the
case of ties, the head output averages the positions
with maximal scores.[3] While their power is not
fully understood, saturated transformers can implement a counting mechanism similarly to LSTMs
(Merrill et al., 2020). In practice, Bhattamishra
et al. (2020) show transformers can learn tasks requiring counting, and that they struggle when more
complicated structural representations are required.
Ebrahimi et al. (2020) find that attention patterns of
certain heads can emulate bounded stacks, but that
this ability falls off sharply for longer sequences.
Thus, the abilities of trained LSTMs and transformers appear to be predicted by the classes of
problems solvable by their saturated counterparts.
Merrill et al. (2020) conjecture that the saturated
capacity might represent a class of tasks implicitly
learnable by GD, but it is unclear a priori why this
should be the case. This work aims to put this conjecture on more solid theoretical footing: we argue
that approximate saturation arises in transformers
as a result of norm growth during training.[4]

### 3 Norm Growth in Transformers

We start with the observation that the parameter
_ℓ2 norm grows during training for practical trans-_
former language models. We first consider the
parameter norm of 104 historical checkpoints from
T5-base (Raffel et al., 2019) pretraining, a 220M

3Hahn (2020) identified weaknesses of strictly hard attention, which is weaker than saturated attention.
4This relates to Correia et al. (2019), who modify the transformer to facilitate approximately sparse attention. In contrast,
we will show that approximate sparsity (i.e., saturation) arises
implicitly in standard transformers.


parameter model, which was trained using the
AdaFactor optimizer (Shazeer and Stern, 2018).
Further details are in §A.

_√_
Fig. 1 shows that the T5 norm follows a _t trend,_

where t is time in training steps. The top right of
Fig. 1 breaks down the growth trend by layer. Generally, the norm grows more quickly in later layers
than in earlier ones, although always at a rate pro_√_
portional to _t.[5]_ Next, in the bottom row of Fig. 1,

we plot the cosine similarity between each parameter checkpoint θt+1 and its predecessor θt. This
rapidly approaches 1, suggesting the “direction” of
the parameters (θt/∥θt∥) converges. The trend in
directional convergence looks similar across layers.
We also train smaller transformer language models with 38M parameters on Wikitext-2 (Merity
et al., 2016) and the Penn Treebank (PTB; Marcus
et al., 1993). We consider two variants of the transformer: pre-norm and post-norm, which vary in the
relative order of layer normalization and residual
connections (cf. Xiong et al., 2020). Every model
exhibits norm growth over training.[6]

Combined, these results provide evidence that
the parameter norm of transformers tends to grow
over the course of training. In the remainder of
this paper, we will discuss the implications of this
phenomenon for the linguistic biases of transformers, and then discuss potential causes of the trend
rooted in the optimization dynamics.

### 4 Effect of Norm Growth

§3 empirically documented that the parameter
_√_
norm grows proportional to _t during T5 pretrain-_

ing. Now, we move to the main contribution of our
paper: the implications of norm growth for understanding transformers’ linguistic inductive biases.
In particular, Prop. 1 says uniform norm growth
across the network guides GD towards saturated
networks. Thus, saturation is not just a useful approximation for analyzing networks, but a state
induced by training with enough time.

**Proposition 1 (Informal) Let θt ∈** R[n] _be parame-_
_ters at step t for f_ (x; θt). If every scalar parameter
_θt[i]_ _[diverges at the same rate up to a constant, then]_
_f converges pointwise to a saturated network._

5We encourage future works that pretrain new transformer
language models to track metrics around norm growth.
6Erratum: In a previous paper version, this footnote reported perplexity numbers that we found to be irreproducible,
as they were likely obtained with a non-standard truncated
version of the PTB dataset. We have thus removed them.


-----

Figure 1: Top: Norm growth during T5 pretraining, with a coefficient r[2] = 1.00. The right is broken down by
layer. Bottom: cosine similarity between subsequent parameter checkpoints.


The proof is in §B. Prop. 1 assumes not just
norm growth, but uniform norm growth, meaning no parameter can asymptotically dominate any
other. Notably, uniform growth implies directional
convergence. Accepting uniform growth for a
given training regimen, we expect transformers to
converge to saturated networks with infinite training. Based on §3, the T5 norm appears to grow
_√_

_t uniformly across the network, suggesting_

_∝_

the uniform growth condition is reasonable. As we
will discuss later in §5, we expect the growth trend
to depend heavily on the learning rate schedule.

**4.1** **Saturated Transformers**

Having established that norm growth should lead
to saturation, we now empirically measure the saturation levels in T5 and other transformer models.

**Large** **transformers** **are** **highly** **saturated.**
Since ∥θt∥ empirically grows during training,
we expect high cosine similarity between the
representations in trained networks and saturated
representations. We estimate this as the cosine
similarity between f (x; θ) and f (x; cθ) for some


large c (in practice, 1,000). We consider the “base”
versions of pretrained BERT, RoBERTa, T5, and
XLNet (pretrained on masked language modeling),
and compute the mean saturation over 100 input
sentences from the Brown corpus (Francis and
Kuˇcera, 1989). To match standard practice, each
sentence is truncated at 512 word pieces. Fig. 2
plots the similarity for each layer of each model.
We compare the pretrained transformers against
a randomly initialized baseline. For every model
type, the similarity is higher for the pretrained
network than the randomly initialized network,
which, except for T5, is 0. For T5 and XLNet,
_∼_
the similarity in the final layer is 0.9, whereas,
_≥_
for RoBERTa, the final similarity is 0.65 (although
0.94 in the penultimate layer). For T5 and XLNet,
similarity is higher in later layers, which is
potentially surprising, as one might expect error
to compound with more layers. This may relate
to the fact that the norm grows faster for later
layers in T5. One question is why the similarity for
BERT is lower than these models. As RoBERTa
is architecturally similar to BERT besides longer


-----

Figure 2: Cosine similarities of the unsaturated and saturated (c = 1,000) transformer representations, by layer.
We compare randomly initialized transformers (left) to pretrained ones (right).


training, we hypothesize that RoBERTa’s higher
similarity is due to longer pretraining.

**Small** **transformers** **reach** **full** **saturation.**
Each of the transformers trained on Wikitext-2
and PTB reached a saturation level of 1.00. It
is unclear why these models saturate more fully
than the pretrained ones, although it might be
because they are smaller.[7] For our LMs, the
feedforward width (512) is less than for T5-base,
while the encoder depth and width are the same.
Other possible explanations include differences in
the initialization scheme, optimizer, and training
objective (masked vs. next-word modeling). See
§A for full hyperparameters.

**4.2** **Power of Saturated Attention**

We have shown that transformer training increases
the parameter norm (§3), creating a bias towards
saturation (§4.1). Now, we discuss the computational capabilities of saturated transformers, and
empirically investigate how they manifest in pretrained transformers. What computation can saturated transformers perform? We review theoretical
background about saturated attention, largely developed by Merrill (2019). Let H (sequence length n
by model dimension d) be the input representation
to a self attention layer. We assume a standard self
attention mechanism with key, query, and value
matrices K, Q, V.[8] Saturated attention resembles
standard attention where softmax is constrained to
a generalization of “argmax” (Merrill, 2019):

s attn(H; Q, K, V ) = arg max(HQK[⊤]H _[⊤])HV._

7Qualitatively, we observed that ∗-small transformers
tended to be more saturated than the _[∗]-base models._
8To simplify presentation, we omit bias terms.


These constructions demonstrate some useful computational abilities of saturated transformers. Due
to the summation in (1), the mean operation (or
near variants of it) can be used to implement


We define this vectorized arg max(A) as

_M(Ai) = {j | aij = max_ _aik}_
_k_

�
1/|M(Ai)| if j ∈M(Ai)

arg max(Ai)j =

0 otherwise.

Crucially, in the case of ties, arg max(A) returns
a uniform distribution over all tied positions. Saturated attention can retrieve the “maximum” value
in a sequence according to some similarity matrix.
It is also capable of restricted counting (Merrill
et al., 2020). Formalizing these observations, we
identify two useful computational operations that
are reducible to saturated self attention: argmax
and mean. Let hi represent the input representation
at each time step 1 _i_ _n._
_≤_ _≤_

1. Argmax: Set V = Id. Then the self attention
mechanism computes a function recovering
the element of H that maximally resembles hi
according to a quadratic form M = KQ[⊤]. If
there is a tie for similarity, a uniform average
of the maximal entries in H is returned.


argmax(H; M ) = arg max _hiMh[⊤]j_ _[.]_
_j_

2. Mean: Parameterize the head to attend uniformly everywhere. Then the head computes
a function taking a uniform average of values:


mean(H; V ) = [1]

_n_


_n_
�

_V hj._ (1)
_j=1_


-----

Figure 3: Distribution of the number of positions attended to for all heads in the PTB language models. The left
plot is pre-norm, and the right is post-norm. Values are averaged over 200 sentences from the development set.


counting, which allows recognizing languages like
_a[n]b[n]c[n]_ (Merrill et al., 2020). Empirically, Bhattamishra et al. (2020) find trained networks can
learn to recognize counter languages that rely on
computing means, failing on more complicated languages like Dyck-2. Our findings partially justify
why transformers can learn these languages: they
lie within the capacity of saturated transformers.

**4.3** **Learned Attention Patterns**

Recall that the small language models trained in
§4.1 reach 1.00 saturation. It follows that we can
convert them to saturated transformers (by multiplying θ by a large constant c) without significantly
shifting the representations in cosine space. We
will evaluate if the saturated attention heads manifest the argmax and mean constructions from §4.2.
As discussed in §4.2, saturated attention can
parameterize both argmax and mean heads. An
argmax head should attend to a small number of
positions. A mean head, on the other hand, attends
uniformly over the full sequence. Are both patterns
acquired in practice by our models? We plot the
distribution of the number of positions attended to
by each head in the saturated PTB models in Fig. 3.
The distribution is bimodal, with one mode at 1,
and the other around 41, representing the mean
sequence length of a 83-length encoder with positional masking to prevent lookahead. The empirical mode around 1 corresponds to heads that are
argmax-like. The mode around 41, on the other
hand, corresponds to mean-like heads, since it implies uniform attention over the masked sequence.
Thus, our analysis suggests that analogs of both
types of attention heads theorized in §4.2 are ac

quired in transformers in practice. In the pre-norm
transformer, which performs substantially better,
there are also a small number of heads lying between the two modes. We defer the investigation
of the function of these heads to future work.

### 5 Explanation for Norm Growth

We have documented norm growth in T5 and other
transformers (§3) and showed how it induces partial saturation in their representations (§4). This
section points towards an understanding of why
the parameter norm grows over the course of training, grounded in results about norm growth from
deep learning theory. We do not analyze specific optimizers directly; instead, we analyze norm
growth within simplified models of training dynamics taken from the literature. We then evaluate how
these candidate dynamics models fit T5’s training.

**5.1** **Setup**

Let δt ∈ R[n] denote the optimizer step at time t,
i.e., δt = θt+1 _θt. We write ηt for the learning_
_−_
rate at t.[9] Let ∇θtL denote the gradient of the loss
with respect to θt. By GD, we refer to the update
_δt = −ηt∇θtL.[10]_ In contrast, we will use the term
_gradient flow to refer to its continuous relaxation,_
specified by an analogous differential equation:

dθt

dt [=][ −][η][t][∇][θ][t][L.]

9Without loss of generality, the arguments presented here
can be seen as applying to an individual parameter in the
network, or the vector of all concatenated network parameters.
10Note that, in practice, T5 was trained with AdaFactor,
whereas the setup in this section assumes simpler optimizers.


-----

Figure 4: Approximate cosine similarity of f (x; cθ)
to sf (x; θ) for randomly initialized transformers f .
sf (x; θ) is approximated as in Fig. 2.

**5.2** **Homogeneity**

We will rely on properties of homogeneous networks, a class of architectures well-studied in deep
learning theory (Ji and Telgarsky, 2020).

**Definition 2 (Homogeneity) A function f** (x; θ) is
_k-homogeneous in θ iff, for all c_ 0, f (x; cθ) =
_≥_
_c[k]f_ (x; θ). We further say that f is homogeneous iff
there exists some k such that f is k-homogeneous.

Many common components of modern neural
networks are homogeneous (Li and Arora, 2019).
Furthermore, as various computations within a neural network preserve homogeneity (§C), some full
networks are also homogeneous. An example of a
fully homogeneous neural network is a feedforward
ReLU network without bias terms.
Why is homogeneity relevant for transformers?
Transformers are not homogeneous, but they are
_almost homogeneous. We formalize this as:_

**Definition 3 (Approx. homogeneity) A scalar[11]**

function f (x; θ) is approximately k-homogeneous
in θ iff there exist d, ρ s.t., for c 1 and _θ_ _ρ,_
_≥_ _∥_ _∥≥_

_f_ (x; cθ) _ckf_ (x; θ) exp( _d_ _θ_ ).
_−_ _≤_ _−_ _∥_ _∥_
��� ���

In other words, as _θ_ grows, f approximates a
_∥_ _∥_
homogeneous function with exponentially vanishing error. In §D, we prove transformer encoders
without biases are approximately 1-homogeneous.
In Fig. 4, we compare the cosine similarity of transformers with and without biases to their saturated
variants, as a function of a constant c scaling their
weights. An approximately homogeneous function

11A vector function is approximately k-homogeneous if this
holds for all its elements.


rapidly approach 1.0 as c increases. We find similar curves for transformers with and without biases,
suggesting biasless transformers are similarly homogeneous to transformers with biases.[12]

Since multiplying two homogeneous functions
adds their homogeneity, a transformer encoder followed by a linear classifier is approximately 2homogeneous. A key property of homogeneous
functions is Euler’s Homogeneity Theorem: the
derivative of a k-homogeneous function is (k 1)_−_
homogeneous. Thus, we will assume the gradients of the linear classifier output are roughly 1homogeneous, which under simple GD implies:
**Assumption 1 Let θt include all encoder and clas-**
_sifier parameters. Let_ _[∝]_ _mean “approximately pro-_
_∼_
_portional to”. For large enough t during trans-_
_former training, ∥δt∥_ _[∝]_ _ηt∥θt∥._
_∼_

**5.3** **Aligned Dynamics**

We now consider the first candidate dynamics
model: aligned dynamics (Ji and Telgarsky, 2020).
Analyzing homogeneous networks with an exponential binary classification loss and gradient flow,
Ji and Telgarsky (2020) show that the parameters
converge in direction, and that the gradients become aligned, meaning that θt[⊤] _[·][ δ][t]_ _[→∥][θ][t][∥∥][δ][t][∥][.]_
While it is unclear whether transformers will follow
aligned dynamics, we entertain this as one hypothesis. Under Ass. 1, alignment implies


_∥θt∥≈_


_t_
�

_∥δi∥_ _[∝]_
_∼_
_i=0_


�
_ηt∥θt∥dt._


_√_
With the ηt = 1/ _t schedule used by T5 (Raffel_

et al., 2019), ∥θt∥ _[∝]_ exp�√t� (see §E.1). This is
_∼_ _√_

asymptotically faster than the observed _t growth,_

suggesting an alternate dynamics might be at play.

**5.4** **Misaligned Dynamics**

Our second candidate model of training is mis**aligned dynamics, which follows largely from Li**
and Arora (2019). This can be derived by assuming the gradients are misaligned (i.e., θt[⊤] _[·][ δ][t]_ [= 0][),]
which hold for scale-invariant networks (Li and
Arora, 2019) and in expectation for random normal
gradients. Misalignment implies (derived in §E.2):


_∥θt∥[2][ ∝]_
_∼_


_t_
�

_∥δi∥[2]._ (2)
_i=0_


12Lyu and Li (2019) find similar results for feedforward
ReLU networks. It is an interesting puzzle why networks with
biases appear similarly homogeneous to those without biases.


-----

Figure 5: Alignment (cosine similarity of δt and θt) and step size (∥δt∥) over training.


We show in §E.2 that, with the T5 learning rate
_√_ _√_
(ηt = 1/ _t), (2) reduces to ∥θt∥_ _[∝]_ _t, as ob-_

_∼_

served empirically for T5. We now further test
whether misaligned dynamics are a good fit for T5.

**5.5** **Evaluation**

We measure the gradient alignment over the course
of training T5. Our alignment metric is the cosine
similarity of δt to θt. As shown on the left of Fig. 5,
the alignment initially rapidly increases to 0.15,
_∼_
and then decays to near 0. This supports the hypothesis that the T5 dynamics are misaligned, since the
similarity is never high, and may be approaching 0.
On the right of Fig. 5, we plot step size over training in order to evaluate the validity of Ass. 1. At
the beginning of training, a chaotic step size seems
reasonable, as it is hard to predict the dynamics
before approximate homogeneity takes hold. For
large t, Ass. 1 combined with the T5 learning rate
schedule predicts step size should be roughly constant.[13] This is not exactly what we find: for large
_t, ∥δt∥_ grows gradually with t. However, the absolute change in step size is small: < 20 across
220M parameters. Thus, we believe Ass. 1 is not
unreasonable, though it would be interesting to
understand what properties of the optimizer can
explain the slight growth in step size.[14]

**5.6** **Weight Decay**

One feature of practical training schemes not considered in this section is weight decay. When applied to standard GD, weight decay can be written
_δt = −ηt∇θtL −_ _λθt. Intuitively, it might hinder_

13 _√_ _√_

14SinceWe believe the sharp drop in ∥δt∥∝∼ _ηt∥θt∥_ = _t/_ _∥tδ = 1t∥_ at the final step is an.
artifact of the original recording of these checkpoints.


norm growth if λ is large.[15] In §F, we report preliminary experiments testing the effect of weight
decay on norm growth. Indeed, if λ is set too large,
weight decay can prevent norm growth, but within
the standard range of values for λ, we find norm
growth even in the face of weight decay. However, it is possible these results may change if the
optimizer or other hyperparameters are varied.

### 6 Conclusion

_√_
We empirically found that ∥θt∥ grows ∝ _t dur-_

ing T5 pretraining—a fact that may be caused by
the approximate homogeneity of the transformer
architecture. We proved that norm growth induces
saturation, and then showed empirically that T5 and
other large transformers become approximately saturated through their pretraining. Examining highly
saturated transformer language models, we found
the attention heads largely split between two distinct behaviors that can be roughly interpreted as
argmax and mean operations. While we lack a
precise formal characterization of “semi-saturated”
transformers, we conjecture their capacity resembles that of the saturated models. Thus, we believe
further analyzing the capabilities of saturated attention may clarify the linguistic biases that emerge in
transformers through training, and the mechanisms
they use to represent linguistic structure.

### Acknowledgments

We thank Colin Raffel for sharing access to the T5
training checkpoints. Additional thanks to Qiang

15Common wisdom says that weight decay improves generalization by keeping ∥θt∥ small; however, recent work challenges the assumption that a bias towards small norm is beneficial (Goldblum et al., 2020), suggesting the benefit of weight
decay may arise from more subtle effects on the GD trajectory.


-----

Ning, Kyle Richardson, Mitchell Wortsman, Martin Lohmann, and other researchers at the Allen
Institute for AI for their comments on the project.

### References

Sanjeev Arora, Nadav Cohen, Wei Hu, and Yuping Luo.
[2019. Implicit regularization in deep matrix factor-](http://arxiv.org/abs/1905.13655)
[ization.](http://arxiv.org/abs/1905.13655)

Satwik Bhattamishra, Kabir Ahuja, and Navin Goyal.
[2020. On the Ability and Limitations of Transform-](https://doi.org/10.18653/v1/2020.emnlp-main.576)
[ers to Recognize Formal Languages.](https://doi.org/10.18653/v1/2020.emnlp-main.576) In Proceed_ings of the 2020 Conference on Empirical Methods_
_in Natural Language Processing (EMNLP), pages_
7096–7116, Online. Association for Computational
Linguistics.

[N. Chomsky. 1956. Three models for the description of](https://doi.org/10.1109/TIT.1956.1056813)
[language. IRE Transactions on Information Theory,](https://doi.org/10.1109/TIT.1956.1056813)
2(3):113–124.

Gonçalo M. Correia, Vlad Niculae, and André F. T.
[Martins. 2019. Adaptively sparse transformers. In](https://doi.org/10.18653/v1/D19-1223)
_Proceedings of the 2019 Conference on Empirical_
_Methods in Natural Language Processing and the_
_9th International Joint Conference on Natural Lan-_
_guage Processing (EMNLP-IJCNLP), pages 2174–_
2184, Hong Kong, China. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. [BERT: Pre-training of](https://doi.org/10.18653/v1/N19-1423)
[deep bidirectional transformers for language under-](https://doi.org/10.18653/v1/N19-1423)
[standing.](https://doi.org/10.18653/v1/N19-1423) In Proceedings of the 2019 Conference
_of the North American Chapter of the Association_
_for Computational Linguistics: Human Language_
_Technologies, Volume 1 (Long and Short Papers),_
pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Javid Ebrahimi, Dhruv Gelda, and Wei Zhang. 2020.

[How can self-attention networks recognize Dyck-n](https://doi.org/10.18653/v1/2020.findings-emnlp.384)
[languages? In Findings of the Association for Com-](https://doi.org/10.18653/v1/2020.findings-emnlp.384)
_putational Linguistics: EMNLP 2020, pages 4301–_
4306, Online. Association for Computational Linguistics.

Winthrop Nelson Francis and Henry Kuˇcera. 1989.
_Manual of information to accompany a standard cor-_
_pus of present-day edited American English, for use_
_with digital computers. Brown University, Depart-_
ment of Linguistics.

Micah Goldblum, Jonas Geiping, Avi Schwarzschild,
[Michael Moeller, and Tom Goldstein. 2020. Truth](https://openreview.net/forum?id=HyxyIgHFvr)
[or backpropaganda? an empirical investigation of](https://openreview.net/forum?id=HyxyIgHFvr)
[deep learning theory. In International Conference](https://openreview.net/forum?id=HyxyIgHFvr)
_on Learning Representations._

Suriya Gunasekar, Blake Woodworth, Srinadh Bhojanapalli, Behnam Neyshabur, and Nathan Srebro.
[2017. Implicit regularization in matrix factorization.](http://arxiv.org/abs/1705.09280)


[Michael Hahn. 2020. Theoretical limitations of self-](https://doi.org/10.1162/tacl_a_00306)
[attention in neural sequence models. Transactions](https://doi.org/10.1162/tacl_a_00306)
_of the Association for Computational Linguistics,_
8:156–171.

[Ziwei Ji and Matus Telgarsky. 2020. Directional con-](https://proceedings.neurips.cc/paper/2020/file/c76e4b2fa54f8506719a5c0dc14c2eb9-Paper.pdf)
[vergence and alignment in deep learning.](https://proceedings.neurips.cc/paper/2020/file/c76e4b2fa54f8506719a5c0dc14c2eb9-Paper.pdf) In Ad_vances in Neural Information Processing Systems,_
volume 33, pages 17176–17186. Curran Associates,
Inc.

[Christo Kirov and Robert Frank. 2012. Processing of](https://doi.org/10.1080/09540091.2011.641939)
[nested and cross-serial dependencies: an automaton](https://doi.org/10.1080/09540091.2011.641939)
[perspective on SRN behaviour. Connection Science,](https://doi.org/10.1080/09540091.2011.641939)
24(1):1–24.

[Zhiyuan Li and Sanjeev Arora. 2019. An exponential](https://openreview.net/forum?id=rJg8TeSFDH)
[learning rate schedule for deep learning. In Proc. of](https://openreview.net/forum?id=rJg8TeSFDH)
_ICLR._

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis,
Luke Zettlemoyer, and Veselin Stoyanov. 2019.
[RoBERTa: A robustly optimized BERT pretraining](http://arxiv.org/abs/1907.11692)
[approach.](http://arxiv.org/abs/1907.11692)

Ilya Loshchilov and Frank Hutter. 2017. [Fixing](http://arxiv.org/abs/1711.05101)
[weight decay regularization in adam.](http://arxiv.org/abs/1711.05101) _CoRR,_
abs/1711.05101.

Charles Lovering, Rohan Jha, Tal Linzen, and Ellie
Pavlick. 2021. [Predicting inductive biases of pre-](https://openreview.net/forum?id=mNtmhaDkAr)
[trained models.](https://openreview.net/forum?id=mNtmhaDkAr) In International Conference on
_Learning Representations._

[Kaifeng Lyu and Jian Li. 2019. Gradient descent maxi-](http://arxiv.org/abs/1906.05890)
[mizes the margin of homogeneous neural networks.](http://arxiv.org/abs/1906.05890)

Mitchell P. Marcus, Beatrice Santorini, and Mary Ann
Marcinkiewicz. 1993. [Building a large annotated](https://www.aclweb.org/anthology/J93-2004)
[corpus of English: The Penn Treebank. Computa-](https://www.aclweb.org/anthology/J93-2004)
_tional Linguistics, 19(2):313–330._

Stephen Merity, Caiming Xiong, James Bradbury, and
[Richard Socher. 2016. Pointer sentinel mixture mod-](http://arxiv.org/abs/1609.07843)
[els.](http://arxiv.org/abs/1609.07843)

[William Merrill. 2019. Sequential neural networks as](https://doi.org/10.18653/v1/w19-3901)
[automata.](https://doi.org/10.18653/v1/w19-3901) _Proceedings of the Workshop on Deep_
_Learning and Formal Languages: Building Bridges._

[William Merrill. 2020. On the linguistic capacity of](http://arxiv.org/abs/2004.06866)
[real-time counter automata.](http://arxiv.org/abs/2004.06866)

William. Merrill, Gail Garfinkel Weiss, Yoav Goldberg, Roy Schwartz, Noah A. Smith, and Eran Yahav.
[2020. A formal hierarchy of RNN architectures. In](https://aclanthology.org/2020.acl-main.43/)
_Proc. of ACL._

Mor Shpigel Nacson, Suriya Gunasekar, Jason D. Lee,
Nathan Srebro, and Daniel Soudry. 2019. [Lexi-](http://arxiv.org/abs/1905.07325)
[cographic and depth-sensitive margins in homoge-](http://arxiv.org/abs/1905.07325)
[neous and non-homogeneous deep models.](http://arxiv.org/abs/1905.07325)

Tomaso Poggio, Andrzej Banburski, and Qianli Liao.
[2019. Theoretical issues in deep networks: Approx-](http://arxiv.org/abs/1908.09375)
[imation, optimization and generalization.](http://arxiv.org/abs/1908.09375)


-----

Tomaso Poggio, Qianli Liao, and Andrzej Banburski.
2020. [Complexity control by gradient descent in](https://www.nature.com/articles/s41467-020-14663-9)
[deep networks. Nature communications, 11(1):1–5.](https://www.nature.com/articles/s41467-020-14663-9)

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine
Lee, Sharan Narang, Michael Matena, Yanqi Zhou,
[Wei Li, and Peter J. Liu. 2019. Exploring the limits](http://arxiv.org/abs/1910.10683)
[of transfer learning with a unified text-to-text trans-](http://arxiv.org/abs/1910.10683)
[former.](http://arxiv.org/abs/1910.10683)

[Noam Razin and Nadav Cohen. 2020. Implicit regular-](http://arxiv.org/abs/2005.06398)
[ization in deep learning may not be explainable by](http://arxiv.org/abs/2005.06398)
[norms.](http://arxiv.org/abs/2005.06398)

Anna Rogers, Olga Kovaleva, and Anna Rumshisky.
2020. [A primer in BERTology: What we know](https://doi.org/10.1162/tacl_a_00349)
[about how BERT works. Transactions of the Associ-](https://doi.org/10.1162/tacl_a_00349)
_ation for Computational Linguistics, 8:842–866._

[Noam Shazeer and Mitchell Stern. 2018. Adafactor:](http://arxiv.org/abs/1804.04235)
[Adaptive learning rates with sublinear memory cost.](http://arxiv.org/abs/1804.04235)
_CoRR, abs/1804.04235._

Chihiro Shibata, Kei Uchiumi, and Daichi Mochihashi.
[2020. How LSTM encodes syntax: Exploring con-](http://arxiv.org/abs/2010.00363)
[text vectors and semi-quantization on natural text.](http://arxiv.org/abs/2010.00363)

[Hava T. Siegelmann and Eduardo D. Sontag. 1992. On](https://doi.org/10.1145/130385.130432)
[the computational power of neural nets. In Proc. of](https://doi.org/10.1145/130385.130432)
_COLT, pages 440–449._

Mirac Suzgun, Yonatan Belinkov, Stuart Shieber, and
[Sebastian Gehrmann. 2019a. LSTM networks can](https://www.aclweb.org/anthology/W19-3905)
[perform dynamic counting. In Proceedings of the](https://www.aclweb.org/anthology/W19-3905)
_Workshop on Deep Learning and Formal Languages:_
_Building Bridges, pages 44–54._

Mirac Suzgun, Sebastian Gehrmann, Yonatan Belinkov,
[and Stuart M. Shieber. 2019b. Memory-augmented](http://arxiv.org/abs/1911.03329)
[recurrent neural networks can learn generalized](http://arxiv.org/abs/1911.03329)
[Dyck languages.](http://arxiv.org/abs/1911.03329)

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
[Kaiser, and Illia Polosukhin. 2017. Attention is all](http://arxiv.org/abs/1706.03762)
[you need.](http://arxiv.org/abs/1706.03762)

[Gail Weiss, Yoav Goldberg, and Eran Yahav. 2018. On](http://arxiv.org/abs/1805.04908)
[the practical computational power of finite precision](http://arxiv.org/abs/1805.04908)
[RNNs for language recognition.](http://arxiv.org/abs/1805.04908)

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R’emi Louf, Morgan Funtowicz, and Jamie Brew. 2019. [Huggingface’s trans-](https://arxiv.org/abs/1910.03771)
[formers: State-of-the-art natural language process-](https://arxiv.org/abs/1910.03771)
[ing. ArXiv, abs/1910.03771.](https://arxiv.org/abs/1910.03771)

Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng,
Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan
[Lan, Liwei Wang, and Tie-Yan Liu. 2020. On layer](http://arxiv.org/abs/2002.04745)
[normalization in the transformer architecture.](http://arxiv.org/abs/2002.04745)

Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V. Le. 2019.
[Xlnet: Generalized autoregressive pretraining for](http://arxiv.org/abs/1906.08237)
[language understanding.](http://arxiv.org/abs/1906.08237)


Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh
Rawat, Sashank Reddi, and Sanjiv Kumar. 2020.
Are transformers [universal](https://openreview.net/forum?id=ByxRM0Ntvr) approximators of
[sequence-to-sequence functions?](https://openreview.net/forum?id=ByxRM0Ntvr) In International
_Conference on Learning Representations._

### A Experimental Details

We provide experimental details for the small language models that we trained. The models were
trained for 5 epochs, and the best performing model
was selected based on development loss. Reported
metrics were then measured on the held-out test
set. We used our own implementation of the standard pre- and post-norm transformer architectures.
We did not do any hyperparameter search, instead
choosing the following hyperparameters:

  - Batch size of 16

  - Model dimension of 768

  - Feedforward hidden dimension of 512

  - 12 heads per layer

  - 12 layers

  - AdamW optimizer with default PyTorch hyperparameters

  - 0 probability of dropout

  - Default PyTorch initialization

**Tokenization** For Wikitext-2, 3 tokens in the
whole test dataset were unattested in the training set
(due to capitalization). To make our model compatible with unseen tokens, we replaced these tokens
with <unk>, the same class that appeared for low
frequency words at training time, when evaluating
the final text perplexity. Due to the small number of
tokens that were affected, the impact of this change
should be negligible.

**Compute** We estimate the experiments in this
paper took several hundred GPU hours on NVIDIA
A100 GPUs over the course of almost two years of
on-and-off research time.

**T5** We used the historical checkpoints of bsl-0,
one of five T5-base models that was trained for the
original paper (Raffel et al., 2019).

**Measuring Norms** As a systematic choice, all
measurements of parameter norm include only en_coder parameters that are not scalars. We advise_
other researchers to follow the practice of excluding embedding parameters, as embedding parameters that are infrequently updated may obscure
general trends in the network parameters.


-----

Product (k1, k2) _k1 + k2_

Table 1: Effects of network components on homogeneity shown by Li and Arora (2019). We write the “k
Output” homogeneity as a function of the “k Input” homogeneity. These facts can be applied recursively to
compute the homogeneity of a network. We will show
that the same facts hold for approximate homogeneity.

### B Norm Growth and Saturation

**Proposition 2 (Formal version of Prop. 1) Let θt ∈**
R[n] _be the parameter vector at train step t for a_
_network f_ (x; θt). Assume that, as t →∞, there
_exists a scalar sequence c(t)_ _and fixed vector_
_→∞_
_θ[′]_ _∈_ (R \ {0})[n] _such that, for all t, θt →_ _θ[′]_ _· c(t)._
_Then f converges pointwise to a saturated network_
_in function space._

_Proof._

lim
_t→∞_ _[f]_ [(][x][;][ θ][t][) = lim]t→∞ _[f]_ [(][x][;][ θ][′][ ·][ c][(][t][))][.]

Now, since c(t) and θ[′] _c(t) contains no_
_→∞_ _·_
indeterminate elements, we can simplify this to

lim
_c→∞_ _[f]_ [(][x][;][ cθ][′][) =][ s][f] [(][x][;][ θ][′][)][.]

### C Approximate Homogeneity

In this section, we will further develop the notion
of approximate homogeneity. We will prove that
is consistent. In other words, every function can
have at most one degree k of approximate homogeneity. Next, we will show that the useful closure
properties applying to full homogeneity also apply
to partial homogeneity.
If _f_ (θ) is approximately _k-homogeneous_
(cf. Def. 3), then f (cθ) = c[k]f (θ)+ϵ for some error
vector ϵ where, for each i, |ϵi| ≤ exp(−d∥θ∥)), for
all c and large enough _θ_ . We use this ϵ notation
_∥_ _∥_
throughout this section.


**C.1** **Consistency**

We first prove that approximate homogeneity is
consistent: in other words, if a function is both
approximately k1 and k2-homogeneous, then k1 =
_k2. This is an important property for establishing_
approximate homogeneity as a meaningful notion.

**Lemma 1 Let k1, k2 ∈** N. Assume that f is both
_approximately k1 and k2-homogeneous. Then k1 =_
_k2._

_Proof. If f is both approximately k1 and k2-_
homogeneous, then we have vanishing terms ϵ1
and ϵ2 such that, for all c,

_f_ (cθ) = c[k][1]f (θ) + ϵ1

_f_ (cθ) = c[k][2]f (θ) + ϵ2.

Subtracting both sides yields

0 = (c[k][1] _−_ _c[k][2])f_ (θ) + ϵ1 − _ϵ2_

∴ _ck1 −_ _ck2_ = |ϵ1 − _ϵ2|_ _._
��� ��� _f_ (θ)

The right-hand side vanishes exponentially in _θ_
_∥_ _∥_
for all c, whereas the left-hand side grows with c
unless k1 = k2. Thus, to satisfy this equation for
all c, it must be the case that k1 = k2.

**C.2** **Closure Properties**

We now prove that effects of various functions on
homogeneity explored by Li and Arora (2019) also
translate to approximate homogeneity.

**Lemma 2 ReLU preserves approximate k-**
_homogeneity, i.e., let f : R[n]_ _→_ R be approxi_mately k-homogeneous. Then ReLU_ _f is approxi-_
_◦_
_mately k-homogeneous._

_Proof._

ReLU �f (cθ)� = ReLU �c[k]f (θ) + ϵ�

ReLU �c[k]f (θ)� + _ϵ_ _._
_≤_ _|_ _|_

Therefore,

ReLU �f (cθ)� ReLU �c[k]f (θ)�[�] _ϵ_ _._
_−_ _≤|_ _|_
��� ��

Set ϵ[′] = _ϵ_, showing ReLU �f (θ)� is approxi_|_ _|_
mately k-homogeneous.

**Lemma 3 Let f, g be vector-valued functions of θ.**
_If f and g are approximately k-homogeneous, then_
_f + g is approximately k-homogeneous._

|Component|k Input|k Output|
|---|---|---|
|Linear Bias Affine LayerNorm LayerNorm + Affine ReLU Sum Product|k 1 0 k k k (k, k) (k , k ) 1 2|k + 1 1 1 0 1 k k k + k 1 2|


-----

_Proof._

_f_ (cθ) + g(cθ) = c[k]f (θ) + ϵf + c[k]g(θ) + ϵg

= c[k]f (θ) + c[k]g(θ) + ϵ[′],

where ϵ[′] = ϵf + ϵg. Thus,
_f_ (cθ) + g(cθ) _ck�f_ (θ) + g(θ)�[�] _ϵ′._
_−_ _≤_
��� ��

**Lemma 4 Let f, g be vector-valued functions**
_of θ. If f and g are approximately kf and kg-_
_homogeneous, then f ·g is approximately (kf +kg)-_
_homogeneous._

_Proof._

_f_ (cθ) · g(cθ) = �c[k]f (θ) + ϵf � _·_ �c[k]g(θ) + ϵg�

= c[k][f] [+][k][g] _f_ (θ)g(θ) + c[k][f] _f_ (θ)ϵg

+ c[k][g] _g(θ)ϵf + ϵf_ _ϵg._

We now rewrite the term c[k][f] _f_ (θ)ϵg as

_θg(x; θ[ˆ])_

� _d[′]_ _θ_ �.
_−_ _∥_ _∥_
exp( _d_ _θ_ )
_−_ _∥_ _∥_ _[≤]_ [exp]

Now, set ϵ[′] = min(exp(−d∥θ∥), ϵf _ϵg)._

_f_ (cθ)g(cθ) _ckf_ +kg _f_ (θ)g(θ) _ϵ′._
_−_ _≤_
��� ���

The analogous results for linear transformation,
bias, and affine transformation directly follow from
the results for sum and product in Lem. 3 and
Lem. 4.
Finally, we show that layer norm converts a
homogeneous function to approximately scaleinvariant function. In order to be numerically stable, practical implementations of layer norm utilize
a small tolerance term so that the denominator is
never zero. We omit this practical detail from our
analysis, instead defining the layer norm LN(x) for
_x ∈_ R[n] according to


_Proof. Since addition preserves approximate k-_
homogeneity, mean (and difference to mean), preserve approximate k-homogeneity. Letting C = c[k],
we can write

_f_ (cθ) _µ(f_ (cθ)) = C�f (θ) _µ(f_ (θ))� + ϵ.
_−_ _−_

We now apply this to the definition of layer norm
to get

LN(f (cθ))i = _[f]_ [(][cθ][)][i][ −] _[µ][(][f]_ [(][cθ][))]

_f_ (cθ) _µ(f_ (cθ))
_∥_ _−_ _∥_

�f (θ)i − _µ(f_ (θ))� + ϵi
= _[C]_

_C_ _f_ (θ) _µ(f_ (θ)) + ϵ [.]
_∥_ _−_ _∥_


We show that the difference between this and the
unscaled layer norm goes to zero. To simplify
notation, we now write f = f (θ), µ = µ(f (θ)),
and ϵ = ϵ in the left-hand side below:


_|LN(f_ (cθ))i − LN(f (θ)i)|

_C�fi −_ _µ�_ + ϵi

=

_C_ _f_ _µ_ + ϵ _f_ _µ_

����� _∥_ _−_ _∥_ _[−]_ _∥[f][i] −[ −]_ _[µ]∥_

= _ϵi∥f −_ _µ∥−_ _ϵ(fi −_ _µ)_
���� _C∥f −_ _µ∥[2]_ + ϵ∥f − _µ∥_ ����

= _ϵi −_ _ϵv_
���� _C∥f −_ _µ∥_ + ϵ ����

_ϵi −_ _ϵv_ _._
_≤_
���� _ϵ_ ����


�����


_µ(x) = [1]_

_n_


_n_
�

_xi_
_i=1_


LN(x)i = _[x][i][ −]_ _[µ][(][x][)]_

_x_ _µ(x)_
_∥_ _−_ _∥_ _[.]_

**Lemma 5 Let f be approximately k-homogeneous**
_for some k._ _Then, LN(f_ ) is approximately 0_homogeneous._


for some v ∈ R[n] which does not grow with ∥θ∥.
Thus, setting ϵ[′] to this final quantity satisfies the
definition of approximate 0-homogeneity, i.e. approximate scale invariance.

**C.3** **Saturating Activation Functions**

We show that the exponentially saturation activation functions σ, softmax, and tanh are approximately scale-invariant in x, i.e. scaling x has an
exponentially diminishing effect on the output. We
start by analyzing the simpler sigmoid, and then
show that the same result holds for softmax. For
completeness, we then present a proof for tanh.
We use Θ (not θ) in the standard sense of asymptotic notation.

**Lemma 6 The scaling error for σ vanishes expo-**
_nentially in the preactivation magnitude, i.e. for all_
_c_ 1,
_≥_

_σ(cx)_ _σ(x)_ Θ(exp( _x_ )).
_|_ _−_ _| ≤_ _−|_ _|_


-----

_Proof. Assume without loss of generality that x_ =
_̸_
0, as if this is the case, the error is 0. When x > 0,
we have

_σ(cx)_ _σ(x)_ = σ(cx) _σ(x)_
_|_ _−_ _|_ _−_

1 _σ(_ _x_ )
_≤_ _−_ _|_ _|_

1
=

exp( _x_ ) + 1
_|_ _|_
= Θ(exp( _x_ )).
_−|_ _|_

When x < 0, we have

_σ(cx)_ _σ(x)_ = σ(x) _σ(cx)_
_|_ _−_ _|_ _−_

1 _σ(_ _x_ ) + 0
_≤_ _−_ _|_ _|_

= Θ(exp( _x_ )).
_−|_ _|_

**Lemma 7 The elementwise scaling error for**
softmax vanishes exponentially in the preactiva_tion norm, i.e. for all c ≥_ 1, x ∈ R[n] _s.t. 1 ≤_ _i ≤_ _n,_

_|softmax(cx)i −_ softmax(x)i| ≤ exp(−Θ(∥x∥)).

_Proof. The proof closely follows that of Lem. 6,_
but is more involved. We consider two cases: xi =
max(x), and xi ̸= max(x).

**Case 1** _xi = max(x)._

_|softmax(cx)i −_ softmax(x)i|

= softmax(cx)i − softmax(x)i
_≤_ 1 − softmax(x)i

exp(xi)
= 1
_−_ �
_j_ [exp(][x][j][)]

exp(max(x))
1
_≤_ _−_

exp(max(x)) + (n 1) exp(min(x))
_−_

1
= 1
_−_

1 + (n 1) exp(min(x) max(x))
_−_ _−_

1
= 1
_−_

1 + exp(min(x) max(x) + d) _[,]_
_−_


Thus, applying these functions to a homogeneous input produces an output that is approximately scale-invariant in the parameters θ. Thus,
these functions act similarly to layer norm, which
maps homogeneous input to scale-invariant output.
But what happens if the input is approximately homogeneous, rather than strictly homogeneous? In
this case, we show that the output is approximately
scale-invariant assuming _θ_ is sufficiently large.
_∥_ _∥_

**Proposition 3 Let f** (x; θ) be approximately k_homogeneous in θ. Then the following functions_
_are approximately scale-invariant in θ:_

_gσ = σ ◦_ _f_

_gsoftmax = softmax ◦f_

_gtanh = tanh ◦f._

_Proof. If f_ (x; θ) is approximately k-homogeneous,
then f (x; cθ) = c[k]f (x; θ) + ϵ where _ϵ_
_∥_ _∥≤_
exp( _O(_ _θ_ )). Crucially, since ϵ vanishes for
_−_ _∥_ _∥_
large norm, there is some ρ where, for all θ such
that ρ < _θ_ :
_∥_ _∥_

sgn �c[k]f (x; θ) + ϵ� = sgn �c[k]f (x; θ)�

arg max �c[k]f (x; θ) + ϵ� = arg max �c[k]f (x; θ)�.


Finally, for completeness, we show that tanh exhibits the same property. The proof is very similar
to sigmoid, following closely from the definition

tanh(x) = [exp(2][x][)][ −] [1]

exp(2x) + 1 _[.]_

**Lemma 8 The scaling error for tanh vanishes**
_exponentially in the preactivation magnitude, i.e._
_for all c_ 1,
_≥_

tanh(cx) tanh(x) exp( Θ( _x_ )).
_|_ _−_ _| ≤_ _−_ _|_ _|_

_Proof._

tanh(cx) tanh(x) 1 tanh(x)
_|_ _−_ _| ≤|_ _−_ _|_

= 1 tanh( _x_ )
_−_ _|_ _|_

= 1
_−_ [exp(2][|][x][|][)][ −] [1]

exp(2 _x_ ) + 1
_|_ _|_

= [exp(2][|][x][|][) + 1][ −] [exp(2][|][x][|][) + 1]

exp(2 _x_ ) + 1
_|_ _|_

2
=

exp(2 _x_ ) + 1
_|_ _|_
= exp( Θ( _x_ )).
_−_ _|_ _|_


for some d ∈ R. As this has the form of σ,

_|softmax(cx)i −_ softmax(x)i|

= 1 _σ(Θ(_ _x_ )) = exp( Θ( _x_ )).
_−_ _∥_ _∥_ _−_ _∥_ _∥_

**Case 2** _xi ̸= max(x)._

_|softmax(cx)i −_ softmax(x)i|

= softmax(x)i − softmax(cx)i
1 max(softmax(x)) 0
_≤_ _−_ _−_

= 1 softmax(max(x)),
_−_

which is identical to case 1.


-----

Therefore, for θ such that _θ_ _> ρ, the bounds_
_∥_ _∥_
used in Lem. 6, Lem. 7, and Lem. 8 hold for
approximately homogeneous f . Thus, we can
conclude that the output is approximately scaleinvariant.

### D Transformers

We introduce the notation _k-homogeneous to_
_∼_
mean approximately k-homogeneous. In this section, we show that the transformer encoder is 1_∼_
homogeneous. A transformer Vaswani et al. (2017)
is made up of three main components: an embedding layer, self attention sublayers, and feedforward sublayers. Since the embedding layer is
just a matrix multiplication, it is a 1-homogeneous
function of the input. Assuming the self attention
and feed-forward sublayers have no bias terms, we
show that they approximate functions preserving
approximate 1-homogeneity. As the full network is
an initial embedding layer followed by these sublayers, the final output is 1-homogeneous. In the
_∼_
main paper, we discuss the connection between
homogeneity and norm growth.
We base our analysis on the HuggingFace implementation[16] of BERT (Wolf et al., 2019). To aid
analysis, we make some simplifying assumptions,
which are discussed along with the definitions. We
later show empirically that homogeneity for the
unsimplified versions is similar.

**D.1** **Transformer Definition**

The transformer encoder is a cascade of alternating multi-head self-attention sublayers and feed_forward sublayers. Each multi-head self-attention_
sublayer can be further broken down as an aggregation of self-attention heads. Let LN( ) denote

_·_
a layer norm followed by a learned affine transformation. Here we will consider the pre-norm
transformer variant (Xiong et al., 2020), meaning
that LN comes before the residual connection wherever it appears.[17] We will also assume that there
are no biases, making all affine transformations
into strict linear transformations.

**Definition 4 (Self-attention head) Given parame-**
ters W _[k], W_ _[q], W_ _[v]_ and input X ∈ R[Tn], we define

[16https://huggingface.co/transformers/](https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertModel)
[_modules/transformers/modeling_bert.](https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertModel)
[html#BertModel](https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertModel)

17The post-norm transformer applies these operations in the
opposite order.


a self-attention head attn as

_K = W_ _[k]X_

_Q = W_ _[q]X_

_V = W_ _[v]X_

�
_A = softmax(QK[⊤]/_ _dk)_

_H = AV,_

where H is the output tensor.

The multi-head self-attention sublayer computes
several attention heads in parallel and aggregates
them into a single sequence of vectors.

**Definition 5 (Multi-head self-attention sublayer)**
Let X ∈ R[Tn] be the input. We now define the
_k-multi-head self-attention sublayer MSAk. First,_
we compute k self-attention heads in parallel to
produce H1, · · ·, Hk. We then concatenate these
along the feature axis to form H, and compute the
sublayer output Y as

MSAk(X) = LN(WH) + X.

Finally, the linear sublayer is the other component of the transformer.

**Definition 6 (Feedforward sublayer) Let X**
_∈_
R[Tn] be the input. We compute the feedforward
_sublayer FF according to_

FF(X) = LN(W _[f]_ ReLU(W _[i]X)) + X._

**D.2** **Results**

**Proposition 4 If X is** 1-homogeneous in
_∼_
_parameters_ _θ,_ _then_ attn(X; W _[k], W_ _[q], W_ _[v])_
_is_ 1-homogeneous in the concatenation of
_∼_
_θ, W_ _[k], W_ _[q], W_ _[v]._

_Proof. Consider a self-attention layer receiving a_
_∼1-homogeneous input matrix X ∈_ R[Tn] where
_T is the sequence length. Using the homogene-_
ity rule for multiplication, K, Q, V are each 2_∼_
homogeneous, as homogeneity is additive over multiplication. By the same argument, QK[⊤] is 4_∼_
homogeneous. In Prop. 3, we show that if the input
to softmax is approximately homogeneous, then
the output is approximately scale-invariant. Thus,
_A is approximately 0-homogeneous. Then AV is_
1-homogeneous.
_∼_

We show that the multi-head component that aggregates multiple heads into a shared representation
also preserves approximate 1-homogeneity.


-----

**Proposition 5 If X is** 1-homogeneous in param_∼_
_eters θ, then MSA is_ 1-homogeneous in the full
_∼_
_parameters._

_Proof. Since Wh is_ 2-homogeneous, LN(WH)
_∼_
is 1-homogeneous. The input X is also 1_∼_ _∼_
homogeneous by assumption, meaning that the sum
is also 1-homogeneous.
_∼_

Finally, we turn to analyzing the feedforward
sublayer of the transformer.

**Proposition 6 If X is** 1-homogeneous, then
_∼_
FF(X; W _[f]_ _, W_ _[i]) is_ 1-homogeneous in the full
_∼_
_parameters._

_Proof. Multiplying by each W increases approx-_
imate homogeneity by 1, and ReLU preserves
approximate homogeneity. So the input to LN
is 3-homogeneous. Thus, its output is 1_∼_ _∼_
homogeneous, and adding X preserves approximate 1-homogeneity.

Together, these results suggest that the pre-norm
transformer output is 1-homogeneous, assuming
_∼_
its input is 1-homogeneous. This precondition
_∼_
for the input holds in the “base case” of standard
embeddings. By induction, we can imagine the
output of a biasless pre-norm transformer encoder
of any depth to be 1-homogeneous.
_∼_
Interestingly, the homogeneity arguments do not
work out if we instead consider the post-norm transformer architecture (Xiong et al., 2020).

### E Sum Derivation

**E.1** **Aligned Case**

Assume that ∥θt∥≈ 0. Then,


_√_
So, in conclusion: ∥θt∥ _[∝]_ _t._
_∼_

### F Weight Decay


**E.2** **Misaligned Case**

First, we derive the sum approximation for ∥θt∥.
We start with the fact that θt+1 = θt + δt and
misalignment, i.e., θt[⊤] _[·][ δ][t]_ [= 0][.]

_∥θt+1∥[2]_ = (θt + δt) · (θt + δt)

= ∥θt∥[2] + θt[⊤][δ][t] [+][ ∥][δ][t][∥][2]

= ∥θt∥[2] + ∥δt∥[2]


_t_
�

= ∥θ0∥[2] + _∥δi∥[2]._

_i=0_


Taking the square root of both sides, ∥θt∥ is roughly
proportional to _i=0_

[�][t] _[∥][δ][i][∥][2][.]_
Next, we show how to solve the integral, similarly to §E.1.


_∥θt∥[2][ ∝]_
_∼_

_∥θt∥[2][ ∝]_
_∼_


�
_∥δt∥[2]dt_

�
_ηt[2][∥][θ][t][∥][2][d][t]_


d

dt _[∥][θ][t][∥][2][ ∝∼]_ _[η]t[2][∥][θ][t][∥][2]_

d∥θt∥[2]

_t_ [d][t]
_∥θt∥[2][ ∝∼]_ _[η][2]_

�

log ∥θt∥[2][ ∝] _ηt[2][d][t.]_
_∼_


_√_
Now, we plug in the ηt = 1/


_t learning rate:_


log ∥θt∥[2][ ∝]
_∼_


� _√_
(1/

� dt


_t)[2]dt_


_∝∼_ _t_

_∝∼_ log t.


_∥θt∥_ _[∝]_
_∼_


�
_ηt∥θt∥dt_


d

dt _[∥][θ][t][∥∝∼]_ _[η][t][∥][θ][t][∥]_

d∥θt∥

_∥θt∥_ _∝∼_ _ηtdt_

�

log ∥θt∥ _[∝]_ _ηtdt_
_∼_

�� �
_∥θt∥_ _[∝]_ exp _ηtdt_ _._
_∼_


_√_
Plugging in ηt = 1/


_t, we get ∥θt∥_ _[∝]_ exp�√
_∼_


_t�._


Weight decay regularizes the loss by the squared ℓ2
norm, modulated by a decay factor λ. For GD, this
can be written

_δt = −ηt∇θtL −_ _λθt._ (3)

Intuitively, the new term −λθt will influence each
step to point towards 0. Thus, large values of λ
might intuitively be expected to hinder or prevent
norm growth. While we leave developing a more
complete theoretical story to future work, here we
empirically investigate the interplay of a constant


-----

10 1 Norm growth with AdamW after 1 epoch by,

10 2

10 3

10 4 Decreasing norm

Increasing norm
PyTorch default

10 5
10 5 10 4 10 3 10 2 10 1

|Norm growth with AdamW after 1 epoch by,|Col2|
|---|---|
|Decreasing norm Increasing norm PyTorch default||
|||


Figure 6: Norm growth over the first epoch, varying
_η, λ. The triangle shows the default AdamW hyperpa-_
rameters in PyTorch.

learning rate η and weight decay λ by training a variety of transformer language models on Wikitext-2
for 1 epoch.[18] We use the AdamW (Loshchilov and
Hutter, 2017) optimizer, varying λ and η across a
range of common values, keeping all other hyperparameters constant. Fig. 6 visualizes the phase transition for norm growth as a function of λ, η. The
norm growth behavior seems to largely depend on
weight decay, with a threshold for λ lying between
0.01 and 0.001. While the trend likely depends on
the optimizer, we can infer for AdamW at least that
norm growth is probable when λ = 0.01, which is
a common choice, e.g., reflecting default settings in
PyTorch. Thus, while large values of λ will indeed
hinder norm growth, we find preliminary empirical evidence that standard choices ( 0.01) do not
_∼_
prevent it.

181 epoch is chosen because of the computational cost of
running this experiment over a large grid. In §3, we found that
growth continued beyond 1 epoch using the default AdamW
settings.


-----

