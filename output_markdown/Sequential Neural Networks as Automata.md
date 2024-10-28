## Sequential Neural Networks as Automata

### William Merrill[∗]
 Yale University, New Haven, CT, USA Allen Institute for Artificial Intelligence, Seattle, WA, USA william.merrill@yale.edu


### Abstract


This work attempts to explain the types of
computation that neural networks can perform
by relating them to automata. We first define
what it means for a real-time network with
bounded precision to accept a language. A
measure of network memory follows from this
definition. We then characterize the classes of
languages acceptable by various recurrent networks, attention, and convolutional networks.
We find that LSTMs function like counter machines and relate convolutional networks to the
subregular hierarchy. Overall, this work attempts to increase our understanding and ability to interpret neural networks through the
lens of theory. These theoretical insights help
explain neural computation, as well as the relationship between neural networks and natural
language grammar.

### 1 Introduction


In recent years, neural networks have achieved
tremendous success on a variety of natural language processing (NLP) tasks. Neural networks
employ continuous distributed representations of
linguistic data, which contrast with classical discrete methods. While neural methods work well,
one of the downsides of the distributed representations that they utilize is interpretability. It is hard
to tell what kinds of computation a model is capable of, and when a model is working, it is hard to
tell what it is doing.
This work aims to address such issues of interpretability by relating sequential neural networks
to forms of computation that are more well understood. In theoretical computer science, the
computational capacities of many different kinds
of automata formalisms are clearly established.
Moreover, the Chomsky hierarchy links natural

_∗_ Work completed while the author was at Yale University.


language to such automata-theoretic languages
(Chomsky, 1956). Thus, relating neural networks
to automata both yields insight into what general
forms of computation such models can perform,
as well as how such computation relates to natural
language grammar.
Recent work has begun to investigate what
kinds of automata-theoretic computations various
types of neural networks can simulate. Weiss et al.
(2018) propose a connection between long shortterm memory networks (LSTMs) and counter automata. They provide a construction by which
the LSTM can simulate a simplified variant of a
counter automaton. They also demonstrate that
LSTMs can learn to increment and decrement their
cell state as counters in practice. Peng et al.

(2018), on the other hand, describe a connection between the gating mechanisms of several recurrent neural network (RNN) architectures and
weighted finite-state acceptors.
This paper follows Weiss et al. (2018) by analyzing the expressiveness of neural network acceptors under asymptotic conditions. We formalize asymptotic language acceptance, as well as an
associated notion of network memory. We use
this theory to derive computation upper bounds
and automata-theoretic characterizations for several different kinds of recurrent neural networks
(Section 3), as well as other architectural variants like attention (Section 4) and convolutional
networks (CNNs) (Section 5). This leads to a
fairly complete automata-theoretic characterization of sequential neural networks.
In Section 6, we report empirical results investigating how well these asymptotic predictions describe networks with continuous activations learned by gradient descent. In some cases,
networks behave according to the theoretical predictions, but we also find cases where there is gap
between the asymptotic characterization and ac

-----

tual network behavior.
Still, discretizing neural networks using an
asymptotic analysis builds intuition about how the
network computes. Thus, this work provides insight about the types of computations that sequential neural networks can perform through the lens
of formal language theory. In so doing, we can
also compare the notions of grammar expressible
by neural networks to formal models that have
been proposed for natural language grammar.

### 2 Introducing the Asymptotic Analysis

To investigate the capacities of different neural
network architectures, we need to first define what
it means for a neural network to accept a language.
There are a variety of ways to formalize language
acceptance, and changes to this definition lead to
dramatically different characterizations.
In their analysis of RNN expressiveness, Siegelmann and Sontag (1992) allow RNNs to perform
an unbounded number of recurrent steps even after the input has been consumed. Furthermore,
they assume that the hidden units of the network
can have arbitrarily fine-grained precision. Under this very general definition of language acceptance, Siegelmann and Sontag (1992) found that
even a simple recurrent network (SRN) can simulate a Turing machine.
We want to impose the following constraints on
neural network computation, which are more realistic to how networks are trained in practice (Weiss
et al., 2018):

1. Real-time: The network performs one iteration of computation per input symbol.

2. Bounded precision: The value of each cell in
the network is representable by O(log n) bits
on sequences of length n.

Informally, a neural sequence acceptor is a network which reads a variable-length sequence of
characters and returns the probability that the input sequence is a valid sentence in some formal
language. More precisely, we can write:

**Definition 2.1 (Neural sequence acceptor). Let X**
be a matrix representation of a sentence where
each row is a one-hot vector over an alphabet Σ.
A neural sequence acceptor 1[ˆ] is a family of functions parameterized by weights θ. For each θ and
**X, the function** 1[ˆ][θ] takes the form

1ˆ[θ] : X _p_ (0, 1).
_�→_ _∈_


Figure 1: With sigmoid activations, the network on the
left accepts a sequence of bits if and only if xt = 1 for
some t. On the right is the discrete computation graph
that the network approaches asymptotically.

In this definition, 1[ˆ] corresponds to a general architecture like an LSTM, whereas 1[ˆ][θ] represents a
specific network, such as an LSTM with weights
that have been learned from data.
In order to get an acceptance decision from
this kind of network, we will consider what happens as the magnitude of its parameters gets very
large. Under these asymptotic conditions, the internal connections of the network approach a discrete computation graph, and the probabilistic output approaches the indicator function of some language (Figure 1).

**Definition 2.2 (Asymptotic acceptance). Let L be**
a language with indicator function 1L. A neural sequence acceptor 1[ˆ] with weights θ asymptotically accepts L if

lim 1ˆ[Nθ] = 1L.
_N_ _→∞_

Note that the limit of 1[ˆ][Nθ] represents the function
that 1[ˆ][Nθ] converges to pointwise.[1]

Discretizing the network in this way lets us analyze it as an automaton. We can also view this
discretization as a way of bounding the precision
that each unit in the network can encode, since it is
forced to act as a discrete unit instead of a continuous value. This prevents complex fractal representations that rely on infinite precision. We will see
later that, for every architecture considered, this
definition ensures that the value of every unit in
the network is representable in O(log n) bits on
sequences of length n.
It is important to note that real neural networks
can learn strategies not allowed by the asymptotic
definition. Thus, this way of analyzing neural networks is not completely faithful to their practical

[1https://en.wikipedia.org/wiki/](https://en.wikipedia.org/wiki/Pointwise_convergence)
[Pointwise_convergence](https://en.wikipedia.org/wiki/Pointwise_convergence)


-----

usage. In Section 6, we discuss empirical studies
investigating how trained networks compare to the
asymptotic predictions. While we find evidence
of networks learning behavior that is not asymptotically stable, adding noise to the network during training seems to make it more difficult for the
network to learn non-asymptotic strategies.
Consider a neural network that asymptotically
accepts some language. For any given length, we
can pick weights for the network such that it will
correctly decide strings shorter than that length
(Theorem A.1).
Analyzing a network’s asymptotic behavior also
gives us a notion of the network’s memory. Weiss
et al. (2018) illustrate how the LSTM’s additive
cell update gives it more effective memory than
the squashed state of an SRN or GRU for solving counting tasks. We generalize this concept
of memory capacity as state complexity. Informally, the state complexity of a node within a network represents the number of values that the node
can achieve asymptotically as a function of the sequence length n. For example, the LSTM cell state
will have O(n[k]) state complexity (Theorem 3.3),
whereas the state of other recurrent networks has
_O(1) (Theorem 3.1)._
State complexity applies to a hidden state sequence, which we can define as follows:
**Definition 2.3 (Hidden state). For any sentence**
**X, let n be the length of X. For 1** _t_ _n, the k-_
_≤_ _≤_
length hidden state ht with respect to parameters
_θ is a sequence of functions given by_

**h[θ]t** [:][ X][ �→] **[v][t]** _[∈]_ [R][k][.]

Often, a sequence acceptor can be written as a
function of an intermediate hidden state. For example, the output of the recurrent layer acts as a
hidden state in an LSTM language acceptor. In recurrent architectures, the value of the hidden state
is a function of the preceding prefix of characters,
but with convolution or attention, it can depend on
characters occurring after index t.
The state complexity is defined as the cardinality of the configuration set of such a hidden state:
**Definition 2.4 (Configuration set). For all n, the**
configuration set of hidden state hn with respect
to parameters θ is given by

� �
_M_ (h[θ]n[) =] lim _n_ [(][X][)][ |][ n][ =][ |][X][|] _._
_N_ _→∞_ **[h][Nθ]**

where **X** is the length, or height, of the sentence
_|_ _|_
matrix X.


**Definition 2.5 (Fixed state complexity). For all n,**
the fixed state complexity of hidden state hn with
respect to parameters θ is given by

m(h[θ]n[) =] _M_ (hθn[)] _._
��� ���

**Definition 2.6 (General state complexity). For all**
_n, the general state complexity of hidden state hn_
is given by

m(hn) = max m(h[θ]n[)][.]
_θ_

To illustrate these definitions, consider a simplified recurrent mechanism based on the LSTM
cell. The architecture is parameterized by a vector
_θ ∈_ R[2]. At each time step, the network reads a bit
_xt and computes_

_ft = σ(θ1xt)_ (1)

_it = σ(θ2xt)_ (2)

_ht = ftht−1 + it._ (3)

When we set θ[+] = ⟨1, 1⟩, ht asymptotically
computes the sum of the preceding inputs. Because this sum can evaluate to any integer between
0 and n, h[θ]n[+] [has a fixed state complexity of]

� �
m _h[θ]n[+]_ = O(n). (4)

However, when we use parameters θ[Id] = 1, 1,
_⟨−_ _⟩_
we get a reduced network where ht = xt asymptotically. Thus,

� �
m _h[θ]n[Id]_ = O(1). (5)

Finally, the general state complexity is the maximum fixed complexity, which is O(n).
For any neural network hidden state, the state
complexity is at most 2[O][(][n][)] (Theorem A.2). This
means that the value of the hidden unit can be
encoded in O(n) bits. Moreover, for every specific architecture considered, we observe that each
fixed-length state vector has at most O(n[k]) state
complexity, or, equivalently, can be represented in
_O(log n) bits._
Architectures that have exponential state complexity, such as the transformer, do so by using
a variable-length hidden state. State complexity
generalizes naturally to a variable-length hidden
state, with the only difference being that ht (Definition 2.3) becomes a sequence of variably sized
objects rather than a sequence of fixed-length vectors.


-----

Now, we consider what classes of languages
different neural networks can accept asymptotically. We also analyze different architectures in
terms of state complexity. The theory that emerges
from these tools enables better understanding of
the computational processes underlying neural sequence models.

### 3 Recurrent Neural Networks

As previously mentioned, RNNs are Turingcomplete under an unconstrained definition of acceptance (Siegelmann and Sontag, 1992). The
classical reduction of a Turing machine to an RNN
relies on two unrealistic assumptions about RNN
computation (Weiss et al., 2018). First, the number of recurrent computations must be unbounded
in the length of the input, whereas, in practice,
RNNs are almost always trained in a real-time
fashion. Second, it relies heavily on infinite precision of the network’s logits. We will see that
the asymptotic analysis, which restricts computation to be real-time and have bounded precision,
severely narrows the class of formal languages that
an RNN can accept.

**3.1** **Simple Recurrent Networks**

The SRN, or Elman network, is the simplest type
of RNN (Elman, 1990):

**Definition 3.1 (SRN layer).**

**ht = tanh(Wxt + Uht−1 + b).** (6)

A well-known problem with SRNs is that they
struggle with long-distance dependencies. One explanation of this is the vanishing gradient problem,
which motivated the development of more sophisticated architectures like the LSTM (Hochreiter
and Schmidhuber, 1997). Another shortcoming of
the SRN is that, in some sense, it has less memory than the LSTM. This is because, while both
architectures have a fixed number of hidden units,
the SRN units remain between 1 and 1, whereas
_−_
the value of each LSTM cell can grow unboundedly (Weiss et al., 2018). We can formalize this
intuition by showing that the SRN has finite state
complexity:

**Theorem 3.1 (SRN state complexity). For any**
_length n, the SRN cell state hn ∈_ R[k] _has state_
_complexity_

m(hn) ≤ 2[k] = O(1).


_Proof. For every n, each unit of hn will be the_
output of a tanh. In the limit, it can achieve either
1 or 1. Thus, for the full vector, the number of
_−_
configurations is bounded by 2[k].

It also follows from Theorem 3.1 that the languages asymptotically acceptable by an SRN are a
subset of the finite-state (i.e. regular) languages.
Lemma B.1 provides the other direction of this
containment. Thus, SRNs are equivalent to finitestate automata.

**Theorem** **3.2** (SRN characterization). _Let_
_L(SRN) denote the languages acceptable by an_
_SRN, and RL the regular languages. Then,_

_L(SRN) = RL._

This characterization is quite diminished compared to Turing completeness. It is also more descriptive of what SRNs can express in practice. We
will see that LSTMs, on the other hand, are strictly
more powerful than the regular languages.

**3.2** **Long Short-Term Memory Networks**

An LSTM is a recurrent network with a complex
gating mechanism that determines how information from one time step is passed to the next.
Originally, this gating mechanism was designed to
remedy the vanishing gradient problem in SRNs,
or, equivalently, to make it easier for the network
to remember long-term dependencies (Hochreiter
and Schmidhuber, 1997). Due to strong empirical performance on many language tasks, LSTMs
have become a canonical model for NLP.

Weiss et al. (2018) suggest that another advantage of the LSTM architecture is that it can use
its cell state as counter memory. They point out
that this constitutes a real difference between the
LSTM and the GRU, whose update equations do
not allow it to increment or decrement its memory
units. We will further investigate this connection
between LSTMs and counter machines.

**Definition 3.2 (LSTM layer).**

**ft = σ(W[f]** **xt + U[f]** **ht−1 + b[f]** ) (7)

**it = σ(W[i]xt + U[i]ht−1 + b[i])** (8)

**ot = σ(W[o]xt + U[o]ht−1 + b[o])** (9)

**˜ct = tanh(W[c]xt + U[c]ht−1 + b[c])** (10)

**ct = ft ⊙** **ct−1 + it ⊙** **˜ct** (11)

**ht = ot** _f_ (ct). (12)
_⊙_


-----

In (12), we set f to either the identity or tanh
(Weiss et al., 2018), although tanh is more standard in practice. The vector ht is the output that is
received by the next layer, and ct is an unexposed
memory vector called the cell state.

**Theorem 3.3 (LSTM state complexity). The**
_LSTM cell state cn ∈_ R[k] _has state complexity_

m(cn) = O(n[k]).

_Proof. At each time step t, we know that the con-_
figuration sets of ft, it, and ot are each subsets of
_{0, 1}[k]. Similarly, the configuration set of ˜ct is a_
subset of 1, 1 . This allows us to rewrite the
_{−_ _}[k]_
elementwise recurrent update as

lim (13)
_N_ _→∞[[][c][t][]][i][ = lim]N_ _→∞[[][f][t][]][i][[][c][t][−][1][]][i][ + [][i][t][]][i][[][˜c][t][]][i]_

= lim (14)
_N_ _→∞_ _[a][[][c][t][−][1][]][i][ +][ b]_

where a 0, 1 and b 1, 0, 1 .
_∈{_ _}_ _∈{−_ _}_
Let St be the configuration set of [ct]i. At each
time step, we have exactly two ways to produce a
new value in St that was not in St−1: either we
decrement the minimum value in St−1 or increment the maximum value. It follows that

_|St| = 2 + |St−1|_ (15)

=⇒|Sn| = O(n). (16)

For all k units of the cell state, we get

m(cn) ≤|Sn|[k] = O(n[k]). (17)

The construction in Theorem 3.3 produces an
automaton closely resembling a classical counter
machine (Fischer, 1966; Fischer et al., 1968). Its
restricted memory give us an upper bound on the
expressive power of the LSTM:

**Theorem 3.4 (LSTM upper bound). Let CL be the**
_real-time log-space languages.[2]_ _Then,_

_L(LSTM)_ CL.
_⊆_

Theorem 3.4 constitutes a very tight upper
bound on the expressiveness of LSTM computation. Asymptotically, LSTMs are not powerful

2Revision: A previous version stated “real-time counter
languages” with no definition. Depending on the definition
of counter languages, the claim may or may not hold. We
clarify that CL is intended to be the “log-space” languages,
where space is measured in bits. See Merrill (2020) for further discussion.


enough to model even the deterministic contextfree language w#w[R].

Weiss et al. (2018) show how the LSTM can
simulate a simplified variant of the counter machine. Combining these results, we see that
the asymptotic expressiveness of the LSTM falls
somewhere between the general and simplified
counter languages. This suggests counting is a
good way to understand the behavior of LSTMs.

**3.3** **Gated Recurrent Units**

The GRU is a popular gated recurrent architecture
that is in many ways similar to the LSTM (Cho
et al., 2014). Rather than having separate forget
and input gates, the GRU utilizes a single gate that
controls both functions.

**Definition 3.3 (GRU layer).**

**zt = σ(W[z]xt + U[z]ht−1 + b[z])** (18)

**rt = σ(W[r]xt + U[r]ht−1 + b[r])** (19)

**ut = tanh** �W[u]xt + U[u](rt ⊙ **ht−1) + b[u][�]**

(20)

**ht = zt ⊙** **ht−1 + (1 −** **zt) ⊙** **ut.** (21)

Weiss et al. (2018) observe that GRUs do not
exhibit the same counter behavior as LSTMs on
languages like a[n]b[n]. As with the SRN, the GRU
state is squashed between 1 and 1 (20). Taken
_−_
together, Lemmas C.1 and C.2 show that GRUs,
like SRNs, are finite-state.

**Theorem 3.5 (GRU characterization).**

_L(GRU) = RL._

**3.4** **RNN Complexity Hierarchy**

Synthesizing all of these results, we get the following complexity hierarchy:

RL = L(SRN) = L(GRU) (22)

SCL _L(LSTM)_ CL. (23)
_⊂_ _⊆_ _⊆_

Basic recurrent architectures have finite state,
whereas the LSTM is strictly more powerful than
a finite-state machine.

### 4 Attention

Attention is a popular enhancement to sequenceto-sequence (seq2seq) neural networks (Bahdanau
et al., 2014; Chorowski et al., 2015; Luong et al.,
2015). Attention allows a network to recall specific encoder states while trying to produce output.


-----

In the context of machine translation, this mechanism models the alignment between words in the
source and target languages. More recent work
has found that “attention is all you need” (Vaswani
et al., 2017; Radford et al., 2018). In other words,
networks with only attention and no recurrent connections perform at the state of the art on many
tasks.
An attention function maps a query vector and a
sequence of paired key-value vectors to a weighted
combination of the values. This lookup function is
meant to retrieve the values whose keys resemble
the query.

**Definition 4.1 (Dot-product attention). For any n,**
define a query vector q ∈ R[l], matrix of key vectors
**K ∈** R[nl], and matrix of value vectors V ∈ R[nk].
Dot-product attention is given by

attn(q, K, V) = softmax(qK[T] )V.

In Definition 4.1, softmax creates a vector of
similarity scores between the query q and the key
vectors in K. The output of attention is thus
a weighted sum of the value vectors where the
weight for each value represents its relevance.
In practice, the dot product qK[T] is often scaled
by the square root of the length of the query vector
(Vaswani et al., 2017). However, this is only done
to improve optimization and has no effect on expressiveness. Therefore, we consider the unscaled
version.
In the asymptotic case, attention reduces to a
weighted average of the values whose keys maximally resemble the query. This can be viewed as
an arg max operation.

**Theorem 4.1 (Asymptotic attention). Let t1, .., tm**
_be the subsequence of time steps that maximize_
**qkt.[3]** _Asymptotically, attention computes_


1
lim
_N_ _→∞_ [attn (][q][,][ K][,][ V][) = lim]N _→∞_ _m_


_m_
�

**vti.**
_i=1_


**Corollary** **4.1.1** (Asymptotic attention with
unique maximum). If qkt has a unique maximum
_over 1_ _t_ _n, then attention asymptotically_
_≤_ _≤_
_computes_

lim
_N_ _→∞_ [attn (][q][,][ K][,][ V][) = lim]N _→∞_ [arg max]vt **[qk][t][.]**

3To be precise, we can define a maximum over the similarity scores according to the order given by

_f > g ⇐⇒_ _Nlim→∞_ _[f]_ [(][N] [)][ −] _[g][(][N]_ [)][ >][ 0][.] (24)


Now, we analyze the effect of adding attention
to an acceptor network. Because we are concerned
with language acceptance instead of transduction,
we consider a simplified seq2seq attention model
where the output sequence has length 1:

**Definition 4.2 (Attention layer). Let the hidden**
state v1, .., vn be the output of an encoder network
where the union of the asymptotic configuration
sets over all vt is finite. We attend over Vt, the
matrix stacking v1, .., vt, by computing

**ht = attn(W[q]vt, Vt, Vt).**

In this model, ht represents a summary of the
relevant information in the prefix v1, .., vt. The
query that is used to attend at time t is a simple
linear transformation of vt.
In addition to modeling alignment, attention improves a bounded-state model by providing additional memory. By converting the state of the
network to a growing sequence Vt instead of a
fixed length vector vt, attention enables 2[Θ(][n][)]

state complexity.

**Theorem 4.2 (Encoder state complexity). The full**
_state of the attention layer has state complexity_

m(Vn) = 2[Θ(][n][)].

The O(n[k]) complexity of the LSTM architecture means that it is impossible for LSTMs to
copy or reverse long strings. The exponential state
complexity provided by attention enables copying,
which we can view as a simplified version of machine translation. Thus, it makes sense that attention is almost universal in machine translation architectures. The additional memory introduced by
attention might also allow more complex hierarchical representations.
A natural follow-up question to Theorem 4.2 is
whether this additional complexity is preserved in
the attention summary vector hn. Attending over
**Vn does not preserve exponential state complex-**
ity. Instead, we get an O(n[2]) summary of Vn.

**Theorem 4.3 (Summary state complexity). The**
_attention summary vector has state complexity_

m(hn) = O(n[2]).

With minimal additional assumptions, we can
show a more restrictive bound: namely, that the
complexity of the summary vector is finite. Appendix D discusses this in more detail.


-----

### 5 Convolutional Networks

While CNNs were originally developed for image
processing (Krizhevsky et al., 2012), they are also
used to encode sequences. One popular application of this is to build character-level representations of words (Kim et al., 2016). Another example is the capsule network architecture of Zhao
et al. (2018), which uses a convolutional layer as
an initial feature extractor over a sentence.

**Definition 5.1 (CNN acceptor).**

**ht = tanh** �W[h](xt−k∥..∥xt+k) + b[h][�] (25)

**h+ = maxpool(H)** (26)

_p = σ(W[a]h+ + b[a])._ (27)

In this network, the k-convolutional layer (25)
produces a vector-valued sequence of outputs.
This sequence is then collapsed to a fixed length
by taking the maximum value of each filter over
all the time steps (26).
The CNN acceptor is much weaker than the
LSTM. Since the vector ht has finite state, we
see that L(CNN) RL. Moreover, simple reg_⊆_
ular languages like a[∗]ba[∗] are beyond the CNN
(Lemma E.1). Thus, the subset relation is strict.

**Theorem 5.1 (CNN upper bound).**

_L(CNN)_ RL.
_⊂_

So, to arrive at a characterization of CNNs, we
should move to subregular languages. In particular, we consider the strictly local languages
(Rogers and Pullum, 2011).

**Theorem 5.2 (CNN lower bound). Let SL be the**
_strictly local languages. Then,_

SL _L(CNN)._
_⊆_

Notably, strictly local formalisms have been
proposed as a computational model for phonological grammar (Heinz et al., 2011). We might take
this to explain why CNNs have been successful at
modeling character-level information.
However, Heinz et al. (2011) suggest that a generalization to the tier-based strictly local languages
is necessary to account for the full range of phonological phenomena. Tier-based strictly local grammars can target characters in a specific tier of the
vocabulary (e.g. vowels) instead of applying to
the full string. While a single convolutional layer
cannot utilize tiers, it is conceivable that a more
complex architecture with recurrent connections
could.


### 6 Empirical Results

In this section, we compare our theoretical characterizations for asymptotic networks to the empirical performance of trained neural networks with
continuous logits.[4]

**6.1** **Counting**

The goal of this experiment is to evaluate which
architectures have memory beyond finite state. We
train a language model on a[n]b[n]c with 5 _n_
_≤_ _≤_
1000 and test it on longer strings (2000 _n_
_≤_ _≤_
2200). Predicting the c character correctly while
maintaining good overall accuracy requires O(n)
states. The results reported in Table 1 demonstrate
that all recurrent models, with only two hidden
units, find a solution to this task that generalizes
at least over this range of string lengths.

Weiss et al. (2018) report failures in attempts
to train SRNs and GRUs to accept counter languages, unlike what we have found. We conjecture
that this stems not from the requisite memory, but
instead from the different objective function we
used. Our language modeling training objective is
a robust and transferable learning target (Radford
et al., 2019), whereas sparse acceptance classification might be challenging to learn directly for long
strings.

Weiss et al. (2018) also observe that LSTMs
use their memory as counters in a straightforwardly interpretable manner, whereas SRNs and
GRUs do not do so in any obvious way. Despite this, our results show that SRNs and GRUs
are nonetheless able to implement generalizable
counter memory while processing strings of significant length. Because the strategies learned by
these architectures are not asymptotically stable,
however, their schemes for encoding counting are
less interpretable.

**6.2** **Counting with Noise**

In order to abstract away from asymptotically unstable representations, our next experiment investigates how adding noise to an RNN’s activations
impacts its ability to count. For the SRN and GRU,
noise is added to ht−1 before computing ht, and
for the LSTM, noise is added to ct−1. In either
case, the noise is sampled from the distribution
_N_ (0, 0.1[2]).

[4https://github.com/viking-sudo-rm/](https://github.com/viking-sudo-rm/nn-automata)
[nn-automata](https://github.com/viking-sudo-rm/nn-automata)


-----

|m|No Noise Acc Acc on c|Noise Acc Acc on c|
|---|---|---|
|SRN O(1) GRU O(1) LSTM O(nk)|100.0 100.0 99.9 100.0 99.9 100.0|49.9 100.0 53.9 100.0 99.9 100.0|


Table 1: Generalization performance of language models trained on a[n]b[n]c. Each model has 2 hidden units.


m Val Acc Gen Acc

LSTM _O(n[k])_ 94.0 51.6

LSTM-Attn 2[Θ(][n][)] 100.0 51.7

LSTM _O(n[k])_ 92.5 73.3

StackNN 2[Θ(][n][)] 100.0 100.0

Table 2: Max validation and generalization accuracies
on string reversal over 10 trials. The top section shows
our seq2seq LSTM with and without attention. The
bottom reports the LSTM and StackNN results of Hao
et al. (2018). Each LSTM has 10 hidden units.

The results reported in the right column of Table 1 show that the noisy SRN and GRU now fail
to count, whereas the noisy LSTM remains successful. Thus, the asymptotic characterization of
each architecture matches the capacity of a trained
network when a small amount of noise is introduced.
From a practical perspective, training neural
networks with Gaussian noise is one way of improving generalization by preventing overfitting
(Bishop, 1995; Noh et al., 2017). From this point
of view, asymptotic characterizations might be
more descriptive of the generalization capacities
of regularized neural networks of the sort necessary to learn the patterns in natural language data
as opposed to the unregularized networks that are
typically used to learn the patterns in carefully curated formal languages.

**6.3** **Reversing**

Another important formal language task for assessing network memory is string reversal. Reversing requires remembering a Θ(n) prefix of
characters, which implies 2[Θ(][n][)] state complexity.
We frame reversing as a seq2seq transduction
task, and compare the performance of an LSTM
encoder-decoder architecture to the same architecture augmented with attention. We also report the
results of Hao et al. (2018) for a stack neural network (StackNN), another architecture with 2[Θ(][n][)]

state complexity (Lemma F.1).
Following Hao et al. (2018), the models were


trained on 800 random binary strings with length
_N_ (10, 2) and evaluated on strings with length
_∼_
_N_ (50, 5). As can be seen in Table 2, the LSTM
_∼_
with attention achieves 100.0% validation accuracy, but fails to generalize to longer strings. In
contrast, Hao et al. (2018) report that a stack neural network can learn and generalize string reversal flawlessly. In both cases, it seems that having
2[Θ(][n][)] state complexity enables better performance
on this memory-demanding task. However, our
seq2seq LSTMs appear to be biased against finding a strategy that generalizes to longer strings.

### 7 Conclusion

We have introduced asymptotic acceptance as a
new way to characterize neural networks as automata of different sorts. It provides a useful and
generalizable tool for building intuition about how
a network works, as well as for comparing the
formal properties of different architectures. Further, by combining asymptotic characterizations
with existing results in mathematical linguistics,
we can better assess the suitability of different architectures for the representation of natural language grammar.
We observe empirically, however, that this discrete analysis fails to fully characterize the range
of behaviors expressible by neural networks. In
particular, RNNs predicted to be finite-state solve
a task that requires more than finite memory. On
the other hand, introducing a small amount of
noise into a network’s activations seems to prevent it from implementing non-asymptotic strategies. Thus, asymptotic characterizations might be
a good model for the types of generalizable strategies that noise-regularized neural networks trained
on natural language data can learn.

### Acknowledgements

Thank you to Dana Angluin and Robert Frank for
their insightful advice and support on this project.

|m|Val Acc Gen Acc|
|---|---|
|LSTM O(nk) LSTM-Attn 2Θ(n)|94.0 51.6 100.0 51.7|
|LSTM O(nk) StackNN 2Θ(n)|92.5 73.3 100.0 100.0|


-----

### References

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2014. [Neural machine translation by jointly](https://arxiv.org/abs/1409.0473)
[learning to align and translate.](https://arxiv.org/abs/1409.0473) _arXiv preprint_
_arXiv:1409.0473._

[Chris M. Bishop. 1995. Training with noise is equiv-](https://doi.org/10.1162/neco.1995.7.1.108)
[alent to Tikhonov regularization. Neural Comput.,](https://doi.org/10.1162/neco.1995.7.1.108)
7(1):108–116.

Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger
Schwenk, and Yoshua Bengio. 2014. [Learning](https://doi.org/10.3115/v1/D14-1179)
[phrase representations using RNN encoder–decoder](https://doi.org/10.3115/v1/D14-1179)
[for statistical machine translation. In Proceedings of](https://doi.org/10.3115/v1/D14-1179)
_the 2014 Conference on Empirical Methods in Nat-_
_ural Language Processing (EMNLP), pages 1724–_
1734, Doha, Qatar. Association for Computational
Linguistics.

[Noam Chomsky. 1956. Three models for the descrip-](https://ieeexplore.ieee.org/abstract/document/1056813)
[tion of language. IRE Transactions on information](https://ieeexplore.ieee.org/abstract/document/1056813)
_theory, 2(3):113–124._

Jan K Chorowski, Dzmitry Bahdanau, Dmitriy
Serdyuk, Kyunghyun Cho, and Yoshua Bengio.
[2015. Attention-based models for speech recogni-](http://papers.nips.cc/paper/5847-attention-based-models-for-speech-recognition)
[tion. In Advances in neural information processing](http://papers.nips.cc/paper/5847-attention-based-models-for-speech-recognition)
_systems, pages 577–585._

[Jeffrey L Elman. 1990. Finding structure in time. Cog-](https://onlinelibrary.wiley.com/doi/pdf/10.1207/s15516709cog1402_1)
_nitive science, 14(2):179–211._

Patrick C Fischer. 1966. [Turing machines with re-](https://www.sciencedirect.com/science/article/pii/S0019995866800037)
[stricted memory access. Information and Control,](https://www.sciencedirect.com/science/article/pii/S0019995866800037)
9(4):364–379.

Patrick C. Fischer, Albert R. Meyer, and Arnold L.
Rosenberg. 1968. [Counter machines and counter](https://doi.org/10.1007/BF01694011)
[languages. Mathematical systems theory, 2(3):265–](https://doi.org/10.1007/BF01694011)
283.

Yiding Hao, William Merrill, Dana Angluin, Robert
Frank, Noah Amsel, Andrew Benz, and Simon
[Mendelsohn. 2018. Context-free transductions with](https://arxiv.org/abs/1809.02836)
[neural stacks. arXiv preprint arXiv:1809.02836.](https://arxiv.org/abs/1809.02836)

Jeffrey Heinz, Chetan Rawal, and Herbert G Tan[ner. 2011. Tier-based strictly local constraints for](https://dl.acm.org/citation.cfm?id=2002750)
[phonology.](https://dl.acm.org/citation.cfm?id=2002750) In Proceedings of the 49th Annual
_Meeting of the Association for Computational Lin-_
_guistics:_ _Human Language Technologies:_ _short_
_papers-Volume 2, pages 58–64. Association for_
Computational Linguistics.

Sepp Hochreiter and J¨urgen Schmidhuber. 1997.

[Long short-term memory.](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735) _Neural computation,_
9(8):1735–1780.

Yoon Kim, Yacine Jernite, David Sontag, and Alexan[der M Rush. 2016. Character-aware neural language](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12489)
[models. In AAAI, pages 2741–2749.](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12489)

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hin[ton. 2012. Imagenet classification with deep con-](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networ)
[volutional neural networks. In Advances in neural](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networ)
_information processing systems, pages 1097–1105._


Minh-Thang Luong, Hieu Pham, and Christopher D
[Manning. 2015. Effective approaches to attention-](https://arxiv.org/abs/1508.04025)
[based neural machine translation.](https://arxiv.org/abs/1508.04025) _arXiv preprint_
_arXiv:1508.04025._

[William Merrill. 2020. On the linguistic capacity of](http://arxiv.org/abs/2004.06866)
[real-time counter automata.](http://arxiv.org/abs/2004.06866)

Hyeonwoo Noh, Tackgeun You, Jonghwan Mun, and
[Bohyung Han. 2017. Regularizing deep neural net-](http://papers.nips.cc/paper/7096-regularizing-deep-neural-networks-by-noise-its-interpretation-and-optimization)
[works by noise: Its interpretation and optimization.](http://papers.nips.cc/paper/7096-regularizing-deep-neural-networks-by-noise-its-interpretation-and-optimization)
In Advances in Neural Information Processing Sys_tems 30: Annual Conference on Neural Information_
_Processing Systems 2017, 4-9 December 2017, Long_
_Beach, CA, USA, pages 5115–5124._

Hao Peng, Roy Schwartz, Sam Thomson, and Noah A
[Smith. 2018. Rational recurrences. arXiv preprint](https://arxiv.org/abs/1808.09357)
_arXiv:1808.09357._

Alec Radford, Karthik Narasimhan, Tim Salimans, and
Ilya Sutskever. 2018. [Improving language under-](https://s3-us-west-2. amazonaws. com/openai-assets/research-covers/language-unsupervised/language_ understanding_paper. pdf)
[standing by generative pre-training.](https://s3-us-west-2. amazonaws. com/openai-assets/research-covers/language-unsupervised/language_ understanding_paper. pdf)

Alec Radford, Jeffrey Wu, Rewon Child, David Luan,
[Dario Amodei, and Ilya Sutskever. 2019. Language](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
[models are unsupervised multitask learners.](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) _URL_
_https://openai. com/blog/better-language-models._

James Rogers and Geoffrey K Pullum. 2011. [Aural](https://link.springer.com/article/10.1007/s10849-011-9140-2)
[pattern recognition experiments and the subregular](https://link.springer.com/article/10.1007/s10849-011-9140-2)
[hierarchy. Journal of Logic, Language and Infor-](https://link.springer.com/article/10.1007/s10849-011-9140-2)
_mation, 20(3):329–342._

[Hava T. Siegelmann and Eduardo D. Sontag. 1992. On](https://doi.org/10.1145/130385.130432)
[the computational power of neural nets. In Proceed-](https://doi.org/10.1145/130385.130432)
_ings of the Fifth Annual Workshop on Computational_
_Learning Theory, COLT ’92, pages 440–449, New_
York, NY, USA. ACM.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
[Kaiser, and Illia Polosukhin. 2017. Attention is all](http://papers.nips.cc/paper/7181-attention-is-all-you-need)
[you need. In Advances in Neural Information Pro-](http://papers.nips.cc/paper/7181-attention-is-all-you-need)
_cessing Systems, pages 5998–6008._

Gail Weiss, Yoav Goldberg, and Eran Yahav. 2018.

[On the practical computational power of finite pre-](http://arxiv.org/abs/1805.04908)
[cision RNNs for language recognition.](http://arxiv.org/abs/1805.04908) _CoRR,_
abs/1805.04908.

Wei Zhao, Jianbo Ye, Min Yang, Zeyang Lei, Suofei
[Zhang, and Zhou Zhao. 2018. Investigating capsule](https://arxiv.org/abs/1804.00538)
[networks with dynamic routing for text classifica-](https://arxiv.org/abs/1804.00538)
[tion. arXiv preprint arXiv:1804.00538.](https://arxiv.org/abs/1804.00538)

### A Asymptotic Acceptance and State Complexity

**Theorem A.1 (Arbitary approximation). Let** 1[ˆ] be
_a neural sequence acceptor for L._ _For all m,_
_there exist parameters θm such that, for any string_
**x1, .., xn with n < m,**

� �
1ˆ[θ][m](X) = 1L(X)


-----

_where [_ ] rounds to the nearest integer.

_·_

_Proof. Consider a string X. By the definition of_
asymptotic acceptance, there exists some number
_MX which is the smallest number such that, for all_
_N ≥_ _MX,_

1ˆNθ(X) − 1L(X) _< 1_ (28)
��� ��� 2

� �
=⇒ 1ˆ[Nθ](X) = 1L(X). (29)


decide to accept or reject in the final layer according to the linearly separable disjunction


�
_at ⇐⇒_

_i∈F_


�

ðt(j, α). (32)
_⟨j,α⟩∈δ[−][1](i)_


Now, let Xm be the set of sentences X with length
less than m. Since Xm is finite, we pick θm just
by taking
_θm = max_ (30)
**X∈Xm** _[M][X][θ.]_

**Theorem A.2 (General bound on state complex-**
ity). Let ht be a neural network hidden state. For
_any length n, it holds that_

m(hn) = 2[O][(][n][)].

_Proof. The number of configurations of hn can-_
not be more than the number of distinct inputs to
the network. By construction, each xt is a one-hot
vector over the alphabet Σ. Thus, the state complexity is bounded according to

m(hn) ≤|Σ|[n] = 2[O][(][n][)].

### B SRN Lemmas

**Lemma B.1 (SRN lower bound).**

RL _L(SRN)._
_⊆_

_Proof. We must show that any language accept-_
able by a finite-state machine is SRN-acceptable.
We need to asymptotically compute a representation of the machine’s state in ht. We do this by
storing all values of the following finite predicate
at each time step:

ðt(i, α) ⇐⇒ _qt−1(i) ∧_ _xt = α_ (31)

where qt(i) is true if the machine is in state i at
time t.
Let F be the set of accepting states for the machine, and let δ[−][1] be the inverse transition relation.
Assuming ht asymptotically computes ðt, we can


We now show how to recurrently compute ðt at
each time step. By rewriting qt−1 in terms of the
previous ðt−1 values, we get the following recurrence:

�
ðt(i, α) ⇐⇒ _xt = α ∧_ ðt(j, β).

_⟨j,β⟩∈δ[−][1](i)_

(33)
Since this formula is linearly separable, we can
compute it in a single neural network layer from
**xt and ht−1.**
Finally, we consider the base case. We need to
ensure that transitions out of the initial state work
out correctly at the first time step. We do this by
adding a new memory unit ft to ht which is always rewritten to have value 1. Thus, if ft−1 = 0,
we can be sure we are in the initial time step.
For each transition out of the initial state, we add
_ft−1 = 0 as an additional term to get_

ðt(0, α) ⇐⇒ _xt = α ∧_
�ft−1 = 0 ∨ � ðt(j, β)�. (34)

_⟨j,β⟩∈δ[−][1](0)_


This equation is still linearly separable and guarantees that the initial step will be computed correctly.

### C GRU Lemmas

These results follow similar arguments to those in
Subsection 3.1 and Appendix B.

**Lemma C.1 (GRU state complexity). The GRU**
_hidden state has state complexity_

m(hn) = O(1).

_Proof. The configuration set of zt is a subset of_
0, 1 . Thus, we have two possibilities for each
_{_ _}[k]_
value of [ht]i: either [ht−1]i or [ut]i. Furthermore,
the configuration set of [ut]i is a subset of {−1, 1}.
Let St be the configuration set of [ht]i. We can
describe St according to

_S0 = {0}_ (35)

_St ⊆_ _St−1 ∪{−1, 1}._ (36)


-----

This implies that, at most, there are only three possible values for each logit: 1, 0, or 1. Thus, the
_−_
state complexity of hn is

m(hn) ≤ 3[k] = O(1). (37)

**Lemma C.2 (GRU lower bound).**

RL _L(GRU)._
_⊆_

_Proof. We can simulate a finite-state machine us-_
ing the ð construction from Theorem 3.2. We
compute values for the following predicate at each
time step:

�
ðt(i, α) ⇐⇒ _xt = α ∧_ ðt−1(j, β).

_⟨j,β⟩∈δ[−][1](i)_

(38)
Since (38) is linearly separable, we can store ðt
in our hidden state ht and recurrently compute its
update. The base case can be handled similarly to
(34). A final feedforward layer accepts or rejects
according to (32).


_Proof. By the general upper bound on state com-_
plexity (Theorem A.2), we know that m(Vn) =
2[O][(][n][)]. We now show the lower bound.
We pick weights θ in the encoder such that vt =
**xt. Thus, m(vt[θ][) =][ |][Σ][|][ for all][ t][. Since the values]**
at each time step are independent, we know that

m(Vn[θ] [) =][ |][Σ][|][n] (41)

∴ m(Vn) = 2[Ω(][n][)]. (42)

**Lemma D.2 (Theorem 4.3 restated). The attention**
_summary vector has state complexity_

m(hn) = O(n[2]).

_Proof. By Theorem 4.1, we know that_


1
lim
_N_ _→∞_ **[h][n][ = lim]N** _→∞_ _m_


_m_
�

**vti.** (43)
_i=1_


By construction, there is a finite set S containing
all possible configurations of every vt. We bound
the number of configurations for each vti by |S| to
get


_n_
�

_S_ _m_ _S_ _n[2]_ = O(n[2]). (44)
_|_ _|_ _≤|_ _|_
_m=1_


### D Attention Lemmas

**Theorem** **D.1** (Theorem 4.1 restated). _Let_
_t1, .., tm be the subsequence of time steps that_
_maximize qkt. Asymptotically, attention computes_


m(hn) ≤


1
lim
_N_ _→∞_ [attn (][q][,][ K][,][ V][) = lim]N _→∞_ _m_


_m_
�

**vti.**
_i=1_


_Proof. Observe that, asymptotically, softmax(u)_
approaches a function

lim � _m1_ if ut = max(u)
_N_ _→∞_ [softmax(][N] **[u][)][t][ =]** 0 otherwise.

(39)
Thus, the output of the attention mechanism reduces to the sum


**Lemma D.3 (Attention state complexity lower**
bound). The attention summary vector has state
_complexity_
m(hn) = Ω(n).

_Proof. Consider the case where keys and values_
have dimension 1. Further, let the input strings
come from a binary alphabet Σ = 0, 1 . We pick
_{_ _}_
parameters θ in the encoder such that, for all t,


lim
_N_ _→∞_ _[v][t][ =]_


�
0 if xt = 0
(45)
1 otherwise


and limN _→∞_ _kt = 1. Then, attention returns_


_n_
�

_vt =_ _[l]_ (46)

_n_

_t=1_


where l is the number of t such that xt = 1. We
can vary the input to produce l from 1 to n. Thus,
we have

m(h[θ]n[) =][ n] (47)

∴ m(hn) = Ω(n). (48)


lim
_N_ _→∞_


lim
_N_ _→∞_


_m_
�

_i=1_


1

(40)
_m_ **[v][t][i][.]**


**Lemma D.1 (Theorem 4.2 restated). The full state**
_of the attention layer has state complexity_

m(Vn) = 2[Θ(][n][)].


-----

**Lemma D.4 (Attention state complexity with**
unique maximum). If, for all X, there exists a
_unique t[∗]_ _such that t[∗]_ = maxt qnkt, then

m(hn) = O(1).

_Proof. If qnkt has a unique maximum, then by_
Corollary 4.1.1 attention returns

lim (49)
_N_ _→∞_ [arg max]vt **[qk][t][ = lim]N** _→∞_ **[v][t][∗][.]**

By construction, there is a finite set S which is a
superset of the configuration set of vt[∗]. Thus,

m(hn) ≤|S| = O(1). (50)

**Lemma D.5 (Attention state complexity with**
ReLU activations). If limN _→∞_ **vt ∈{0, ∞}[k]** _for_
1 _t_ _n, then_
_≤_ _≤_

m(hn) = O(1).

_Proof. By Theorem 4.1, we know that attention_
computes


**Definition E.1 (Strictly k-local grammar). A**
strictly k-local grammar over an alphabet Σ is a
set of allowable k-grams S. Each s _S takes the_
_∈_
form
_s_ �Σ # �k
_∈_ _∪{_ _}_

where # is a padding symbol for the start and end
of sentences.

**Definition E.2 (Strictly local acceptance). A**
strictly k-local grammar S accepts a string σ if,
at each index i,

_σiσi+1..σi+k−1 ∈_ _S._

**Lemma E.2 (Implies Theorem 5.2). A k-CNN can**
_asymptotically accept any strictly 2k+1-local lan-_
_guage._

_Proof. We construct a k-CNN to simulate a_
strictly 2k +1-local grammar. In the convolutional
layer (25), each filter identifies whether a particular invalid 2k +1-gram is matched. This condition
is a conjunction of one-hot terms, so we use tanh
to construct a linear transformation that comes out
to 1 if a particular invalid sequence is matched,
and 1 otherwise.
_−_
Next, the pooling layer (26) collapses the filter
values at each time step. A pooled filter will be
1 if the invalid sequence it detects was matched
somewhere and 1 otherwise.
_−_
Finally, we decide acceptance (27) by verifying
that no invalid pattern was detected. To do this,
we assign each filter a weight of 1 use a thresh_−_
old of _K +_ [1]
_−_ 2 [where][ K][ is the number of invalid]

patterns. If any filter has value 1, then this sum
will be negative. Otherwise, it will be [1]

2 [. Thus,]
asymptotic sigmoid will give us a correct acceptance decision.


1
lim
_N_ _→∞_ **[h][n][ = lim]N** _→∞_ _m_


_m_
�

**vti.** (51)
_i=1_


This sum evaluates to a vector in 0,, which
_{_ _∞}[k]_
means that

m(hn) ≤ 2[k] = O(1). (52)

Lemma D.5 applies if the sequence v1, .., vn is
computed as the output of ReLU. A similar result holds if it is computed as the output of an unsquashed linear transformation.

### E CNN Lemmas

**Lemma E.1 (CNN counterexample).**

_a[∗]ba[∗]_ _/_ _L(CNN)._
_∈_

_Proof. By contradiction._ Assume we can write
a network with window size k that accepts any
string with exactly one b and reject any other
string. Consider a string with two bs at indices i
and j where _i_ _j_ _> 2k + 1. Then, no column_
_|_ _−_ _|_
in the network receives both xi and xj as input.
When we replace one b with an a, the value of
**h+ remains the same. Since the value of h+ (26)**
fully determines acceptance, the network does not
accept this new string. However, the string now
contains exactly one b, so we reach a contradiction.


### F Neural Stack Lemmas

Refer to Hao et al. (2018) for a definition of the
StackNN architecture. The architecture utilizes a
differentiable data structure called a neural stack.
We show that this data structure has 2[Θ(][n][)] state
complexity.

**Lemma F.1 (Neural stack state complexity). Let**
**Sn ∈** R[nk] _be a neural stack with a feedforward_
_controller. Then,_

m(Sn) = 2[Θ(][n][)].


-----

_Proof. By the general state complexity bound_
(Theorem A.2), we know that m(Sn) = 2[O][(][n][)]. We
now show the lower bound.
The stack at time step n is a matrix Sn ∈ R[nk]

where the rows correspond to vectors that have
been pushed during the previous time steps. We
set the weights of the controller θ such that, at
each step, we pop with strength 0 and push xt with
strength 1. Then, we have

m(S[θ]n[) =][ |][Σ][|][n] (53)

∴ m(Sn) = 2[Ω(][n][)]. (54)


-----

