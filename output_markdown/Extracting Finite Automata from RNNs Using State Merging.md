## Extracting Finite Automata from RNNs Using State Merging

**William Merrill** [* 1] **Nikolaos Tsilivis** [* 1]


### Abstract


One way to interpret the behavior of a blackbox
recurrent neural network (RNN) is to extract from
it a more interpretable discrete computational
model, like a finite state machine, that captures its
behavior. In this work, we propose a new method
for extracting finite automata from RNNs inspired
by the state merging paradigm from grammatical
inference. We demonstrate the effectiveness of
our method on the Tomita languages benchmark,
where we find that it is able to extract faithful
automata from RNNs trained on all languages in
the benchmark. We find that extraction performance is aided by the number of data provided
during the extraction process, as well as, curiously,
whether the RNN model is trained for additional
epochs after perfectly learning its target language.
We use our method to analyze this phenomenon,
finding that training beyond convergence is useful because it leads to compression of the internal
state space of the RNN. This finding demonstrates
how our method can be used for interpretability
and analysis of trained RNN models.

### 1. Introduction


Interpretability poses a problem for deep-learning based
sequence models like recurrent neural networks (RNNs).
When trained on language data and other structured discrete sequences, such models implicitly acquire structural
rules over the input sequences that modulate their classification decisions. However, it is often difficult to recover
the discrete rules encoded in the parameters of the network.
Traditionally, it is useful to be able to understand the rules a
model is using to reach its classification decision, as models
like decision trees or finite automata allow. Not only does
this address practical deployment concerns like being able
to explain model decisions or debug faulty inputs, but it
also has more foundational scientific uses: e.g., inducing

*Equal contribution 1Center for Data Science, NYU. Correspondence to: William Merrill <willm@nyu.edu>, Nikolaos Tsilivis
_<nt2231@nyu.edu>._


a model of grammar over natural sentences can help test
and build linguistic theories of the syntax of natural language. Another potential use of extracted automata is to
study the training dynamics of neural networks: automata
can be extracted from different checkpoints and compared
to understand how the strategy of an RNN evolves over
training. In §5.5, we will discuss analysis of our method
that may provide insight on the implicit regularization of
RNN training.

How, then, can we gain insight into the discrete rules acquired by RNN models? One family of approaches is to
extract finite automata that capture the behavior of an RNN,
and use the extracted state machine for interpretability. This
problem can be seen as a special case of grammar induction
(Gold, 1978; Angluin, 1987): the task of, given samples
from a formal language, inferring a small automaton that
will generate the data. Thus, past work on RNN extraction
has generally adapted techniques from the grammar induction literature. For example, Weiss et al. (2018b) do this
by adapting the L[∗] query learning algorithm for grammar
induction to work with an RNN oracle. Lacroce et al. (2021)
leverage spectral learning, a different framework for grammar induction, to infer weighted automata from RNNs, as
opposed to the more standard deterministic automata. Other
work has used k-means clustering on the RNN hidden states
to extract a graph of states (Wang et al., 2018).

_L[∗]_ is an active learning approach that learns via queries of
two forms: membership of strings in L, and equivalence
queries comparing a candidate DFA and the true underlying
DFA. Thus, Weiss et al. (2018b) assume blackboxes computing these oracles are available at train time, which may
be problematic for the potentially expensive equivalence
queries. The k-means method of Wang et al. (2018) does
not have this problem, although it comes with no theoretical
guarantees of faithfulness, and requires that the number of
states must be picked as a hyperparameter. In this work,
we will present an alternative extraction method that does
not require expensive equivalence queries, and where the
number of states does not need to be set in advance.

To meet these goals, we will leverage state merging (Oncina
& Garc´ıa, 1992; Lang et al., 1998; Sebban & Janodet, 2003),
another grammar induction paradigm, to extract deterministic finite automata (DFAs) from RNNs. State merging


-----

**Extracting Finite Automata from RNNs Using State Merging**


works by first building a prefix tree from a finite dataset: a
deterministic automaton that simply memorizes the training
data, and will not recognize any held-out strings beyond
the finite set used to build the prefix tree. The next step
of the process is to compress this prefix tree by merging
states together, using a strategy (or ‘policy’) π. This process
both reduces the automaton size and introduces loops between states. Through this, the automaton gains the ability
to generalize to an infinite set of held-out strings. Of course,
the nature of this generalization depends on how π is computed. For grammatical inference, π is generally computed
by verifying simple constraints are met: in order to merge
two states, the states must agree in whether or not they are
final states. We will add an additional constraint that the
RNN representations associated with each state must be
close, thus enforcing that our learned automaton reflects the
structure of the RNN’s implicit state space.

In summary, we introduce state merging as a method to extract DFAs from blackbox RNNs. We first show in §5.2 that
our state merging method enables RNN extraction on all 7
Tomita languages (Tomita, 1982), the standard benchmark
for evaluating RNN extraction. As an additional contribution, we use our method to show that continuing to train an
RNN past convergence in development accuracy makes it
easier to extract a DFA from it, and the implicit state space
of the resulting DFA is simplified (§5.5). We discuss speculatively how this phenomenon may have implications for
understanding the implicit regularization of RNN training.

### 2. Background

**2.1. Recurrent Neural Networks**

For our purposes, a generalized RNN is a function mapping
a sequence of symbols {wi}i[n]=1 [to a sequence of labels]
_{yi}i[n]=1[. In the abstract, the RNN has a state vector][ h][i][ ∈]_ [R][d]

that satisfies the following form for some gating function f :

_hi+1 = f_ (hi, wi+1)

**yi+1 = argmax(w[⊤]hi+1 + b).**

In principle, our method can be applied to RNNs with any
gating function f, but, in the paper, we will use the simple
recurrent gating (Elman, 1990):

**hi+1 = tanh(U** **hi + V xi+1),** (1)

where xi is a vector embedding of token wi. Other common variants include Long Short-Term Memory networks
(LSTMs; Hochreiter & Schmidhuber, 1997) and Gated Recurrent Units (GRUs; Cho et al., 2014).

**2.2. Deterministic Finite Automata**

Automata have a long history of study in theoretical computer science, linguistics, and related fields, originally hav

_a_

start _q0_ _q1_

_b_

_Figure 1. A DFA specified as ⟨Σ, Q, q0, δ, F_ _⟩_ with Σ = {a, b},
_Q_ = _{q0, q1}, δ such that δ(q0, a)_ = _q1, δ(q1, b)_ = _q0,_
and F = _{q0}._ It recognizes the language (ab)[∗] =
_{ϵ, ab, abab, ababab, · · · }. The ∗_ symbol denotes Kleene star,
i.e., 0 or more repetitions of a string.

ing been formalized in part as a discrete model of neural
networks (Kleene et al., 1956; Minsky, 1956). A deterministic finite automaton (DFA) can be specified as a tuple
_A = ⟨Σ, Q, q0, δ, F_ _⟩, where:_

  - Σ is a finite input alphabet (set of tokens);

  - Q is a set of states, along with a special “undefined”
state, where _Q;_
_∅_ _∅̸∈_

  - q0 ∈ _Q is an initial state;_

  - δ : (Q ) Σ (Q ) is a transition function
_∪{∅}_ _×_ _→_ _∪{∅}_
such that _σ_ Σ, δ( _, σ) =_ ;
_∀_ _∈_ _∅_ _∅_

 - F _Q is a set of accepting states._
_⊆_


Now that we have formally specified this model, how does
one do computation with it? Informally, when processing
a string w ∈ Σ[n], A starts in state q0, and each token in the
input string causes it to transition to a different state according to δ. Once all input tokens have been consumed, the
machine either accepts or rejects the input string depending
on whether the final state qn ∈ _F_ . More formally, we define
the state after the prefix w:i as:

_qi = δ(qi−1, wi)._

We then say that A accepts a string w Σ[n] if and only if
_∈_
_qn ∈_ _F_ . The regular language recognized by A is the set of
strings it accepts, i.e.,

_L(A) = {w | qn(w) ∈_ _F_ _}._

**Example 1. Consider the DFA in Figure 1. It recognizes**
the language (ab)[⋆] = _ϵ, ab, abab, ababab,_, and it is
_{_ _· · · }_
the minimal automaton that does so.

  - For the string ab, the computation would start from
_q0 (the initial state - common for any string), then the_
automaton would traverse to q1 (δ(q0, a) = q1), and,
subsequently, to q0 (δ(q1, b) = q0). Since the final
state q1 belongs to F after consuming all input tokens,
we say that the DFA accepts ab.

  - For the string aba, the DFA identically reaches state
_q0 after consuming the prefix ab. However, the final_
_a causes the DFA to transition back to q1. Because_
_q1 ̸∈_ _F_, the DFA rejects aba.


-----

**Extracting Finite Automata from RNNs Using State Merging**



  - For the string abb, the DFA also reaches state q0 after
consuming the prefix ab. At this point, the transition
_δ(q0, b) = ∅, so the state will be ∅_ (“error”) for the rest
of the string. Since _F_, the DFA rejects abb.
_∅̸∈_

**2.3. Power of DFAs**

DFAs are equivalent to nondeterministic finite automata,
both recognizing the regular languages (Kleene et al., 1956).
The regular languages form the lowest level of the Chomsky hierarchy (Chomsky, 1956), and intuitively represent
languages that can be recognized with memory that does
not grow with the sequence length. In contrast, more powerful classes allow the memory of the recognizer to grow
with the length of the input string. For example, contextfree languages correspond to nondetermistic finite automata
augmented with a stack data structure (Chomsky, 1956), enabling O(n) memory on strings of length n. Other classes
in the Chomsky hierarchy correspond to the languages recognizable by even more complex automata: for example,
the recursive languages correspond to the set of languages
that can be recognized by a Turing machine.

**2.4. Connections of RNNs to DFAs**

At a high level, DFAs and RNNs can both be used to match
the language recognition task specification: essentially, binary classification over strings. RNNs with continuous
activation functions and unbounded computation time and
precision have been shown to be Turing-complete, meaning
they can recognize languages that are not regular (Siegelmann & Sontag, 1992). However, more recent literature
has argued that these assumptions differ substantially from
the type of RNNs trained in modern deep learning (Weiss
et al., 2018a; Merrill, 2019). The same work suggests that
the regular languages are a much more reasonable model
for the capacity of RNNs as trainable deep learning model.
We now briefly summarize this line of research.

Some of the original motivation for formalizing finite automata came from trying to develop a model of computation
for early connectionist versions of neural networks (Kleene
et al., 1956; Minsky, 1956). Thus, by design, RNNs with
threshold activation functions are equivalent in terms of
the set of languages they can recognize to finite automata
(Merrill et al., 2020). More recent work has shown that the
infinite parameter norm limits of simple RNNs and GRUs
are equivalent to finite automata in expressive power (Merrill, 2019), and found that language learning experiments
with these networks can often be predicted by the theoretical
capacity of these “saturated” infinite-norm networks (Merrill et al., 2020). For instance, Weiss et al. (2018a) found
that RNNs and GRUs cannot “count” (a capability requiring
more than finite state), unlike the more complicated LSTM.
Combining this theoretical and empirical evidence suggests


that simple RNNs and GRUs behave as finite-state models,
rather than models whose states grow with the input length.
This perspective supports using deterministic finite-state
automata as a target for extraction with RNNs.

We note, however, that if we would like to do extraction
for LSTMs or other complex RNNs, it could make sense to
extract a counter automaton (Fischer et al., 1968) rather than
a finite automaton, which we believe state merging could be
adapted to accommodate in future work.

### 3. Method

We now describe our state merging method for extracting
DFAs from RNNs. Our method assumes a blackbox RNN
model that supplies the following desiderata:

1. Hidden States Given an input string x ∈ Σ[n], and for
each 0 ≤ _i ≤_ _n, the RNN produces a vector hi ∈_ R[k]

that encodes the full state of the model after processing
the prefix of w up to index i. Thus, h0 corresponds to
a representation for the empty string ϵ. We will write
**H to mean the full (n + 1)** _k hidden state matrix._
_×_

2. Recognition Decisions Given an input string x ∈ Σ[n],
the RNN produces a vector ˆy (0, 1)[n][+1] that scores
_∈_
the probability that each prefix of x is a valid string in
the formal language defined by the RNN.

Our method can be applied to any model satisfying these
properties. However, as discussed in the previous section,
it is most motivated to apply it to simple RNNs or GRUs,
which have been shown to resemble finite state machines. If
our method is applied to an LSTM or other complex RNN
variant, the extracted DFA will potentially be a finite-state
approximation of more complex behavior.

Our state merging algorithm has two parts: first, we construct a prefix tree using the recognition decisions ˆy. Next,
we merge states in the prefix tree according to the RNN
hidden states H.

**3.1. Building the Prefix Tree**

A prefix tree, or trie, is a DFA that can be built to correctly
recognize any language L over all prefixes of a finite support
of strings {wi}i[m]=1[. Each state in the tree represents a prefix]
of some wi, and is labelled according to whether that prefix
is a valid string in L. Paths of transitions are added to the
tree to connect prefixes together in the natural way, e.g.,
_wi = ab would induce three states qϵ, qa, qab and the path_
_qϵ_ _a qa_ _b qab (see top row in Figure 3 for an example)._
_→_ _→_

To build the prefix tree, we sample a new training set of
strings {wi}i[m]=1[, and record as labels][ ˆy][(][w][i][)][, i.e., whether]
each prefix of every wi is a valid string in L. Note that this


-----

**Extracting Finite Automata from RNNs Using State Merging**


training set is distinct (and generally much smaller) than the
training set used to train the RNN. After its construction, we
identify with each state qj a feature vector φ(qj) = h|w|(w),
where w is the prefix corresponding to qj.

**3.2. Merging States**

Once the prefix tree is built, we define a policy π(qi, qj) that
compares states, and predicts whether or not to merge them.
Let κ (0, 1) be a hyperparameter. We specify π to merge
_∈_
_qi_ _qj when both of the following two constraints are met:_
_→_

1. Consistency: qi ∈ _F ⇐⇒_ _qj ∈_ _F_

2. Similarity: cos(φ(qi), φ(qj)) > 1 − _κ_

The consistency constraint is standard in grammar induction:
it guarantees that each step preserves the performance of the
automaton across observed positive and negative examples,
and thus that the new automaton is consistent with the behavior of the RNN on the training set. We add the similarity
constraint to enforce that the automaton’s representations
reflect the true structure of the underlying state space in the
RNN. Thus, two states are merged if and only if doing so
would preserve recognition behavior on the training set and
reflects the internal structure of the RNN state space.

If both of these conditions are met, then we merge qi → _qj._
To do this, we delete the state qi from the graph, and choose
the transitions to/from qj by taking the union of all transitions involving qi or qj.[1]. This merge operation is not fully
symmetric, since the representation φ(q[′]) is inherited from
_qj after merging qi and qj. On the other hand, the conditions_
to merge two states are defined symmetrically. Thus, the
algorithm will potentially reach different results depending
on the enumeration order for qi and qj. In practice, this will
not be an issue, as long as κ is set sufficiently high, since
in this case, the vectors φ(qi) and φ(qj) will be effectively
equivalent from the point of view of the algorithm.

**3.3. Postprocessing**

Finally, after reducing the automaton via state merging, we
can apply a DFA minimization step (Hopcroft, 1971) to
reduce the size of the extracted DFA while preserving the
language it recognizes. DFA minimization is an operation
that takes a regular language defined by a DFA and returns
the DFA with the smallest number of states that recognizes
that regular language, which is unique up to isomorphism.
Thus, DFA minimization is semantically different than state
merging: while state merging preserves recognition decisions over the training set, minimization is guaranteed to
preserve recognition decisions over all strings. Thus, apply
1This procedure may yield a non-deterministic finite automaton
which is equivalent though to a DFA (Kleene et al., 1956).


ing minimization alone to the initial prefix tree is not able
to produce a DFA that generalizes beyond the training set.
Our goal in applying minimization after state merging is
to make the behavior of the resulting automaton easier to
visualize and evaluate without changing it.

**3.4. Theoretical Motivation**

Our proposed algorithm is justified in the sense that it extracts the state transitions of the saturated version of the
RNN it receives as input. A saturated RNN with a tanh
non-linearity will have state vectors h 1 . Thus, a
_∈{±_ _}[d]_
saturated RNN has a finite number of states over which the
update rule (1) acts as DFA transition function (cf. Merrill,
2019). Given the discontinuity of the RNN state space, a
cosine similarity greater than _[d][−]d_ [1] ensures that two state vec
tors are the same. Trained RNNs have been found to become
approximately saturated (Karpathy et al., 2015a), suggesting
the saturated network should closely capture their behavior.
The following proposition, whose details and proof can be
found in the Appendix, captures this intuition, while it also
provides a way to select the similarity hyperparameter κ
based on the level of the saturation of the RNN.

**Proposition 1. Let h1, h2 ∈** R[d] be two normalized state
vectors, **h[˜]1,** **h[˜]2 ∈{±1}[d]** their saturated versions and assume that the RNN is ϵ-saturated with respect to these states,
i.e., ∥hi − **h[˜]i∥2 ≤** _ϵ, i ∈{1, 2}. Then, if cos(h1, h2) ≥_

1 − _κ with_ _[√]κ <_ _√2_ � _√1d_ �, the two vectors represent

_[−]_ _[ϵ]_

the same state on the DFA / saturated RNN (h[˜]1 = h[˜]2).


In practice, one can measure the level of saturation (Merrill
et al., 2021) and select κ from the expression above, but in
our experiments we found that is not necessary, as a very
small value of κ, together with the postprocessing step of
DFA minimization suffices for successful DFA extraction.

### 4. Data and Models

**4.1. Tomita Languages**

The Tomita languages are a standard formal language
benchmark used for evaluating grammar induction systems
(Tomita, 1982) and RNN extraction (Weiss et al., 2018b;
Wang et al., 2018). Specifically, the benchmark consists of
seven regular languages. All languages are defined over the
binary alphabet Σ2 = {a, b}. The languages are numbered
1-7 such that the difficulty of learnability (in an intuitive,
informal sense) increases with number. Slight variation exists in the definition of these languages; we use the version
reported by Weiss et al. (2018b), which is fully documented
in Table 1.


-----

**Extracting Finite Automata from RNNs Using State Merging**

### 5. Extraction Results

|#|Definition|
|---|---|
|1 2 3 4 5 6 7|a∗ (ab)∗ Odd # of a’s must be followed by even # of b’s All strings without the trigram aaa Strings w where # (w) and # (w) are even a b Strings w where # a(w) # b(w) ≡3 b∗a∗b∗a∗|


_Table 1. Definitions of the Tomita languages. Let #σ(w) denote_
the number of occurrences of token σ in string w. Let ≡3 denote equivalence mod 3. |Q| denotes the number of states in the
minimum DFA for each language.

**4.2. Training Details**

To train RNN language recognizers for some formal language L, we need data that supervises which strings fall in
_L. Fixing a maximum sequence length n, we sample data_
_{(x, y)}, where x ∈_ Σ[n]2 [is a string, and][ y][ ∈{][0][,][ 1][}][n][+1][ is a]
zero-indexed vector of language recognition decisions for
each prefix of n. For example, given x = ab,

_y0 = 1 ⇐⇒_ _ϵ ∈_ _L_

_y1 = 1 ⇐⇒_ _a ∈_ _L_

_y2 = 1 ⇐⇒_ _ab ∈_ _L._

where ϵ denotes the empty string. To enforce that the dataset
is roughly balanced across sequence lengths, we sample half
the x uniformly over Σ[n]2 [, and, for the other half, enforce]
that the full string x must be valid in L. Given some x, the
**y’s are deterministic to compute. We use a string length of**
_n = 100 for the training set (with 100, 000 examples), and_
_n = 200 for a development set (with 1, 000 examples)._

We train the RNNs for 22 epochs, choosing the best model
by validating with accuracy on a development set.[2] The
architecture consists of an embedding layer (dimension 10),
followed by an RNN layer (dimension 100), followed by a
linear classification head that predicts language membership
for each position in the sequence. We use the AdamW
optimizer with default hyperparameters.

Before evaluating our method for RNN extraction, we verify
that our trained RNNs reach 100% accuracy. We do this
using a held-out generalization set. The sequence length is
greater in the generalization set than in training, so strong
performance requires generalizing to new lengths. In practice, we find that all RNN language recognizers converge to
100% generalization accuracy within a few epochs.

2As the RNNs converge to 100% accuracy quickly, we break
ties by taking the highest epoch to achieve 100%, which in all
cases turns out to be the final epoch.


In this section, we evaluate the proposed merging method
on how well it describes the behavior of the original RNN.
More specifically, we first assess whether the extracted DFA
matches the predictions of the RNN on the Tomita languages, and then investigate how different hyperparameters,
like the number of data used to built the trie and the dissimilarity tolerance κ, affect the final output of our algorithm.
In summary, we find that our method can extract DFAs
matching the original RNN across all 7 languages.

**5.1. Extraction Details**

To build the prefix tree for each language, we use training
strings of length 10 where each one is either a member of
the language or a random string (with equal probability),
and, then, use the trained RNN to compute labels and the
representations for each state. We then apply state merging
to compress the trie, and the final DFA is evaluated on a
held-out set that contains 1, 000 strings of uniform random
length between 0 and 50. Unless otherwise stated, we set
_κ = 0.01. Finally, a note on the training data: we vary the_
number of them to evaluate our algorithm’s dependence on
it, but the number of examples is always orders of magnitude
smaller than the number used to train the RNN.

**5.2. Extraction Faithfulness**

As expected, the merged DFA retains the initial performance
over the training set, ensured by the Consistency constraint,
while the Similarity one furnishes it with generalization
capabilities. As a sanity check, see Figure 2 (left) for the accuracy of the extracted DFA for Tomita 2, one of the “easy”
languages, vs. the accuracy of the initial prefix tree. For
Tomita 5, a harder language, the method requires approximately 25 training strings in the prefix tree to merge down
to the perfectly correct DFA (right side of Figure 2). In
general, while the prefix tree may reject previously unseen
negative examples, it will never accept previously unseen
positive ones. This is the reason why we observe such a
large gap in development accuracy between the initial prefix
tree and the final merged DFA.

In Table 2, we summarize the results of the extraction on
all languages for a fixed number of training data (n = 300).
We see that in almost all cases the extracted DFA matches
the predictions of the RNN. For Tomita 1-6, the algorithm
always finds the correct DFA. For Tomita 7, it achieves near
100% accuracy on every run and recovers the fully correct
DFA 3/5 times. We conclude that our algorithm returns a
faithful descriptor of the RNN language recognizers for the
Tomita languages. Table 2 also compares against a k-means
baseline (Wang et al., 2018). See §A for details. As shown,
_k-means finds the correct DFA for Tomita 1-6, but performs_


-----

**Extracting Finite Automata from RNNs Using State Merging**

State merging _k-means_ right side of Figure 4, we show that the number of states in

# Acc _Q[ˆ]_ Acc _Q[ˆ]_ _Q_ the final DFA decreases with the number of data, reaching
_|_ _|_ _|_ _|_ _|_ _|_

1 100. 0. 1 100. 0. 1 1 the true minimum DFA for Tomita 1-6. For Tomita 7, we
_±_ _±_

2 100. 0. 2 100. 0. 2 2 first recover the minimum DFA when using 135 data points,
_±_ _±_

3 100. 0. 4 100. 0. 4 4 but state merging does not always produce the true minimum
_±_ _±_

4 100. 0. 3 100. 0. 3 3 DFA, even at 500 data points.
_±_ _±_

5 100. 0. 4 100. 0. 4 4
_±_ _±_

6 100. ± 0. 3 100. ± 0. 3 3 **5.5. Effect of Training Beyond Convergence**

7 **99.62 ± 0.55** **4** 57.35 ± 0.25 1 4

In all cases, our RNNs converged to 100% development

_Table 2. Mean accuracy together with standard deviation of the_ accuracy within one or two epochs. However, we find that
extracted DFA on the 7 Tomita languages. Randomness induced by continuing to train beyond this point can improve the ability
5 random seeds for sampling data to build the prefix tree. “|Q[ˆ]|” is of our method to extract the true gold-standard DFA.
the smallest extracted DFA size after minimization. “|Q|” reports

First, we investigate the effect of continued training on

the size of the true minimum DFA for each language. State merging
is our method; k-means is a baseline based on Wang et al. (2018). learning Tomita 6, one of the “difficult” Tomita languages,

using representations extracted from an RNN after 2 and
20 epochs of training respectively. The results are shown
in Figure 5. Although the RNN development accuracy is

roughly at chance for Tomita 7.

100% at both checkpoints, the extraction results are now

Our results are not directly comparable with Weiss et al. different; as the figure illustrates, a correct automaton can
(2018b) on the same data, as their method learns from active be extracted from the “overly” trained network with much
membership and equivalence queries, while ours is designed fewer data than its less trained ancestor. This suggests that
for the more constrained setting of a static dataset. However, additional training is somehow improving or simplifying
we note that Weiss et al. (2018b) were also able to extract the quality of the representations, even after development
faithful DFAs for all 7 Tomita languages using their L[⋆] accuracy has converged to 100%. The behavior across all 7

method. Unlike L[∗], however, our method does not make languages is similar; see §B.
use of potentially expensive equivalence queries.

Next, we study the effect of continued training on the complexity of the extracted automaton, measured in the number

**5.3. Effect of Similarity Threshold**

of states. As shown in the left side of Figure 6, the number

Now, we assess qualitatively the role of the similarity thresh- of states in DFA obtained through state merging (but before
old κ on the output of the state merging algorithm. We use minimization) tends to decrease gradually with more addias case study Tomita 2, whose recognizing automata are tional training for Tomita 1-6, despite some rapid upward
easy to visualize and interpret. As we see in Figure 3, the spikes. This suggests that additional training is perhaps simchoice of κ affects crucially the output of the algorithm. plifying the structure of the RNN’s state space by merging
With large tolerance κ = .5, we “overmerge” states result- redundant states together. On the right, we can see that for
ing with a trivial 2-state DFA, that accepts only the empty Tomita 1-6, training for enough additional epochs brings
string. Gradually decreasing κ produces the desirable ef- the size of the minimized DFA down to the ideal minimum
fect. For instance, for κ = .4 we find an almost minimal DFA size. Together, these results suggest that additional
DFA that describes our language. Finally, for a very strict training may be simplifying the implicit RNN state space
threshold 1 0.99 that decides only to merge states whose to remove redundant state representations, thereby improv_−_
100-dimensional representations are very well aligned, we ing our ability to extract the true minimum DFA for the
recover a correct, but highly redundant DFA. Applying min- language it recognizes. We call this phenomenon implicit
imization to this DFA produces the correct 2-state DFA. The _merging induced by the training procedure._
connection between the number of states in the unminimized
DFA and the quality of the representations afforded by the

**Speculative Explanations** Why should training beyond

RNN is further discussed in Section 5.5.

convergence lead to easier extraction? One potential explanation is the “saturation” phenomenon of neural net training

**5.4. Effect of Data Size**

(Karpathy et al., 2015b; Merrill et al., 2021): if training

Across all 7 Tomita languages, using more data to build the consistency increases the parameter 2-norm (which we find
prefix tree improves our ability to extract the correct DFA. to hold for our RNNs), then training for more time should
As seen in Figure 4, our method reaches 100% accuracy lead the RNN to more closely approximate infinite-norm
at matching the predictions of the RNN for all languages, RNNs. Infinite-norm RNNs can be directly viewed as DFAs
although Tomita 7 has high variance even at the end. On the (Merrill, 2019), which may explain why it is easier to ex
|#|State merging Acc Qˆ | ||k-means Acc Qˆ | ||Q | ||
|---|---|---|---|
|1 2 3 4 5 6 7|100. 0. 1 ± 100. 0. 2 ± 100. 0. 4 ± 100. 0. 3 ± 100. 0. 4 ± 100. 0. 3 ± 99.62 ± 0.55 4|100. 0. 1 ± 100. 0. 2 ± 100. 0. 4 ± 100. 0. 3 ± 100. 0. 4 ± 100. 0. 3 ± 57.35 0.25 1 ±|1 2 4 3 4 3 4|


-----

**Extracting Finite Automata from RNNs Using State Merging**


100

95

90

85

80

75

70

65

60


40


100

90


80

70


60

50

|Col1|Col2|Col3|Tomita 2|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
|||||||||
|||||||train initial trai dev|n|
|||||||||
|||||||||
|||||||initial dev||
|||||||||
|||||||||
|||||||||
|||||||||

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Tomita 5|Col11|Col12|Col13|Col14|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||train initial trai dev|n|
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||initial dev||
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||


0 20 40 60 80 100

#data


0 20 40 60 80 100

#data


_Figure 2. Faithfulness of extracted DFA. For Tomita 2 (left) and Tomita 5 (right), we extract DFAs that (i) are consistent with the training_
set, and (ii) reach 100% dev accuracy. Notice that the initial prefix tree records trivial accuracy on the dev set (red line). Trend line shows
the average across 3 random seeds (different datasets), and the shaded region denotes one std deviation.


start _q0_


(a) Prefix tree


_a_


tract a DFA from a network trained significantly beyond
convergence. Implicit merging is also consistent with the information bottleneck theory of deep learning (Shwartz-Ziv
& Tishby, 2017) that identifies two phases during training;
first, a data-fitting period and then a compressing one. Interestingly, it has been observed that it is the saturating
nature of non-linearities that yields the compression phase
(Saxe et al., 2018), further supporting our previous explanation. More speculatively, it is possible that the benefit
of training beyond convergence for extraction could be related to “grokking” (Power et al., 2021): a phenomenon
where generalization on synthetic formal language tasks
begins to improve after continuing to train for hundreds of
epochs past convergence in training accuracy. Along these
lines, it would be interesting for future work to continue
investigating the mechanism through which training beyond
convergence can improve the ease of RNN extraction, as
this may provide interesting insight into the implicit regularization that RNNs receive during training.

### 6. Conclusion


_a, b_

_a, b_

start _q0_ _q1_


start _q0_ _q1_

_b_

_b_

_b_

_q2_ _q3_ _a, b_


(b) κ = 0.5

_q1_

_a_

start _q0_


_b_


(c) κ = 0.4

_q5_

_b_ _q10_
_a_

_q8_

_b_

_a_ _q9_

_a_


(d) κ = 0.01

_Figure 3. Initial prefix tree and resulting merged DFAs for different_
values of κ. Language: Tomita 2.


We have shown how state merging can be used to extract
automata from RNNs that capture both the decision behavior and representational structure of the original blackbox
RNN. Using state merging, we were able to extract faithful
automata from RNNs trained on all 7 Tomita languages,
demonstrating the effectiveness of our method. For future
work, it would be useful to find ways to extend the state
merging extraction algorithm to scale to larger state spaces
and alphabet sizes. One interesting empirical finding is that
continuing to train an RNN after it has perfectly learned the
target language improves the sample efficiency of extracting
an automaton from it. Our analysis of this implicit merg

-----

**Extracting Finite Automata from RNNs Using State Merging**


100

90


80

70


25

20


60

50


15

10


#states in minimized DFA

Tom1
Tom2
Tom3
Tom4
Tom5
Tom6
Tom7

0 100 200 300 400 500

#data


40


dev acc of merged DFA

Tom1
Tom2
Tom3
Tom4
Tom5
Tom6
Tom7

0 100 200 300 400 500

#data


5

0

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
||||||Tom|||
|||||||Tom|1|
|||||||Tom Tom Tom|2 3 4|
|||||||Tom Tom|5 6|
||||||Tom||7|

|Col1|Col2|Col3|Col4|Col5|Tom Tom|Col7|1 2|
|---|---|---|---|---|---|---|---|
|||||||Tom Tom Tom|3 4 5|
||||||Tom Tom|Tom Tom|6 7|
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||


_Figure 4. Left: The DFA’s accuracy at reconstructing the RNN on the development set as the number of data used to build the prefix_
tree increases. Right: the number of states in the minimized DFA obtained through state merging, which plateaus for Tomita 1-6, and
sometimes reaches the ideal value for Tomita 7. Trend line shows the median across 5 random seeds (different datasets), and the 0.25–0.75
percentile region is shaded. The prefix tree is built from sentences of length 15 here.


100

90


100

90


80

70


80

70


60

50


60

50


40


Tomita 6, epoch2

train
initial train
dev
initial dev

0 20 40 60 80 100

#data


40


Tomita 6, epoch20

train
initial train
dev
initial dev

0 20 24 40 60 80 100

#data

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||train initial trai dev initial dev|n|
||||||||

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
||||||||||
||||||||||
||||||||train initial trai dev initial dev|n|
||||||||||


_Figure 5. Accuracy of extracted DFA vs number of data for two different sets of representations for Tomita 6. Left: Representations from_
an RNN trained for 2 epochs. Right: Representations from an RNN trained for 20 epochs. The RNNs record the same dev accuracy, but
the automata extracted from the longer-trained RNN has higher accuracy. Trend line shows the average across 3 random seeds (different
datasets), and the shaded region denotes one std deviation. κ was set equal to .99.


_ing phenomenon suggests that training past convergence_
may lead to a more robust representation of the underlying
state space within the RNN through an implicit regularizing effect where neighborhoods representing the same state
converge to single vector representations. Under this view,
gradient-descent-like training itself may be viewed as a state
merging process: training may exude a pressure to compress
similar states together, producing a simpler model that may
generalize better to unseen strings.


### References

Angluin, D. Learning regular sets from queries and counterexamples. Inf. Comput., 75:87–106, 1987.


Cho, K., van Merrienboer, B., Bahdanau, D., and Bengio, Y.
On the properties of neural machine translation: Encoderdecoder approaches, 2014.

Chomsky, N. Three models for the description of language.
_IRE Trans. Inf. Theory, 2:113–124, 1956._


Elman, J. L. Finding structure in time. _Cognitive_
_Science, 14(2):179–211, 1990._ ISSN 0364-0213.


-----

**Extracting Finite Automata from RNNs Using State Merging**


200

175

150

125

100

75

50

25


30

25


20

15


10

5


0


#states in merged DFA (200 train points)

Tom1
Tom2
Tom3
Tom4
Tom5
Tom6
Tom7

0 5 10 15 20

#epochs


0


#states in minimized DFA (200 train points)

Tom1
Tom2
Tom3
Tom4
Tom5
Tom6
Tom7

0 5 10 15 20

#epochs

|Col1|Tom1 Tom2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||Tom3 Tom4||||||
||Tom5 Tom6||||||
||Tom7||||||
||||||||
||||||||
||||||||
||||||||
||||||||

|Col1|Col2|Col3|Col4|Col5|Col6|Tom1 Tom2 Tom3|
|---|---|---|---|---|---|---|
||||||||
||||||||
|||||||Tom4 Tom5 Tom6|
||||||||
||||||||
|||||||Tom7|
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||


_Figure 6. Number of states in the DFA achieved through state merging (left) and the minimized version of that DFA (right) training beyond_
convergence. As an artifact of our training logic, Tomita 7 was trained for 24 epochs, and epochs 0-1 were lost. The merged DFA size
tend to gradually decrease for Tomita 1-6 with more training epochs, suggesting implicit merging during training. The minimized DFA
reliably reaches the theoretical minimum DFA for all languages besides Tomita 7, for which it finds the correct minimum DFA 3/5 times.


doi: https://doi.org/10.1016/0364-0213(90)90002-E.
URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/036402139090002E)
[science/article/pii/036402139090002E.](https://www.sciencedirect.com/science/article/pii/036402139090002E)

Fischer, P. C., Meyer, A. R., and Rosenberg, A. L. Counter
machines and counter languages. Mathematical systems
_theory, 2(3):265–283, Sep 1968. ISSN 1433-0490. doi:_
[10.1007/BF01694011. URL https://doi.org/10.](https://doi.org/10.1007/BF01694011)
[1007/BF01694011.](https://doi.org/10.1007/BF01694011)

Gold, E. M. Complexity of automaton identification
from given data. _Information and Control, 37(3):_
302–320, jun 1978. doi: 10.1016/s0019-9958(78)
90562-4. [URL https://doi.org/10.1016%](https://doi.org/10.1016%2Fs0019-9958%2878%2990562-4)
[2Fs0019-9958%2878%2990562-4.](https://doi.org/10.1016%2Fs0019-9958%2878%2990562-4)

Hochreiter, S. and Schmidhuber, J. Long Short-Term Memory. _Neural Computation, 9(8):1735–1780, 11 1997._
ISSN 0899-7667. doi: 10.1162/neco.1997.9.8.1735.
[URL https://doi.org/10.1162/neco.1997.](https://doi.org/10.1162/neco.1997.9.8.1735)
[9.8.1735.](https://doi.org/10.1162/neco.1997.9.8.1735)

Hopcroft, J. An n log n algorithm for minimizing states in
a finite automaton. In Theory of machines and computa_tions, pp. 189–196. Elsevier, 1971._

Karpathy, A., Johnson, J., and Fei-Fei, L. Visualizing and
[understanding recurrent networks, 2015a. URL https:](https://arxiv.org/abs/1506.02078)
[//arxiv.org/abs/1506.02078.](https://arxiv.org/abs/1506.02078)


Lacroce, C., Panangaden, P., and Rabusseau, G. Extracting weighted automata for approximate minimization in
language modelling. arXiv preprint arXiv:2106.02965,
2021.

Lang, K. J., Pearlmutter, B. A., and Price, R. A. Results
of the abbadingo one dfa learning competition and a new
evidence-driven state merging algorithm. In Honavar,
V. and Slutzki, G. (eds.), Grammatical Inference, pp. 1–
12, Berlin, Heidelberg, 1998. Springer Berlin Heidelberg.
ISBN 978-3-540-68707-8.

Merrill, W. Sequential neural networks as automata.
In Proceedings of the Workshop on Deep Learning
_and Formal Languages: Building Bridges, pp. 1–13,_
[August 2019. URL https://www.aclweb.org/](https://www.aclweb.org/anthology/W19-3901)
[anthology/W19-3901.](https://www.aclweb.org/anthology/W19-3901)

Merrill, W., Weiss, G., Goldberg, Y., Schwartz, R., Smith,
N. A., and Yahav, E. A formal hierarchy of RNN
architectures. In Proceedings of the 58th Annual
_Meeting of the Association for Computational Linguis-_
_tics, pp. 443–459, Online, July 2020. Association for_
Computational Linguistics. doi: 10.18653/v1/2020.
acl-main.43. [URL https://www.aclweb.org/](https://www.aclweb.org/anthology/2020.acl-main.43)
[anthology/2020.acl-main.43.](https://www.aclweb.org/anthology/2020.acl-main.43)

Merrill, W., Ramanujan, V., Goldberg, Y., Schwartz, R.,
and Smith, N. A. Effects of parameter norm growth
during transformer training: Inductive bias from gradient descent. In Proceedings of the 2021 Conference
_on Empirical Methods in Natural Language Process-_
_ing, pp. 1766–1781, Online and Punta Cana, Domini-_
can Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.


Karpathy, A., Johnson, J., and Fei-Fei, L. Visualizing and
understanding recurrent networks, 2015b.

Kleene, S. C., Shannon, C. E., and McCarthy, J. Automata
studies. Princeton, NJ, 1956.


-----

**Extracting Finite Automata from RNNs Using State Merging**


[133. URL https://aclanthology.org/2021.](https://aclanthology.org/2021.emnlp-main.133)
[emnlp-main.133.](https://aclanthology.org/2021.emnlp-main.133)

Minsky, M. L. Some universal elements for finite automata.
_Annals of Mathematics studies, 35, 1956._

Oncina, J. and Garc´ıa, P. Inferring Regular Languages
_in Polynomial Time, pp. 49–61._ World Scientific,
1992. doi: 10.1142/9789812797902 0004. URL

[https://www.worldscientific.com/doi/](https://www.worldscientific.com/doi/abs/10.1142/9789812797902_0004)
[abs/10.1142/9789812797902_0004.](https://www.worldscientific.com/doi/abs/10.1142/9789812797902_0004)

Power, A., Burda, Y., Edwards, H., Babuschkin, I., and
Misra, V. Grokking: Generalization beyond overfitting on
small algorithmic datasets. In ICLR MATH-AI Workshop,
2021.

Saxe, A. M., Bansal, Y., Dapello, J., Advani, M., Kolchinsky,
A., Tracey, B. D., and Cox, D. D. On the information bottleneck theory of deep learning. In 6th International Con_ference on Learning Representations, ICLR 2018, Van-_
_couver, BC, Canada, April 30 - May 3, 2018, Conference_
_[Track Proceedings. OpenReview.net, 2018. URL https:](https://openreview.net/forum?id=ry_WPG-A-)_
[//openreview.net/forum?id=ry_WPG-A-.](https://openreview.net/forum?id=ry_WPG-A-)

Sebban, M. and Janodet, J.-C. On state merging in grammatical inference: A statistical approach for dealing with
noisy data. In Proceedings of the 20th International Con_ference on Machine Learning (ICML-03), pp. 688–695,_
2003.

Shwartz-Ziv, R. and Tishby, N. Opening the black
box of deep neural networks via information. CoRR,
[abs/1703.00810, 2017. URL http://arxiv.org/](http://arxiv.org/abs/1703.00810)
[abs/1703.00810.](http://arxiv.org/abs/1703.00810)

Siegelmann, H. T. and Sontag, E. On the computational
power of neural nets. In COLT ’92, 1992.

Tomita, M. Dynamic construction of finite-state automata
from examples using hill-climbing. In Proceedings of
_the Fourth Annual Conference of the Cognitive Science_
_Society, pp. 105–108, 1982._

Wang, Q., Zhang, K., au2, A. G. O. I., Xing, X., Liu, X., and
Giles, C. L. An empirical evaluation of rule extraction
from recurrent neural networks, 2018.

Weiss, G., Goldberg, Y., and Yahav, E. On the practical computational power of finite precision RNNs for language
[recognition, 2018a. URL http://arxiv.org/abs/](http://arxiv.org/abs/1805.04908)
[1805.04908.](http://arxiv.org/abs/1805.04908)

Weiss, G., Goldberg, Y., and Yahav, E. Extracting automata
from recurrent neural networks using queries and counterexamples. In Dy, J. and Krause, A. (eds.), Proceed_ings of the 35th International Conference on Machine_


_Learning, volume 80 of Proceedings of Machine Learn-_
_ing Research, pp. 5247–5256. PMLR, 10–15 Jul 2018b._
[URL https://proceedings.mlr.press/v80/](https://proceedings.mlr.press/v80/weiss18a.html)
[weiss18a.html.](https://proceedings.mlr.press/v80/weiss18a.html)


-----

**Extracting Finite Automata from RNNs Using State Merging**


### A. Baseline Method

We describe the details of the k-means extraction method that we use as a baseline (Wang et al., 2018).


Taking a train set of the same form as state merging, we collect all the hidden states from every prefix of every train string,
each of which is associated with a label, i.e., whether the prefix is in L. This yields a dataset of the form {(hij, yij)}, where
**hij is the RNN hidden state on word j of example i, and yij records whether wi,:j** _L._
_∈_

We apply k-means clusters to the hidden states, where k is a hyperparameter that we set to 20. Manual inspection reveals
that the results are not particularly sensitive to k, which makes sense given the fact that all Tomita languages require at most
7 states. We identify each cluster with a state in that DFA that will be extracted. We then need to decide which cluster is the
initial state, and, for each cluster, whether it is accepting, and to which clusters it transitions for each input token. We find
the initial state by checking which cluster is assigned to the <bos> symbol by the RNN. We compute whether a cluster is
accepting or rejecting by taking a majority vote for each yij in the cluster. Finally, for a token σ, we assign the transition
out of a cluster by collecting all hidden states that are achieved after observing σ in that cluster, finding the corresponding
clusters, and taking a majority vote.

### B. Missing Plots


100

95

90

85

80

75

70

65

60


Tomita 1, epoch2

train
initial train
dev
initial dev

0 2 20 40 60 80 100

#data


100

95

90

85

80

75

70

65

60


Tomita 1, epoch20

train
initial train
dev
initial dev

0 2 20 40 60 80 100

#data


100

95

90

85

80

75

70

65

60


Tomita 2, epoch2

train
initial train
dev
initial dev

0 2 20 40 60 80 100

#data


100

95

90

85

80

75

70

65

60


Tomita 2, epoch20

train
initial train
dev
initial dev

0 2 20 40 60 80 100

#data

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
||||train initial train|
||||dev initial dev|
|||||
|||||
|||||

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
||||train initial train|
||||dev initial dev|
|||||
|||||
|||||

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
||||train initial train|
||||dev initial dev|
|||||
|||||
|||||

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
||||train initial train|
||||dev initial dev|
|||||
|||||
|||||


Tomita 3, epoch20

train
initial train
dev
initial dev

0 20 40 60 80 100

#data


100

90

80

70

60

50

40


Tomita 4, epoch20

train
initial train
dev
initial dev

0 16 20 40 60 80 100

#data


100

90

80

70

60

50


Tomita 3, epoch2

train
initial train
dev
initial dev

0 20 28 40 60 80 100

#data


100

90

80

70

60

50


Tomita 4, epoch2

train
initial train
dev
initial dev

0 2022 40 60 80 100

#data


100

90

80

70

60

50

40

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
||||train initial train dev|
||||initial dev|
|||||
|||||

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
||||train initial train dev|
||||initial dev|
|||||
|||||

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
||||train initial train dev|
||||initial dev|
|||||
|||||

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
||||train initial train dev|
||||initial dev|
|||||
|||||


Tomita 7, epoch20

train
initial train
dev
initial dev

0 25 50 75 100 125 150 175178 200

#data


Tomita 7, epoch2

train
initial train
dev
initial dev

0 25 50 75 100 125 150 175180 200

#data


100

90

80

70

60

50


Tomita 5, epoch20

train
initial train
dev
initial dev

0 20 26 40 60 80 100

#data


100

90

80

70

60

50


100

90

80

70

60

50

40


Tomita 5, epoch2

train
initial train
dev
initial dev

0 20 28 40 60 80 100

#data


100

90

80

70

60

50

40

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
||||train initial train dev|
||||initial dev|
|||||
|||||

|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
|||||train initial train dev|
|||||initial dev|
||||||
||||||

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||tra ini de|in tial train v|||||
|||ini|tial dev|||||
|||||||||
|||||||||

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
|||t i d|rain nitial train ev||||
|||i|nitial dev||||
||||||||
||||||||


_Figure 7. The implicit merging of RNN training as captured by our DFA extraction algorithm._


### C. Proof

**Lemma 1. If h1, h2 ∈** R[d] both have unit norm, then ∥h1 − **h2∥2[2]** [= 2(1][ −] [cos(][h][1][,][ h][2][))][.]


_Proof._


_∥h1 −_ **h2∥2[2]** [=][ ∥][h][1][∥]2[2] _[−]_ [2][h]1[⊤][h][2] [+][ ∥][h][2][∥][2]2
= 2 − 2h[⊤]1 **[h][2]**

= 2(1 − **h[⊤]1** **[h][2][)]**

= 2(1 − cos(h1, h2)).


-----

**Extracting Finite Automata from RNNs Using State Merging**

Let **h[˜] be the saturated version of vector h, i.e., viewing h as a function of the inputs x and parameters θ (cf. Merrill, 2019),**

**h˜(x, θ) = lim**
_ρ→∞_ **[h][(][x, ρθ][)][.]**

**Proposition 2. Let h1, h2 ∈** R[d] be two normalized state vectors, **h[˜]1,** **h[˜]2 ∈{±1}[d]** their saturated versions and assume
that the RNN is ϵ-saturated with respect to these states, ∥hi − **h[˜]i∥2 ≤** _ϵ, i ∈{1, 2}. Then, if cos(h1, h2) ≥_ 1 − _κ with_
_√_ _√_ � 1 �
_κ <_ 2 _√d_, the two vectors represent the same state on the DFA / saturated RNN (h[˜]1 = h[˜]2).

_[−]_ _[ϵ]_

_Proof. By the triangle inequality,_


_∥h[˜]1 −_ **h[˜]2∥2 = ∥h[˜]1 −** **h1 + h1 −** **h2 + h2 −** **h[˜]2∥2**

_≤∥h[˜]1 −_ **h1∥2 + ∥h1 −** **h2∥2 + ∥h2 −** **h[˜]2∥2**
_≤_ 2ϵ + ∥h1 − **h2∥2.**

Saturated RNN state vectors take discrete values in 1, 1, and thus two state vectors must be equal if the norm of their
_{−_ _}[d]_
_√_
difference is < 2/ _d. By the transitivity of inequalities,_ **h[˜]1 = h[˜]2 if**


2
2ϵ + ∥h1 − **h2∥2 <** _√_ _._

_d_

Applying Lemma 1 and using the fact that cos(h1, h2) ≥ 1 − _κ,_


�
2ϵ +


2
2(1 − cos(h1, h2)) < _√_

_d_

_√_ � 1 �
∴ _[√]κ <_ 2 _√_ _−_ _ϵ_ _._

_d_


-----

