## A Formal Hierarchy of RNN Architectures

### William Merrill[∗] Gail Weiss[†] Yoav Goldberg[∗‡]


### Roy Schwartz[∗§] Noah A. Smith[∗§] Eran Yahav[†]

_∗_ Allen Institute for AI _† Technion_ _‡ Bar Ilan University_ _§ University of Washington_
### willm,yoavg,roys,noah @allenai.org { } sgailw,yahave @cs.technion.ac.il { }

 Abstract


We develop a formal hierarchy of the expressive capacity of RNN architectures. The hierarchy is based on two formal properties:
space complexity, which measures the RNN’s
memory, and rational recurrence, defined as
whether the recurrent update can be described
by a weighted finite-state machine. We place
several RNN variants within this hierarchy.
For example, we prove the LSTM is not rational, which formally separates it from the related QRNN (Bradbury et al., 2016). We also
show how these models’ expressive capacity is
expanded by stacking multiple layers or composing them with different pooling functions.
Our results build on the theory of “saturated”
RNNs (Merrill, 2019). While formally extending these findings to unsaturated RNNs is left
to future work, we hypothesize that the practical learnable capacity of unsaturated RNNs
obeys a similar hierarchy. Experimental findings from training unsaturated networks on formal languages support this conjecture. We re**port updated experiments in Appendix H.**

### 1 Introduction


While neural networks are central to the performance of today’s strongest NLP systems, theoretical understanding of the formal properties of different kinds of networks is still limited. It is established, for example, that the Elman (1990) RNN
is Turing-complete, given infinite precision and
computation time (Siegelmann and Sontag, 1992,

1994; Chen et al., 2018). But tightening these unrealistic assumptions has serious implications for
expressive power (Weiss et al., 2018), leaving a significant gap between classical theory and practice,
which theorems in this paper attempt to address.
Recently, Peng et al. (2018) introduced rational
**RNNs, a subclass of RNNs whose internal state**
can be computed by independent weighted finite
automata (WFAs). Intuitively, such models have


Figure 1: Hierarchy of state expressiveness for saturated RNNs and related models. The y axis represents
increasing space complexity. means provably empty.
_∅_
Models are in bold with qualitative descriptions in gray.

a computationally simpler recurrent update than
conventional models like long short-term memory
networks (LSTMs; Hochreiter and Schmidhuber,

1997). Empirically, rational RNNs like the quasirecurrent neural network (QRNN; Bradbury et al.,
2016) and unigram rational RNN (Dodge et al.,
2019) perform comparably to the LSTM, with a
smaller computational budget. Still, the underlying
simplicity of rational models raises the question of
whether their expressive power is fundamentally
limited compared to other RNNs.
In a separate line of work, Merrill (2019) introduced the saturated RNN[1] as a formal model for
analyzing the capacity of RNNs. A saturated RNN
is a simplified network where all activation functions have been replaced by step functions. The
saturated network may be seen intuitively as a “sta

1Originally referred to as the asymptotic RNN.


-----

ble” version of its original RNN, in which the internal activations act discretely. A growing body
of work—including this paper—finds that the saturated theory predicts differences in practical learnable capacity for various RNN architectures (Weiss
et al., 2018; Merrill, 2019; Suzgun et al., 2019a).
We compare the expressive power of rational and
non-rational RNNs, distinguishing between state
_expressiveness (what kind and amount of informa-_
tion the RNN states can capture) and language
_expressiveness (what languages can be recognized_
when the state is passed to a classifier). To do this,
we build on the theory of saturated RNNs.

**State expressiveness** We introduce a unified hierarchy (Figure 1) of the functions expressible by
the states of rational and non-rational RNN encoders. The hierarchy is defined by two formal
properties: space complexity, which is a measure of
network memory,[2] and rational recurrence, whether
the internal structure of the RNN can be described
by WFAs. The hierarchy reveals concrete differences between LSTMs and QRNNs, and further
separates both from a class containing convolutional neural networks (CNNs, Lecun and Bengio,

1995; Kim, 2014), Elman RNNs, and gated recurrent units (GRU; Cho et al., 2014).
We provide the first formal proof that LSTMs
can encode functions that rational recurrences cannot. On the other hand, we show that the saturated
Elman RNN and GRU are rational recurrences with
constant space complexity, whereas the QRNN has
unbounded space complexity. We also show that
an unrestricted WFA has rich expressive power beyond any saturated RNN we consider—including
the LSTM. This difference potentially opens the
door to more expressive RNNs incorporating the
computational efficiency of rational recurrences.

**Language expressiveness** When applied to classification tasks like language recognition, RNNs
are typically combined with a “decoder”: additional layer(s) that map their hidden states to a prediction. Thus, despite differences in state expressiveness, rational RNNs might be able to achieve
comparable empirical performance to non-rational
RNNs on NLP tasks. In this work, we consider
the setup in which the decoders only view the fi
2Space complexity measures the number of different configurations an RNN can reach as a function of input length.
Formal definition deferred until Section 2.


nal hidden state of the RNN.[3] We demonstrate that
a sufficiently strong decoder can overcome some
of the differences in state expressiveness between
different models. For example, an LSTM can recognize a[n]b[n] with a single decoding layer, whereas
a QRNN provably cannot until the decoder has two
layers. However, we also construct a language that
an LSTM can recognize without a decoder, but a
QRNN cannot recognize with any decoder. Thus,
no decoder can fully compensate for the weakness
of the QRNN compared to the LSTM.

**Experiments** Finally, we conduct experiments
on formal languages, justifying that our theorems
correctly predict which languages unsaturated recognizers trained by gradient descent can learn.
Thus, we view our hierarchy as a useful formal
tool for understanding the relative capabilities of
different RNN architectures.

**Roadmap** We present the formal devices for our
analysis of RNNs in Section 2. In Section 3 we
develop our hierarchy of state expressiveness for
single-layer RNNs. In Section 4, we shift to study
RNNs as language recognizers. Finally, in Section 5, we provide empirical results evaluating the
relevance of our predictions for unsaturated RNNs.

### 2 Building Blocks

In this work, we analyze RNNs using formal models from automata theory—in particular, WFAs and
counter automata. In this section, we first define the
basic notion of an encoder studied in this paper, and
then introduce more specialized formal concepts:
WFAs, counter machines (CMs), space complexity,
and, finally, various RNN architectures.

**2.1** **Encoders**

We view both RNNs and automata as encoders:
machines that can be parameterized to compute a
set of functions f : Σ[∗] _→_ Q[k], where Σ is an input
alphabet and Q is the set of rational reals. Given
an encoder M and parameters θ, we use Mθ to represent the specific function that the parameterized
encoder computes. For each encoder, we refer to
the set of functions that it can compute as its state
_expressiveness. For example, a deterministic finite_
state acceptor (DFA) is an encoder whose parameters are its transition graph. Its state expressiveness
is the indicator functions for the regular languages.

3This is common, but not the only possibility. For example,
an attention decoder observes the full sequence of states.


-----

**2.2** **WFAs**

Formally, a WFA is a non-deterministic finite automaton where each starting state, transition, and
final state is weighted. Let Q denote the set of
states, Σ the alphabet, and Q the rational reals.[4]

This weighting is specified by three functions:

1. Initial state weights λ : Q → Q
2. Transition weights τ : Q × Σ × Q → Q
3. Final state weights ρ : Q → Q
The weights are used to encode any string x ∈ Σ[∗]:

**Definition 1 (Path score). Let π be a path of the**
form q0 →x1 q1 →x2 · · · →xt qt through WFA A.
The score of π is given by

_A[π] = λ(q0)_ ��ti=1 _[τ]_ [(][q][i][−][1][, x][i][, q][i][)]� _ρ(qt)._

By Π(x), denote the set of paths producing x.

**Definition 2 (String encoding). The encoding com-**
puted by a WFA A on string x is

_A[x] =_ [�]π∈Π(x) _[A][[][π][]][.]_

**Hankel matrix** Given a function f : Σ[∗] _→_ Q
and two enumerations α, ω of the strings in Σ[∗], we
define the Hankel matrix of f as the infinite matrix

[Hf ]ij = f (αi·ωj). (1)

where denotes concatenation. It is sometimes con_·_
venient to treat Hf as though it is directly indexed
by Σ[∗], e.g. [Hf ]αi,ωj = f (αi·ωj), or refer to a
sub-block of a Hankel matrix, row- and columnindexed by prefixes and suffixes P, S ⊆ Σ[∗]. The
following result relates the Hankel matrix to WFAs:

**Theorem 1 (Carlyle and Paz, 1971; Fliess, 1974).**
_For any f : Σ[∗]_ _→_ Q, there exists a WFA that
_computes f if and only if Hf has finite rank._

**Rational series (Sakarovitch, 2009)** For all k
_∈_
N, f : Σ[∗] _→_ Q[k] is a rational series if there exist
WFAs A1, · · ·, Ak such that, for all x ∈ Σ[∗] and
1 ≤ _i ≤_ _k, Ai[x] = fi(x)._

**2.3** **Counter Machines**

We now turn to introducing a different type of encoder: the real-time counter machine (CM; Merrill,
2020; Fischer, 1966; Fischer et al., 1968). CMs are
deterministic finite-state machines augmented with
finitely many integer counters. While processing
a string, the machine updates these counters, and
may use them to inform its behavior.

4WFAs are often defined over a generic semiring; we consider only the special case when it is the field of rational reals.


We view counter machines as encoders mapping
Σ[∗] _→_ Z[k]. For m ∈ N, ◦∈{+, −, ×}, let ◦m
denote the function f (n) = n _m._
_◦_

**Definition 3 (General CM; Merrill, 2020). A k-**
counter CM is a tuple ⟨Σ, Q, q0, u, δ⟩ with

1. A finite alphabet Σ
2. A finite set of states Q, with initial state q0
3. A counter update function

_u : Σ_ _Q_ 0, 1 0, 1, +0, +1
_×_ _× {_ _}[k]_ _→{×_ _−_ _}[k]_

4. A state transition function

_δ : Σ_ _Q_ 0, 1 _Q_
_×_ _× {_ _}[k]_ _→_

A CM processes input tokens {xt}t[n]=1 [sequen-]
tially. Denoting ⟨qt, ct⟩∈ _Q × Z[k]_ a CM’s configuration at time t, define its next configuration:

� �
_qt+1 = δ_ _xt, qt,_ _[⃗]1=0 (ct)_ (2)

� �
**ct+1 = u** _xt, qt,_ _[⃗]1=0 (ct)_ (ct), (3)

where _[⃗]1=0 is a broadcasted “zero-check” opera-_
tion, i.e., _[⃗]1=0(v)i ≜_ 1=0(vi). In (2) and (3), note
that the machine only views the zeroness of each
counter, and not its actual value. A general CM’s
encoding of a string x is the value of its counter
vector ct after processing all of x.

**Restricted CMs**

1. A CM is Σ-restricted iff u and δ depend only
on the current input σ Σ.
_∈_

2. A CM is (Σ _Q)-restricted iff u and δ de-_
_×_
pend only on the current input σ Σ and the
_∈_
current state q _Q._
_∈_

3. A CM is Σ[w]-restricted iff it is (Σ _Q)-_
_×_
restricted, and the states Q are windows over
the last w input tokens, e.g., Q = Σ[≤][w].[5]


These restrictions prevent the machine from being
“counter-aware”: u and δ cannot condition on the
counters’ values. As we will see, restricted CMs
have natural parallels in the realm of rational RNNs.
In Subsection 3.2, we consider the relationship between counter awareness and rational recurrence.

**2.4** **Space Complexity**

As in Merrill (2019), we also analyze encoders in
terms of state space complexity, measured in bits.

5The states q ∈ Σ<w represent the beginning of the sequence, before w input tokens have been seen.


-----

**Definition 4 (Bit complexity). An encoder M :**
Σ[∗] _→_ Q[k] has T (n) space iff


max
_θ_


��{sMθ (x) | x ∈ Σ≤n}�� = 2T (n),


where sMθ (x) is a minimal representation[6] of M ’s
internal configuration immediately after x.
We consider three asymptotic space complexity
classes: Θ(1), Θ(log n), and Θ(n), corresponding
to encoders that can reach a constant, polynomial,
and exponential (in sequence length) number of
configurations respectively. Intuitively, encoders
that can dynamically count but cannot use more
complex memory like stacks–such as all CMs–are
in Θ(log n) space. Encoders that can uniquely encode every input sequence are in Θ(n) space.

**2.5** **Saturated Networks**

A saturated neural network is a discrete approximation of neural network considered by Merrill (2019), who calls it an “asymptotic network.”
Given a parameterized neural encoder Mθ(x), we
construct the saturated network s-Mθ(x) by taking

s-Mθ(x) = lim (4)
_N_ _→∞_ _[M][Nθ][(][x][)]_

where Nθ denotes the parameters θ multiplied by
a scalar N . This transforms each “squashing” function (sigmoid, tanh, etc.) to its extreme values (0,
1). In line with prior work (Weiss et al., 2018;
_±_
Merrill, 2019; Suzgun et al., 2019b), we consider
saturated networks a reasonable approximation for
analyzing practical expressive power. For clarity,
we denote the saturated approximation of an architecture by prepending it with s, e.g., s-LSTM.

**2.6** **RNNs**

A recurrent neural network (RNN) is a parameterized update function gθ : Q[k] _×Q[d][x]_ _→_ Q[k], where θ
are the rational-valued parameters of the RNN and
_dx is the dimension of the input vector. gθ takes_
as input a current state h ∈ Q[k] and input vector
**x ∈** Q[d][x], and produces the next state. Defining the
initial state as h0 = 0, an RNN can be applied to
an input sequence x ∈ (Q[d][x])[∗] one vector at a time
to create a sequence of states {ht}t≤|x|, each representing an encoding of the prefix of x up to that
time step. RNNs can be used to encode sequences
over a finite alphabet x ∈ Σ[∗] by first applying a
mapping (embedding) e : Σ → Q[d][x].

6I.e., the minimal state representation needed to compute
_Mθ correctly. This distinction is important for architectures_
like attention, for which some implementations may retain
unusable information such as input embedding order.


**Multi-layer RNNs** “Deep” RNNs are RNNs
that have been arranged in L stacked layers
_R1, ..., RL. In this setting, the series of output_
states h1, h2, ..., h|x| generated by each RNN on
its input is fed as input to the layer above it, and
only the first layer receives the original input sequence x ∈ Σ[∗] as input.
The recurrent update function g can take several
forms. The original and most simple form is that of
the Elman RNN. Since then, more elaborate forms
using gating mechanisms have become popular,
among them the LSTM, GRU, and QRNN.

**Elman RNNs (Elman, 1990)** Let xt be a vector
embedding of xt. For brevity, we suppress the bias
terms in this (and the following) affine operations.

**ht = tanh(Wxt + Uht−1).** (5)

We refer to the saturated Elman RNN as the s-RNN.
The s-RNN has Θ(1) space (Merrill, 2019).

**LSTMs (Hochreiter and Schmidhuber, 1997)** An
LSTM is a gated RNN with a state vector ht ∈ Q[k]

and memory vector ct ∈ Q[k]. [7]

**ft = σ(W[f]** **xt + U[f]** **ht−1)** (6)

**it = σ(W[i]xt + U[i]ht−1)** (7)

**ot = σ(W[o]xt + U[o]ht−1)** (8)

**˜ct = tanh(W[c]xt + U[c]ht−1)** (9)

**ct = ft ⊙** **ct−1 + it ⊙** **˜ct** (10)

**ht = ot** tanh(ct). (11)
_⊙_

The LSTM can use its memory vector ct as a register of counters (Weiss et al., 2018). Merrill (2019)
showed that the s-LSTM has Θ(log n) space.

**GRUs (Cho et al., 2014)** Another kind of gated
RNN is the GRU.

**zt = σ(W[z]xt + U[z]ht−1)** (12)

**rt = σ(W[r]xt + U[r]ht−1)** (13)

**ut = tanh** �W[u]xt + U[u](rt ⊙ **ht−1)�** (14)

**ht = zt ⊙** **ht−1 + (1 −** **zt) ⊙** **ut.** (15)

Weiss et al. (2018) found that, unlike the LSTM, the
GRU cannot use its memory to count dynamically.
Merrill (2019) showed the s-GRU has Θ(1) space.

7 With respect to our presented definition of RNNs, the
concatenation of ht and ct can be seen as the recurrently
updated state. However in all discussions of LSTMs we treat
only ht as the LSTM’s ‘state’, in line with common practice.


-----

Figure 2: Diagram of the relations between encoders.
Neural networks are underlined. We group by asymptotic upper bound (O), as opposed to tight (Θ).

**QRNNs** Bradbury et al. (2016) propose QRNNs
as a computationally efficient hybrid of LSTMs
and CNNs. Let denote convolution over time, let
_∗_
**W[z], W[f]** _, W[o]_ _∈_ Q[d][x][×][w][×][k] be convolutions with
window length w, and let X ∈ Q[n][×][d][x] denote the
matrix of n input vectors. An ifo-QRNN (henceforth referred to as a QRNN) with window length
_w is defined by W[z], W[f]_, and W[o] as follows:

**Z = tanh(W[z]** **X)** (16)
_∗_

**F = σ(W[f]** **X)** (17)
_∗_

**O = σ(W[o]** **X)** (18)
_∗_

**ct = ft ⊙** **ct−1 + it ⊙** **zt** (19)

**ht = ot** **ct** (20)
_⊙_

where zt, ft, ot are respectively rows of Z, F, O. A
QRNN Q can be seen as an LSTM in which all
uses of the state vector ht have been replaced with
a computation over the last w input tokens–in this
way it is similar to a CNN.
The s-QRNN has Θ(log n) space, as the analysis
of Merrill (2019) for the s-LSTM directly applies.
Indeed, any s-QRNN is also a (Σ[w])-restricted CM
extended with = 1 (“set to 1”) operations.
_±_ _±_

### 3 State Expressiveness

We now turn to presenting our results. In this section, we develop a hierarchy of single-layer RNNs
based on their state expressiveness. A set-theoretic
view of the hierarchy is shown in Figure 2.
Let be the set of rational series. The hierarchy
_R_
relates Θ(log n) space to the following sets:

**RR As in Peng et al. (2018), we say that**

_•_
An encoder is rationally recurrent (RR) iff
its state expressiveness is a subset of .
_R_

**RR-hard An encoder is RR-hard iff its state**

_•_
expressiveness contains . A Turing machine
_R_
is RR-hard, as it can simulate any WFA.


**RR-complete Finally, an encoder is RR-**

_•_
_complete iff its state expressiveness is equiv-_
alent to . A trivial example of an RR_R_
complete encoder is a vector of k WFAs.

The different RNNs are divided between the intersections of these classes. In Subsection 3.1, we
prove that the s-LSTM, already established to have
Θ(log n) space, is not RR. In Subsection 3.2, we
demonstrate that encoders with restricted counting ability (e.g., QRNNs) are RR, and in Subsection 3.3, we show the same for all encoders with
finite state (CNNs, s-RNNs, and s-GRUs). In Subsection 3.4, we demonstrate that none of these
RNNs are RR-hard. In Appendix F, we extend
this analysis from RNNs to self attention.

**3.1** **Counting Beyond RR**

We find that encoders like the s-LSTM—which,
as discussed in Subsection 2.3, is “aware” of its
current counter values—are not RR. To do this, we
construct f0 : {a, b}[∗] _→_ N that requires counter
awareness to compute on strings of the form a[∗]b[∗],
making it not rational. We then construct an sLSTM computing f0 over a[∗]b[∗].
Let #a−b(x) denote the number of as in string
_x minus the number of bs._

**Definition 5 (Rectified counting).**


Therefore rank(An) = n−1. Thus, for all n, there
is a sub-block of Hf with rank n − 1, and so
rank(Hf ) is unbounded. It follows from Theorem 1 that there is no WFA computing f .

**Theorem 2. The s-LSTM is not RR.**


_f0 : x �→_


�
#a−b(x) if #a−b(x) > 0
0 otherwise.


**Lemma 1. For all f : {a, b}[∗]** _→_ N, if f (a[i]b[j]) =
_f0(a[i]b[j]) for all i, j ∈_ N, then f ̸∈R .

_Proof. Consider the Hankel sub-block An of Hf_
with prefixes Pn = {a[i]}i≤n and suffixes Sn =
_{b[j]}j≤n. An is lower-triangular:_










0 0 0

_· · ·_
1 0 0

_· · ·_
2 1 0

_· · ·_
... ... ... ...





 (21)

 _[.]_


-----

_a/+1_

start _q0_

_b,_ =0/ 1
_̸_ _−_


_b, =0/+0_


Figure 3: A 1-CM computing f0 for x ∈{a[i]b[j] _| i, j ∈_
N}. Let σ/±m denote a transition that consumes σ and
updates the counter by _m. We write σ, =0/_ _m (or_
_±_ _±_
=) for a transition that requires the counter is 0.
_̸_

_Proof. Assume the input has the form a[i]b[j]_ for
some i, j. Consider the following LSTM [8]:

_it = σ�10Nht−1 −_ 2N 1=b(xt) + N � (22)

_c˜t = tanh_ �N 1=a(xt) − _N_ 1=b(xt)� (23)

_ct = ct−1 + itc˜t_ (24)

_ht = tanh(ct)._ (25)

Let N →∞. Then it = 0 iff xt = b and
_ht−1 = 0 (i.e. ct−1 = 0). Meanwhile, ˜ct = 1 iff_
_xt = a. The update term becomes_


1 if xt = a
_−1_ if xt = b and ct−1 > 0
0 otherwise.


The WFA in Figure 4 also underlies unigram rational RNNs (Peng et al., 2018). Thus, Σ-restricted
CMs are actually a special case of unigram WFAs.
In Appendix A, we show the more general result:

**Theorem 4. Any (Σ** _Q)-restricted CM is RR._
_×_

In many rational RNNs, the updates at different
time steps are independent of each other outside
of a window of w tokens. Theorem 4 tells us this
independence is not an essential property of rational encoders. Rather, any CM where the update
is conditioned by finite state (as opposed to being
conditioned by a local window) is in fact RR.
Furthermore, since (Σ[w])-restricted CMs are a
special case of (Σ _Q)-restricted CMs, Theorem 4_
_×_
can be directly applied to show that the s-QRNN is
RR. See Appendix A for further discussion of this.

**3.3** **Finite-Space RR**

Theorem 4 motivates us to also think about finitespace encoders: i.e., encoders with no counters”
where the output at each prefix is fully determined
by a finite amount of memory. The following
lemma implies that any finite-space encoder is RR:

**Lemma 2. Any function f : Σ[∗]** _→_ Q computable
_by a Θ(1)-space encoder is a rational series._

_Proof. Since f is computable in Θ(1) space, there_
exists a DFA Af whose accepting states are isomorphic to the range of f . We convert Af to a WFA
by labelling each accepting state by the value of f
that it corresponds to. We set the starting weight of
the initial state to 1, and 0 for every other state. We
assign each transition weight 1.

Since the CNN, s-RNN, and s-GRU have finite
state, we obtain the following result:

**Theorem 5. The CNN, s-RNN, and s-GRU are RR.**

While Schwartz et al. (2018) and Peng et al. (2018)
showed the CNN to be RR over the max-plus semiring, Theorem 5 shows the same holds for ⟨Q, ·, +⟩.

**3.4** **RR Completeness**

While “rational recurrence” is often used to indicate the simplicity of an RNN architecture, we find
in this section that WFAs are surprisingly computationally powerful. Figure 5 shows a WFA mapping
binary string to their numeric value, proving WFAs
have Θ(n) space. We now show that none of our
RNNs are able to simulate an arbitrary WFA, even
in the unsaturated form.


_itc˜t =_










(26)


For a string a[i]b[j], the update in (26) is equivalent
to the CM in Figure 3. Thus, by Lemma 1, the
s-LSTM (and the general CM) is not RR.

**3.2** **Rational Counting**

While the counter awareness of a general CM enables it to compute non-rational functions, CMs
that cannot view their counters are RR.

**Theorem 3. Any Σ-restricted CM is RR.**

_Proof. We show that any function that a Σ-_
restricted CM can compute can also be computed
by a collection of WFAs. The CM update operations ( 1, +0, +1, or 0) can all be reexpressed
_−_ _×_
in terms of functions r(x), u(x) : Σ[∗] _→_ Z[k] to get:

**ct = r(xt)ct−1 + u(xt)** (27)

_t_ �
**ct =** [�]i[t]=1 ��j=i+1 **[r][(][x][j][)]** **u(xi).** (28)

A WFA computing [ct]i is shown in Figure 4.

8In which ft and ot are set to 1, such that ct = ct−1 +itc˜t.


-----

_σ/1_
_∀_

_∀σ/ui(σ)_

start _q0_ _q1_


_∀σ/ri(σ)_


Figure 4: WFA simulating unit i of a Σ-restricted CM.
Let _σ/w(σ) denote a set of transitions consuming_
_∀_
each token σ with weight w(σ). We use standard DFA
notation to show initial weights λ(q0) = 1, λ(q1) = 0
and accepting weights ρ(q0) = 0, ρ(q1) = 1.

_σ/1_
_∀_


_∀_

start _q0_ _q1_


_σ/2_
_∀_


Figure 5: A WFA mapping binary strings to their numeric value. This can be extended for any base > 2.
Cortes and Mohri (2000) present a similar construction.
Notation is the same as Figure 4.

**Theorem 6. Both the saturated and unsaturated**
_RNN, GRU, QRNN, and LSTM[9]_ _are not RR-hard._

_Proof. Consider the function fb mapping binary_
strings to their value, e.g. 101 5. The WFA in
_�→_
Figure 5 shows that this function is rational.
The value of fb grows exponentially with the
sequence length. On the other hand, the value of the
RNN and GRU cell is bounded by 1, and QRNN
and LSTM cells can only grow linearly in time.
Therefore, these encoders cannot compute fb.

In contrast, memory networks can have Θ(n)
space. Appendix G explores this for stack RNNs.

**3.5** **Towards Transformers**

Appendix F presents preliminary results extending saturation analysis to self attention. We show
saturated self attention is not RR and consider its
space complexity. We hope further work will more
completely characterize saturated self attention.

### 4 Language Expressiveness

Having explored the set of functions expressible
internally by different saturated RNN encoders, we
turn to the languages recognizable when using them
with a decoder. We consider the following setup:

1. An s-RNN encodes x to a vector ht ∈ Q[k].
2. A decoder function maps the last state ht to
an accept/reject decision, respectively: 1, 0 .
_{_ _}_

9As well as CMs.


We say that a language L is decided by an
encoder-decoder pair e, d if d(e(x)) = 1 for every sequence x _L and otherwise d(e(x)) = 0._
_∈_
We explore which languages can be decided by
different encoder-decoder pairings.
Some related results can be found in Cortes and
Mohri (2000), who study the expressive power of
WFAs in relation to CFGs under a slightly different
definition of language recognition.

**4.1** **Linear Decoders**

Let d1 be the single-layer linear decoder

**d1(ht) ≜** 1>0(w · ht + b) ∈{0, 1} (29)

parameterized by w and b. For an encoder architecture E, we denote by D1(E) the set of languages
decidable by E with d1. We use D2(E) analogously for a 2-layer decoder with 1>0 activations,
where the first layer has arbitrary width.

**4.2** **A Decoder Adds Power**

We refer to sets of strings using regular expressions,
e.g. a[∗] = {a[i] _| i ∈_ N}. To illustrate the purpose
of the decoder, consider the following language:

_L≤_ = {x ∈{a, b}[∗] _| #a−b(x) ≤_ 0}. (30)

The Hankel sub-block of the indicator function
for L over P = a[∗], S = b[∗] is lower triangular.
_≤_
Therefore, no RR encoder can compute it.
However, adding the D1 decoder allows us to
compute this indicator function with an s-QRNN,
which is RR. We set the s-QRNN layer to compute
the simple series ct = #a−b(x) (by increasing on
_a and decreasing on b). The D1 layer then checks_
_ct ≤_ 0. So, while the indicator function for L≤ is
not itself rational, it can be easily recovered from a
rational representation. Thus, L≤ _∈_ _D1(s-QRNN)._

**4.3** **Case Study: a[n]b[n]**

We compare the language expressiveness of several
rational and non-rational RNNs on the following:

_a[n]b[n]_ ≜ _{a[n]b[n]_ _| n ∈_ N} (31)

_a[n]b[n]Σ[∗]_ ≜ _{a[n]b[n](a|b)[∗]_ _| 0 < n}._ (32)

_a[n]b[n]_ is more interesting than L≤ because the D1
decoder cannot decide it simply by asking the encoder to track #a−b(x), as that would require it to
compute the non-linearly separable =0 function.
Thus, it appears at first that deciding a[n]b[n] with D1


-----

might require a non-rational RNN encoder. However, we show below that this is not the case.
Let denote stacking two layers. We will go on
_◦_
to discuss the following results:

_a[n]b[n]_ _∈_ _D1(WFA)_ (33)

_a[n]b[n]_ _∈_ _D1(s-LSTM)_ (34)

_a[n]b[n]_ _̸∈_ _D1(s-QRNN)_ (35)

_a[n]b[n]_ _∈_ _D1(s-QRNN ◦_ s-QRNN) (36)

_a[n]b[n]_ _∈_ _D2(s-QRNN)_ (37)

_a[n]b[n]Σ[∗]_ _∈_ _D1(s-LSTM)_ (38)

_a[n]b[n]Σ[∗]_ _/_ _D (s-QRNN) for any D_ (39)
_∈_

_a[n]b[n]Σ[∗]_ _∪{ϵ} ∈_ _D1(s-QRNN ◦_ s-QRNN) (40)

**WFAs (Appendix B)** In Theorem 8 we present a
function f : Σ[∗] _→_ Q satisfying f (x) > 0 iff x ∈
_a[n]b[n], and show that Hf has finite rank. It follows_
that there exists a WFA that can decide a[n]b[n] with
the D1 decoder. Counterintuitively, a[n]b[n] can be
recognized using rational encoders.

**QRNNs** (Appendix C) Although _a[n]b[n]_
_∈_
_D1(WFA), it does not follow that every rationally_
recurrent model can also decide a[n]b[n] with the
help of D1. Indeed, in Theorem 9, we prove
that a[n]b[n] _∈/_ _D1(s-QRNN), whereas a[n]b[n]_ _∈_
_D1(s-LSTM) (Theorem 13)._
It is important to note that, with a more complex
decoder, the QRNN could recognize a[n]b[n]. For example, the s-QRNN can encode c1 = #a−b(x) and
set c2 to check whether x contains ba, from which
a D2 decoder can recognize a[n]b[n] (Theorem 10).
This does not mean the hierarchy dissolves as the
decoder is strengthened. We show that a[n]b[n]Σ[∗]—
which seems like a trivial extension of a[n]b[n]—is
not recognizable by the s-QRNN with any decoder.
This result may appear counterintuitive, but in
fact highlights the s-QRNN’s lack of counter awareness: it can only passively encode the information
needed by the decoder to recognize a[n]b[n]. Failing
to recognize that a valid prefix has been matched,
it cannot act to preserve that information after additional input tokens are seen. We present a proof in
Theorem 11. In contrast, in Theorem 14 we show
that the s-LSTM can directly encode an indicator
for a[n]b[n]Σ[∗] in its internal state.

**Proof sketch:** _a[n]b[n]Σ[∗]_ _/_ _D(s-QRNN). A se-_
_∈_
quence s1 ∈ _a[n]b[n]Σ[∗]_ is shuffled to create s2 /∈
_a[n]b[n]Σ[∗]_ with an identical multi-set of counter up

dates.[10] Counter updates would be order agnostic
if not for reset operations, and resets mask all history, so extending s1 and s2 with a single suffix s
containing all of their w-grams reaches the same
final state. Then for any D, D(s-QRNN) cannot
separate them. We formalize this in Theorem 11.
We refer to this technique as the suffix attack,
and note that it can be used to prove for multiple
other languages L ∈ _D2(s-QRNN) that L·Σ[∗]_ is
not in D(s-QRNN) for any decoder D.

**2-layer QRNNs** Adding another layer overcomes the weakness of the 1-layer s-QRNN, at
least for deciding a[n]b[n]. This follows from the
fact that a[n]b[n] _∈_ _D2(s-QRNN): the second QRNN_
layer can be used as a linear layer.
Similarly, we show in Theorem 10 that a 2-layer
s-QRNN can recognize a[n]b[n]Σ[∗] _ϵ_ . This sug_∪{_ _}_
gests that adding a second s-QRNN layer compensates for some of the weakness of the 1-layer
s-QRNN, which, by the same argument for a[n]b[n]Σ[∗]

cannot recognize a[n]b[n]Σ[∗] _ϵ_ with any decoder.
_∪{_ _}_

**4.4** **Arbitrary Decoder**

Finally, we study the theoretical case where the
decoder is an arbitrary recursively enumerable (RE)
function. We view this as a loose upper bound of
stacking many layers after a rational encoder. What
information is inherently lost by using a rational
encoder? WFAs can uniquely encode each input,
making them Turing-complete under this setup;
however, this does not hold for rational s-RNNs.

**RR-complete** Assuming an RR-complete encoder, a WFA like Figure 5 can be used to encode
each possible input sequence over Σ to a unique
number. We then use the decoder as an oracle to
decide any RE language. Thus, an RR-complete
encoder with an RE decoder is Turing-complete.

**Bounded space** However, the Θ(log n) space
bound of saturated rational RNNs like the s-QRNN
means these models cannot fully encode the input.
In other words, some information about the prefix
_x:t must be lost in ct. Thus, rational s-RNNs are_
not Turing-complete with an RE decoder.

### 5 Experiments

In Subsection 4.3, we showed that different saturated RNNs vary in their ability to recognize a[n]b[n]

and a[n]b[n]Σ[∗]. We now test empirically whether

10Since QRNN counter updates depend only on the wgrams present in the sequence.


-----

Figure 6: Accuracy recognizing L5 and a[n]b[n]Σ[∗].
“QRNN+” is a QRNN with a 2-layer decoder, and
“2QRNN” is a 2-layer QRNN with a 1-layer decoder.
**Experimental results updated in Appendix H.**


these predictions carry over to the learnable capacity of unsaturated RNNs.[11] We compare the QRNN
and LSTM when coupled with a linear decoder D1.
We also train a 2-layer QRNN (“QRNN2”) and a
1-layer QRNN with a D2 decoder (“QRNN+”).
We train on strings of length 64, and evaluate
generalization on longer strings. We also compare
to a baseline that always predicts the majority class.
The results are shown in Figure 6. We provide
further experimental details in Appendix E.

**Experiment 1** We use the following language,
which has similar formal properties to a[n]b[n], but
with a more balanced label distribution:

_L5 =_ �x ∈ (a|b)[∗] _| |#a−b(x)| < 5�._ (41)

In line with (34), the LSTM decides L5 perfectly
for n 64, and generalizes fairly well to longer
_≤_
strings. As predicted in (35), the QRNN cannot
fully learn L5 even for n = 64. Finally, as predicted in (36) and (37), the 2-layer QRNN and the
QRNN with D2 do learn L5. However, we see
that they do not generalize as well as the LSTM

[11https://github.com/viking-sudo-rm/](https://github.com/viking-sudo-rm/rr-experiments)
[rr-experiments](https://github.com/viking-sudo-rm/rr-experiments)


for longer strings. We hypothesize that these multilayer models require more epochs to reach the same
generalization performance as the LSTM.[12]

**Experiment 2** We also consider a[n]b[n]Σ[∗]. As
predicted in (38) and (40), the LSTM and 2-layer
QRNN decide a[n]b[n]Σ[∗] flawlessly for n = 64. A
1-layer QRNN performs at the majority baseline
for all n with both a 1 and 2-layer decoder. Both of
these failures were predicted in (39). Thus, the only
models that learned a[n]b[n]Σ[∗] were exactly those predicted by the saturated theory.

### 6 Conclusion

We develop a hierarchy of saturated RNN encoders,
considering two angles: space complexity and rational recurrence. Based on the hierarchy, we formally distinguish the state expressiveness of the
non-rational s-LSTM and its rational counterpart,
the s-QRNN. We show further distinctions in state
expressiveness based on encoder space complexity.
Moreover, the hierarchy translates to differences
in language recognition capabilities. Strengthening
the decoder alleviates some, but not all, of these
differences. We present two languages, both recognizable by an LSTM. We show that one can be
recognized by an s-QRNN only with the help of a
decoder, and that the other cannot be recognized
by an s-QRNN with the help of any decoder.
While this means existing rational RNNs are fundamentally limited compared to LSTMs, we find
that it is not necessarily being rationally recurrent
that limits them: in fact, we prove that a WFA can
perfectly encode its input—something no saturated
RNN can do. We conclude with an analysis that
shows that an RNN architecture’s strength must
also take into account its space complexity. These
results further our understanding of the inner working of NLP systems. We hope they will guide the
development of more expressive rational RNNs.

### Acknowledgments

We appreciate Amir Yehudayoff’s help in finding the WFA used in Theorem 8. We also thank
our anonymous reviewers, Tobias Jaroslaw, Ana
Marasovic, and other researchers at the Allen In-´
stitute for AI. The project was supported in part
by NSF grant IIS-1562364, Israel Science Foundation grant no.1319/16, and the European Research

12As shown by the baseline, generalization is challenging
because positive labels become less likely as strings get longer.


Figure 6: Accuracy recognizing L5 and a[n]b[n]Σ[∗].
“QRNN+” is a QRNN with a 2-layer decoder, and
“2QRNN” is a 2-layer QRNN with a 1-layer decoder.
**Experimental results updated in Appendix H.**


-----

Council under the EU’s Horizon 2020 research and
innovation program, grant agreement No. 802774
(iEXTRACT).

### References

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hin[ton. 2016. Layer normalization.](http://arxiv.org/abs/1607.06450)

Borja Balle, Xavier Carreras, Franco M. Luque, and
Ariadna Quattoni. 2014. [Spectral learning of](https://doi.org/10.1007/s10994-013-5416-x)
[weighted automata.](https://doi.org/10.1007/s10994-013-5416-x) _Machine Learning, 96(1):33–_
63.

James Bradbury, Stephen Merity, Caiming Xiong, and
[Richard Socher. 2016. Quasi-recurrent neural net-](http://arxiv.org/abs/1611.01576)
[works.](http://arxiv.org/abs/1611.01576)

J. W. Carlyle and A. Paz. 1971. [Realizations by](https://doi.org/10.1016/S0022-0000(71)80005-3)
[stochastic finite automata.](https://doi.org/10.1016/S0022-0000(71)80005-3) _J. Comput. Syst. Sci.,_
5(1):26–40.

Yining Chen, Sorcha Gilroy, Andreas Maletti, Jonathan
[May, and Kevin Knight. 2018. Recurrent neural net-](https://doi.org/10.18653/v1/N18-1205)
[works as weighted language recognizers. In Proc. of](https://doi.org/10.18653/v1/N18-1205)
_NAACL, pages 2261–2271._

Kyunghyun Cho, Bart van Merri¨enboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger
Schwenk, and Yoshua Bengio. 2014. [Learning](https://doi.org/10.3115/v1/D14-1179)
[phrase representations using RNN encoder–decoder](https://doi.org/10.3115/v1/D14-1179)
[for statistical machine translation.](https://doi.org/10.3115/v1/D14-1179) In Proc. of
_EMNLP, pages 1724–1734._

Corinna Cortes and Mehryar Mohri. 2000. [Context-](https://doi.org/10.1023/A:1009911903208)
[free recognition with weighted automata.](https://doi.org/10.1023/A:1009911903208) _Gram-_
_mars, 3(2/3):133–150._

Jesse Dodge, Roy Schwartz, Hao Peng, and Noah A.
[Smith. 2019. RNN architecture learning with sparse](https://doi.org/10.18653/v1/D19-1110)
[regularization.](https://doi.org/10.18653/v1/D19-1110) In Proc. of EMNLP, pages 1179–
1184.

Jeffrey L Elman. 1990. Finding structure in time. Cog_nitive Science, 14(2):179–211._

Patrick C Fischer. 1966. Turing machines with restricted memory access. Information and Control,
9(4):364–379.

Patrick C. Fischer, Albert R. Meyer, and Arnold L.
Rosenberg. 1968. [Counter machines and counter](https://doi.org/10.1007/BF01694011)
[languages. Mathematical Systems Theory, 2(3):265–](https://doi.org/10.1007/BF01694011)
283.

Michel Fliess. 1974. Matrices de Hankel. _J. Math._
_Pures Appl, 53(9):197–222._

Matt Gardner, Joel Grus, Mark Neumann, Oyvind
Tafjord, Pradeep Dasigi, Nelson F. Liu, Matthew Peters, Michael Schmitz, and Luke Zettlemoyer. 2018.
[AllenNLP: A deep semantic natural language pro-](https://doi.org/10.18653/v1/w18-2501)
[cessing platform. Proceedings of Workshop for NLP](https://doi.org/10.18653/v1/w18-2501)
_Open Source Software (NLP-OSS)._


[Michael Hahn. 2020. Theoretical limitations of self-](https://doi.org/10.1162/tacl_a_00306)
[attention in neural sequence models. Transactions](https://doi.org/10.1162/tacl_a_00306)
_of the Association for Computational Linguistics,_
8:156–171.

Sepp Hochreiter and J¨urgen Schmidhuber. 1997.
Long short-term memory. _Neural Computation,_
9(8):1735–1780.

[Yoon Kim. 2014. Convolutional neural networks for](https://doi.org/10.3115/v1/D14-1181)
[sentence classification. In Proc. of EMNLP, pages](https://doi.org/10.3115/v1/D14-1181)
1746–1751.

Yann Lecun and Yoshua Bengio. 1995. The Handbook
_of Brain Theory and Neural Networks, chapter “Con-_
volutional Networks for Images, Speech, and Time
Series”. MIT Press.

[William Merrill. 2019. Sequential neural networks as](https://www.aclweb.org/anthology/W19-3901)
[automata. In Proceedings of the Workshop on Deep](https://www.aclweb.org/anthology/W19-3901)
_Learning and Formal Languages: Building Bridges,_
pages 1–13.

[William Merrill. 2020. On the linguistic capacity of](http://arxiv.org/abs/2004.06866)
[real-time counter automata.](http://arxiv.org/abs/2004.06866)

Hao Peng, Roy Schwartz, Sam Thomson, and Noah A.
Smith. 2018. [Rational recurrences.](https://doi.org/10.18653/v1/D18-1152) In Proc. of
_EMNLP, pages 1203–1214._

Jacques Sakarovitch. 2009. Rational and recognisable
power series. In Handbook of Weighted Automata,
pages 105–174. Springer.

Roy Schwartz, Sam Thomson, and Noah A. Smith.
[2018. Bridging CNNs, RNNs, and weighted finite-](https://doi.org/10.18653/v1/P18-1028)
[state machines. In Proc. of ACL, pages 295–305.](https://doi.org/10.18653/v1/P18-1028)

[Hava T. Siegelmann and Eduardo D. Sontag. 1992. On](https://doi.org/10.1145/130385.130432)
[the computational power of neural nets. In Proc. of](https://doi.org/10.1145/130385.130432)
_COLT, pages 440–449._

Hava T. Siegelmann and Eduardo D. Sontag. 1994.
Analog computation via neural networks. Theoret_ical Computer Science, 131(2):331–360._

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
Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need. In Advances in Neural Information Pro_cessing Systems, pages 5998–6008._

[Gail Weiss, Yoav Goldberg, and Eran Yahav. 2018. On](http://arxiv.org/abs/1805.04908)
[the practical computational power of finite precision](http://arxiv.org/abs/1805.04908)
[RNNs for language recognition.](http://arxiv.org/abs/1805.04908)


-----

### A Rational Counting

We extend the result in Theorem 3 as follows.

**Theorem 7. Any (Σ** _Q)-restricted CM is ratio-_
_×_
_nally recurrent._

_Proof. We present an algorithm to construct a WFA_
computing an arbitrary counter in a (Σ _Q)-_
_×_
restricted CM. First, we create two independent
copies of the transition graph for the restricted CM.
We refer to one copy of the CM graph as the add
_graph, and the other as the multiply graph._
The initial state in the add graph receives a starting weight of 1, and every other state receives a
starting weight of 0. Each state in the add graph
receives an accepting weight of 0, and each state
in the multiply graph receives an accepting weight
of 1. In the add graph, each transition receives a
weight of 1. In the multiply graph, each transition
receives a weight of 0 if it represents 0, and 1 oth_×_
erwise. Finally, for each non-multiplicative update
_σ/+m[13]_ from qi to qj in the original CM, we add
a WFA transition σ/m from qi in the add graph to
_qj in the multiply graph._
Each counter update creates one path ending in
the multiply graph. The path score is set to 0 if
that counter update is “erased” by a 0 operation.
_×_
Thus, the sum of all the path scores in the WFA
equals the value of the counter.

This construction can be extended to accommodate =m counter updates from qi to qj by adding
an additional transition from the initial state to qj
in the multiplication graph with weight m. This
allows us to apply it directly to s-QRNNs, whose
update operations include =1 and = 1.
_−_

### B WFAs

We show that while WFAs cannot directly encode
an indicator for the language a[n]b[n] = _a[n]b[n]_
_{_ _| |_
_n ∈_ N}, they can encode a function that can be
thresholded to recognize a[n]b[n], i.e.:

**Theorem 8. The language a[n]b[n]** = _a[n]b[n]_ _n_
_{_ _|_ _∈_
N} over Σ = {a, b} is in D1(WFA).

We prove this by showing a function whose Hankel matrix has finite rank that, when combined with
the identity transformation (i.e., w = 1, b = 0) followed by thresholding, is an indicator for a[n]b[n].
Using the shorthand σ(x) = #σ(x), the function

13Note that m = −1 for the −1 counter update.


is:

�
0.5 − 2(a(x) − _b(x))[2]_ if x ∈ _a[∗]b[∗]_

_f_ (w) =

0.5 otherwise.
_−_
(42)
Immediately f satisfies 1>0(f (x)) ⇐⇒ _x ∈_
_a[n]b[n]. To prove that its Hankel matrix, Hf_, has
finite rank, we will create 3 infinite matrices of
ranks 3, 3 and 1, which sum to Hf . The majority
of the proof will focus on the rank of the rank 3
matrices, which have similar compositions.
We now show 3 series r, s, t and a set of series
they can be combined to create. These series will
be used to create the base vectors for the rank 3
matrices.

_ai =_ _[i][(][i][ + 1)]_ (43)

2

_bi = i[2]_ _−_ 1 (44)

_ri = fix0(i, ai−2)_ (45)

_si = fix1(i, −bi−1)_ (46)

_ti = fix2(i, ai−1)_ (47)

where for every j 2,
_≤_


_x_ if i > 2
1 if i = j
0 otherwise.


fixj(i, x) =










(48)


**Lemma 3. Let ci = 1 −** 2i[2] _and {c[(][k][)]}k∈N be the_
_set of series defined c[(]i[k][)]_ = c|i−k|. Then for every
_i, k ∈_ N,

_c[(]i[k][)]_ = c[(]0[k][)][r][i][ +][ c]1[(][k][)][s][i][ +][ c]2[(][k][)][t][i][.]

_Proof. For i ∈{0, 1, 2}, ri, si and ti collapse_
to a ‘select’ operation, giving the true statement
_c[(]i[k][)]_ = c[(]i[k][)] _· 1. We now consider the case i > 2._
Substituting the series definitions in the right side
of the equation gives

_ckai−2 + c|k−1|(−bi−1) + ck−2ai−1_ (49)

which can be expanded to

(1 2k[2]) _[i][2][ −]_ [3][i][ + 2] +
_−_ _·_

2

(1 2(k 1)[2]) (1 (i 1)[2]) +
_−_ _−_ _·_ _−_ _−_


(1 2(k 2)[2]) [(][i][ −] [1)][i] _._
_−_ _−_ _·_

2


-----

Reordering the first component and partially opening the other two gives

( 2k[2] + 1) _[i][2][ −]_ [3][i][ + 2] +
_−_

2

( 2k[2] + 4k 1)(2i _i[2])+_
_−_ _−_ _−_

( _k[2]_ + 4k 3.5)(i[2] _i)_
_−_ _−_ _−_


and a further expansion gives

_k[2]i[2]+_ 0.5i[2] + 3k[2]i 1.5i 2k[2] + 1+
_−_ _−_ _−_

2k[2]i[2] 4ki[2]+ _i[2]_ 4k[2]i + 8ki 2i+
_−_ _−_ _−_

_k[2]i[2]_ + 4ki[2] 3.5i[2] + k[2]i 4ki + 3.5i
_−_ _−_ _−_

which reduces to

_−2i[2]_ + 4ki − 2k[2] + 1 = 1 − 2(k − _i)[2]_ = c[(]i[k][)].

We restate this as:

**Corollary 1. For every k ∈** N, the series c[(][k][)] _is a_
_linear combination of the series r, s and t._

We can now show that f is computable by a
WFA, proving Theorem 8. By Theorem 1, it is
sufficient to show that Hf has finite rank.

**Lemma 4. Hf has finite rank.**

_Proof. For every P, S_ _a, b_, denote
_⊆{_ _}[∗]_



[Hf _|P,S]u,v =_


�

[Hf ]u,v if u ∈ _P and v ∈_ _S_

0 otherwise


Using regular expressions to describe P, S, we create the 3 finite rank matrices which sum to Hf :

_A = (Hf + 0.5)|a∗,a∗b∗_ (50)

_B = (Hf + 0.5)|a∗b+,b∗_ (51)

_C = (−0.5)|u,v._ (52)

Intuitively, these may be seen as a “split” of Hf
into sections as in Figure 7, such that A and B
together cover the sections of Hf on which u·v
does not contain the substring ba (and are equal on
them to Hf + 0.5), and C is simply the constant
matrix −0.5. Immediately, Hf = A + B + C, and
rank(C) = 1.
We now consider A. Denote PA = a[∗], SA =
_a[∗]b[∗]. A is non-zero only on indices u ∈_ _PA, v ∈_
_SA, and for these, u·v ∈_ _a[∗]b[∗]_ and Au,v = 0.5 +
_f_ (u _v) = 1_ 2(a(u) + a(v) _b(v))[2]. This gives_

_·_ _−_ _−_
that for every u ∈ _PA, v ∈_ _SA,_

_Au,v = c|a(u)−(b(v)−a(v))| = c[(]b[a](v[(][u])−[))]a(v)[.]_ (53)


Figure 7: Intuition of the supports of A, B and C.

For each τ ∈{r, s, t}, define ˜τ ∈ Q[{][a,b][}][∗] as

_τ˜v = 1v∈a∗b∗_ _· τb(v)−a(v)._ (54)

We get from Corollary 1 that for every u ∈ _a[∗],_
the uth row of A is a linear combination of ˜r, ˜s, and
_t˜. The remaining rows of A are all 0 and so also a_
linear combination of these, and so rank(A) 3.
_≤_
Similarly, we find that the nonzero entries of B
satisfy

_Bu,v = c|b(v)−(a(u)−b(u))| = c[(]a[b]([(]u[v])[))]−b(u)_ (55)

and so, for τ _r, s, t_, the columns of B are
_∈{_ _}_
linear combinations of the columns τ _[′]_ _∈_ Q[{][a,b][}][∗]

defined

_τu[′]_ [=][ 1]u∈a[∗]b[+][ ·][ τ]a(u)−b(u)[.] (56)

Thus we conclude rank(B) 3.
_≤_
Finally, Hf = A + B + C, and so by the subadditivity of rank in matrices,

�
rank(Hf ) ≤ rank(M ) = 7. (57)

_M_ =A,B,C


In addition, the rank of _H[˜]f ∈_ Q[{][a,b][}][≤][2][,][{][a,b][}][≤][2]

defined [ H[˜]f ]u,v = [Hf ]u,v is 7, and so we can
conclude that the bound in the proof is tight, i.e.,
rank(Hf ) = 7. From here _H[˜]f is a complete sub-_
block of Hf and can be used to explicitly construct
a WFA for f, using the spectral method described
by Balle et al. (2014).

### C s-QRNNs

**Theorem 9. No s-QRNN with a linear threshold**
_decoder can recognize a[n]b[n]_ = {a[n]b[n] _| n ∈_ N},
_i.e., a[n]b[n]_ _∈/_ _D1(s-QRNN)._


-----

_Proof. An ifo s-QRNN can be expressed as a Σ[k]-_
restricted CM with the additional update operations
:= 1, := 1, where k is the window size of the
_{_ _−_ _}_
QRNN. So it is sufficient to show that such a machine, when coupled with the decoder D1 (linear
translation followed by thresholding), cannot recognize a[n]b[n].
Let be some such CM, with window size k
_A_
and h counters. Take n = k + 10 and for every
_m ∈_ N denote wm = a[n]b[m] and the counter values
of A after wm as c[m] _∈_ Q[h]. Denote by ut the vector
of counter update operations made by this machine
on input sequence wm at time t ≤ _n + m. As A is_
dependent only on the last k counters, necessarily
all uk+i are identical for every i ≥ 1.
It follows that for all counters in the machine
that go through an assignment (i.e., :=) operation
in uk+1, their values in c[k][+][i] are identical for every
_i ≥_ 1, and for every other counter j, c[k]j [+][i] _−_ _c[k]j_ [=]
_i · δ for some δ ∈_ Z. Formally: for every i ≥ 1
there are two sets I, J = [h] _I and constant_
_\_
vectors u ∈ N[I] _, v ∈_ N[J] s.t. c[k][+][i]|I = u and

[c[k][+][i] _−_ _c[k]]|J = i · v._
We now consider the linear thresholder, defined
by weights and bias w, b. In order to recognise
_a[n]b[n], the thresholder must satisfy:_

**w** _c[k][+9]+b_ _< 0_ (58)
_·_

**w** _c[k][+10]+b_ _> 0_ (59)
_·_

**w** _c[k][+11]+b_ _< 0_ (60)
_·_

Opening these equations gives:

**w|J** (·c[k]|J +9v|J ) + w|I · u _< 0_ (61)

**w|J** (·c[k]|J +10v|J ) + w|I · u _> 0_ (62)

**w|J** (·c[k]|J +11v|J ) + w|I · u _< 0_ (63)


but this gives 9w|J _·v|J_ _<_ 10w|J _·v|J_ _>_
11w|J _·v|J_, which is impossible.

However, this does not mean that the s-QRNN is
entirely incapable of recognising a[n]b[n]. Increasing
the decoder power allows it to recognise a[n]b[n] quite
simply:

**Theorem 10. For the two-layer decoder D2,**
_a[n]b[n]_ _∈_ _D2(s-QRNN)._

_Proof. Let #ba(x) denote the number of ba 2-_
grams in x. We use s-QRNN with window size


2 to maintain two counters:

[ct]1 = #a−b(x) (64)

[ct]2 = #ba(x). (65)

[ct]2 can be computed provided the QRNN window
size is 2. A two-layer decoder can then check
_≥_

0 ≤ [ct]1 ≤ 0 ∧ [ct]2 ≤ 0. (66)

**Theorem 11 (Suffix attack). No s-QRNN and**
_decoder can recognize the language a[n]b[n]Σ[∗]_ =
_a[n]b[n](a_ _b)[∗], n > 0, i.e., a[n]b[n]Σ[∗]_ _/_ _L(s-QRNN) for_
_|_ _∈_
_any decoder L._

The proof will rely on the s-QRNN’s inability
to “freeze” a computed value, protecting it from
manipulation by future input.

_Proof. As in the proof for Theorem 9, it is suffi-_
cient to show that no Σ[k]-restricted CM with the
additional operations := 1, :=1 can recognize
_{_ _−_ _}_
_a[n]b[n]Σ[∗]_ for any decoder L.
Let be some such CM, with window size k
_A_
and h counters. For every w Σ[n] denote by
_∈_
_c(w) ∈_ Q[h] the counter values of A after processing w. Denote by ut the vector of counter update
operations made by this machine on an input sequence w at time t _w_ . Recall that is Σ[k]
_≤|_ _|_ _A_

restricted, meaning that ui depends exactly on the
window of the last k tokens for every i.
We now denote j = k + 10 and consider
the sequences w1 = _a[j]b[j]a[j]b[j]a[j]b[j], w2_ =
_a[j]b[j][−][1]a[j]b[j][+1]a[j]b[j]. w2 is obtained from w1 by re-_
moving the 2j-th token of w1 and reinserting it at
position 4j.
As all of w1 is composed of blocks of ≥ _k iden-_
tical tokens, the windows preceding all of the other
tokens in w1 are unaffected by the removal of the
2j-th token. Similarly, being added onto the end of
a substring b[k], its insertion does not affect the windows of the tokens after it, nor is its own window
different from before. This means that overall, the
set of all operations ui performed on the counters
is identical in w1 and in w2. The only difference is
in their ordering.
_w1 and w2 begin with a shared prefix a[k], and so_
necessarily the counters are identical after processing it. We now consider the updates to the counters
after these first k tokens, these are determined by
the windows of k tokens preceding each update.


-----

First, consider all the counters that undergo some
assignment (:=) operation during these sequences,
and denote by _w_ the multiset of windows in
_{_ _}_
_w ∈_ Σ[k] for which they are reset. w1 and w2 only
contain k-windows of types a[x]b[k][−][x] or b[x]a[k][−][x], and
so these must all re-appear in the shared suffix
_b[j]a[j]b[j]_ of w1 and w2, at which point they will be
synchronised. It follows that these counters all
finish with identical value in c(w1) and c(w2).
All the other counters are only updated using
addition of 1, 1 and 0, and so the order of the
_−_
updates is inconsequential. It follows that they
too are identical in c(w1) and c(w2), and therefore
necessarily that c(w1) = c(w2).
From this we have w1, w2 satisfying w1
_∈_
_a[n]b[n]Σ[∗], w2 /∈_ _a[n]b[n]Σ[∗]_ but also c(w1) = c(w2).
Therefore, it is not possible to distinguish between
_w1 and w2 with the help of any decoder, despite_
the fact that w1 ∈ _a[n]b[n]Σ[∗]_ and w2 /∈ _a[n]b[n]Σ[∗]. It_
follows that the CM and s-QRNN cannot recognize
_a[n]b[n]Σ[∗]_ with any decoder.

For the opposite extension Σ[∗]a[n]b[n], in which the
language is augmented by a prefix, we cannot use
such a “suffix attack”. In fact, Σ[∗]a[n]b[n] can be recognized by an s-QRNN with window length w 2
_≥_
and a linear threshold decoder as follows: a counter
counts #a−b(x) and is reset to 1 on appearances of
_ba, and the decoder compares it to 0._
Note that we define decoders as functions from
the final state to the output. Thus, adding an additional QRNN layer does not count as a “decoder”
(as it reads multiple states). In fact, we show
that having two QRNN layers allows recognizing
_a[n]b[n]Σ[∗]._

**Theorem 12. Let ϵ be the empty string. Then,**

_a[n]b[n]Σ[∗]_ _∪{ϵ} ∈_ _D1(s-QRNN ◦_ s-QRNN).

_Proof. We construct a two-layer s-QRNN from_
which a[n]b[n]Σ[∗] can be recognized. Let $ denote
the left edge of the string. The first layer computes
two quantities dt and et as follows:

_dt = #ba(x)_ (67)

_et = #$b(x)._ (68)

Note that et can be interpreted as a binary value
checking whether the first token was b. The second
layer computes ct as a function of dt, et, and xt
(which can be passed through the first layer). We
will demonstrate a construction for ct by creating


linearly separable functions for the gate terms ft
and zt that update ct.


Now, the update function ut to ct can be expressed


+0 if 0 < dt


_ut = ftzt =_ +1 if dt ≤ 0 ∧ (xt = a ∨ _et)_


 1 otherwise.

_−_
(71)
Finally, the decoder accepts iff ct ≤ 0. To justify
this, we consider two cases: either x starts with b or
_a. If x starts with b, then et = 0, so we increment_
_ct by 1 and never decrement it. Since 0 < ct for_
any t, we will reject x. If x starts with a, then we
accept iff there exists a sequence of bs following
the prefix of as such that both sequences have the
same length.

### D s-LSTMs

In contrast to the s-QRNN, we show that the sLSTM paired with a simple linear and thresholding
decoder can recognize both a[n]b[n] and a[n]b[n]Σ[∗].

**Theorem 13.**

_a[n]b[n]_ _∈_ _D1(s-LSTM)._

_Proof. Assuming a string a[i]b[i], we set two units of_
the LSTM state to compute the following functions
using the CM in Figure 3:

[ct]1 = ReLU(i − _j)_ (72)

[ct]2 = ReLU(j − _i)._ (73)

We also add a third unit [ct]3 that tracks whether the
2-gram ba has been encountered, which is equivalent to verifying that the string has the form a[i]b[i].
Allowing ht = tanh(ct), we set the linear threshold layer to check

[ht]1 + [ht]2 + [ht]3 0. (74)
_≤_

**Theorem 14.**

_a[n]b[n]Σ[∗]_ _∈_ _D1(s-LSTM)._


_ft =_

_zt =_


�
1 if dt ≤ 0 (69)
0 otherwise

�
1 if xt = a ∨ _et_
(70)
1 otherwise.
_−_


-----

_Proof. We use the same construction as Theo-_
rem 13, augmenting it with

[ct]4 ≜ [ht−1]1 + [ht−1]2 + [ht−1]3 ≤ 0. (75)

We decide x according to the (still linearly separable) equation
�0 < [ht]4� _∨_ �[ht]1 + [ht]2 + [ht]3 ≤ 0�. (76)

### E Experimental Details

Models were trained on strings up to length 64,
and, at each index t, were asked to classify whether
or not the prefix up to t was a valid string in the
language. Models were then tested on independent datasets of lengths 64, 128, 256, 512, 1024,
and 2048. The training dataset contained 100000
strings, and the validation and test datasets contained 10000. We discuss task-specific schemes for
sampling strings in the next paragraph. All models
were trained for a maximum of 100 epochs, with
early stopping after 10 epochs based on the validation cross entropy loss. We used default hyperparameters provided by the open-source AllenNLP
framework (Gardner et al., 2018). The code is avail[able at https://github.com/viking-sudo-rm/](https://github.com/viking-sudo-rm/rr-experiments)

[rr-experiments.](https://github.com/viking-sudo-rm/rr-experiments)

**Sampling strings** For the language L5, each token was sampled uniformly at random from Σ =
_{a, b}. For a[n]b[n]Σ[∗], half the strings were sampled_
in this way, and for the other half, we sampled n
uniformly between 0 and 32, fixing the first 2n
characters of the string to a[n]b[n] and sampling the
suffix uniformly at random.

**Experimental cost** The originally reported experiments were run for 20 GPU hours on Quadro
RTX 8000.

### F Self Attention

**Architecture** We place saturated self attention
(Vaswani et al., 2017) into the state expressiveness
hierarchy. We consider a single-head self attention
encoder that is computed as follows:

1. At time t, compute queries qt, keys kt, and
values vt from the input embedding xt using
a linear transformation.

2. Compute attention head ht by attending over
the keys and values up to time t (K:t and V:t)
with query qt.


Applying layer norm to this quantity preserves
equality of the first and second elements. Thus,
we set the layer in (77) to independently check
0 < [h[0]t []][1] _[−]_ [[][h]t[0][]][2] [and][ [][h][0]t []][1] _[−]_ [[][h]t[0][]][2] _[<][ 0][ using]_
ReLU. The final layer ct sums these two quantities, returning 0 if neither condition is met, and 1
otherwise.
Since saturated self attention can represent f /
_∈_
, it is not RR.
_R_


3. Let ∥·∥L denote a layer normalization operation (Ba et al., 2016).

**h[′]t** [= ReLU] �W[h] _· ∥ht∥L�_ (77)

**ct =** ��Wch′t��L[.] (78)

This simplified architecture has only one attention head, and does not incorporate residual connections. It is also masked (i.e., at time t, can
only see the prefix X:t), which enables direct comparison with unidirectional RNNs. For simplicity,
we do not add positional information to the input
embeddings.

**Theorem 15. Saturated masked self attention is**
_not RR._

_Proof. Let #σ(x) denote the number of oc-_
curences of σ Σ in string x. We construct a
_∈_
self attention layer to compute the following function over _a, b_ :
_{_ _}[∗]_


_f_ (x) =


�
0 if #a(x) = #b(x)
(79)
1 otherwise.


Since the Hankel sub-block over P = a[∗], S = b[∗]

has infinite rank, f .
_̸∈R_
Fix vt = xt. As shown by Merrill (2019),
saturated attention over a prefix of input vectors
**X:t reduces to sum of the subsequence for which**
key-query similarity is maximized, i.e., denoting
_I = {i ∈_ [t] | ki · qt = m} where m =
max{ki · qt|i ∈ [t]}:


**ht = [1]**

_I_
_|_ _|_


�

**xti.** (80)
_i∈I_


For all t, set the key and query kt, qt = 1. Thus, all
the key-query similarities are 1, and we obtain:

_t_
�

**ht = [1]** **xt′** (81)

_t_

_t[′]=1_

= [1] �#a(x), #b(x)�⊤. (82)

_t_


-----

**Space Complexity** We show that self attention
falls into the same space complexity class as the
LSTM and QRNN. Our method here extends Merrill (2019)’s analysis of attention.

**Theorem 16. Saturated single-layer self attention**
_has Θ(log n) space._

_Proof. The construction from Theorem 15 can_
reach a linear (in sequence length) number of different outputs, implying a linear number of different
configurations, and so that the space complexity of
saturated self attention is Ω(log n). We now show
the upper bound O(log n).
A sufficient representation for the internal state
(configuration) of a self-attention layer is the unordered group of key-value pairs over the prefixes
of the input sequence.
Since fk : xt **kt and fv : xt** **vt have finite**
_�→_ _�→_
domain (Σ), their images K = image(fk), V =
image(fv) are finite.[14] Thus, there is also a finite number of possible key-value pairs ⟨kt, vt⟩∈
_K_ _V . Recall that the internal configuration can be_
_×_
specified by the number of occurrences of each possible key-value pair. Taking n as an upper bound
for each of these counts, we bound the number of
configurations of the layer as n[|][K][×][V][ |]. Therefore
the bit complexity is

log2 �n[|][K][×][V][ |][�] = O(log n). (83)

Note that this construction does not apply if
the “vocabulary” we are attending over is not finite. Thus, using unbounded positional embeddings, stacking multiple self attention layers, or
applying attention over other encodings with unbounded state might reach Θ(n).
While it eludes our current focus, we hope future work will extend the saturated analysis to self
attention more completely. We direct the reader to
Hahn (2020) for some additional related work.

### G Memory Networks

All of the standard RNN architectures considered
in Section 3 have O(log n) space in their saturated
form. In this section, we consider a stack RNN
encoder similar to the one proposed by Suzgun
et al. (2019b) and show how it, like a WFA, can
encode binary representations from strings. Thus,

14Note that any periodic positional encoding will also have
finite image.


the stack RNN has Θ(n) space. Additionally, we
find that it is not RR. This places it in the upperright box of Figure 1.
Classically, a stack is a dynamic list of objects to
which elements v _V can be added and removed_
_∈_
in a LIFO manner (using push and pop operations).
The stack RNN proposed in Suzgun et al. (2019b)
maintains a differentiable variant of such a stack,
as follows:

**Differentiable Stack** In a differentiable stack,
the update operation takes an element st to push
and a distribution πt over the update operations
push, pop, and no-op, and returns the weighted average of the result of applying each to the current
stack. The averaging is done elementwise along
the stacks, beginning from the top entry. To facilitate this, differentiable stacks are padded with
infinite ‘null entries’. Their elements must also
have a weighted average operation defined.

**Definition 6 (Geometric k-stack RNN encoder).**
Initialize the stack S to an infinite list of null entries,
and denote by St the stack value at time t. Using
1-indexing for the stack and denoting [St−1]0 ≜ **st,**
the geometric k-stack RNN recurrent update is:[15]

**st = fs(xt, ct−1)**

_πt = fπ(xt, ct−1)_


This encoding gives preference to the latest values
in the stack, giving initial stack encoding c0 = 0.

**Space Complexity** The memory introduced by
the stack data structure pushes the encoder into
Θ(n) space. We formalize this by showing that,
like a WFA, the stack RNN can encode binary
strings to their value.

**Lemma 5. The saturated stack RNN can com-**
_pute the converging binary encoding function, i.e.,_
101 1 1 + 0.5 0 + 0.25 1 = 1.25.
_�→_ _·_ _·_ _·_

15Intuitively, [πt]a corresponds to the operations push, noop, and pop, for the values a = 1, 2, 3 respectively.


_∀i ≥_ 1 [St]i =


3
�

[πt]a[St−1]i+a−2.
_a=1_


In this work we will consider the case where the
null entries are 0 and the encoding ct is produced
as a geometric-weighted sum of the stack contents,


**ct =**


_∞_
�

_i=1_


� 1 �i−1[St]i.

2


-----

_Proof. Choose k = 1. Fix the controller to always_
push xt. Then, the encoding at time t will be


**ct =**


_t_
�

_i=1_


� 1 �i−1xi. (84)

2


This is the value of the prefix x:t in binary.

**Rational Recurrence** We provide another construction to show that the stack RNN can compute
non-rational series. Thus, it is not RR.

**Definition 7 (Geometric counting). Define f2 :**
_{a, b}[∗]_ _→_ N such that


_f2(x) = exp 12_ �#a−b(x)� _−_ 1.

Like similar functions we analyzed in Section 3,
the Hankel matrix Hf2 has infinite rank over the
sub-block a[i]b[j].

**Lemma 6. The saturated stack RNN can compute**
_f2._

_Proof. Choose k = 1. Fix the controller to push 1_
for xt = a, and pop otherwise.


### H Erratum

We present corrections for experimental results
originally reported in Section 5. Thanks to David
Chiang for helping to identify these mistakes. The
QRNN used in our theoretical analysis was the ifoQRNN, whereas QRNN used for the original experimental results was the fo-QRNN (by our definition,
a QRNN where i = 1 **f** ). We redo our experi_−_
ments as originally intended with the ifo-QRNN
instead of the weaker fo-QRNN.
Results are presented in Figure 8. Overall, the
trend is similar to what was originally reported. For
_L5, all four models achieve 100% accuracy at the_
training length of 64. However, the QRNN performance drops earlier than for the other networks.
This matches the theoretical result that the s-QRNN
cannot recognize L5, whereas the other three saturated networks can. For a[n]b[n]Σ[∗], the LSTM and
2-layer QRNN reach similar accuracy at all lengths.
On the other hand, the 1-layer QRNN, with either
a 1 or 2-layer decoder, performs worse. This is
predicted by the fact that the s-QRNN cannot recognize a[n]b[n]Σ[∗] for any decoder.
While the results are mostly similar to the original results, one difference is that the ifo-QRNN
reaches 100% accuracy on L5 whereas the original
QRNN did not reach 100% even at n = 64. We


100

95

90

85

80

75

70

65

|anbn|*|
|---|---|
|anbn|*|
|QRNN QRNN (D2) LSTM 2-QRNN||


64 128 256 512 1024 2048

Length


Figure 8: Updated results for L5 (top) and a[n]b[n]Σ[∗]

(bottom). All networks use a D1 decoder, except for
“QRNN (D2)”. 2-QRNN is a 2-layer QRNN.


-----

consider the generalization accuracy for n > 64
to be a better indicator of whether the network has
learned the language rather than the in-distribution
test accuracy on strings of length 64. This is because, if we evaluate at the same length, a finitestate model can still in principle do well since it is
unlikely that the test set will contain prefixes with
configurations unseen during training.
We formalize this for L5, defined as

_L5 =_ �x ∈ (a|b)[∗] _| |#a−b(x)| < 5�._ (85)

Define the configuration c(x) of a string x
_∈_
_{a, b}[∗]_ as #a(x) − #b(x). Intuitively, c(x) represents all the information needed solve the recognition task. As a function of string length n, c(x)
follows a random walk where the motion of each
discrete time step is 1 with probability 1/2 and 1
_−_
otherwise. Thus, c(x) is a random variable with
a binomial distribution with mean 0 and variance
_n/4. So, roughly 95% of strings with length 64_
_√_
will have _c(x)_ 64 = 8. Only by increasing
_|_ _| ≤_

the length n can we force the model to contend
with new configurations.


-----

