### The Illusion of State in State-Space Models


**William Merrill** [1] **Jackson Petty** [1] **Ashish Sabharwal** [2]

#### Abstract


State-space models (SSMs) have emerged as a
potential alternative to transformers. One theoretical weakness of transformers is that they cannot
express certain kinds of sequential computation
and state tracking (Merrill & Sabharwal, 2023a),
which SSMs are explicitly designed to address
via their close architectural similarity to recurrent
neural networks. But do SSMs truly have an ad_vantage (over transformers) in expressive power_
_for state tracking? Surprisingly, the answer is_
no. Our analysis reveals that the expressive power
of S4, Mamba, and related SSMs is limited very
similarly to transformers (within TC[0]), meaning
these SSMs cannot solve simple state-tracking
problems like permutation composition and consequently are provably unable to accurately track
chess moves with certain notation, evaluate code,
or track entities in a long narrative. To supplement our formal analysis, we report experiments
showing that S4 and Mamba indeed struggle with
state tracking. Thus, despite their recurrent formulation, the “state” in common SSMs is an illusion: S4, Mamba, and related models have similar expressiveness limitations to non-recurrent
models like transformers, which may fundamentally limit their ability to solve real-world statetracking problems. Moreover, we show that only
a minimal change allows SSMs to express and
learn state tracking, motivating the development
of new, more expressive SSM architectures.

#### 1. Introduction


8
## qlrlqZ0Z
7
## Z0Z0Z0Z0
6
## 0Z0Z0Z0Z
5
## Z0Z0Z0Z0
4
## 0Z0Z0Z0Z
3
## Z0Z0Z0Z0
2
## PO0Z0Z0Z
1
## J0Z0Z0Zk
a b c d e f g h

x = [ 0, 0, 1, 0, 0]
x[1], x[3] = x[3], x[1] # Swap 1, 3

_Alice, Bob, Carl, Dan, and Emma each have a coin. All_
_are dimes except Carl’s. Alice and Carl trade coins._

Figure 1: We prove that SSMs, like transformers, cannot
solve inherently sequential problems like permutation composition (S5), which lies at the heart of state-tracking problems like tracking chess moves in source-target notation
(see Section 3.2), evaluating Python code, or entity tracking.
Thus, SSMs cannot, in general, solve these problems either.


Recent theoretical work has shown that transformer architecture based models are incapable of expressing inherently sequential computation (Merrill & Sabharwal, 2023a). These
results reveal a surprising limitation of transformers: they

1New York University 2Allen Institute for AI. Correspondence to: William Merrill <willm@nyu.edu>, Jackson Petty
_<petty@nyu.edu>, Ashish Sabharwal <ashishs@allenai.org>._

_Proceedings of the 41_ _[st]_ _International Conference on Machine_
_Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by_
the author(s).


**[Code: http://jpetty.org/ssm-illusion](http://jpetty.org/ssm-illusion)**

cannot express simple kinds of state tracking problems, such
as composing sequences of permutations, which even simple recurrent neural networks (RNNs) can naturally express.
In a different line of work, state space model (SSM) architectures (Gu et al., 2021; 2022a; Fu et al., 2023; Gu & Dao,
2023; Wang et al., 2024) have been introduced as an alternative to transformers, with the goal of achieving RNN-like
expressive power for handling problems that are naturally
stateful and sequential (Gu et al., 2021; 2022b). But does the
_seemingly stateful design of SSMs truly enable them to solve_
_sequential and state-tracking problems that transformers_
_cannot? If so, this would be a promising property of SSMs_
because state tracking is at the heart of large language model
(LLM) capabilities such as tracking entities in a narrative


1


-----

**The Illusion of State in State-Space Models**

(Heim, 1983; Kim & Schuster, 2023), playing chess under
certain notation[1], or evaluating code. This would motivate

_S5_

further research on SSM architectures and their deployment
in the next generation of LLMs.


In this work, we show that the apparent stateful design of
SSMs is an illusion as far as their expressive power is concerned. In contrast to the suggestion by Gu et al. (2021;
2022b) (and, perhaps, a broader belief in the community)
that SSMs have expressive power for state tracking similar to
RNNs, we prove theoretically that linear and Mamba-style
SSMs, like transformers, cannot express inherently sequential problems, including state-tracking problems like composing permutations that RNNs can easily express. Further,
our experiments confirm this prediction: both transformers and these SSMs cannot learn to compose permutations
with a fixed number of layers, whereas RNNs can compose
permutations with just a single layer. Our results imply
that arguments that current SSMs have an advantage over
transformers due to being “more recurrent” or capable of
tracking state are misguided. In fact, the SSM architectures
we consider are just as theoretically unequipped for state
tracking and recurrent computation as transformers are.

We first establish the theoretical weakness of linear SSMs
and near generalizations by proving they are in the complexity class L-uniform TC[0], which has been previously shown
for transformers (Merrill & Sabharwal, 2023a). This implies
these SSMs cannot solve inherently sequential problems
(formally, problems that are NC[1]-hard), including statetracking problems like permutation composition (Liu et al.,
2023). Permutation composition is a fundamental problem
at the heart of many real-world state-tracking problems such
as playing chess, evaluating code, or tracking entities in a
narrative (Figure 1), implying solutions to these problems,
too, cannot be expressed by SSMs, at least in the worst case.

At first glance, our results may appear to contradict Gu et al.
(2021)’s claim that linear SSMs can simulate general recurrent models, which can express permutation composition.
But the contradiction is resolved by a difference in assumptions: Gu et al. (2021) relied on infinite depth (number of
layers) to show that SSMs could simulate RNNs. We, on
the other hand, analyze the realistic setting with a bounded
number of layers, under which we find that SSMs cannot
simulate the recurrent state of an RNN and, in fact, suffer
from similar limitations as transformers for state tracking.

Empirically, we find that S4 (Gu et al., 2022a) and S6 (Gu
& Dao, 2023) SSMs, as well as transformers, do not learn to
solve the permutation composition state-tracking problem
with a fixed number of layers, while simple RNNs can do so
with just one layer. This provides empirical support for our

1The hardness of chess state tracking holds with (source, target)
notation, but standard notation may make state tracking easier.


Figure 2: Complexity hierarchy within NC[1]. Transformers
can only recognize languages within TC[0] (Merrill & Sabharwal, 2023a), and we show the same for SSMs (Theorems 4.2
and 4.4). Thus, both architectures cannot express the “hard
state tracking” captured by NC[1]-complete problems like S5,
which can be straightforwardly expressed by RNNs. The
figure assumes the widely held conjecture TC[0] = NC[1].
_̸_

theoretical separation in expressive power for state tracking
between SSMs and true recurrent models. We also find that
both transformers and SSMs struggle compared to RNNs
on state-tracking problems less complex than permutation
composition where it is not known whether they can express a solution. Thus, in practice, SSMs may struggle not
just on the hardest state-tracking problems like permutation
composition but also on easier variants.

Finally, we consider a minimal extension of a linear SSM
which makes the transition matrix input dependent, similar to Liquid S4 (Hasani et al., 2023). We show that this
extension has sufficient expressive power for state tracking and permutation composition. Empirically, we show
that our implementation of this extension learns to solve
permutation composition with a single layer, just like an
RNN, while being similarly parallelizable to other SSMs. It
is an open question whether such SSM architectures with
greater expressivity for state tracking are practically viable
for large-scale language modeling.

#### 2. Background

We first present the SSM architectures we will analyze (Section 2.1). Our analysis of the state tracking capabilities of
SSMs borrows deeply from the circuit complexity and algebraic formal language theory literature. We thus review
how circuit complexity can be used to analyze the power of
neural networks (Section 2.3) and how state-tracking problems can be captured algebraically and analyzed within the
circuit complexity framework (Section 3).

**2.1. Architecture of State-Space Models**

SSMs are a neural network architecture for processing sequences similar in design to RNNs or linear dynamical
systems. SSMs have been suggested to have two potential
advantages compared to transformers owing to their recur

2


-----

**The Illusion of State in State-Space Models**


rent formulation: faster inference and, possibly, the ability
to better express inherently sequential or stateful problems
(Gu et al., 2021; 2022b). Several architectural variants of
SSMs have been proposed, including S4 (Gu et al., 2022a)
and Mamba (Gu & Dao, 2023). Recently, SSMs have been
shown to achieve strong empirical performance compared to
transformers in certain settings, particularly those involving
a long context (Gu & Dao, 2023; Wang et al., 2024).

SSMs consist of SSM layers, which can be thought of as
simplified RNN layers. We define a generalized linear SSM
_layer that encapsulates both S4 (Gu et al., 2022a) and the S6_
layer used by Mamba (Gu & Dao, 2023) as special cases.
**Definition 2.1 (Generalized linear SSM layer). Given a**
sequence[2] **x1, . . ., xn ∈** R[k], the recurrent form of a linear
SSM layer defines a new sequence of states h1, . . ., hn ∈
R[d] using projections **A[¯]** _i ∈_ R[d][×][d] and **B[¯]** _i ∈_ R[d][×][k], which
can themselves depend on xi. For each 1 ≤ _i ≤_ _n,_

**hi = A[¯]** _ihi−1 + B[¯]_ _ixi._ (Recurrent form)

The convolutional form of the SSM layer defines the same[3]

**h1, . . ., hn computed differently as a summation:**


**2.2. Numeric Datatype**

Circuit-complexity analysis of neural networks depends to
some degree on low-level details about arithmetic and the
underlying datatype D used in the network’s computation
graph. We can think of D as parameterized by the number
of bits available to represent a number in D. For instance,
non-negative integers in [0, 2[p]] use p bits, signed integers in

[ 2[p], 2[p]] use p + 1 bits, FP16 uses 16 bits, etc.
_−_

Our main results (Theorems 4.2 and 4.4) will go through
for any datatype D for which the following operations are
efficiently parallel-computable (i.e., are in the complexity
class L-uniform TC[0], to be defined shortly in Section 2.3):

1. Iterated addition, i.e., summing n numbers in D

2. Iterated product, i.e., multiplying n numbers in D

3. Matrix powering, i.e., computing the n-th power of a
fixed-size d × d matrix over D

When D is any finite-precision datatype, i.e., has a fixed
number of bits available (e.g., 16 or 64), then these operations are easily seen to be in L-uniform TC[0]. As Merrill & Sabharwal (2023b) argue, however, finite-precision
datatypes severely limit the expressivity of neural architectures from a formal perspective (e.g., finite-precision transformers cannot represent uniform attention), motivating the
use of parameterized datatypes that can (approximately)
represent any number with a sufficiently large parameter.
Interestingly, when D is the datatype of n-bit integers, all
of the above operations are known to be in L-uniform TC[0]

(Hesse, 2001; Mereghetti & Palano, 2000). Realistically,
however, neural model implementations use floating point
numbers with much fewer than n bits. Following Merrill &
Sabharwal (2023b), we use the log-precision floating point
model, i.e., c log n bit floats where c is some fixed constant
(see Appendix A for a formal definition). Merrill & Sabharwal (2023a) showed that iterated addition over log-precision
floats is in L-uniform TC[0]. We extend the arguments of
Hesse (2001) and Mereghetti & Palano (2000) to show that
iterated product and matrix powering over log-precision
floats are also in L-uniform TC[0] (see Appendix A).

**2.3. Limits of Transformers via Circuit Complexity**

A line of recent work has used circuit complexity and logic
formalisms to identify expressivity limitations of transformers on reasoning problems (Angluin et al., 2023; Merrill
& Sabharwal, 2023a; Liu et al., 2023; Chiang et al., 2023;
Merrill & Sabharwal, 2023b; Hao et al., 2022); see Strobl
et al., 2024 for a survey. In particular, Merrill & Sabharwal
(2023a) showed transformers can only solve problems in the
complexity class TC[0], which is the set of problems that can
be recognized by constant-depth, polynomial-size threshold
circuit families. Such circuits, in addition to having standard


 _i_

 � **A¯** _k_

_k=j+1_




 **B¯** _jxj._ (Convolutional form)


**hi =**


_i_
�

_j=1_


The layer outputs yi = Cihi + Dixi R[k], where Ci
_∈_ _∈_
R[k][×][d] and Di ∈ R[k][×][k] depend on xi.

Two common cases of this layer are when **A[¯]** _i does not_
depend on the input (“non-gated”; Section 4.2) and when
**A¯** _i is diagonal (Section 4.3). In both of these cases, we will_
show that the SSM can be simulated in TC[0].

A generalized linear SSM is made up of multiple such
layers, with a linear projection and a non-linearity applied
after every layer (Rush & Karamcheti, 2022). Layer-norm
can also be applied, either before or after the layer.

**Practical Details.** In S4 and related SSMs, Definition 2.1

is applied elementwise (k = 1) across all m elements of
the previous layer output (Gu et al., 2022a). In practice,
the weight matrix initialization is crucial for training. Our
expressivity results (Theorems 4.2 and 4.4) apply for any
generalized linear SSM (including S4 and S6), independent
of initialization. In contrast to S4 and S6, H3 (Fu et al.,
2023) does not meet Definition 2.1 because the context is
not represented by a single vector. Rather, it resembles a
transformer with SSM components.

2In practice, this sequence is often a vector x1, . . . xn ∈ Rm

and the SSM is applied elementwise on each feature.
3The two forms express the same function over R or any other
distributive datatype. Over floating points (Section 2.2), they are
not guaranteed to be the same, but we must assume the error is
negligible for them to be well-defined and usable in practice.


3


-----

**The Illusion of State in State-Space Models**


AND, OR, and NOT gates (of arbitrary fan-in), can also use
threshold gates that output 1 iff at least k of the inputs are 1,
where k is a parameter of the gate. Informally, TC[0] can be
thought of as the class of problems that can be solved with
extremely parallel (constant-depth) computation.[4]

Problems outside TC[0], corresponding to problems that are
inherently sequential and thus cannot be parallelized, cannot
be solved by transformers. No problems in polynomial time
are known unconditionally to be outside TC[0], but unless the
widely held conjecture that TC[0] = NC[1] is false, many sim_̸_
ple NC[1]-hard problems are outside TC[0]. In particular, this
includes simulating finite automata (NC[1]-complete), evaluating boolean formulas (NC[1]-complete), determining graph
connectivity (L-complete), and solving linear equations (Pcomplete). These problems have already been shown to be
inexpressible by transformers (Merrill & Sabharwal, 2023a).
By showing that SSMs can be simulated in TC[0], we will
establish that they also cannot be solved by SSMs.

#### 3. State Tracking

Informally, a state-tracking problem is a problem where the
text specifies some sequence of updates to the state of the
world, and the goal of the problem is to determine what
the world state is after the updates have been applied in sequence. The circuit complexity view on the power of neural
networks can be combined with other insights from algebraic formal language theory to analyze the kinds of state
tracking that SSMs can express. In particular, this theory
reveals which kinds of state-tracking problems are (likely)
not in TC[0]. This will, in turn, allow us to find examples of
hard state tracking that models like SSMs cannot express.

**3.1. State Tracking as a Monoid Word Problem**

From the perspective of algebraic formal language theory,
state tracking over a finite world can be captured as a word
_problem on a finite monoid (Liu et al., 2023).[5]_ Different updates to the world become different elements in the monoid,
and resolving the final world state after all the updates have
been applied is equivalent to computing the product of a
sequence of elements (also called a “word”).

**Definition 3.1 (Word problem). Let M be a finite set, and**
(M, ) a finite monoid (i.e., M with identity and associa_·_
tive multiplication). The word problem for M is to re
4We use TC0 to mean L-uniform TC0, meaning the circuit
family is constructible by a Turing machine that runs in space logarithmic in the size of the input (cf. Merrill & Sabharwal, 2023a;
Strobl et al., 2024). We believe our results could be extended
from L-uniform TC[0] to DLOGTIME-uniform TC[0] using techniques similar to Merrill & Sabharwal (2023b) for composing TC[0]

circuits in a way that preserves DLOGTIME uniformity.
5We consider finite monoids for simplicity, but the approach
may be extendable to infinite (e.g., finitely generated) monoids.


duce sequences in M _[∗]_ under multiplication; that is, send
_m0m1 · · · mk to m0 · m1 · . . . · mk ∈_ _M_ . Solving the word
problem requires reducing sequences of arbitrary length.

_Example 3.2. Consider the monoid_ 0, 1 where is addi_{_ _}_ _·_
tion modulo 2. The word problem is to compute the parity
of a string, e.g., 0011 0. From a state-tracking perspec_�→_
tive, this monoid captures a world with a single light switch.
Identity 0 corresponds to no action, and 1 flips the switch.

Modeling state tracking with word problems lets us draw
connections between circuit complexity and algebra to understand which word problems are hard to solve. Krohn &
Rhodes (1965) established that not all word problems are
created equal: some, like Example 3.2, are in TC[0], while
others are NC[1]-complete, requiring recurrent processing
to solve (Immerman & Landau, 1989; Barrington, 1989).
Because we will show SSMs can be simulated in TC[0], it
follows that NC[1]-complete state-tracking problems cannot
be expressed by SSMs (cf. Figure 2).

Whether or not a word problem is NC[1]-complete depends on
the algebraic structure of the underlying monoid. Barrington
(1989) showed that the word problem of every finite nonsolvable[6] group is NC[1]-complete. That non-solvable groups
have NC[1]-complete word problems is notable because of
the ubiquity with which non-solvable groups show up in
tasks involving state tracking. The canonical example of an
NC[1]-complete word problem is that of S5, the symmetric
group on five elements that encodes the permutations over
five objects. As an immediate instantiation of this, consider
a document describing a sequence of transpositions: “swap
_ball 1 and 3, swap ball 3 and 5, swap ball 4 and 2, ...”._
Being able to answer the question “where does ball 5 end
_up?” for all possible swap sequences requires solving the S5_
word problem.[7] Beyond permutations, Figure 1 shows how
many natural state-tracking problems like tracking chess
moves, evaluating code, or tracking entities also encode the
structure of S5, meaning these state-tracking problems also
cannot be expressed by a model in TC[0]. Rather, in order to
solve these problems, the depth of the model would have to
be expanded to accommodate longer inputs.

Although the S5 word problem is canonical, in this paper we
will consider the word problem on a closely related group
_A5: the alternating group on five elements. We do this for_
simplicity: A5 is a subgroup of S5 containing only even
permutations, and is the smallest non-solvable subgroup.
We will compare the word problem on A5 to two other
baseline groups: A4 × Z5, a non-abelian but solvable group;

6We focus on word problems on groups, which are monoids
with inverses. Formally, a group G is solvable exactly when there
is a series of subgroups 1 = G0 < G1 < · · · < Gk = G such
that Gi−1 is normal in Gi and Gi/Gi−1 is abelian.
7W.l.o.g., any permutation can be factored into a sequence of
transpositions, or swaps.


4


-----

**The Illusion of State in State-Space Models**


and Z60, an abelian group encoding mod-60 addition. We
choose these groups as points of comparison because they
all have 60 distinct elements, meaning that the difficulty
in learning their word problems will come only from the
complexity of learning the group multiplication operation.

**3.2. Encoding S5 in Chess State Tracking**

Figure 1 already gives some intuition into how state-tracking
problems encode S5. Out of these examples, the most intricated case is chess. We now give a proper reduction from
_S5 to tracking chess moves, showing formally that not just_
_S5, but chess state tracking as well, is NC[1]-complete. We_
define the chess state-tracking problem as follows:

  - Input: A chessboard state and sequence of chess
**moves, where each move is written in UCI notation**
as a tuple (source square, target square). This differs
from the standard SAN notation that represents other
information like piece type (Toshniwal et al., 2021).

  - Output: The resulting board state after starting in the
initial board state and applying the sequence of moves
one after another, ignoring draws. If any move is illegal
given the previous board state, a null state is returned.

We show that S5 can be reduced to chess state tracking,
establishing its NC[1]-completeness:

**Proposition 3.3. S5 can be reduced to chess state tracking**
_in UCI notation via NC[0]_ _reductions._

_Proof. Without loss of generality, we consider the variant_
of S5 where the output is true if and only if the original first
element returns to the first position after the given sequence
of permutations has been applied.

The idea, as illustrated in Figure 1, is to map each element
of S5 to a fixed sequence of chess moves that permutes five
pieces accordingly on the chessboard. Given an instance
of the S5 word problem, we will construct an initial board
state and a sequence of moves such that the final chessboard
state encodes the output of that S5 problem instance.

Let M denote the set of chess moves in the UCI, i.e., (source
square, target square), notation.

**Initial Board State. We construct a chessboard similar to**
Figure 1 but with a black rook at a8 and black queens at b8
to e8.

**Chess Move Sequence. We then construct a finite function**
_f : S5 →_ _M_ _[∗]_ that encodes a permutation π as a sequence
of chess moves. We first factor each permutation π to a
sequence of transpositions τ1(π) · · · τmπ (π). Each transposition τ _T can in turn be expressed as a sequence of_
_∈_
chess moves analogously to Figure 1. For example, transposing items 1 and 3 can be expressed as the move sequence:


(a8, a7), (a1, b1), (c8, c6), (b1, a1), (a7, c7), (a1, b1),
(c6, a6), (b1, a1), (c7, c8), (a1, b1), (a6, a8), (b1, a1),
which has the crucial property that it transposes a8 with c8.
We denote the mapping from transpositions to chess move
sequences as f : T → _M_ _[∗]. Putting it all together, we have_

_mπ_


**Putting It All Together. We call our oracle for chess state**
tracking with the constructed initial board state and f (w) as
the sequence of chess moves. By construction, we can then
return true if and only if the rook is at a8. The reduction can
be implemented in NC[0] because it is a simple elementwise
mapping of the input tokens, and decoding from the output
chessboard is a finite table lookup.

As a fun aside, we note that the chess board constructed
in the above proof is reachable in a standard chess game.
The chess sequences encoding permutation sequences are
all valid in the game of chess, except that they ignore the
fact that repeated board states in chess technically lead to a
draw.

Since S5 is NC[1]-complete under AC[0] reductions and NC[0] _⊆_
AC[0], we have:

**Corollary 3.4. The chess state-tracking problem is NC[1]-**
_complete under AC[0]_ _reductions._

Theorem 3.2 of Feng et al. (2023) uses a similar reduction
to prove formula evaluation is NC[1]-complete. Reductions
can be constructed for evaluating Python or tracking entities
in a dialog, as suggested by Figure 1. As for chess, the
task formatting for entity tracking affects its hardness. For
instance, the formatting used by Kim & Schuster (2023)
in their Figure 1 is not NC[1]-complete, whereas the variant
shown in our Figure 1 is. This underscores the value of
theory for constructing examples of hard state tracking.

#### 4. SSMs Can be Simulated in TC[0]

In this section, we show that the convolutional form of
common variants of SSM can be simulated in TC[0]. Assuming the convolutional form of the model computes the
same function as the recurrent form, this implies such SSMs
cannot solve inherently sequential problems, despite their
appearance of recurrence and statefulness. We first show
containment in TC[0] for non-gated SSMs (Theorem 4.2), and
then show the same holds for diagonal SSMs (Theorem 4.4).


_f_ (π) =


_f_ (τj(π)).
_j=1_


To reduce a sequence of permutations w ∈ _S5[∗][, we let]_

_n_


_f_ (w) =


_f_ (wi).
_i=1_


5


-----

**The Illusion of State in State-Space Models**


**4.1. Conditions for Linear SSMs in TC[0]**

Before characterizing specific SSM architectures, we first
show that the complexity of computing transition matrix
products essentially determines the complexity of simulating an SSM with a circuit family.

**Lemma 4.1. Let M be a log-precision generalized linear**
_SSM. Then there exists an L-uniform TC[0]_ _circuit family that_
_computes M_ _’s convolutional form if:_

_1. For any integer interval [j, k], the matrix product_
�k
_i=j_ **[A][¯]** _[i][ can be computed in][ L][-uniform][ TC][0][ as a func-]_
_tion of_ **A[¯]** _j, . . .,_ **A[¯]** _k (to c log n precision for any c > 0)._

_2. For 1 ≤_ _i ≤_ _n,_ **A[¯]** _i,_ **B[¯]** _i, Ci, and Di can be computed_
_in L-uniform TC[0]_ _as a function of xi._

_Proof. Following the proof structure of Merrill & Sabhar-_
wal (2023a), we describe how to construct a log-space
bounded Turing machine TM that, given x1, . . ., xn as input, prints a circuit that simulates M on this input. We
first note that for all processing done before or after an
SSM layer (projection, non-linearity, layer norm, etc.), TM
can follow known simulations of such operations for transformers (Merrill & Sabharwal, 2023a; 2024) to output a
TC[0] circuit simulating this processing. We thus focus on
simulating an individual SSM layer.

Recall from Definition 2.1 that M ’s convolutional form
_i_ �
requires computing hi = [�]j[i] =1 ��k=j+1 **[A][¯]** _[k]_ **B¯** _jxj and_
**yi = Cihi + Dixi. By the second precondition, TM can**
print a TC[0] circuit that computes all matrices involved here.
Further, by the first precondition, TM can also print a TC[0]

circuit that computes the innermost product in the computation of each hidden state hi, namely [�]k[i] =j+1 **[A][¯]** _[k][. It]_
can now print a TC[0] circuit to multiply the resulting product[8] with **B[¯]** _j and xj, and then print a circuit to compute_
an iterated sum over the i resulting vectors to compute hi
(cf. iterated addition in Appendix A). It can similarly print a
(simpler) circuit to compute yi. Thus, the entire SSM layer
can be simulated by an L-uniform TC[0] circuit.

We will use Lemma 4.1 to show that any non-gated or diagonal generalized linear SSM can be simulated in TC[0].

**4.2. Non-Gated SSMs are in TC[0]**

**Theorem 4.2 (Non-gated SSM). Let M be a log-precision**
_generalized linear SSM such that, for any i,_

**A¯** _i = ¯A,_ **B¯** _i = ¯B,_ **Ci = C,** **Di = D.**

8Let c log n be the SSM’s precision. We compute [�]k **[A][¯]** _[k][ to]_

_c[′]_ log n precision for a large enough c[′] (similar to the proof of
Lemma A.5) such that the full product ��k **[A][¯]** _[k]�_ **B¯** _jxj is correct_

to at least c log n bits, as technically required by Definition A.3.


_Then there exists an L-uniform TC[0]_ _circuit family that com-_
_putes M_ _’s convolutional form._

_Proof. We prove this by showing that both conditions from_
Lemma 4.1 are satisfied. Computing the matrix product
reduces to powering **A[¯]** _[k][−][j]._ Crucially, we can use the
fact that matrix powering over floats is in L-uniform TC[0]

(Lemma A.8, extending Mereghetti & Palano, 2000). Finally, **A[¯]** _i,_ **B[¯]** _i, Ci, and Di can be computed in L-uniform_
TC[0] because they are constants.

As S4 satisfies the premises of Theorem 4.2, we obtain:

**Corollary 4.3. There exists an L-uniform TC[0]** _circuit family_
_that computes S4’s convolutional form._

**4.3. Diagonal SSMs are in TC[0]**

**Theorem 4.4 (Diagonal SSM). Let M be a log-precision**
_generalized linear SSM where for 1_ _i_ _n:_
_≤_ _≤_

_1. the transition matrix_ **A[¯]** _i is diagonal, denoted diag(¯ai)_
_where ¯ai ∈_ R[d];

_2. each of ¯ai,_ **B[¯]** _i, Ci and Di can be computed in L-_
_uniform TC[0]_ _as a function of xi._

_Then there exists an L-uniform TC[0]_ _circuit family that com-_
_putes M_ _’s convolutional form._

_Proof. By the first condition,_ [�]i **[A][¯]** _[i][ =][ �]i_ [diag(¯][a][i][)][. It-]

erated multiplication of diagonal matrices is reducible to
several iterated scalar multiplications, placing this product
in L-uniform TC[0] (Lemma A.5). The second condition
from Lemma 4.1 is satisfied by assumption. Thus, M ’s
convolutional form is computable in L-uniform TC[0].


Since S6 satisfies the premises of Theorem 4.4, we have:

**Corollary 4.5. There exists an L-uniform TC[0]** _circuit family_
_that computes S6’s convolutional form (used by Mamba)._

_Proof. For the first condition, note that S6’s transition ma-_
trix **A[¯]** _i is defined as exp(δiA) for a fixed diagonal A. The_
set of diagonal matrices is closed under scalar multiplication and matrix exponentiation, so **A[¯]** _i is also diagonal. See_
Appendix B for a proof that the second condition is satisfied
by the S6 parameterization.

Appendix C extends Theorem 4.4 to hold even when {A[¯] _i}_
are simultaneously diagonalizable, rather than just diagonal. Specifically, we prove the following generalization:

**Theorem 4.6 (Simultaneously diagonalizable SSM). Let**
**W be a fixed matrix. Let M be a log-precision generalized**
_linear SSM such that, for 1_ _i_ _n,_
_≤_ _≤_


6


-----

**The Illusion of State in State-Space Models**


_1. the transition matrix_ **A[¯]** _i is computable to log precision_
_by the expression W diag(¯ai)W[−][1], where ¯ai ∈_ R[d];

_2. each of ¯ai,_ **B[¯]** _i, Ci and Di can be computed in L-_
_uniform TC[0]_ _as a function of xi._

_Then there exists an L-uniform TC[0]_ _circuit family that com-_
_putes M_ _’s convolutional form._

This, in turn, allows us to prove that a simultaneously diagonalizable transition matrix generalization of the S6 layer is
also in L-uniform TC[0] (Corollary C.7).

**4.4. Discussion**

Theorems 4.2 and 4.4 establish that common SSM variants,
like transformers, can only express solutions to problems in
the class TC[0]. This means these SSMs cannot solve NC[1]hard problems like evaluating boolean formulas or graph
connectivity. In particular, it shows that they are limited as
far as their state tracking capabilities as they are unable to
compose permutations (solve the S5 word problem):

**Corollary 4.7. Assuming TC[0]** ≠ NC[1], no log-precision
_SSM with the S4 or S6 architecture can solve the word_
_problem for S5 or any other NC[1]-hard problem._

In contrast, RNNs can easily express S5 via standard constructions that encode finite-state transitions into an RNN
(Minsky, 1954; Merrill, 2019). This shows that SSMs cannot express some kinds of state tracking and recurrence that
RNNs can. This tempers the claim from Gu et al. (2021,
Lemma 3.2) that SSMs have the expressive power to simulate RNNs, which relied on the assumption that SSMs
can have infinite depth. In a more realistic setting with
a bounded number of layers, our results show SSMs cannot express many state-tracking problems, including those
which can be solved by fixed-depth RNNs.

#### 5. Extending the Expressive Power of SSMs

We have shown that S4 and S6, despite their seemingly
“stateful” design, cannot express problems outside TC[0],
which includes state tracking like S5. We show how SSMs
can be extended to close the gap in expressive power with
RNNs, allowing them to express S5. Two simple extensions
can bring about this increase in expressive power, assuming
layer input dimension k > 1. First, adding a nonlinearity
makes the SSM into an RNN, adding expressive power but
degrading parallelism. On the other hand, allowing **A[¯]** _i to_
be input-dependent makes the SSM more like a weighted
finite automaton (WFA; Mohri, 2009), adding expressive
power while remaining parallelizable.


**5.1. Via Nonlinearities**

One extension to the SSM is to add a nonlinearity, effectively
making it an RNN. We call this an RNN-SSM layer:

**hi = sgn** �Ah¯ _i−1 + ¯Bxi�_ _._

A model with this architecture can solve the S5 word problem when the input dimension k > 1:

**Theorem 5.1. For any regular language L ⊆** Σ[∗] _(includ-_
_ing the word problem for S5), there exists a one-layer log-_
_precision RNN-SSM with k =_ Σ _that recognizes L._
_|_ _|_

_Proof. The standard constructions for simulating automata_
with RNNs (cf. Minsky, 1954; Merrill, 2019) apply. The
condition k = Σ comes from needing to represent token
_|_ _|_
types with linearly independent vectors.

Adding a nonlinearity to the output of an SSM layer (as in
Mamba) is not the same thing as an RNN-SSM. Rather, an
RNN-SSM applies the nonlinearity at each recurrent update.
A downside of this approach is that it becomes nonlinear
to parallelize the RNN-SSM computation graph with the
SCAN algorithm used by linear SSMs (Blelloch, 1990).

**5.2. Via Input-Dependent Transition Matrices**

Another way to get greater expressive power is to let the
transition matrix **A[¯]** _i be fully input-dependent, as explored_
by Liquid S4 (Hasani et al., 2023). To illustrate this, we
define a minimally different SSM called Input-Dependent
S4 (IDS4) that achieves greater expressive power for state
tracking. Let πA : R[k] _→_ R[d][×][d] be some affine transformation where the output vector is interpreted as a d _d matrix,_
_×_
and let **A[¯]** _i = πA(xi). Let_ **B[¯]** _, C, D be fixed (w.r.t. i). By_
Definition 2.1, the IDS4 convolutional form computes an
_iterated product of non-diagonal, input-dependent matrices:_


 _i_

�
 _πA(xi)_

_k=j+1_




 **Bx¯** _j._


**hi =**


_i_
�

_j=1_


In contrast to matrix powers or iterated products of diagonal
matrices, iterated products of general matrices cannot be
computed in TC[0] (Mereghetti & Palano, 2000). This means
that the arguments from Theorems 4.2 and 4.4 will not
go through for IDS4. In fact, we can show IDS4 gains
expressive power beyond TC[0]:

**Theorem 5.2. For any regular language L ⊆** Σ[∗] _(includ-_
_ing the word problem for S5), there exists a one-layer log-_
_precision IDS4 SSM with k =_ Σ _that recognizes $L, where_
_|_ _|_
$ Σ is a special beginning-of-string symbol.
_̸∈_

_Proof. It suffices to show that IDS4 can simulate a deter-_
ministic finite automaton (DFA). We do this via a transition


7


-----

**The Illusion of State in State-Space Models**


RNN S4 IDS4 Mamba Transformer

|A × 60 4 5 4 layers 3 of # 2 min. 1 5 10 15 20 5 10 15 20 sequence length sequence length TC|A 5 5 10 15 20 sequence length NC¹|
|---|---|


Figure 3: Minimum number of layers (lower is better) required to attain > 90% validation accuracy on group multiplication
problems by sequence length and group. RNN and IDS4 models of constant depth can solve arbitrarily long sequences,
while transformer, S4, and Mamba models require depths monotonically increasing in sequence length.


monoid construction. For any w ∈ Σ[∗], let δw : Q → _Q be_
the function mapping a state to its eventual destination state
after w is read from that state. For any DFA, this set of functions forms a finite monoid (the transition monoid) under
composition, following from the Myhill-Nerode theorem
(Hopcroft et al., 2001). Further, each monoid element δw
can be represented as a boolean transition matrix, making
matrix multiplication isomorphic to monoid composition.
Computing the transition monoid of a DFA allows recognizing valid words: compute the monoid element for a word
by multiplying the elements for its tokens and then check
whether the initial state maps to an accepting state.

Fix a DFA and its transition monoid δ. To complete the
proof, we show there exists an SSM that, for all w ∈ Σ[∗],
computes δw given input x = $w. Let **A[¯]** _i be the transition_
matrix representation of δxi. Matrix multiplication is isomorphic to composition of transition monoid elements. We
view indices in hi as states and define **B[¯]** $ as 1 at the initial
state q0 and 0 elsewhere. For other σ, let **B[¯]** _σ =_ _[⃗]0. This_
yields the following convolutional form:


�


�


**hi =**


� _i_
� **A¯** _i_

_k=2_


**B$**
_≡_


� _i_

_δxk_
_k=2_


(q0).


Since x = $w, we conclude that h|x| ≡ _δw(q0)._

**5.3. Discussion**

Theorems 5.1 and 5.2 show that two minimal extensions of
the SSM enable expressive power outside TC[0], allowing the
model to solve hard state-tracking problems:

**Corollary 5.3. There exist a one-layer log-precision RNN-**
_SSM and WFA-SSM that express the word problem for S5_
_(with a beginning-of-string symbol), and these these SSMs_
_cannot be simulated in TC[0]._


But would these variants of SSMs be feasible to use in practice? Besides expressive power, there are two competing
practical concerns that might make these extensions problematic: parallelism and the impact on learning dynamics.

**Parallelism. To be used effectively in an LLM, a model**
architecture must be parallelizable on practical hardware.
Architectures in TC[0] are parallelizable by design (Merrill
& Sabharwal, 2023a), but architectures in NC[1] may still be
parallelizable to log depth even if they cannot be parallelized
to constant depth. For IDS4, the bottleneck would be computing iterated matrix product with a log-depth computation
graph. This could be achieved with the SCAN algorithm
(Blelloch, 1990) similar to S4 and S6. In contrast, it is less
clear how to parallelize a model with a nonlinearity.

**Learning Dynamics. Another potential concern for IDS4**
is that learning dynamics could be degraded. In particular,
an iterated product of matrices may lead to vanishing or
exploding gradients. However, this is already potentially
an issue for the S6 architecture, where the selective gating
involves computing an iterated product of scalars.

#### 6. Can SSMs Learn Permutations in Practice?

Having established theoretical limitations of SSMs for state
tracking, we empirically test how well SSMs can learn such
tasks, focusing on the A5 word problem. Since this problem
is NC[1]-complete and transformers, S4, and Mamba can only
express functions in TC[0], these models should require a
depth that grows with the input length to solve this problem.

**Task. We model word problems (see Section 3.1) as a**
token-tagging task. Models are given as input a sequence
_g0g1g2_ _gn drawn from one of A5, A4_ Z5, or Z60. At
_· · ·_ _×_
each step i, the label is the product of the first i elements of


8


-----

**The Illusion of State in State-Space Models**


the sequence. Modeling the problem as a tagging task rather
than as a sequence classification task provides the models
with more supervision during training, making it as easy as
possible to learn the correct function. We tokenize inputs
such that each element gets a unique token.

**Models. We train a transformer as a TC[0]** baseline, an RNN
that we expect can perform state tracking, and three SSMs:
S4 (Gu et al., 2022a), Mamba (Gu & Dao, 2023), and IDS4
(Section 5.2). For IDS4, we initialize the affine projection α
as a random normal centered around the identity: α(xi) ∼
**I +** (0, σ[2]). This ensures that, at initialization, input_N_
dependent transitions tend to propagate the previous state,
which we expect to aid learning efficiency.

**Experimental Setup. We train models on sequences of**
length n for successively larger values of n and report fullsequence accuracy on a test set.[9] To validate the prediction
that SSMs and transformers require growing depth to solve
longer A5 word problems, we plot the minimum depth with
90% test accuracy as a function of input sequence length.

**Results. Figure 3 shows single-layer RNN and IDS4 models**
learn the word problem for arbitrarily long sequences for
all three groups. In contrast, transformer, S4, and Mamba
models require depth monotonically increasing in sequence
length to attain good test accuracy for the non-commutative
groups. We draw three conclusions from this:

1. As expected, S4 and Mamba show the same limitations
as transformers on the A5 word problem. Longer A5 sequences require deeper models, consistent with these models
being in TC[0]. In contrast, RNNs (Theorem 5.1) and IDS4
(Theorem 5.2) can efficiently solve the A5 word problem.

2. Transformers, S4, and Mamba require greater depth
even for A4 × Z5, which can be theoretically expressed by
TC[0] circuits. Although transformer and Mamba models
of a given depth perform as good or better on A4 × Z5
as they on A5, they still require increasingly many layers
to handle proportionally longer sequences. There are two
possible interpretations of this. First, it could be that while
these word problems are expressible in TC[0], they cannot be
expressed by S4, Mamba, or transformers (which can each
likely recognize only a proper subset of TC[0]). On the other
hand, it is possible that these word problems are expressible
by transformers, S4, and Mamba but that effectively learning
a constant-depth solution is difficult.

3. Despite this limitation, S4 and Mamba appear empirically
_better than transformer at approximate state tracking on_
the non-commutative tasks. For length-n sequences from
_A4_ Z5 or A5, the transformer requires at least as many
_×_
(and frequently more) layers as S4 or Mamba.

9We always include all 3600 pairwise sequences of length 2 in
the training data along with the training split of length-n sequences.


#### 7. Conclusion

We formally analyzed a family of generalized linear SSMs
and showed that, like transformers, common SSM variants
including S4 and Mamba can only express computation
within the complexity class L-uniform TC[0] of highly parallel computations. This means they cannot solve inherently sequential problems like graph connectivity, boolean
formula evaluation, and—of particular interest for state
tracking—the permutation composition problem S5. S5
can be naturally expressed by true recurrent models like
RNNs and captures the essence of hard state tracking due
to its NC[1]-completeness. In practice, one-layer RNNs can
easily learn a task capturing S5 while linear SSMs require
depth growing with the sequence length. These results reveal that S4, Mamba, and related SSMs cannot truly track
state: rather, they can only solve simple state-tracking problems for which shallow shortcuts exist (Liu et al., 2023).

On the other hand, we showed that an input-dependent SSM
similar to Hasani et al.’s (2023) Liquid S4 can both express
and learn the S5 word problem, providing evidence that
the expressiveness limitations of current SSMs can be overcome. Ultimately, this line of work could unlock new neural
architectures that balance the parallelism of transformers
and SSMs with full expressive power for state tracking, enabling LLMs that can benefit from scale while enjoying a
greater capacity to reason about games, code, and language.

#### Impact Statement

This paper aims to advance the foundational understanding
of state-space architectures for deep learning. Such work
can affect the development and deployment of deep learning
models in a variety of ways, which in turn can have societal
impacts. However, we find it difficult to meaningfully speculate about or anticipate these downstream impacts here.

#### Acknowledgments

This work benefited from discussions with and valuable
feedback from Chris Barker, Stefano Ermon, and Charles
Foster. It was supported in part through the NYU IT High
Performance Computing resources, services, and staff expertise. It was funded by NSF award 1922658, and WM
was supported by an NSF graduate research fellowship, AI2,
and Two Sigma.

#### References

Angluin, D., Chiang, D., and Yang, A. Masked hardattention transformers and Boolean RASP recognize exactly the star-free languages, 2023. arXiv:2310.13897.

Barrington, D. A. Bounded-width polynomial

9


-----

**The Illusion of State in State-Space Models**


size branching programs recognize exactly
those languages in nc1. _Journal of Computer_
_and_ _System_ _Sciences,_ 38(1):150–164, 1989.
URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/0022000089900378)
[science/article/pii/0022000089900378.](https://www.sciencedirect.com/science/article/pii/0022000089900378)

Blelloch, G. E. Prefix sums and their applications. Technical
Report CMU-CS-90-190, School of Computer Science,
Carnegie Mellon University, November 1990.

Chiang, D., Cholak, P., and Pillay, A. Tighter bounds on the
expressivity of transformer encoders. In ICML, 2023.

Feng, G., Zhang, B., Gu, Y., Ye, H., He, D., and Wang, L.
Towards revealing the mystery behind chain of thought:
A theoretical perspective. In NeurIPS, 2023.

Fu, D. Y., Dao, T., Saab, K. K., Thomas, A. W., Rudra, A.,
and Re, C. Hungry hungry hippos: Towards language
modeling with state space models. In ICLR, 2023.

Gu, A. and Dao, T. Mamba: Linear-time sequence modeling
with selective state spaces, 2023. arXiv:2312.00752.

Gu, A., Johnson, I., Goel, K., Saab, K. K., Dao, T., Rudra,
A., and Re, C. Combining recurrent, convolutional, and
continuous-time models with linear state space layers. In
_NeurIPS, 2021._

Gu, A., Goel, K., and Re, C. Efficiently modeling long
sequences with structured state spaces. In ICLR, 2022a.

Gu, A., Goel, K., Saab, K., and Re, C.´ Structured state spaces: Combining continuous-time, recurrent, and convolutional models, January 2022b.
[URL https://hazyresearch.stanford.edu/](https://hazyresearch.stanford.edu/blog/2022-01-14-s4-3)
[blog/2022-01-14-s4-3. Blog post accessed Jan-](https://hazyresearch.stanford.edu/blog/2022-01-14-s4-3)
uary 31, 2024.

Hao, S., Angluin, D., and Frank, R. Formal language recognition by hard attention transformers: Perspectives from
circuit complexity. TACL, 10:800–810, 2022.

Hasani, R., Lechner, M., Wang, T.-H., Chahine, M., Amini,
A., and Rus, D. Liquid structural state-space models. In
_ICLR, 2023._

Heim, I. File change semantics and the familiarity theory of
definiteness. Semantics Critical Concepts in Linguistics,
pp. 108–135, 1983.

Hesse, W. Division is in uniform TC [0]. In International
_Colloquium on Automata, Languages, and Programming,_
pp. 104–114, 2001.

Hesse, W., Allender, E., and Barrington, D. A. M. Uniform
constant-depth threshold circuits for division and iterated
multiplication. J. Comput. Syst. Sci., 65:695–716, 2002.


Hopcroft, J. E., Motwani, R., and Ullman, J. D. Introduction
to automata theory, languages, and computation. ACM
_SIGACT News, 32(1):60–65, 2001._

Immerman, N. and Landau, S. The complexity of iterated
multiplication. In [1989] Proceedings. Structure in Com_plexity Theory Fourth Annual Conference, pp. 104–111,_
1989. doi: 10.1109/SCT.1989.41816.

Kim, N. and Schuster, S. Entity tracking in language models.
In Rogers, A., Boyd-Graber, J., and Okazaki, N. (eds.),
_ACL, July 2023._

Krohn, K. and Rhodes, J. Algebraic theory of machines. i.
prime decomposition theorem for finite semigroups and
machines. Transactions of the American Mathematical
_Society, 116:450–464, 1965._

Liu, B., Ash, J. T., Goel, S., Krishnamurthy, A., and Zhang,
C. Transformers learn shortcuts to automata. In ICLR,
2023.

Mereghetti, C. and Palano, B. Threshold circuits for iterated
matrix product and powering. RAIRO-Theor. Inf. Appl.,
34(1):39–46, 2000. doi: 10.1051/ita:2000105. URL
[https://doi.org/10.1051/ita:2000105.](https://doi.org/10.1051/ita:2000105)

Merrill, W. Sequential neural networks as automata. In Eisner, J., Galle, M., Heinz, J., Quattoni, A., and Rabusseau,´
G. (eds.), Proceedings of the Workshop on Deep Learn_ing and Formal Languages: Building Bridges, Florence,_
August 2019. ACL.

Merrill, W. and Sabharwal, A. The parallelism tradeoff:
Limitations of log-precision transformers. _TACL, 11,_
2023a.

Merrill, W. and Sabharwal, A. A logic for expressing logprecision transformers. In NeurIPS, 2023b.

Merrill, W. and Sabharwal, A. The expressive power of
transformers with chain of thought. In ICLR, 2024.

Minsky, M. Neural nets and the brain-model problem. Un_published doctoral dissertation, Princeton University, NJ,_
1954.

Mohri, M. _Weighted Automata Algorithms, pp. 213–_
254. Springer Berlin Heidelberg, Berlin, Heidelberg,
2009. ISBN 978-3-642-01492-5. doi: 10.1007/
[978-3-642-01492-5 6. URL https://doi.org/10.](https://doi.org/10.1007/978-3-642-01492-5_6)
[1007/978-3-642-01492-5_6.](https://doi.org/10.1007/978-3-642-01492-5_6)

Reif, J. H. and Tate, S. R. On threshold circuits and
polynomial computation. _SIAM Journal on Comput-_
_ing, 21(5):896–908, 1992. doi: 10.1137/0221053. URL_
[https://doi.org/10.1137/0221053.](https://doi.org/10.1137/0221053)


10


-----

**The Illusion of State in State-Space Models**


Rush, S. and Karamcheti, S. The annotated S4. In
_Blog Track at ICLR 2022, 2022._ [URL https://](https://openreview.net/forum?id=xDaLPsMBZv-)
[openreview.net/forum?id=xDaLPsMBZv-.](https://openreview.net/forum?id=xDaLPsMBZv-)

Strobl, L., Merrill, W., Weiss, G., Chiang, D., and Angluin,
D. What formal languages can transformers express? A
survey. TACL, 12, 2024.

Toshniwal, S., Wiseman, S., Livescu, K., and Gimpel, K.
Chess as a testbed for language model state tracking. In
_AAAI, 2021._

Wang, J., Gangavarapu, T., Yan, J. N., and Rush, A. M.
Mambabyte: Token-free selective state space model,
2024. arXiv:2401.13660.

#### A. Floating-Point Arithmetic

Our results use the log-precision floating point model used
by Merrill & Sabharwal (2023b) to analyze transformers.
For some fixed constant c ∈ Z[+], a c log n precision float
is a tuple _m, e_ where m, e are signed integers together
_⟨_ _⟩_
taking c log n bits. Using _x_ to mean the number of bits
_|_ _|_
used to represent integer x, this float represents the value
_m_ 2[e][−|][m][|][+1].
_·_

Unlike for integers, arithmetic operations over log-precision
floats are not closed. That is, the product ϕ1 _ϕ2 of two_
_×_
_p-precision floats is a well-defined number but may not_
be exactly representable as a p-precision float. It is thus
necessary to define approximate versions of these operations
when formalizing log-precision floating-point arithmetic.
To this end, Merrill & Sabharwal (2023a) define a natural
notion of approximate iterated addition over log-precision
floats and show that it is computable in L-uniform TC[0].
We can naturally apply their definition of iterated addition
for floats to matrices of floating points, defining iterated
summation over matrices of datatype D as the result of
treating the numbers as reals, performing exact arithmetic,
and casting the exact output ϕ back to D, denoted castD(ϕ).
Formally:

**Definition A.1 (Iterated D-matrix sum; Merrill & Sabhar-**
wal, 2023a). For matrices M1, . . ., Mn over D with the
same size, their iterated D-sum is


**Definition A.2 (Iterated D-matrix product). For square ma-**
trices M1, . . ., Mz over D, their iterated D-product is


�


_z_
�

**Mi ≜** castD
_i=1_


� _z_
�

castR(Mi)
_i=1_


_._


�


_z_
�

**Mi ≜** castD
_i=1_


� _z_
�

castR(Mi)
_i=1_


_._


Here castR converts a number in D to the corresponding
real number. D is implicit in the notations castR and [�].
Integer addition can be obtained as a special case for 1dimensional matrices. We can also analogously defined
iterated summation, which will be necessary for formalizing
SSMs:


Merrill & Sabharwal (2023a) showed that iterated addition
from for log-precision floats is in L-uniform TC[0]. It naturally follows from their argument that iterated addition
over log-precision float matrices is also in L-uniform TC[0].
In general, iterated matrix products are not necessarily computable in TC[0]. However, we extend the arguments of Hesse
(2001) and Mereghetti & Palano (2000) for integers to show
that two special cases (iterated scalar multiplication and matrix powering) over log-precision floats are also computable
in L-uniform TC[0].

Finally, we define a canonical value for a compositional
arithmetic expression over floats that enjoys the associative
property.

**Definition A.3 (Flattened expression evaluation). Let ϕ be**
a compositional expression over floats, which may contain
alternating sums and products as well as other operations
like exp. We define the canonical value of ϕ as the value
returned by the computation graph obtained by flattening
all adjacent sums into a single sum (and analogously for
products).

Definition A.3 has the nice effect of making Definition A.2
associative. The only results that rely on this assumption
are our analysis of diagonalizable SSMs in Appendix C. We
also deal with the details of this assumption in Lemma 4.1,
though the proof there also goes through directly without
handling these details.

**A.1. Complexity of Iterated Scalar Multiplication**

The first special case of iterated matrix products we analyze
is when the matrices are simply scalars (or, w.l.o.g., diagonal
matrices). In this case, the iterated product can be computed
in L-uniform TC[0].

**Lemma A.4 (Iterated D-product). Let ϕ1, . . ., ϕz ∈** D be
_such that z ≤_ _n and each ϕi can be represented as an n-bit_
_integer. If operators castD and castR are in L-uniform TC[0],_
_then the iterated D-product_ [�]i[z]=1 _[ϕ][i][ can be computed in]_
L-uniform TC[0].

_Proof. By preconditions of the lemma, we can compute_
_yi = castR(ϕi) for each i in L-uniform TC[0]. Since each_
_ϕi is equivalent to an n-bit integer, yi can be viewed as an_
_n-bit integer. The iterated integer product y =_ [�]i[z]=1 _[y][i]_
can be computed with an L-uniform TC[0] circuit (Hesse,
2001). Finally, by a precondition of the lemma, we can cast
the result back to D, i.e., compute castD(y) which equals


11


-----

**The Illusion of State in State-Space Models**


the iterated D-product [�]i[z]=1 _[ϕ][i][, with an][ L][-uniform][ TC][0]_

circuit.

**Lemma A.5 (Iterated float product). Let ϕ1, . . ., ϕz be**
_c log n precision floats and z_ _n. Then the iterated float_
_≤_
_product_ [�]i[z]=1 _[ϕ][i][ can be computed in][ L][-uniform][ TC][0][.]_

_Proof. The idea is to convert (by scaling up) the sequence_
of ϕi to another sequence of floats that are all representable
as integers, apply Lemma A.4, reverse the scaling, and cast
the result back to a c log n precision float.

Let e be the smallest exponent across all ϕi and q =
max{0, −e}. Construct re-scaled floats ψi = ϕi2[q] by
adding q to the exponent of ϕi, using up to c log n additional
bits in the exponent if necessary to keep the computation
exact. Note that e, q, and all ψi can easily be computed exactly by an L-uniform TC[0] circuit as they involve fixed-arity
arithmetic operations. Further, by construction, every ψi
has a non-negative exponent and thus represents an integer.

The maximum number representable by each c log n precision float ϕi is upper bounded by 2[n][c] . Thus, the maximum
number representable by each entry ψi is 2[n][c] _×_ 2[q] = 2[n][c][+][q].
Let m = n[c] + q. It follows that each ψi can be equivalently
represented as an m-bit integer. Further, this integer can be
computed by left-shifting the mantissa of ψi by a number
of bits equal to the value of the exponent of ψi (which is
non-negative). Finally, this left-shift, and thus the castR operation over m-precision floats, can be easily computed by
an L-uniform threshold circuit of size poly(m). In the other
direction, casting from reals to m-precision floats can also
be easily accomplished by an L-uniform threshold circuit of
size poly(m).

Observing that ψ1, . . ., ψz is a sequence of floats each representable as an m-bit integer, we now apply Lemma A.4
with D being ‘float’ to conclude that iterated float product
_τ =_ [�]i[z]=1 _[ψ][i][ can be computed by an][ L][-uniform threshold]_
circuit of size poly(m). Since m 2n[c], this circuit is also
_≤_
of size poly(n).

Finally, to compute the original iterated float product
�z
_i=1_ _[ϕ][i][, we divide][ τ][ by][ 2][qz][. This can be accomplished]_
by subtracting qz from the exponent of τ ; again, we do this
computation exactly, using up to (c + 1) log n additional
bits in the exponent if necessary. We then cast the resulting
float back to a c log n precision float. All this can be done
in L-uniform TC[0], finishing the proof that [�]i[z]=1 _[ϕ][i][ can be]_
computed in L-uniform TC[0].

**A.2. Complexity of Matrix Powering**

The second special case we analyze is matrix powering: i.e.,
a matrix product where all the matrices being powered are
the same. Mereghetti & Palano (2000) showed that when the
datatype D is n-bit integers, one can compute M[n] in TC[0].


We note that their construction also works for computing
**M[z]** for any z ≤ _n, z ∈_ Z[+]. Further, as they remark,
their construction can, in fact, be done in uniform TC[0].
Specifically, we observe most of their construction involves
sums and products of constantly many n-bit integers, which
can be done in L-uniform TC[0]. The only involved step is
dividing a polynomial of degree (up to) n by a polynomial of
degree (up to) d 1 and returning the remainder. It turns out
_−_
that this “polynomial division with remainder” operation can
also be performed in L-uniform TC[0] (see Corollary 6.5 of
Hesse et al., 2002 and an explanation in Appendix A.3). We
thus have the following extension of Mereghetti & Palano’s
result:

**Lemma A.6 (Integer matrix power, derived from Mereghetti**
& Palano, 2000). Let d ∈ Z[+] _be a fixed constant. Let M be_
_a d_ _×_ _d matrix over n-bit integers and z ≤_ _n, z ∈_ Z[+]. Then
_integer matrix power M[z]_ _can be computed in L-uniform_
TC[0].

We extend this to matrix powers over D rather than integers:

**Lemma A.7 (D-matrix power). Let d ∈** Z[+] _be a fixed_
_constant. Let M be a d × d matrix over a datatype D with_
_entries equivalently representable as n-bit integers. Let_
_z ≤_ _n, z ∈_ Z[+]. If operators castD and castR are in L_uniform TC[0], then D-matrix power M[z]_ _can be computed in_
L-uniform TC[0].

_Proof. By preconditions of the lemma, we can compute_
castR(M) in L-uniform TC[0]. Since the entries of M are
equivalent to n-bit integers, castR(M) can be viewed as a
_d_ _d integer matrix of n-bit integers. By Lemma A.6, we_
_×_
can compute castR(M)[z] using an L-uniform TC[0] circuit.
Finally, by a precondition of the lemma, we can cast the
result back to D, i.e., compute castD(castR(M)[z]) which
equals M[z], with an L-uniform TC[0] circuit.

**Lemma A.8 (Float matrix power). Let d, c ∈** Z[+] _be fixed_
_constants. Let M be a d_ _d matrix over c log n precision_
_×_
_floats. Let z ≤_ _n, z ∈_ Z[+]. Then float matrix power M[z] _can_
_be computed in L-uniform TC[0]._

_Proof. The idea is to convert (by scaling up) M to another_
float matrix all whose entries are representable as integers,
apply Lemma A.7, reverse the scaling, and cast the result
back to c log n precision floats.

Let e be the smallest exponent across all float entries of M
and q = max 0, _e_ . Construct a re-scaled float matrix
_{_ _−_ _}_
**M˜** = M2[q] by adding q to the exponent of every entry of
**M, using up to c log n additional bits in the exponent if**
necessary to keep the computation exact. Note that e, q,
and **M[˜]** can easily be computed exactly by an L-uniform
TC[0] circuit as they involve fixed-arity arithmetic operations.
Further, by construction, **M[˜]** has non-negative exponents in


12


-----

**The Illusion of State in State-Space Models**


all its float entries. Thus, every entry of **M[˜]** represents an
integer.

The maximum number representable by each c log n precision float in M is upper bounded by 2[n][c]. Thus, the
maximum number representable by each entry of **M[˜]** is
2[n][c] 2[q] = 2[n][c][+][q]. Let m = n[c] + q. It follows that each
_×_
entry ϕ of **M[˜]** can be equivalently represented as an m-bit integer. Further, this integer can be computed by left-shifting
the mantissa of ϕ by a number of bits equal to the value
of the exponent of ϕ (which is non-negative). Finally, this
left-shift, and thus the castR operation over m-precision
floats, can be easily computed by an L-uniform threshold
circuit of size poly(m). In the other direction, casting from
reals to m-precision floats can also be easily accomplished
by an L-uniform threshold circuit of size poly(m).

Note that 2[q] [0, n[c]] and hence m [n[c], 2n[c]]. In partic_∈_ _∈_
ular, m _n. Thus z_ _n (a precision) implies z_ _m._
_≥_ _≤_ _≤_
Observing that **M[˜]** is a matrix of floats each representable as
an m-bit integer, we now apply Lemma A.7 with D being
‘float’ to conclude that float matrix power **M[˜]** _[z]_ can be computed by an L-uniform threshold circuit of size poly(m).
Since m 2n[c], this circuit is also of size poly(n).
_≤_

Finally, to compute M[z], we first divide each entry of **M[˜]** _[z]_

by 2[qz]. This can be accomplished by subtracting qz from
the exponent of each entry of **M[˜]** ; again, we do this computation exactly, using up to (c + 1) log n additional bits in
the exponent if necessary. We then cast all entries of the
resulting matrix back to c log n precision floats. All this can
be done in L-uniform TC[0], finishing the proof that M[z] can
be computed in L-uniform TC[0].

**A.3. L-Uniformity of Polynomial Division in TC[0]**

Hesse et al. (2002) state that polynomial division is in Luniform TC[0] in Corollary 6.5. For historical reasons, this
claim is preceded by weaker claims in older papers. We
briefly clarify this situation to help understand why the
stronger claim is valid.

Reif & Tate (1992) establish that polynomial division can be
performed in P-uniform TC[0], whereas we state our results
for L-uniform TC[0], which is a smaller class. However, the
only issue preventing the polynomial division result from
originally going through in the L-uniform case is that, at
the time of Reif & Tate’s publication, it was not known
whether integer division and iterated integer multiplication
are computable in L-uniform TC[0]. However, Hesse (2001)
later proved exactly this. Combining the two results, Theorem 3.2 of Reif & Tate (1992) goes through even with
L-uniformity (not just P-uniformity). Its Corollary 3.3 then
allows us to conclude that integer polynomial division can
be solved by L-uniform TC[0] circuits because the output of
integer polynomial division is an analytic function whose


Taylor expansion has a finite number of terms (Reif & Tate,
1992).

#### B. S6 Parameterization

To justify that the S6 architecture used by Mamba is computable in TC[0], we justify that **A[¯]** _i,_ **B[¯]** _i, Ci, Di can be com-_
puted as a function of xi in TC[0].

We begin by summarizing how exactly is S6 parameterized.
S6 first defines continuous-time parameters:

1. A is a fixed, diagonal matrix that is invertible (each
_aii ̸= 0);_

2. Bi = πB(xi) is computed via a projection;

3. Ci = πC(xi) is computed via a projection;

4. Di = I .

Next, we need to discretize the matrices A and B. S6 does
this using an input-dependent discretization factor δi:

_δi = softplus(δ + πδ(xi))._

The discretized matrices are then defined as:

**A¯** _i = exp(δiA)_
**B¯** _i = (δiA)[−][1][ �]A¯_ _i_ **I�** _δiBi._
_−_

It is clear to see that the diagonalizability condition of Theorem 4.4 is satisfied because **A[¯]** _i itself is diagonal. Addition-_
ally, all the relevant matrices can be computed in TC[0].

**Proposition B.1.** **A[¯]** _i,_ **B[¯]** _i, Ci, and Di can all be computed_
_as functions of xi in L-uniform TC[0]._

To prove this, observe that A, Bi, Ci, Di can all be computed in L-uniform TC[0] because they are either constants
or linear transformations of xi. To justify that **A[¯]** _i and_ **B[¯]** _i_
can be computed in L-uniform TC[0], we just need to justify
that we can invert diagonal matrices and compute softplus
and exp in L-uniform TC[0].

**Lemma B.2. Diagonal matrices over log-precision floats**
_can be inverted in L-uniform TC[0]._

_Proof. Inverting a diagonal matrix just involves forming the_
reciprocal along the diagonal. Scalar reciprocals can be approximated to error at most 2[−][n][c] (for any c) in TC[0] (Hesse
et al., 2002). This means we can compute the reciprocal
of a log-precision float (cf. Appendix A) exactly up to log
precision.

In Appendix D, we show that we can compute the nonlinearities exp and softplus over a bounded domain in TC[0].


13


-----

**The Illusion of State in State-Space Models**


#### C. Diagonalizable SSMs

We extend Theorem 4.4 to cover the case when the SSMs
transition matrices are simultaneously diagonalizable, rather
than just diagonal. This requires us to note that when working with log-precision floating point representations of matrices, a diagonal matrix A and its diagonalized decomposition
**W diag(a)W[−][1]** are numerically substitutable.

**Theorem 4.6 (Simultaneously diagonalizable SSM). Let**
**W be a fixed matrix. Let M be a log-precision generalized**
_linear SSM such that, for 1_ _i_ _n,_
_≤_ _≤_

_1. the transition matrix_ **A[¯]** _i is computable to log precision_
_by the expression W diag(¯ai)W[−][1], where ¯ai ∈_ R[d];

_2. each of ¯ai,_ **B[¯]** _i, Ci and Di can be computed in L-_
_uniform TC[0]_ _as a function of xi._

_Then there exists an L-uniform TC[0]_ _circuit family that com-_
_putes M_ _’s convolutional form._

_Proof. When the first condition is satisfied, the following_
equality holds over log-precision floats:

� **A¯** _i =_ � �W diag(¯ai)W[−][1][�] _._

_i_ _i_

By the associativity of D-matrix products, we can remove
the parentheses to get

� **A¯** _i =_ � **W diag(¯ai)W[−][1]**

_i_ _i_

�� �

= W diag(¯ai) **W[−][1].**

_i_

Iterated multiplication of diagonal matrices is reducible to
several iterated scalar multiplications, which is in L-uniform
TC[0] (Lemma A.5). Then the product of all **A[¯]** _i is the product_
of three L-uniform TC[0]-computable matrices, so is itself
L-uniform TC[0]-computable. The second condition from
Lemma 4.1 is satisfied by assumption. Thus, the convolutional form for M can be computed in L-uniform TC[0].

**C.1. Diagonalizable S6**

We can define an extension of S6 which satisfies these conditions to show that it is also in L-uniform TC[0].

**Definition C.1. Diagonalizable S6 has continuous-time**
parameters:

1. A is a fixed matrix diagonalizable as W diag(a)W[−][1]

that is invertible (each aii ̸= 0);

2. Bi = πB(xi) is computed via a projection;

3. Ci = πC(xi) is computed via a projection;

4. D = I.


As in the standard S6, the discretization of A and B is done
by an input-dependent discretization factor δi:

_δi = softplus(δ + πδ(xi))._

The discretized matrices are then defined as

**A¯** _i = exp(δiA),_
**B¯** _i = (δiA)[−][1]( ¯Ai_ **I)δiBi.**
_−_

To prove that **A[¯]** _i and_ **B[¯]** _i have the necessary properties, we_
first introduce some lemmas dealing with matrix-valued
functions of diagonalizable matrices.

**Lemma C.2. If a matrix A is diagonalizable, then we can**
_substitute its diagonalized decomposition W diag(a)W[−][1]_

_in a computation graph over log-precision floats involving_
**A without incurring any meaningful error.**

_Proof. Let A be diagonalizable. Then there exists invertible_
**W and diagonal diag(a) such that A = W diag(a)W[−][1].**
Note that the product of a fixed number of matrices is
in L-uniform TC[0], and so the first c log n bits of A and
**W diag(a)W[−][1]** are identical.

**Lemma C.3. Let A be diagonalizable as W diag(a)W[−][1],**
_where a ∈_ R[d]. Then c · A is simultaneously diagonalizable
_with A via c_ **A = Wc** diag(a)W[−][1].
_·_ _·_

_Proof. Scalar multiplication commutes around matrix mul-_
tiplication.

**Lemma C.4. Let A be diagonalizable as W diag(a)W[−][1],**
_where a ∈_ R[d]. Then exp(A) = W exp(diag(a))W[−][1].

_Proof. The matrix exponential is defined as a power series,_
so for diagonalizable A it follows that

exp(A) = exp(W diag(a)W[−][1])

�∞ 1

=

_k!_ [(][W][ diag(][a][)][W][−][1][)][k]

_k=0_

�∞ 1

=

_k!_ **[W][ diag(][a][)][k][W][−][1]**

_k=0_

��∞ 1 �

= W **W[−][1]**

_k! [diag(][a][)][k]_

_k=0_


= W exp(diag(a))W[−][1].

The expressions in Lemma C.4 are equivalent not just over
real numbers but also over log-precision floats. This is
because we know both expressions can be approximated in
TC[0] with error at most 2[−][n][c], which means the c log n bits
of the approximation must be equivalent.


14


-----

**The Illusion of State in State-Space Models**


**Lemma C.5. Diagonalizable matrices over log-precision**
_floats can be inverted in L-uniform TC[0]._

_Proof. Let A_ = **W diag(a)W[−][1].** Then A[−][1] =
**W[−][1]** diag(a)[−][1]W. We are guaranteed that each of these
matrices exists, and furthermore by Lemma B.2 we know
that diag(a)[−][1] is computable in L-uniform TC[0]. Their product, involving a finite number of additions and multiplies, is
also computable in L-uniform TC[0].

**Proposition C.6.** **A[¯]** _i and_ **B[¯]** _i can be computed as functions_
_of xi in L-uniform TC[0]._

_Proof. We first show that_ **A[¯]** _i is L-uniform TC[0]_ computable.
By definition,
**A¯** _i = exp(δiA)._

By Corollary D.2, δi is computable in L-uniform TC[0]. The
product δiA is simultaneously diagonalizable with A so

**A¯** _i = exp(Wδi diag(a)W[−][1])_ (Lemma C.3)

= W exp(diag(a))W[−][1]. (Lemma C.4)

Since the exponential of scalars is L-uniform TC[0] computable by Corollary D.2, then **A[¯]** _i is as well._

Turning to **B[¯]** _i, note that the term (δiA)[−][1]_ is L-uniform
TC[0] computable by Lemma C.5 since δiA is diagonalizable.
Since **A[¯]** _i is L-uniform TC[0]_ computable, the difference **A[¯]** _i_ _−_
**I is as well. Then every term in**

**B¯** _i = (δiA)[−][1]( ¯Ai_ **I)δiBi**
_−_

is L-uniform TC[0] computable, and so their product is as
well.

_Remark. Since Ci and Di are unchanged between the stan-_
dard and diagonalizable versions of S6, the proofs of their
computability as functions of xi in L-uniform TC[0] pass
through from Appendix B.

**Corollary C.7. There exists an L-uniform TC[0]** _circuit fam-_
_ily that computes Diagonalizable S6’s convolutional form._

_Proof. Note that since A = W diag(a)W[−][1]_ is fixed the
set of transition matrices {A[¯] _i} is simultaneously diagonal-_
izable via W for all i.

Then Diagonalizable S6 meets the conditions for Theorem 4.6.

#### D. Nonlinearities in L-Uniform TC[0]

.

The parameterization of SSMs (and transformers) involves
computing nonlinearities like exp and softplus. We leverage existing circuit complexity results (Reif & Tate, 1992) to


show that, in general, any well-behaved nonlinearity should
be computable in L-uniform TC[0] when used in conjunction
with pre- or post-layer norm.

**Lemma D.1 (Adapts Corollary 3.3, Reif & Tate, 1992). Let**
_X = (_ _B, B) be a bounded interval. Let f be a function_
_−_
_over X with a convergent Taylor series:_


_f_ (x) =


_∞_
�

_n=0_


_an_
(x − _x0)[n],_
_bn_


_where an, bn are integers with magnitude at most 2[n][O][(1)]_

_computable in L-uniform TC[0]. Then f can be approximated_
_over X by L-uniform TC[0]_ _circuits to log precision (error at_
_most 2[−][n][c]_ _for any c_ 1).
_≥_

_Proof. Reif & Tate (1992) give a proof when X = (_ 1, 1).
_−_
We generalize to X = ( _B, B), assuming w.l.o.g. B = 2[k]._
_−_
The idea is to transform f to have domain ( 1, 1) via
_−_

_g(x) = f_ (Bx).

Then, we can apply Corollary 3.3 of Reif & Tate (1992)
to approximate g with error at most 2[−][n][c]. Reif & Tate
(1992) state their result for P-uniform TC[0], but through
advances in circuit complexity since the time of publication (Appendix A.3), their construction naturally applies for
L-uniform TC[0] as well.

To approximate f, compute z = x/B, which can be done
exactly since B = 2[k]. We conclude by computing g(z) =
_f_ (x), which, as established, has error at most 2[−][n][c] .

Because of pre- and post-norm layers, the elements of xi in
an SSM will remain in a bounded domain ( _B, B). Thus,_
_−_
the following lemma shows we can compute them:

**Corollary D.2. The pointwise nonlinearities exp, log, and**
softplus are computable over ( _B, B) in L-uniform TC[0]._
_−_

_Proof. By Reif & Tate (1992, Corollary 3.3) know that the_
Taylor series for exp and log is convergent with an, bn computable in L-uniform TC[0]. Then exp and log are themselves
computable in L-uniform TC[0].

Since softplus(x) = log (1 + exp(x)) is a fixed composition of L-uniform TC[0]-computable functions, it too is
computable in L-uniform TC[0].


15


-----

