## The Parallelism Tradeoff: Limitations of Log-Precision Transformers


### William Merrill Center for Data Science New York University, New York, NY willm@nyu.edu

 Abstract


### Ashish Sabharwal Allen Institute for AI Seattle, WA ashishs@allenai.org

Early theoretical work on transformers established their Turing completeness, albeit with assumptions like infinite precision and arbitrarily
powerful feedforward subnets (Pérez et al., 2019;
Dehghani et al., 2019). On the other hand, a
strand of more recent work uses techniques from
circuit complexity theory to derive strong limitations on the types of problems transformers can
solve given restrictions on the form of attention
allowed in the transformer. Specifically, Hahn

(2020) and Hao et al. (2022) showed transformers restricted to hard attention are very limited:
they can only solve problems in a weak complexity class (non-uniform AC[0]) that doesn’t even contain basic problems like majority of n bits. Merrill
et al. (2022) extended this to a more general class
of “saturated attention” transformers with a floating point datatype, and showed a larger class of
problems (non-uniform TC[0]) as an upper bound.
This motivates analyzing a setting that strikes a
middle ground: Can we characterize transformers
_whose precision and feedforward nets’ computa-_
_tional power are realistically bounded, but where_
_attention is also realistically expressive?_

An important practical limitation of these prior
results is the “non-uniform” nature of the considered circuit classes, which makes these classes
non-realizable and the findings difficult to interpret. This is because non-uniform AC[0] and TC[0],
while highly limited in computation, also contain some problems that are not even decidable,
i.e., for which there doesn’t exist any exact algorithm. Thus, non-uniform classes cannot be directly compared with standard algorithmic complexity classes such as P, NP, etc. This motivates
our second key question: Can we derive uniform
_upper bounds on transformers?_

We show that one can achieve both of these
goals by making the modest assumption that all
values in the transformer have O(log n) precision (where n is the number of input tokens),


Despite their omnipresence in modern NLP,
characterizing the computational power of
transformer neural nets remains an interesting open question. We prove that transformers whose arithmetic precision is logarithmic in the number of input tokens (and
whose feedforward nets are computable using space linear in their input) can be simulated by constant-depth logspace-uniform
threshold circuits. This provides insight
on the power of transformers using known
results in complexity theory. For example, if L = P (i.e., not all poly-time
_̸_
problems can be solved using logarithmic
space), then transformers cannot even accurately solve linear equalities or check membership in an arbitrary context-free grammar with empty productions. Our result intuitively emerges from the transformer architecture’s high parallelizability. We thus
speculatively introduce the idea of a fundamental parallelism tradeoff: any model
architecture as parallelizable as the transformer will obey limitations similar to it.
Since parallelism is key to training models
at massive scale, this suggests a potential inherent weakness of the scaling paradigm.

### 1 Introduction


This work aims to characterize the computational
model implicit in transformer neural networks
(Vaswani et al., 2017), which form the basis of recent breakthroughs in large language models such
as BERT (Devlin et al., 2019), T5 (Raffel et al.,
2020), and GPT-3 (Brown et al., 2020). What
computational primitives can the transformer’s
components implement, and what problems can
the full system solve in aggregate? These questions are important for interpreting transformers in
a principled way, understanding potential limitations of their reasoning capabilities, and building
trust in deployed transformer-based systems.


-----

and, similarly, that transformer’s subnetworks are
computable in O(log n) space. Log precision is
enough to represent the positional encodings at the
input layer of the transformer, and to encode pointers to all other positions in the sequence at later
transformer layers. Assuming log precision across
all layers captures the idea that the hidden representations contain a constant number of hidden
states whose precision (16 or 32 bits) is small relative to the length of the input (2048 in GPT-3). On
long sequences, the precision will not be enough
to losslessly encode the full input sequence into
a single vector. Instead, the processing of the sequence must somehow be distributed in each layer
and performed in parallel.

**Upper Bound on Transformers.** Our main contribution is proving that log-precision transformers can be simulated by uniform constant-depth
threshold circuits. Thus, such transformers can
**only solve problems in uniform TC[0]. This char-**
acterization is strikingly weak compared to the
Turing-completeness of infinite-precision transformers. Since we believe log precision is more realistic for practical transformers than infinite precision, these results point to the conclusion that
transformers are not Turing-complete in practice.
In contrast to past results, our upper bound on
transformers is a uniform circuit class, enabling
direct comparison of log-precision transformers to
many natural complexity classes. These connections reveal specific problems that define the upper
limits of log-precision transformers’ capabilities,
as discussed further in §2.
Intuitively, our upper bound says that logprecision transformers are computationally shallow, and that this shallowness can be understood
to emerge from their parallelizability. Transformers’ inherent parallelism is useful for training them
efficiently at massive scale, but may limit the complexity of the computations they can express. We
introduce the term parallelism tradeoff to capture
this idea, which represents a potential fundamental
weakness of the current paradigm of scaling language models. Formally characterizing reasoning
capabilities relevant to language models and understanding whether they likely fall outside upper
bounds implied by the tradeoff would clarify the
practical implications of this limitation of scaling.
It could also be that the limitations of parallelism are not a curse but a blessing, if they constrain the hypothesis space in a way useful for


learning. We have no evidence that this is true,
but mention it as an alternate interpretation of the
results that could be clarified in future work.

**Instruction Following and Advice Transform-**
**ers.** We also consider an instruction following
setting (Brown et al., 2020) where the transformer
is provided the description of a task along with an
input on which to execute the instruction. We construct a practically parameterizable transformer
that can execute instructions perfectly if they are
provided in the form of TC[0] circuits. This complements recent work that studies transformers’ ability to follow other forms of instructions such as
regular expressions (Finlayson et al., 2022).
Based on the fundamental property that transformers can correctly evaluate any given TC[0] circuit on a given input, we introduce the notion of
_advice transformers akin to advice taking Turing_
machines. We show that transformers can recognize any (non-uniform) TC[0] language if provided
appropriate poly-size advice.

In summary, our findings provide new insights
on both the abilities and the limitations of transformers, and bring out bounded precision, threshold computations, and parallelism as key notions for understanding the implicit computational
model of transformers in practice.

**Roadmap.** Before diving into technical details,
we discuss in §2 the implications of our results
on both fundamental as well as practical abilities
of transformers. §3 provides a brief primer on
circuits as a model of computation. It then discusses a way of serializing a circuit into a string;
we later show how to generate such serializations
using a resource-bounded algorithm, which is the
key to proving containment of transformers in uni_form circuit classes. §4 defines our formal model_
of bounded-precision transformers. §5 derives our
first formal bound on log-precision transformers.
This bound involves non-uniform circuit families,
similar in spirit to prior results in this area. §6
proves our more technical main result: the first
_uniform circuit complexity upper bound for trans-_
formers (specifically, uniform TC[0]). Finally, §7
provides a lower bound on transformers, introduces the notion of an Advice Transformer, and
connects these to the machine learning problems
of Instruction Learning and Following.


-----

### 2 Implications of Our Findings

Before diving into technical details, we discuss the
general implications of our findings on the abilities and limitations of transformers. We will focus
here on our main result (Thm. 2), which shows that
log-precision transformers are in the complexity
class logspace-uniform TC[0].

**The Parallelism Tradeoff.** One interpretation
of complexity classes such as NC[0], AC[0], and TC[0]

is sets of poly-time solvable problems that are
parallelizable to a very high degree—they can be
solved in parallel in constant time with enough
parallel processors. This gives some intuitive explanation of our result: log-precision transformers
end up in TC[0] because they were designed to be
highly parallelizable. Since parallelism is an important property of today’s dominant paradigm of
training models at massive scale, this points to the
conclusion that any massively scaled up model—
transformer or otherwise—will likely obey restrictions similar to the ones derived here for logprecision transformers. There is thus an important
tradeoff between the massive parallelizability of
today’s networks and their representation power.

**What Transformers Can/Cannot Compute.**
Our result places log-precision transformers in the
complexity class logspace-uniform TC[0]. This has
immediate implications on the kinds of problems
such transformers can and cannot accurately solve.
Consider any problem X that is complete for a
complexity class C that contains logspace-uniform
TC[0]. By definition of completeness, every problem log-precision transformers can solve perfectly
is efficiently reducible to X and is thus no harder
than X. This implies that—despite their massive
size—the computation performed by such transformers is, for instance, no harder than solving basic L-complete problems like graph connectivity:
the problem of checking whether there is a path
between two nodes in an undirected graph (Lewis
and Papadimitriou, 1982; Reingold, 2008).
By the same token, if C is strictly larger than
logspace-uniform TC[0], then such transformers
_cannot perfectly solve X._ Thus, log-precision
transformers cannot perfectly solve the following
reasoning problems:

  - Linear equalities: find x s.t. Ax = b[1]

1Assuming logspace-uniform TC0 ̸= P. Follows because
these problems are P-complete (Greenlaw et al., 1991).



  - Universal context-free recognition[1][,][2]

  - Propositional satisfiability (SAT)[3]

  - Horn-clause satisfiability (HORN-SAT)[1]

  - AI planning (Bylander, 1991)

  - Permanent computation[4]

This highlights the limits of practical transformers
with limited-precision arithmetic, indicating that
they are far from being universal or all-powerful
as suggested by some prior studies.
One important caveat about these negative results is that they are asymptotic in nature—they
apply for “large enough” input size n. It’s possible
for log-precision transformers to solve such problems easily when n is small. Further, these negative results are about exact solutions, but they often also extend beyond this when formal hardnessof-approximation results are known.

**Limitations of Our Formal Model.** Prior formal characterizations of transformers either make
unrealistically strong assumptions (Pérez et al.,
2019; Dehghani et al., 2019) or place unrealistic
restrictions (Hahn, 2020; Hao et al., 2022; Merrill et al., 2022). In contrast, we make only one
assumption—namely, all intermediate values in
the transformer are limited to O(log n) bits, where
_n is the number of input tokens. We next discuss_
some implications of this assumption and what our
findings mean for practical transformers.
As mentioned above, our bounds are asymptotic in nature and thus apply when n is sufficiently large. In practice, transformers use fixed
precision at each computation node, which is more
restrictive than precision growing with the input
sequence length n, as O(log n) bits. However,
this constant could be large and thus, for relatively small n, our results do not rule out practical transformers solving difficult problems. Our
results, however, do show that as n grows sufficiently large, log-precision transformers are fundamentally limited to problems within TC[0] and
cannot accurately solve various commonly studied
problems mentioned earlier under “What Transformers Cannot Compute”. Extending our analysis to small n will help close the gap to practice.

2Takes both a grammar and a string as input and return
whether the grammar generates the string. Jones and Laaser
(1976) demonstrate P-completeness.
3Assuming logspace-uniform TC0 ̸= NP. Follows because SAT is NP-complete (cf. Biere et al., 2009).
4Assuming logspace-uniform TC0 ̸= #P. Follows because permanent is #P-complete (Valiant, 1979). Allender
(1999) shows permanent is not in logtime-uniform TC[0].


-----

Our formal model is based on a binary classification view of transformers. However, our results apply directly to multi-class classification as
well and can be extended to generation problems
by viewing, for instance, next word prediction in
NLP as a multi-class classification problem. However, if the transformer decoder is allowed to condition on its previous output in a generation problem, then this would violate our formal setup.

**2.1** **Potential Applications**

**Extracting Circuits from Transformers.** Elhage et al. (2021) propose extracting circuits[5] that
capture the computational structure of transformers. Our results suggest threshold circuit families are a good formalism for expressing mechanisms extracted from transformers. Constructively
converting transformers to threshold circuits is beyond the scope of the current paper, although we
hope to explore this in more detail in future work.

**Testing Separation Candidates in Complexity**
**Theory.** Thm. 2 also motivates a paradigm for
quickly testing complexity theory conjectures. If
a problem is believed to separate TC[0] and NC[1], a
transformer can be trained on problem instances.
If the transformer generalizes perfectly to harder
instances than it was trained on, this gives an empirical hint that the problem is in TC[0], providing
evidence against the conjecture.

### 3 Circuit Computation

Let 0, 1 be the set of finite binary strings. For
_{_ _}[∗]_
_x_ 0, 1, let _x_ be its length. We refer to
_∈{_ _}[∗]_ _|_ _|_
a function from 0, 1 to 0, 1 as a boolean
_{_ _}[∗]_ _{_ _}[∗]_
function. Boolean functions can implement arithmetic operations if we define a semantics for binary strings as numbers. We will treat the intermediate values in a transformer as binary strings,
and the internal operations as boolean functions.
_Circuits are a model of computation for com-_
puting boolean functions of fixed-length binary
strings.[6] Formally, a circuit is a directed acyclic
computation graph. The leaf nodes represent binary variables and their negations. The internal
nodes represent functions in some set, and the
_G_

5Their sense of “circuit” is not exactly the formal sense
we use in this paper, though the goal of capturing transformers’ implicit computational mechanism is the same.
6For a mini-tutorial on circuit complexity theory and its
relevance to transformers, see Merrill et al. (2022).


directed edges represent the flow of function outputs into inputs of other functions. One or more
nodes in the circuit are marked such that their
value is the output of the circuit.

**Definition 1. For a set of functions**, a -circuit
_G_ _G_
is a directed acyclic computation graph where the
internal nodes have labels from .
_G_

**Complexity Measures.** The size of a circuit is
the total number of gates in it, including negation.
The depth of a circuit is the length of the longest
path from any input node to any output node.

**Circuit Families.** A circuit family generalizes a
circuit to take variable-length binary strings as input. Formally, a circuit family is a sequence of
circuits Cn : {0, 1}[n] _→{0, 1} for n ∈_ N. A
circuit family implicitly recognizes a formal language defined as follows:

**Definition 2. A circuit family Cn recognizes L ⊆**
_{0, 1}[∗]_ if, for all x ∈{0, 1}[∗], C|x|(x) = 1 if and
only if x _L._
_∈_

We now define classes of languages by constraining the complexity of the circuit families
needed to recognize them:

**Definition 3. Let non-uniform AC[0]** be the set of
_L_ 0, 1 such that L is recognizable by a poly_⊆{_ _}[∗]_
size, constant-depth _,_ _,_ -circuit family.
_{¬_ _∧_ _∨}_

For k ∈ N, a threshold gate θ≤k takes m input
bits and returns whether [�]i[m]=1 _[x][i][ ≤]_ _[k][. We define]_
_θ≥k analogously. For example, θ≤3(110011) = 0._

**Definition 4. Let TC[0]** be the set of L
_⊆_
0, 1 such that L is recognizable by a poly-size,
_{_ _}[∗]_
constant-depth {θ≤k, θ≥k}k∈N-circuit.

The gates,, and are all just special cases
_¬_ _∧_ _∨_
of thresholds, so we can imagine TC[0] circuits to
have access to these as well. Thus, TC[0] circuits
can implement AC[0] circuits.

**Circuit Serialization.** We identify a circuit with
its serialization in a formal language that identifies each node’s label and adjacency list. We will
adopt a specific grammar for concreteness, but our
construction can be adapted to other string representations of circuits.
We define a circuit serialization as a traversal
of a circuit ordered by some topological sort. In
this serialization, leaf nodes (variables) are represented by the string X. An internal node (gate) is
represented in Polish notation by the function it


-----

computes (AND, OR, or NOT) followed by a list of
pointers to its arguments. Each argument &1[j] of
gate i encodes (in a unary) a zero-indexed pointer
to the j-th gate in the circuit, where j < i. The
final node is interpreted as the circuit output.
To serialize _,_ -circuits, we use the follow_{∧_ _∨}_
ing grammar, where the i parameter is passed
through Gate[i] nonterminals to track the index of
the gate in left-to-right order:

Circuit Gate[1] Gate[2] Gate[g]
_→_ _· · ·_

Gate[i] → X | NOT Arg[i] | Op Arg[i][∗]

Arg[i] &1[j] s.t. j < i
_→_

Op AND OR
_→_ _|_

In the Arg[i] rule, we enforce that j < i so that arguments must be pointers to already defined gates.
As an example of this serialization language, the
circuit for x1 _x2_ _x3 is represented as[7]_
_∨¬_ _∨_

X X X NOT &1 OR & &111 &11

By convention (cf. §3), negations in AC[0] circuits
are usually taken to occur at the beginning of the
circuit, rather than after or nodes.[8] Our seri_∧_ _∨_
alization grammar does not enforce this property,
but of course any circuit with this property can be
serialized by our grammar.
It is a bit more complicated to serialize threshold circuits. Formally, a threshold circuit serialization is generated by the following grammar:

Circuit Gate[1] Gate[2] Gate[g]
_→_ _· · ·_

Gate[i] X Dir 1[k]0[m][−][k] Arg[i][m]
_→_ _|_

Arg[i] &1[j] s.t. j < i
_→_

Dir <= >=
_→_ _|_

In the rewrite rule for Gate[i], m ∈ N is the
arity of the gate, and k _m is its threshold._
_≤_
The span 1[k] after Dir can be interpreted semantically as a unary encoding of the parameter k for a
threshold gate, padded by 0’s to the number of total arguments of gate i. For simplicity, we imagine
_¬ gates are represented as unary θ≤0 gates. Thus,_
the circuit θ≥1(x1, ¬x2) would be represented as

X X <= 00 &1 >= 10 & &11

We say a threshold circuit serialization is in pre_fix form if all inputs (X) come before all threshold_
gates (<= or >=), as is the case in this example.

7Spaces here (and in the grammar) are added for readability. We will ignore these spaces when passing circuit serializations as inputs to a transformer in §7.
8We can apply De Morgan’s laws to force any AC0 circuit
to have this property.


**Uniformity.** The circuit families we have defined above are non-uniform, meaning that we do
not enforce that the circuits processing different
input sizes must be related in any way. In degenerate cases, non-uniform circuit families can solve
undecidable problems[9] because they have infinite
description length, making them a physically unrealizable model of computation. Complexity theorists have thus introduced uniform circuit families.
Uniform circuit families are a realizable model of
computation with relations to classes in computational complexity and formal language theory.
Intuitively, in a uniform circuit family, the circuits for different input sizes must be “somewhat
similar” to each other. We formalize this (cf.

Arora and Barak, 2009) by saying that there exists
a resource-constrained Turing machine that maps
the input 1[n] to a serialization of circuit Cn.

**Definition 5. A language L is (S(n), I(n))-space**
uniformly computable by a circuit model M iff
there exists a Turing machine that, for all n 0,
_≥_
uses S(n) space to map 1[n] to an M -circuit recognizing L on inputs of size I(n).

This notion of uniformity is more general than
the standard notion in that the input size I(n) is a
function of the problem complexity n. The reason
for this is that we will apply uniformity to subcomputations with different input sizes I(n) within a
larger computation of input size n. The standard
notion of uniformity corresponds to I(n) = n.
Furthermore, we will refer to a circuit family as uniform if it is uniformly computable with
_S(n) = O(log n) (cf. Arora and Barak, 2009)._
We can define uniform versions of AC[0] and TC[0]

by adopting the previous definitions exactly, but
also enforcing uniformity. For the rest of the paper we will clarify whether we mean the uniform
or non-uniform variant of TC[0] when unclear from
context, since both classes will come up.

### 4 Bounded-Precision Transformers

A transformer (Vaswani et al., 2017) is a neural
network architecture made up of a constant number of transformer layers. A transformer layer is
a module that computes self-attention over a sequence followed by an elementwise transformation of the output vectors.

9Consider the unary language 1n such that Turing machine n (under some arbitrary enumeration) halts. This problem is in non-uniform AC[0] since we can hard-code the right
answer for each n in Cn.


-----

**4.1** **Precision and Space**

We will assume that each transformer is resource
bounded in terms of the precision of each value it
computes and, for some of our results, the space
it uses for the computation of key operations such
as embedding, attention, and activation. Specifically, we will assume precision p, i.e., the values
at all layers, as well as the outputs of all key intermediate operations in it (attention, activation,
arithmetic operators, etc.), are represented using
_p bits. This is a realistic assumption as, in prac-_
tice, today’s transformers are typically limited to
the 64-bit precision of the underlying hardware.
Formally, we define p-precision as follows:

**Definition 6. A k-ary function f : x1, . . ., xk �→**
_y is p-precision if x1, . . ., xk, y ∈{0, 1}[∗]_ have
size at most p bits, and f can be computed by a
_p-space-bounded Turing machine._

This says the size of the function input and output are bounded below p. Similarly, the intermediate space used by the computation must also be
bounded below p. Thus, higher precision computations cannot somehow be hidden inside f .

Def. 6 naturally applies to functions with
bounded arity k. We will also need to define p
precision for the summation operator in the transformer, which adds n different floats of size p.[10]

Adding n floats can blow up the precision needed
to represent their sum. For example, imagine
adding the floating points 1 2[0] + 1 2[c]. We obtain
_·_ _·_
(2[c] +1) 2[0], whose mantissa takes c+1 bits to rep
_·_
resent. In practice, computers do not preserve full
precision in such situations: instead, small terms
like 1 2[0] are discarded. Thus, we define the trans
_·_
former’s addition operation to be similarly ap_⊕_
proximate (and thus preserve precision); see §A.

**4.2** **Transformer Definition**

**4.3** **Attention Heads**

The core building block of a transformer is an attention head. We define this at a high level of abstraction as follows:

**Definition 7. A p-precision attention head is spec-**
ified by a binary p-precision similarity function
_s :_ 0, 1 0, 1 0, 1 .
_{_ _}[p]_ _× {_ _}[p]_ _→{_ _}[p]_

Let h1, . . ., hn ∈{0, 1}[p] be the input sequence
to a p-precision attention head, and let be ap_⊕_
proximate floating-point addition (§A).

10Our proof also goes through if the transformer weights
are integers, as is sometimes done (Dettmers et al., 2022).


**Definition 8. For all ℓ** 0, a p-precision attention
_≥_
head Hh[ℓ][+1] computes a vector a[ℓ]ih[+1] _∈{0, 1}[p]_ via


where Zi = [�]j[n]=1 _[s][(][h]i[ℓ][,][ h][ℓ]j[)][.]_

Standard transformer attention heads (Vaswani
et al., 2017) are a special case of this definition
where s is scaled dot-product similarity between
keys and queries. Standard transformers also have
a linear or affine value function applied to each h[ℓ]j
in the sum over j. By its affineness, the value function can, without loss of generality, be removed
from the attention head and considered to be part
of the transformer layer (i.e., applied to the output
of the attention head).

**4.4** **Transformer Layers**

A p-precision transformer layer is then a tuple of
heads and a function f used to combine them.

**Definition 9 (p-precision transformer layer). A**
_p-precision transformer layer is a tuple L[ℓ][+1]_ =
_⟨H1, · · ·, Hk, f_ _⟩, where each Hh is an attention_
head and f : ( 0, 1 )[k] 0, 1 0, 1 is a
_{_ _}[p]_ _× {_ _}[p]_ _→{_ _}[p]_
_p-precision activation function._

A _p-precision_ transformer layer can be
understood to define a sequence of vectors
**h[ℓ]1[+1], . . ., h[ℓ]n[+1]** in terms of an input sequence of
vectors h[ℓ]1[, . . .,][ h]n[ℓ] [(coming from the previous]
layer in the transformer) by first computing k
attention heads in parallel and then combining
their output using f . The first k inputs to f will
correspond to the attention head outputs, and the
additional input is the original input from the
previous layer. Recall that a[ℓ]ih[+1] is the output of
head Hih[ℓ][+1] on input h[ℓ] at position i. The function
computed by a transformer layer can be described
formally as follows.

**Definition 10 (Transformer layer computation).**
For ℓ 0, a p-precision transformer layer
_≥_
_L[ℓ][+1]_ recurrently computes the output sequence
**h[ℓ]1[+1], . . ., h[ℓ]n[+1]** as a function of the inputs
**h[ℓ]1[, . . .,][ h]n[ℓ]** [, where, for][ 1][ ≤] _[i][ ≤]_ _[n][, the][ i][-th com-]_
ponent is computed according to

**h[ℓ]i[+1]** = f (a[ℓ]i1[+1][, . . .,][ a]ik[ℓ][+1][,][ h]i[ℓ][)][.]

_f can be understood to encapsulate layernorm,_
residual connections, and the feedforward sublayer of a standard transformer (Vaswani et al.,


**a[ℓ]ih[+1]** =


_n_
�

_j=1_


_s(h[ℓ]i[,][ h][ℓ]j[)]_

_· h[ℓ]j[,]_
_Zi_


-----

2017). h[ℓ]i [is given to][ f][ to allow residual connec-]
tions. As mentioned in §4.3, f can also encapsulate the value function for each head.

**4.5** **Transformer Encoder**

Finally, we define a transformer of depth d as a
cascade of d transformer layers:

**Definition 11 (p-precision transformer). A p-**
precision transformer over alphabet Σ is a pair
consisting of a p-precision position embedding
function[11] _φ : Σ × N →{0, 1}[p]_ and a d-tuple
of p-precision transformer layers _L[1], . . ., L[d]_ .
_⟨_ _⟩_

For a position embedding function φ and w
_∈_
Σ[n], let φ(w) be the position-wise broadcasted embedding of w: for 1 ≤ _i ≤_ _n, φi(w) ≜_ _φ(wi, i)._

**Definition 12 (Transformer computation). A**
transformer �φ, _L[1],_ _L[d]_ � computes the fol_⟨_ _· · ·_ _⟩_
lowing function of a string w ∈ Σ[∗]:

_T_ (w) = (L[d] _L[d][−][1]_ _L[1])(φ(w))._
_◦_ _◦· · · ◦_

We will use n to denote the length of w, and
take the transformer’s depth d to be fixed w.r.t. n.
The input to the transformer can thus be represented with N = n log Σ bits using a binary
_|_ _|_
encoding for the vocabulary. The circuits we construct in subsequent sections to simulate transformers will also have input size N . We will
assume transformers have log-precision relative
to the size of the input, specifically O(log N )precision. Since Σ is fixed (typically 30000
_|_ _|_
in practice), we will think in terms of O(log n)precision. Thus, by Def. 6, all of the intermediate
functions of such transformers are computable in
O(log n) space and output (at most) these many
bits. Note that this is enough precision to represent positional encodings and for each position to
point to a constant number of other values, but not
enough precision for non-lossy pooling of the entire input into a single value.

**Relationship to Practical Transformers.** Our
log-precision transformers do not enforce that s
(Def. 7) and f (Def. 9) follow the transformer
structure. However, a feedforward net whose
primitive operations (e.g., scalar multiplication)
are defined over O(log n)-size numbers can be

11To apply the normal notion of p-precision to inputs outside {0, 1}[∗], we imagine elements of Σ are encoded as integers ≤|Σ| in binary, and natural numbers are represented as
integers ≤ _n. Thus, we assume log|Σ| + log n ≤_ _p._


computed in O(log n) space. Thus, boundedprecision practical transformers are a special case
of our log-precision transformers. This makes
our setup appropriate for proving upper bounds on
transformers, which is our main contribution.

### 5 Log-Precision Transformers as Non-Uniform Threshold Circuits

We first show that log-precision transformers can
be simulated by non-uniform threshold circuits,
before presenting the more technical uniform version of the results in §6. The initial non-uniform
result extends the findings of Merrill et al. (2022),
who showed that saturated attention transformers[12] can be simulated in TC[0]. Here, we remove
the simplifying saturated attention assumption and
other restrictions on the underlying datatype. Instead, we show that our log-precision assumption
is enough to prove that a transformer can be simulated in TC[0] with any attention function.

Hao et al. observed that any boolean function of
O(log n) bits can be computed by a poly(n) size
circuit. We extend this to m-bit outputs, which
is both more convenient and more efficient than
constructing m separate boolean circuits:

**Lemma 1 (Extended from Hao et al., 2022). Let**
_f :_ 0, 1 0, 1 _be a function. For all c_
_{_ _}[∗]_ _→{_ _}[m]_ _∈_
R[+] _and n ∈_ N, there exists an AND/OR circuit
_of size at most n[c]_ + c log n + m and depth 3 that
_computes f on inputs of size c log n._

_Proof. Like Hao et al. (2022), we construct a cir-_
cuit using a DNF representation of f on inputs of
size c log n, except we use a combined DNF representation for all output bits of f . The DNF formula has at most 2[c][ log][ n] = n[c] terms. The circuit
has a NOT gate for each input bit, an AND gate for
each DNF term, and, for each of the m output bits,
an OR gate combining the outputs of those AND
gates (i.e., DNF terms) for which that bit is 1.

We now use Lem. 1 to prove the following
non-uniform result. We note that the proof goes
through even if the notion of p-precision (Def. 6)
is relaxed to not require computability in space p.
This requirement will, however, become important
for our subsequent result in §6.

**Theorem 1 (Non-uniform). Any c log n-precision**
_depth-d transformer operating on inputs in Σ[n]_ _can_

12Saturated attention is uniform attention over a subset of
the prior layer nodes.


-----

_be simulated by a threshold circuit family of depth_
3 + (9 + 2d )d.
_⊕_

_Proof. Let w_ Σ[n] be the input of a c log n_∈_
precision transformer. We show by induction that
we can construct a composition of constant-depth,
poly-size threshold circuits to compute each layer
of this transformer. Thus, any constant-depth
transformer will be computable by a constantdepth threshold circuit.
In the base case of layer 0 and token i, we construct gates representing the constant i encoded in
binary. We can then compute h[0]i [=][ φ][(][w][i][, i][)][ using]

Lem. 1, yielding a poly-size depth-3 circuit.
In the inductive case of computing layer h[ℓ]i[+1]
for 1 _ℓ_ + 1 _d, we note that each vector output_
_≤_ _≤_
of layer h[ℓ]i [has size (at most)][ c][ log][ n][ bits because]
of the log-precision assumption.
We first fix a head a[ℓ]ik[+1] (Def. 8) to simulate.
Applying Lem. 1, we can compute s(h[ℓ]i[,][ h][ℓ]j[)][ with]
a poly-size depth-3 circuit, in parallel for all j.
Since n floats with c log n precision can be approximately added in TC[0] (§A), we can construct
a TC[0] circuit of depth d⊕ to compute Zj. Since
_s(h[ℓ]i[,][ h][ℓ]j[)][, Z][i][, and][ h][ℓ]i_ [all have][ c][ log][ n][ bits, we can]

_s(h[ℓ]i_ _[,][h][ℓ]j_ [)]
compute _Zi_ **h[ℓ]j** [with a poly-size depth-3 cir-]

cuit;[13] we do this in parallel for all j. Next, we
again use the fact that approximate addition of n
floats is in TC[0] to compute a[ℓ]ih[+1] as the approximate sum over j with a depth-d circuit.
_⊕_
We now simulate a layer h[ℓ]i[+1] (Def. 10) in terms
of its constituent heads. Since all arguments of
_g have size c log n, we apply Lem. 1 to compute_
_g with a poly-size depth-3 circuit, yielding h[ℓ]i[+1]._
We repeat this in parallel for all i. This completes
the inductive step new to compute all values in the
_ℓ_ + 1-st layer with a circuit depth of 9 + 2d .
_⊕_
Aggregating the circuit over all d layers, the
overall circuit depth is 3 + (9 + 2d )d.
_⊕_


**Corollary 1.1 (Non-uniform). Any log-precision**
_transformer can be simulated by a non-uniform_
TC[0] _circuit family.[14]_

13This may seem counterintuitive since multiplication of
two n-precision numbers is outside AC[0]. However, here we
leverage the fact that the precision is c log n.
14Here, a TC0 circuit family is a constant-depth, poly-size
circuit family computing some function {0, 1}[∗] _→{0, 1}[∗]._
While we define TC[0] for decision problems in Def. 4, it is
standard and well-defined to extend the same term to refer to
circuit families computing functions as well (Hesse, 2001).


### 6 Log-Precision Transformers as Uniform Threshold Circuits

We will now extend the argument from the last
section to show that O(log n)-precision transformers can be simulated by uniform constant-depth
threshold circuits by capitalizing on the assumption that φ, s, and f are log-precision, and thus can
be computed in O(log n) space. The overall proof
idea is similar, but due to the uniformity condition, the proof becomes substantially more technical. We must not just show the existence of a
threshold circuit family computing a transformer,
but also show that this circuit family can be generated by a log-space Turing machine.
We first extend Lem. 1 to respect uniformity:

**Lemma 2. Let f :** 0, 1 0, 1 _be a linear-_
_{_ _}[∗]_ _→{_ _}[m]_
_space computable function. There exists a Turing_
_machine that, for all n ∈_ N and c ∈ R[+], uses at
_most c log n + log m space to map input 1[n]_ _to a_
_circuit of size at most n[c]_ + c log n + m and depth
3 that computes f on inputs of size at most c log n.

_Proof. We give the proof in the form of an algo-_
rithm to construct a circuit as a function of n and
then justify its correctness and space complexity.

Algorithm. We first print 2c log n nodes representing unnegated and negated input nodes.[15]

Now, we need to show how to construct nodes
corresponding to n[c] DNF terms. To this end, we
loop over all possible inputs x 0, 1 by
_∈{_ _}[c][ log][ n]_
maintaining the c log n bit binary representation
of x (initialized with 0[c][ log][ n]) and incrementing it
by 1 at each step of the loop. We create a new
_∧_
node i with c log n arguments, defined as follows.
For j [c log n], we create an argument pointer
_∈_
to (unnegated) node j if xj = 1 and to (negated)
node c log n + j otherwise.
Now, we construct nodes computing each of the
_m output nodes. We loop over k_ [m], construct_∈_
ing a single node for each k. We loop over all
_x_ 0, 1 analogously above to construct a
_∈{_ _}[c][ log][ n]_
list of arguments. By our linear-space computability assumption and because x has c log n bits, we
can compute f (x) as a subroutine in O(log n)space to obtain fk(x). If fk(x) = 1, we print node
2c log n + j as an argument of node k.

Correctness. We show that this Turing machine
maps input n to a serialized circuit computing f

15We ignore the initial unnegated input nodes when considering the size of the circuit.


-----

on inputs of size n. The first layer simply produces unnegated and negated input values. The
second layer then produce all possible DNF terms.
Finally, node k of the third layer computes the disjunction over all terms x such that fk(x) = 1.
Thus, node k of the third layer computes fk.

Log Space. To complete the proof, we justify
that M uses O(log n+log m) space. Looping over
_x_ 0, 1 is accomplished by treating x as
_∈{_ _}[c][ log][ n]_
a binary number initialized to 0 and incrementing
it at each step. Thus, the loop pointer for building
the DNF terms takes c log n space to store. For
building the m output nodes, we maintain a similar loop pointer as well as an index k _m, taking_
_≤_
_c log n + log m space. Thus, the overall algorithm_
uses c log n + log m space.

Thus, M uses c log n + log m space to map 1[n]

to a circuit of size at most n[c] + c log n + m and
depth 3 that computes f on size c log n inputs.

We can leverage this lemma to derive the uni_form analog of Thm. 1, as follows._

**Theorem 2 (Uniform, main result). Any c log n-**
_precision depth-d transformer operating on inputs_
_in Σ[n]_ _can be simulated by a logspace-uniform_
_threshold circuit family of depth 3 + (9 + 2d_ )d.
_⊕_

_Proof. We will provide a proof by induction over_
transformer layers ℓ that there is a Turing machine
_M operating in O(log n) space that, on input 1[n],_
outputs a circuit that simulates the transformer’s
computation on inputs of size n. This circuit is
identical to the one in the proof of Thm. 1, and
thus has the same circuit depth.
In the base case, we use log space to track a
counter maintaining the current token i (between
1 and n) throughout the circuit construction. We
construct gates encoding the constant i in binary.
We can then apply Lem. 2 to construct a Turing
machine that maps 1[n] to a constant-depth threshold circuit computing h[0]i [=][ φ][(][w][i][, i][)][.]
In the inductive case, we assume we can output
in O(log n) space a circuit computing every value
**h[ℓ]i** [in the previous layer][ ℓ][. We will show that we]
can, in O(log n) space, now output a circuit computing every value in layer ℓ + 1.
As in Thm. 1, we first fix a head a[ℓ]ih[+1] to simulate. Recall (Def. 8) that


By Lem. 2, we can generate a depth-3 circuit of
size at most z = n[c][′] + c[′] log n + 1, where c[′] = 2c
(since the input to f is of size 2c log n) that computes s(h[ℓ]i[,][ h][ℓ]j[)][ for specific][ i, j][. We do this se-]
quentially for 1 _j_ _n and 1_ _h_ _k, padding_
_≤_ _≤_ _≤_ _≤_
each circuit with unused nodes so that each one
has size exactly z, and the z-th node corresponds
to the output. Thus, the indices of the output nodes
for each of the columns will be wℓ + _z(jk +_ _h) for_
1 ≤ _j ≤_ _n, where wℓ_ is the index of the last output
node h[ℓ]n [of the previous layer.]

At this point, we use the fact that for p =
_c log n, the p-precision approximate sum of n p-_
precision numbers can be computed by a uniform
threshold circuit (§A). We can thus use a Turing machine as a sub-routine to generate, on input 1[n], a k threshold circuits, where each has size
_z[′]_ that computes an gate over n items of pre_⊕_
cision p each. We set the inputs of circuit h to
be nodes wℓ + z(jk + h) for 1 ≤ _j ≤_ _n. By_
construction, this yields the normalizing constants
_Zi =_ [�]j[n]=1 _[s][(][h]i[ℓ][,][ h][ℓ]j[)][, whose value is located at]_
the node at index wℓ + znk + z[′] for head h.

Using p-precision arithmetic operator circuits,
we can now also generate a circuit to compute
_s(h[ℓ]i_ _[,][h][ℓ]j_ [)]

_Zi_ **h[ℓ]j** [for each][ 1][ ≤] _[j][ ≤]_ _[n][ and][ 1][ ≤]_ _[h][ ≤]_ _[k][,]_
by using index wℓ + z(jk + h) as before for the
value of s(h[ℓ]i[,][ h][ℓ]j[)][ and index][ w][ℓ] [+][ znk][ +][ z][′][h]
for the normalizing constant Zi of head h. Here
too we use circuits of identical size z[′′], making
_wℓ_ +k(zn+z[′] +z[′′]i) the index of the output nodes
of these n circuits. Next, we again employ a cir_⊕_
cuit of size z[′], similar to the computation of Zi, to
compute the sum of these n values. Finally, we
compute h[ℓ]i[+1] by applying f via Lem. 2.

Note that this requires keeping only ℓ, i, and n
in memory, each of which takes O(log n) bits.

We repeat this process for all 1 _i_ _n to_
_≤_ _≤_
compute the entire ℓ + 1 layer, which finishes the
inductive step: if we can output a circuit computing layer ℓ in O(log n) space, then we can do the
same for layer ℓ + 1.

Because the depth derived in Thm. 2 is constant
with respect to n, it follows that:

**Corollary 2.1 (Uniform, main result). Any log-**
_precision transformer can be simulated by a uni-_
_form TC[0]_ _circuit family._


**a[ℓ]ih[+1]** =


_n_
�

_j=1_


_s(h[ℓ]i[,][ h][ℓ]j[)]_

_· h[ℓ]j[.]_
_Zi_


-----

### 7 Lower Bounds for Instruction Following and Advice Transformers

So far, we have shown that uniform TC[0] is an upper bound for log-precision transformers. Is this
upper bound tight, i.e., also a lower bound? While
we do not answer this question here, we address
a related question as a first step: we construct a
transformer that can evaluate TC[0] circuits on binary inputs, showing that transformers can compute any TC[0] function when their input is augmented with the right “instructions”.
More formally, we consider the Circuit Value
**Problem (CVP) (Ladner, 1975), also referred to**
as the Circuit Evaluation Problem, where the input
is a boolean circuit C and a string x 0, 1, and
_∈{_ _}[n]_
the task is to return the value of C(x) 0, 1 .
_∈{_ _}_
This problem is known to be complete for the
class P under AC[0] reductions (Ladner, 1975). We
will assume C is serialized as described in §3 and
prove that log-precision transformers can evaluate
any TC[0] circuit. Note that this is an extension
of the typical CVP since the circuit has threshold
gates, not just standard AND/OR gates.
It is known that LSTMs cannot evaluate boolean
formulae (Merrill, 2020), a special case of the
CVP. In contrast, we show that transformers can.
To demonstrate the practicality of our lower
bound construction, we will not just prove the existence of transformers that can evaluate TC[0] circuits but also specify concrete choices for the positional embedding scheme and the class of attention
functions that are sufficient to do so.

**Fractional Positional Embeddings.** For a vector x and scalar y, let **x, y** be the vector append_⟨_ _⟩_
ing y onto x.[16] For σ Σ, let v(σ) be the one-hot
_∈_
embedding of σ into R[|][Σ][|]. For w ∈ Σ[∗] and i ≥ 1,
the fractional positional embedding at token i is

_φ(wi, i) = ⟨v(wi), i/n⟩._

**Saturated Attention.** We imagine f (h[ℓ]i[,][ h][ℓ]j[)][ is]
computed via saturated attention (cf. Merrill et al.,
2022), which provides a simple model of the types
of attention we can expect to be learned in transformers (Merrill et al., 2021). First, queries are
computed as qi = Qh[ℓ]i[, and then keys][ k][j][ =]
**Kh[ℓ]j** [Define the dot-product attention score][ σ][ij][ =]
**q[⊤]i** **[k][j][. We can then define saturated attention as]**

�
1 if σij = maxk σik

_s(h[ℓ]i[,][ h][ℓ]j[) =]_

0 otherwise.

16I.e., ⟨x, y⟩i = xi for 1 ≤ _i ≤|x|, and y if i = |x| + 1._


After normalization, saturated attention creates a
distribution that is uniform over a subset of positions. Thus, it is capable of parameterizing hard
attention, uniform attention over the full sequence,
and various attention patterns in between.

**Simple Pooling Functions.** For simplicity, we
assume pooling functions f are thresholded linear
functions of their inputs. Thus, they could be implemented by a feedforward neural net. Without
loss of generality, we let attention heads have a
value function, which can be folded into the pooling function from the last layer (see §4).

**Terminology.** We use input node to mean a token of type X and gate node to mean a token of
type Dir. We call a token of type & an argument.

We are now ready to present the main result.
Our construction below is specific to circuits serialized in prefix form (see §3), but it can be extended to other serializations as well.

**Lemma 3. For all d, there exists a transformer**
_with fractional positional embeddings, saturated_
_attention, thresholded linear pooling functions,_
_and depth 2d that, for any threshold circuit C_
_of depth d serialized in prefix form, maps input_
_C, x_ _to the value C(x)._
_⟨_ _⟩_

_Proof. We will construct a pair of two transformer_
layers that evaluate all the nodes at depth ℓ in the
threshold circuit, for any ℓ. It follows that a transformer of depth 2d can compute the value C(x).

Base Case: Input Nodes. We use an attention
layer to attend uniformly over all positions with
value returns 1 if wi = X and 0 otherwise. This
head computes #(X)/n, where #(X) is the number of occurrences of X in w. A second layer, then,
at input node i, computes the positional embedding of the token representing input value xi:

1 #(X) + i
_−_

_._
_n_

We attend to this position to retrieve xi. After
these layers, each input node i stores its value xi.
We also use the base-case layers to construct
an attention head that, at the i-th node, counts the
fraction of tokens (out of n) that are nodes to the
left of the current node. Thus, the column corresponding to node i stores the value i/n.
At each gate node i, we use two more attention
heads to find the index of the next & to the right
and then count the fraction of tokens before it that


-----

are 1. This head thus computes ki/mi where ki is
the threshold value of gate i and mi is its arity.
Finally, using the first attention layer, we have
each 1 node attend to the first argument symbol
& to its left and retrieve its index p/n. Then, in
the second attention layer, each argument attends
uniformly over all nodes with values p/n. The net
effect is for each argument to store j/n, i.e., the
pointer it is encoding in unary as &1[j].

Inductive Case: Gate Nodes. By our inductive
assumption over prior layers, all tokens corresponding to circuit nodes at depth _ℓ_ contain
_≤_
their appropriate value. We now construct 2 transformer layers to evaluate gate nodes at depth ℓ +1.
In the first attention layer, each argument token
attends to the closest gate node i to its left, which
is the gate it belongs to. Recall from the base case
that argument token & already stores j/n, where
_j is the pointer value it encodes. Each argument_
token now attends with query j/n to retrieve from
node j its already computed value.
The second attention layer applies at gate nodes,
not arguments. At gate i of arity mi, we set the attention s(i, j) to indicate whether argument j belongs to gate node i, which holds for exactly mi
arguments. We set the attention value at argument
_j to be the binary value of node j, which was re-_
trieved in the previous paragraph. Thus, the attention head computes ci/mi, where ci is the number
of arguments of node i that are 1. We repeat this
for all gate nodes.
At this point, we have both the count of true
inputs to gate node i (ci/mi) and, from the base
case, the threshold parameter of gate i (ki/mi).
Thresholding (ci _ki)/mi at 0 allows us to de-_
_−_
cide, based on whether Dir is <= or >=, whether
the current gate node should output a 0 or a 1. Repeating this for all gates at layer ℓ + 1 completes
the inductive step: we can evaluate all gate nodes
in this layer.

**Theorem 3. Depth-2d transformers can solve**
_CVP for depth-d TC[0]_ _circuits._

**7.1** **Instruction Following**

CVP is closely related to instruction learn_ing (Brown et al., 2020) and instruction following_
tasks (Finlayson et al., 2022). The latter task setup
provides a transformer two inputs: a regular expression r as an “instruction”, and z 0, 1 .
_∈{_ _}[∗]_
The goal of the task is to return whether z belongs


to the regular language represented by r. Viewed
from this lens, the circuit evaluation setup asks:
_Can transformers follow instructions provided in_
_the form of a circuit? As discussed below, our re-_
sult says the answer is yes for all constant depth
threshold circuits. This, to the best of our knowledge, provides the first non-trivial lower bound for
transformers in the instruction learning setting.
Formally, an instruction I is any description[17]

of a function fI of {0, 1}[∗]. We say a transformer correctly follows an instruction I if, for all
_x ∈{0, 1}[∗], it correctly computes fI_ (x) on input
_I, x_ . A non-uniform instruction description is
_⟨_ _⟩_
a family of length-specific descriptions {In}n[∞]=1[.]
We say a transformer correctly follows a nonuniform instruction family {In} if, for all n and
all x ∈{0, 1}[n], it correctly computes fI (x) on
input ⟨In, x⟩. The non-uniform description {In}
may take any form. When it forms a TC[0] circuit
family, we refer to it as a TC[0] instruction description. Since Thm. 3 constructs a transformer that
can evaluate any TC[0] circuit, it follows that:

**Corollary 3.1. There exists a depth-2d trans-**
_former that can correctly follow any depth-d TC[0]_

_instruction description._

Thus, transformers with simple position embeddings, attention, and pooling functions can simulate any instruction provided in the form of a TC[0]

circuit. We note that while it is unknown whether
the class of regular languages, considered by Finlayson et al. (2022), is contained in TC[0], the other
side is known: there are problems computable by
TC[0] circuits that are not computable by a regular language. These include problems involving
counting and arithmetic, which are beyond regular languages. Our results thus expand the known
kinds of instructions transformers are able to follow, at least with hand-constructed weights.

**7.2** **Advice Transformers**

We can also view circuit evaluation abilities of
transformers (Lem. 3) from the lens of advice tak_ing Turing machines which, in addition to their_
usual input, are also provided an input length dependent (but input independent) advice string. For
instance, P/poly is the class of problems decidable in polynomial time when the Turing machine
is given an advice string of size polynomial in the
input length (cf. Arora and Barak, 2009).

17Formally, a function description is a fixed-size program
to compute that function under some model of computation.


-----

In the same vein, let T/poly be the class of logprecision, constant-depth transformers with polynomial advice strings. In other words, on an input of size n, we allow the transformer to receive
an additional poly(n) bits of input that cannot depend on the standard input. Now let {Cn}n[∞]=1 [be]
a circuit family demonstrating that a problem is in
non-uniform TC[0]. Then, by passing the description of Cn as advice for input length n, it immediately follows from Lem. 3 that advice transformers
can simulate non-uniform TC[0]:

**Corollary 3.2. Non-uniform TC[0]** T/poly .
_⊆_

Since non-uniform TC[0] even contains some
undecidable languages (Arora and Barak, 2009,
Claim 6.8), T/poly is clearly a very powerful class
and a strict superset of T, the class of decision
problems recognized by transformers (which are
all decidable). Thus, a problem in T/poly cannot always be solved by a transformer on its own.
However, if given a description of how to do so
(“advice”) in the form of a TC[0] circuit, our result
shows that a transformer could solve that problem.

### 8 Conclusion

Answering two open questions from Merrill et al.
(2022), we prove log-precision transformers with
any (including soft) attention can be simulated by
_uniform constant-depth threshold circuits._ This
establishes thresholded addition as a fundamental operation for understanding the computational
model of transformers: any log-precision transformer can be re-expressed as a polynomial number of threshold gates stacked to a constant depth.
This result also establishes potential limits on the
computational power of log-precision transformers; e.g., if L P, transformers cannot com_⊂_
pute all poly-time functions. They are certainly
very far from being universal. The intuition at
the heart of this result is that forcing a model
to be highly parallelizable likely sacrifices its expressiveness. Since parallelism seems essential to
pretraining any massive model at scale, any large
language model—transformer or otherwise—may
suffer from a similar tradeoff.

### Acknowledgments

The authors are grateful for the valuable feedback
from the anonymous reviewers and the TACL action editor Dan Gildea. They also thank Paul
Beame and colleagues at AI2 including Kyle
Richardson, Michal Guerquin, Peter Clark, Tushar


Khot, and especially Matthew Finlayson, whose
empirical findings about instruction learning inspired §7. Feedback from Sam Bowman, Arya
McCarthy, Roma Patel, and Lena Strobl, and discussions with the FLaNN, ML for Code (MILA),
and Foundations of Language Processing (Umeå)
research groups helped improve earlier drafts. The
authors also appreciate Rahul Santhanam’s feedback. This work was funded in part by NSF award
1922658. William Merrill was supported by an
NSF graduate research fellowship and by AI2.

### References

Eric Allender. 1999. The permanent requires large
uniform threshold circuits. Chicago Journal of
_Theoretical Computer Science._

[Sanjeev Arora and Boaz Barak. 2009. Computa-](https://books.google.com/books/about/Computational_Complexity.html?id=8Wjqvsoo48MC)
_[tional Complexity: A Modern Approach. Cam-](https://books.google.com/books/about/Computational_Complexity.html?id=8Wjqvsoo48MC)_
bridge University Press.

Arin Biere, Marijn Heule, Hans van Maaren, and
Toby Walsh. 2009. Handbook of Satisfiability:
_Volume 185 Frontiers in Artificial Intelligence_
_and Applications. IOS Press._

Tom Brown, Benjamin Mann, Nick Ryder,
Melanie Subbiah, Jared D Kaplan, Prafulla
Dhariwal, Arvind Neelakantan, Pranav Shyam,
Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger,
Tom Henighan, Rewon Child, Aditya Ramesh,
Daniel Ziegler, Jeffrey Wu, Clemens Winter,
Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess,
Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario
[Amodei. 2020. Language models are few-shot](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
[learners.](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) In Advances in Neural Information
_Processing Systems, volume 33, pages 1877–_
1901. Curran Associates, Inc.

Tom Bylander. 1991. Complexity results for planning. In Proceedings of the International Joint
_Conference on Artificial Intelligence._

Andrew Chiu, George I. Davida, and Bruce E.
Litow. 2001. Division in logspace-uniform nc1.
_RAIRO Theor. Informatics Appl., 35:259–275._

Mostafa Dehghani, Stephan Gouws, Oriol
Vinyals, Jakob Uszkoreit, and Lukasz Kaiser.


-----

2019. Universal transformers. In International
_Conference on Learning Representations._

Tim Dettmers, Mike Lewis, and Luke Zettle[moyer. 2022. GPT3.int8(): 8-bit matrix multi-](https://openreview.net/forum?id=dXiGWqBoxaD)
[plication for transformers at scale. In Advances](https://openreview.net/forum?id=dXiGWqBoxaD)
_in Neural Information Processing Systems._

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
[Kristina Toutanova. 2019. BERT: Pre-training](https://doi.org/10.18653/v1/N19-1423)
[of deep bidirectional transformers for language](https://doi.org/10.18653/v1/N19-1423)
[understanding.](https://doi.org/10.18653/v1/N19-1423) In Proceedings of the 2019
_Conference of the North American Chapter of_
_the Association for Computational Linguistics:_
_Human Language Technologies._

Nelson Elhage, Neel Nanda, Catherine Olsson,
Tom Henighan, Nicholas Joseph, Ben Mann,
Amanda Askell, Yuntao Bai, Anna Chen, Tom
Conerly, Nova DasSarma, Dawn Drain, Deep
Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane
Lovitt, Kamal Ndousse, Dario Amodei, Tom
Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. 2021. [A mathemati-](https://transformer-circuits.pub/2021/framework/index.html)
[cal framework for transformer circuits. Trans-](https://transformer-circuits.pub/2021/framework/index.html)
_former Circuits Thread._

Matthew Finlayson, Kyle Richardson, Ashish
[Sabharwal, and Peter Clark. 2022. What makes](https://aclanthology.org/2022.emnlp-main.27/)
[instruction learning hard? An investigation and](https://aclanthology.org/2022.emnlp-main.27/)
[a new challenge in a synthetic environment. In](https://aclanthology.org/2022.emnlp-main.27/)
_Proceedings of the 2022 Conference on Empir-_
_ical Methods in Natural Language Processing._

Raymond Greenlaw, James M. Hoover, and Wal[ter L. Ruzzo. 1991. A compendium of prob-](https://era.library.ualberta.ca/items/403292c5-460b-49e6-8b05-9a5a7b45b0d6)
[lems complete for P. Technical Report TR91-](https://era.library.ualberta.ca/items/403292c5-460b-49e6-8b05-9a5a7b45b0d6)
11, University of Alberta.

Michael Hahn. 2020. [Theoretical limitations](https://www.aclweb.org/anthology/2020.tacl-1.11)
[of self-attention in neural sequence models.](https://www.aclweb.org/anthology/2020.tacl-1.11)
_Transactions of the Association for Computa-_
_tional Linguistics, 8:156–171._

Yiding Hao, Dana Angluin, and Robert Frank.
[2022. Formal language recognition by hard at-](https://doi.org/10.1162/tacl_a_00490)
[tention transformers: Perspectives from circuit](https://doi.org/10.1162/tacl_a_00490)
[complexity. Transactions of the Association for](https://doi.org/10.1162/tacl_a_00490)
_Computational Linguistics, 10:800–810._

William Hesse. 2001. Division is in uniform TC[0].
In International Colloquium on Automata, Lan_guages, and Programming, pages 104–114._


Neil Immerman. 2012. _Descriptive complexity._
Springer Science & Business Media.

[Neil D. Jones and William T. Laaser. 1976. Com-](https://doi.org/https://doi.org/10.1016/0304-3975(76)90068-2)
[plete problems for deterministic polynomial](https://doi.org/https://doi.org/10.1016/0304-3975(76)90068-2)
[time. Theoretical Computer Science, 3(1):105–](https://doi.org/https://doi.org/10.1016/0304-3975(76)90068-2)
117.

Richard E Ladner. 1975. The circuit value problem is log space complete for P. ACM SIGACT
_News, 7(1):18–20._

Harry R. Lewis and Christos H. Papadimitriou.
1982. Symmetric space-bounded computation.
_Theoretical Computer Science, 19:161–187._

William Merrill, Vivek Ramanujan, Yoav Goldberg, Roy Schwartz, and Noah A. Smith. 2021.
[Effects of parameter norm growth during trans-](https://doi.org/10.18653/v1/2021.emnlp-main.133)
[former training: Inductive bias from gradient](https://doi.org/10.18653/v1/2021.emnlp-main.133)
[descent.](https://doi.org/10.18653/v1/2021.emnlp-main.133) In Proceedings of the 2021 Confer_ence on Empirical Methods in Natural Lan-_
_guage Processing._

[William Cooper Merrill. 2020. On the linguistic](https://arxiv.org/abs/2004.06866)
[capacity of real-time counter automata. ArXiv,](https://arxiv.org/abs/2004.06866)
abs/2004.06866.

William Cooper Merrill, Ashish Sabharwal, and
[Noah A. Smith. 2022. Saturated transformers](https://aclanthology.org/2022.tacl-1.49/)
[are constant-depth threshold circuits. Transac-](https://aclanthology.org/2022.tacl-1.49/)
_tions of the Association for Computational Lin-_
_guistics, 10._

Jorge Pérez, Javier Marinkovi´c, and Pablo
[Barceló. 2019. On the Turing completeness of](https://openreview.net/forum?id=HyGBdo0qFm)
[modern neural network architectures. In Inter-](https://openreview.net/forum?id=HyGBdo0qFm)
_national Conference on Learning Representa-_
_tions._

Colin Raffel, Noam M. Shazeer, Adam Roberts,
Katherine Lee, Sharan Narang, Michael
Matena, Yanqi Zhou, Wei Li, and Peter J. Liu.
[2020. Exploring the limits of transfer learning](http://jmlr.org/papers/v21/20-074.html)
[with a unified text-to-text transformer. Journal](http://jmlr.org/papers/v21/20-074.html)
_of Machine Learning Research, 21(140)._

Omer Reingold. 2008. Undirected connectivity in
log-space. Journal of the ACM, 55:17:1–17:24.

Leslie G. Valiant. 1979. The complexity of computing the permanent. _Theoretical Computer_
_Science, 8:189–201._


-----

Ashish Vaswani, Noam Shazeer, Niki Parmar,
Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
[Łukasz Kaiser, and Illia Polosukhin. 2017. At-](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
[tention is all you need.](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) In Advances in Neu_ral Information Processing Systems, volume 30._
Curran Associates, Inc.

### A Iterated p-Precision Float Addition

We interpret a p-bit string x as a p-precision float
by taking the first p/2 bits[18] of x as a signed integer m encoding the mantissa and the remaining
_p/2 bits of x as another signed integer e encoding_
the exponent. A float with mantissa m and exponent e, denoted _m, e_, encodes m 2[e].
_⟨_ _⟩_ _·_
Computing the sum of n n-bit integers (known
as iterated addition or simply summation) is wellknown to be in uniform TC[0] (Hesse, 2001; Chiu
et al., 2001). We leverage this fact to show that the
same holds for the sum of n O(log n)-precision
floats. A subtlety of adding p-precision floats is
that their sum can require more than p bits to represent precisely as a float. For instance, while each
of 2[r] and 1 is representable with a (signed) mantissa of only 2 bits, their exact sum, 2[r] + 1, requires a mantissa of r + 1 bits. Hence, p-precision
transformers must sacrifice some precision when
performing summation.
We define float addition by mapping the floats
to integers, adding the integers exactly, and then
mapping the sum back to a float (with possible loss
of precision). Let Iq[max] = 2[q] _−_ 1 be the greatest q-bit signed integer, and Iq[min] = −Iq[max]. Let
_Fp[max]_ be the greatest value representable by a pprecision float. Since the exponent of a float φ can
be negative and represent a fraction, we rescale φ
by 2[−][I]p/[min]2 when mapping it to an integer gp(φ):

**Definition 13. The integer mapping of a p-bit float**
_φ = ⟨m, e⟩_ is defined as gp(φ) = m · 2[e][−][I]p/[min]2 .

**Definition 14. The p-truncated float mapping of**
an integer z is defined as fp(z) = ⟨m, e⟩ where[19]

_m = rshift(z, max_ 0, sizeof(z) _p/2_ )
_{_ _−_ _}_

_e = sizeof(z) −_ sizeof(m) + Ip/[min]2

when e ≤ _Ip/[max]2_ [; otherwise (i.e., when][ z > F]p[ max]),
we set m = e = I [max]
_p/2_ [to properly handle overflow.]

18We assume w.l.o.g. that p is even.
19For x ̸= 0, sizeof(x) = ⌊log|x|⌋ + 2; sizeof(0) = 2.
For y ≥ 0, rshift(x, y) right-shifts x by y bits


**Definition 15 (Iterated p-precision float addition).**
We define the sum of k p-precision floats as


�


_k_
�

_φi = fp_
_i=1_


� _k_
�

_gp(φi)_
_i=1_


_._


We first verify that Def. 14 closely approximates exact addition.

**Lemma 4. Let φ =** _e, m_ _be a float such that_
_⟨_ _⟩_
_|φ| ≤_ _Fp[max]_ _and e ≥_ _Ip/[min]2_ _[. Then][ φ][ and][ f][p][(][g][p][(][φ][))]_

_differ by a factor of at most 1_ 2[−][p/][2+2].
_±_

_Proof. Let z = gp(φ), which is well-defined be-_
cause of the precondition e _I_ [min]
_≥_ _p/2_ [of the lemma.]
Let φ[′] = ⟨m[′], e[′]⟩ = fp(z).
First consider the easy case where sizeof(z)
_≤_
_p/2. Then m[′]_ = z and e[′] = I [min]
_p/2_ [from][ Def. 14][.]

Since z = m 2[e][−][I]p/[min]2 by Def. 13, it follows that
_·_
_φ and φ[′]_ have exactly the same value.
Now assume sizeof(z) > p/2. It follows from
the precondition |φ| ≤ _Fp[max]_ of the lemma that
there is no overflow when applying Def. 14 to
compute _m[′], e[′]_ . Thus m[′] consists of the p/2
_⟨_ _⟩_
highest-order bits (including the sign bit) of z and
_e[′]_ = ℓ + I [min]
_p/2_ [, where][ ℓ] [= sizeof(][z][)][ −] _[p/][2][ is the]_
number of bits truncated from z to obtain m[′]. Let
_δ denote the (non-negative) integer formed by the_
_ℓ_ lowest-order bits of z that are truncated. Then
_δ_ 2[ℓ] 1 = 2[sizeof(][z][)][−][p/][2] 1 < z 2[−][p/][2+2].
_≤_ _−_ _−_ _·_
Recall that the value of φ is gp(φ) · 2[−][I]p/[min]2 = z ·

_p/2_
2[−][I] [min] . By the above argument, we also have that
the value of φ[′] is within (z _δ)_ 2[−][I]p/[min]2, which is
_±_ _·_
within z _·(1±2[−][p/][2+2])·2[−][I]p/[min]2 . Thus, φ and φ[′]_ are
within a factor of 1 2[−][p/][2+2] of each other.
_±_

Finally, we show that, with log precision, computing (Def. 14) is in uniform TC[0].
_⊕_

**Lemma 5. Let p ≤** _c log n and φ =_ [�]i[k]=1 _[φ][i][,]_
_where k ≤_ _n and each φi is p-precision. Then φ is_
_computable by a constant-depth uniform threshold_
_circuit of size poly(n)._

_Proof. Let N_ = c log n + 2n[c]. We first use

Lem. 1 to map each φi = ⟨mi, ei⟩ to the integer

_p/2_
_zi = mi_ 2[e][i][−][I] [min], which has size sizeof(mi) +
_·_
(ei − _I_ [min]) ≤ _p/2+2_ _·_ 2[p/][2] _≤_ _c log n_ +2n[c] = N .
For 1 ≤ _i ≤_ _k, we pad zi to N bits, and for_
_k < i ≤_ _N_, we create an N -bit integer zi = 0. We
can then compute z = [�]i[k]=1 _[z][i][ with a constant-]_
depth uniform threshold circuit of size poly(N )
using the classical construction to sum N N -bit


-----

integers (cf. Immerman, 2012, exercise 5.29). The
size of this circuit is also polynomial in n by the
definition of N . Finally, we compute f _[†](z) using_
a constant-depth AND/OR circuit.


-----

