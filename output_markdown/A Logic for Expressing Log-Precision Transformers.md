## A Logic for Expressing Log-Precision Transformers


**William Merrill**
New York University
```
 willm@nyu.edu

```

**Ashish Sabharwal**
Allen Institute for AI
```
ashishs@allenai.org

```

#### Abstract

One way to interpret the reasoning power of transformer-based language models
is to describe the types of logical rules they can resolve over some input text.
Recently, Chiang et al. (2023) showed that finite-precision transformer classifiers
can be equivalently expressed in a generalization of first-order logic. However,
finite-precision transformers are a weak transformer variant because, as we show,
a single head can only attend to a constant number of tokens and, in particular,
cannot represent uniform attention. Since attending broadly is a core capability for
transformers, we ask whether a minimally more expressive model that can attend
universally can also be characterized in logic. To this end, we analyze transformers
whose forward pass is computed in log n precision on contexts of length n. We
prove any log-precision transformer classifier can be equivalently expressed as
a first-order logic sentence that, in addition to standard universal and existential
quantifiers, may also contain majority-vote quantifiers. This is the tightest known
upper bound and first logical characterization of log-precision transformers.

```
    aaaabbbb ✓ aaabbbbb ✗ baaaabbb ✗

```
Figure 1: A first-order logic with majority (FO(M)) sentence for a[m]b[m]. In addition to standard ∀
and quantifiers over string indices, FO(M) allows majority quantifiers (M) that take a majority-vote
_∃_
across indices. a(i) indicates whether token i is a (and analogously for b). We prove FO(M) can
express any function computed by a log-precision transformer.


#### 1 Introduction

The incredible success of deep learning models, especially very large language and vision transformers
with hundreds of billions of parameters (Brown et al., 2020; Thoppilan et al., 2022), has come at the
cost of increasingly limited understanding of how these models actually work and when they might
fail. This raises many concerns, such as around their safe deployment, fairness, and accountability.
Does the inner working of a transformer defy description in a simpler symbolic system that we
can better understand? Or can transformer computation be described using a familiar symbolic
_formalism? Understanding how to view the reasoning process of a transformer in terms of logic could_
potentially expand our ability to formally reason about their behavior over large domains of inputs.

Chiang et al. (2023) provide a partial answer to this question, showing that any finite-precision
transformer classifier can be expressed as a sentence in a variant of first-order logic with counting


37th Conference on Neural Information Processing Systems (NeurIPS 2023).


-----

quantifiers and modular arithmetic over input position indices. Specifically, counting quantifiers take
the form _i : ϕ(i) where x is a count variable and i is a position index. They show that there exists_
_∃[=][x]_
a single sentence in this logic that computes the output of the transformer for any input string of any
length. This is a powerful result because it shows that a simple logical formalism is fully sufficient to
describe all the complexity of a massive finite-precision transformer. It also provides an upper bound
on finite-precision transformers: any function that cannot be defined in first-order counting logic with
modular indexing cannot be expressed by the transformer.

However, Chiang et al.’s result is not fully general because it relies on the transformer precision being
fixed with respect to the transformer’s context length. More generally, as we will demonstrate in
Section 3, finite-precision transformers are a fundamentally weak variant of transformers: crucially,
cannot express uniform attention patterns, which are a core algorithmic primitive of transformers
(Weiss et al., 2018). In fact, we show that they can only attend to a constant number of input
positions, which may be seen as a rather limited generalization of hard attention.[1] For example,
Chiang et al. show that their logic for finite-precision transformers cannot recognize a[m]b[m], whereas
in practice, transformers can (Bhattamishra et al., 2020).[2] This motivates studying a formal model
of transformers where precision grows with context length (which we formalize as log-precision),
making it possible to capture uniform attention as well as other broad attention patterns. This is
useful both for recognizing a[m]b[m] and more generally for reasoning globally over the input.

We demonstrate that log-precision transformer classifiers can also be expressed as sentences in a
simple logic: first-order logic with majority, or FO(M), over inputs strings (Barrington et al., 1990).
In addition to standard existential and universal quantifiers, FO(M) has majority quantifiers that
return true iff at least half the propositions they quantify are true. It also allows comparing input
positions (e.g., ℓ< k in Figure 1) and accessing their individual bits. Our main result is as follows:

**Theorem 1 (Informal version of Theorem 2). For any log-precision transformer T, there exists an**
FO(M) sentence ϕ that computes the same function as _, i.e., ϕ(x) =_ (x) for any input string x.
_T_ _T_

**Upper bound.** Theorem 2 shows transformers with more than finite precision can also be expressed
in a simple extension of first-order logic, going beyond Chiang et al. (2023)’s result. On the other
hand, FO(M) is a strict superset of Chiang et al.’s counting logic; it can simulate counting quantifiers
(see Section 2.2) and allows non-modular position comparisons. Thus, handling a more general class
of transformers powerful enough to express uniform attention slightly weakens the bound.

Still, our result constitutes (to our knowledge) the tightest upper bound on log-precision transformers
and the first defined in terms of logic, building on a line of complexity-theoretic work analyzing the
power of transformers (Hahn, 2020; Merrill et al., 2022; Liu et al., 2023; Merrill & Sabharwal, 2023).
In particular, FO(M) strengthens the upper bound of log-space-uniform TC[0] by Merrill & Sabharwal
(2023). The refined bound adds to the limitations of transformers identified by Merrill & Sabharwal
(2023): for example, it establishes unconditionally that log-precision transformers cannot compute
boolean matrix permanents, and shows that, in a certain formal sense, integer division and matching
parentheses are among the formally hardest problems that transformers can solve (see Section 4).[3]

**Mechanistic interpretability.** Beyond providing an upper bound on the reasoning problems solvable by transformers, we believe Theorem 1 could guide the design of “transformer-complete”
programming languages similar in spirit to RASP (Weiss et al., 2018). RASP is a declarative programming language designed to capture transformer computation, and Lindner et al. (2023) implement a
compiler from RASP into transformers. Unlike RASP, FO(M) can provably express any transformer
(Theorem 1), which we believe justifies using it (or an equivalent but more user-friendly variant) as a
target language for programs extracted from transformers.

Similar to a decision tree, an FO(M) sentence has the interpretable property that each sub-sentence
corresponds to a constraint on input (see Figure 1). In contrast, the internal modules of a transformer
or circuit do not satisfy this since they map between arbitrary latent spaces. We speculate this property

1Hard attention is provably substantially weaker than general attention (Hao et al., 2022; Merrill et al., 2022).
2Technically, the empirical results of Bhattamishra et al. (2020) are for ambmcm, a harder variant of ambm.
3To be clear, Theorem 1 is one-sided: every transformer can be expressed as an FO(M) sentence, but
not necessarily the other way. Moreover, we believe that many FO(M) sentences cannot be expressed by
transformers. An exact logical characterization of transformers remains an open problem.


-----

could facilitate interpreting models by translating them to FO(M), though a careful exploration of the
algorithmic and HCI aspects of this idea lies outside the current paper’s theoretical scope.

**Contributions.** Our results shed new light on how to view the computation inside transformers in
terms of logic. Specifically, our main contributions are to prove the following:

1. Fixed-precision transformers can only attend to a fixed number of tokens, and those with
precision less than log log n cannot uniformly attend over length-n contexts (Proposition 1).

2. Log-precision transformer classifiers can be expressed as sentences in FO(M) (Theorem 2).

#### 2 Preliminaries: Transformers and FO(M)

Let Σ be a finite alphabet. We denote by _[∗]_ the Kleene star operator, i.e., for a set X, X _[∗]_ = [�]n[∞]=0 _[X]_ _[n][.]_
We will view transformers and FO(M) sentences both as functions from Σ[∗] 0, 1, and show that
_→{_ _}_
any function a transformer computes can also be computed by an FO(M) sentence.

**2.1** **Transformers**

We view the transformer precision p as a function of the context length n, writing p(n) where
appropriate. Let Dp be the datatype of p-precision floats, i.e., tuples ⟨m, e⟩ where m, e are signed
integers together taking p bits. Using _x_ to mean the size of integer x, a float represents the value
_|_ _|_
_m_ 2[e][−|][m][|][+1].[4] Following Appendix A of Merrill & Sabharwal (2023), we define p-truncated addition

_·_
(+, [�]), multiplication (·), and division (/) over Dp. We now define a transformer encoder binary
_classifier over Dp, largely adopting Merrill & Sabharwal’s notation.[5]_

**Definition 1. A p-precision transformer T with h heads, d layers, model dimension m (divisible by**
_h), and feedforward width w is specified by:_

1. An embedding function ϕ : Σ × N → D[m]p [whose form is defined in Appendix C.1;][6]

2. For each 1 ≤ _ℓ_ _≤_ _d and 1 ≤_ _k ≤_ _h, a head similarity function s[ℓ]k_ [:][ D]p[m] _[×][ D]p[m]_ _[→]_ [D][p] [whose]
form is defined in Appendix C.2;

3. For each 1 ≤ _ℓ_ _≤_ _d and 1 ≤_ _k ≤_ _h, a head value function vk[ℓ]_ [:][ D]p[m] _[→]_ [D]p[m/h] whose form is
defined in Appendix C.2;

4. For each 1 ≤ _ℓ_ _≤_ _d, an activation function f_ _[ℓ]_ : (Dp[m/h])[h] _× D[m]p_ _[→]_ [D]p[m] [whose form is]
defined in Appendix C.3 and implicitly uses the feedforward dimension w;

5. An output classifier head κ : D[m]p _[→{][0][,][ 1][}][ whose form is defined in Appendix C.4.]_

**Definition 2. We define the transformer computation and output as a function of an input x ∈** Σ[n].

1. Embeddings: For 1 ≤ _i ≤_ _n, h[0]i_ [=][ ϕ][(][x][i][, i][)][.][6]

2. Self Attention: For 0 _ℓ_ _d_ 1, (multihead) self-attention block ℓ + 1 computes h
_≤_ _≤_ _−_
attention heads:


_n_
�

_s[ℓ]k[+1](h[ℓ]i_ _[,][ h][ℓ]j[)][.]_
_j=1_


**a[ℓ]i,k[+1]** [=]


_n_
�

_j=1_


_s[ℓ]k[+1](h[ℓ]i_ _[,][ h][ℓ]j[)]_ _· vk[ℓ][+1](h[ℓ]j[)][,]_ where Zi,k =

_Zi,k_


3. Activation Block: For 0 _ℓ_ _d_ 1, activation block ℓ + 1 aggregates the head outputs to
_≤_ _≤_ _−_
produce h[ℓ][+1]:

**h[ℓ]i[+1]** = f _[ℓ][+1](a[ℓ]i,[+1]1_ _[, . . .,][ a]i,h[ℓ][+1][,][ h]i[ℓ][)][.]_

4. Classifier Head: The network prediction on x ∈ Σ[n] is κ(h[d]n[)][.]

4⟨101, 010⟩ represents 1.012 × 2102 . This is closer to the IEEE standard than the m · 2e semantics used in
Merrill & Sabharwal (2023), letting us define the minimum representable float more realistically in Proposition 1.
5Increasing the classifier’s output space arity (e.g., a transformer that predicts the next token) or switching
to causal attention of a decoder-only model would not change our results. However, our proof no longer goes
through if the decoder can generate tokens that get added to the input at the next step (cf. Pérez et al., 2019).
6 ϕ, like p, is actually a function of the context length n, and Appendix C.1 enforces that ϕ is computable in
O(log n) time, as standard choices of positional embeddings would satisfy.


-----

We say T (x) = κ(h[d]|x|[)][ and][ L][T][ is the language of][ x][ ∈] [Σ][∗] [such that][ T][ (][x][) = 1][. We refer to]
_ϕ, s[ℓ]k[, v]h[ℓ]_ _[, f][ ℓ][, and][ κ][ as the][ core functions][ in][ T][, and to embeddings, self attention, activation, and the]_
classifier head as the components of T . We write θT for the concatenated vector of parameters for
the functions ϕ, s[ℓ]k[, v]h[ℓ] _[, f][ ℓ][, and][ κ][, for all][ 1][ ≤]_ _[ℓ]_ _[≤]_ _[d][ and][ 1][ ≤]_ _[k][ ≤]_ _[h][.]_

We define a log-precision transformer as one where p is at most O(log n) and is a “simple” function,
i.e., computable in O(log n) time. In our model, the weights θT defining T are fixed, but the precision
_p used to compute the forward pass can depend on n (see Footnote 13 for a generalization)._

**2.2** **First-Order Logic with Majority**

As we will show, transformers can be translated into sentences in FO(M). But what do such sentences
look like? Informally, FO(M) is first-order logic extended to also have majority (M) quantifiers.
Following Barrington et al. (1990), our sense of FO(M) takes strings in Σ[∗] as input and returns 0 or
1 to define a formal language. In this setting, quantifiers range over indices (positions) into the string.
Predicates can be applied to the variables introduced by these quantifiers.
**Definition 3 (FO(M) index). Indices in FO(M) are integers denoting positions in the input string:**

1. The constant 1, representing the first token’s position.
2. The constant n, representing the last token’s position.
3. Strings (e.g., i, j, k) representing variables ranging over positions 1 to n.
4. Any index built by applying addition or subtraction to other indices.[7]

**Definition 4 (FO(M) formula). Formulas in FO(M) are constructed as follows:[8]**

1. Let Σ be a finite alphabet. For each σ ∈ Σ and any index i, σ(i), e.g., a(i), is a formula that
is true if the i-th input token is σ.[9]

2. For any indices i, j, the formula bit(i, j) returns the j-th bit of the binary expansion of i.[10]
3. For two indices i, j, i = j, i _j, and i_ _j are formulas with their conventional semantics._
_≤_ _≥_
4. For two formulas ϕ, ψ,ϕ _ψ and ϕ_ _ψ are formulas with their conventional semantics._
_∧_ _∨_
5. For any formula ϕ (which may refer to i), the following are valid formulas:


(a) _i. ϕ means some value of i in [1, n] makes ϕ true._
_∃_

(b) _i. ϕ means all values of i in [1, n] make ϕ true._
_∀_

(c) Mi. ϕ means _n/2 values of i in [1, n] make ϕ true._
_≥_

We use parentheses where necessary to disambiguate the order of operations. General formulas may
contain free (i.e., unbound) variables: e.g., _i. i = j. A sentence is an FO(M) formula ϕ with no free_
_∀_
variables. Sentences represent functions from from Σ[∗] to 0, 1 and thus define a formal language.[11]
_{_ _}_

**Extensions.** Beyond Definition 4, FO(M) can express counting and threshold quantifiers in terms
of majority quantifiers (Barrington et al., 1990). Given a formula ϕ, a counting quantifier creates a
new formula _i : ϕ that is true iff ϕ is true across exactly k values of i. Threshold quantifiers_
_∃[k]_ _∃[≤][k]_
and work similarly but check if ϕ is true for at least or at most k values of i. In addition, we
_∃[≥][k]_
show in Appendix A that FO(M) can express conditional majority quantifiers, which create a formula
Mi : ϕ [ψ] that is true iff ψ is true for at least half the values of i that make ϕ true.

**2.2.1** **Examples**

To illustrate the formalism, we provide example languages definable in FO(M) with Σ = {a, b}.
First, we show two languages that do not require majority quantifiers to express:
**Example 1 (Bigram matching). Strings containing the bigram ab: ∃i [a(i) ∧** `b(i + 1)] .`
**Example 2 (Skip-bigram matching). Strings containing the long-distance pattern a . . . b (cf. “induc-**
tion heads” of Elhage et al. 2021): ∃i [b(i) ∧∃j [j ≤ _i ∧_ `a(j)]] .`

7Barrington et al. (1990) did not introduce this as a primitive, but it can be simulated using the ≤ predicate.
8We write parentheses to indicate the order of operations.
9Barrington et al. (1990) define Qb(i) for b ∈{0, 1}. We generalize this to an arbitrary vocabulary Σ by
assuming each token is one-hot-encoded: σ(i) = Q1(|Σ|i + s) where s is the index of σ in the vocabulary.
10This predicate is included in the logic for technical reasons; see Barrington et al. (1990).
11One can also take multiple sub-sentences within ϕ to be labeled as ordered outputs, thus allowing ϕ to be a
function from Σ[∗] to {0, 1}[k] for some fixed constant k.


-----

In contrast, Example 3 is a simple example that requires majority quantifiers (Furst et al., 1984):
**Example 3 (Majority). Strings with more b’s than a’s: Mi [b(i)] .**

Figure 1 showed how FO(M) can be used to recognize patterns like a[m]b[m]. A similar idea can be
used to model parentheses matching (Barrington et al., 1990):
**Example 4 (1-Dyck). The well-balanced parentheses language (with a opening and b closing):**

_∀i. (∃a, b. ((∃[a]j : a(j) ∧_ _j ≤_ _i) ∧_ (∃[b]j : b(j) ∧ _j ≤_ _i) ∧_ _b ≤_ _a)) ∧_ Mi. a(i) ∧ Mj. b(j).

**Example 5 (Integer Arithmetic). Iterated addition (i.e., summing n n-bit numbers), iterated multipli-**
cation, and division (Hesse, 2001) can all be expressed in FO(M).

#### 3 Finite Precision Transformers Cannot Attend Universally

Attention heads that spread attention weight uniformly across inputs have been observed in transformer LMs (Merrill et al., 2021) and make soft attention fundamentally more powerful than hard
attention (Hao et al., 2022; Merrill et al., 2022). In particular, uniform attention is an important
primitive that transformers can use to solve tasks involving counting (Bhattamishra et al., 2020;
Chiang et al., 2023), taking majority votes (Merrill et al., 2022), and matching parentheses or sorting
(Weiss et al., 2021). A transformer with sufficient precision can easily implement uniform attention
by setting the keys and queries across all positions to be constant. However, attention heads with finite
precision cannot represent uniform attention over long sequences as a consequence of the following:
**Proposition 1. Let a ∈** R[n] _s.t._ [�]i[n]=1 _[a][i][ = 1][ and][ ˜][a][ its nearest][ p][-precision float approximation.]_

_1. Then the number of nonzero entries of ˜a is upper bounded by its precision: specifically, ˜a_
_has at most 2[2][p]_ _nonzero entries._

_2. Moreover, if p < log log n and a is uniform (i.e., ai = 1/n), then ˜a =_ _[⃗]0._

_Proof. The smallest positive value representable by a p-precision float is 2[−][(][p][m][−][2+2][pe]_ _[−][1][)]_ which is
bounded below by 2[−][2][p][+1]. Letting k = 2[2][p], it holds that 2[−][2][p][+1] = 2/k. So if ˜ai gets the minimum
value, then ai ≥ 1/k. Since [�]i _[a][i][ = 1][, there can be at most][ k][ indices satisfying this property. This]_

implies there can be at most k nonzero entries in ˜a. If n > k and a is uniform, 1/n is less than half
of the minimum representable value of 2/k. Thus, ˜a = _[⃗]0._

Proposition 1 says that fixed-precision transformers are artificially limited because they can only
attend over bounded-length windows, making them similar to hard-attention transformers (Hao
et al., 2022). Morever, they cannot compute uniform attention over contexts of length n with less
than log log n precision. This explains why Chiang et al. (2023) prove finite-precision transformers
provably cannot recognize a[m]b[m], while in practice transformers have been shown to learn even its
harder variant a[m]b[m]c[m] even with long context lengths (Bhattamishra et al., 2020). In essence, their
upper bound only applies in the asymptotic regime when n > 2[2][p] .

In contrast, transformers in practice have enough precision both to compute uniform attention and
recognize a[m]b[m] on practical context lengths. More concretely, the bfloat16 representation allows
uniform attention over 2[6+2][7] 10[42] tokens and normal float16[12] allows 2[10+2][4] 10[8] tokens, both
_≈_ _≈_
well above the typical context window of transformers. This motivates a formal model of transformers
with enough precision to compute uniform attention and recognize languages such as a[m]b[m].

#### 4 Main Result: Expressing Log-Precision Transformers in FO(M)

By Proposition 1, precision must grow with the context length n (p > log log n) for a transformer
to compute uniform attention and other attention patterns with unbounded range, like practical
transformers. In this paper, we analyze any transformer with up to O(log n) precision. We show that
any function computable by log-precision transformers can be expressed in FO(M):

12We account for the division of p into pm and pe rather than treating them together. Our minimum value
differs slightly from numpy but is on the same order of magnitude. Moving to float8 lowers the length upper
bound for uniform attention to 2[3+2][3] _≈_ 2048, which suggests float8 LMs will have limited length generalization.


-----

**Theorem 2. Let T be a log-precision transformer with a parameter vector θT fixed for all context**
_lengths n.[13]_ _Then, there exists an FO(M) sentence ϕ that computes the same function as_ _, i.e.,_
_T_
_ϕ(x) =_ (x) for any input string x.
_T_

Theorem 2 is the tightest known upper bound for log-precision transformers and shows that it is still
possible to characterize transformers in a simple variant of first-order logic even with log-precision
and uniform attention. As alluded to earlier, Theorem 2 immediately implies that any problem
complete for FO(M) (or a larger class) is also transformer-hard. Since integer division and Dyck
language membership are known to be FO(M)-complete (Hesse, 2001; Aaronson et al., 2022), it
follows, perhaps surprisingly, that the entire computation of any transformer on input x can be
reduced to a single integer division or a finite number of Dyck-language queries:
**Corollary 2.1. Let T be a transformer satisfying Theorem 2. For any input x, there exist first-order**
_definable integers a, b, and i (dependent on_ _and x) such that_ (x) equals the i-th bit of _a/b_ _. For_
_T_ _T_ _⌊_ _⌋_
_any x, there also exist first-order definable strings w1, . . ., wm such that T (x) is first-order definable_
_in terms of the membership of the wi’s in k-Dyck._

#### 5 Preliminaries for Proving Theorem 2

**5.1** **Computation Graphs**

A computation graph G over a datatype D ⊆{0, 1}[∗] and a countable set of primitive functions
F ⊆ D[∗] _× D is a directed acyclic graph where:_

1. Each node is labelled by a node type: a function f ∈ F computed by this node.
2. Each edge represents a value D flowing as output from one node into another node. We
consider the edges flowing into node j to have an order, i.e., be numbered.

3. F contains the special symbol input, which designates k nodes as input nodes. We refer to k
as the arity and assume w.l.o.g. that nodes 0, . . ., k 1 are inputs.[14]
_−_

4. A single node is taken as the output node (w.l.o.g., the node with the largest index).

A computation graph G of arity k parameterizes a function D[k] _→_ D in the standard way: the input
nodes are assigned the input values, and the value of each node is computed (traversing the graph in a
bottom-up topological order) as a function of the values of its children until the output node receives
a value. The value of the output node is considered the output of the function. It is worth noting that
computation graphs can only process inputs of bounded length. To process arbitrary-length inputs,
we will need to generalize them to computation graph families (Section 5.2).

For a computation graph G, size(G) is the number of nodes, depth(G) is the length of the longest
path from an input node to the output, and arity(G, i) is the number of inputs to node i.

**Threshold circuits. A threshold circuit is a special case of a computation graph where D = {0, 1}**
and F is the set of threshold functions of the form θ≤∆ and θ≥∆ over D[∗], defined as follows:
_θ≤∆(x) = 1 if_ [�]σ∈x _[σ][ ≤]_ [∆] [and][ 0][ otherwise;][ θ][≥][∆][(][x][)][ is defined analogously. Typical AND, OR,]

and NOT gates are a special case of threshold gates, as is an IDENTITY gate.[15]

We allow nodes with the k[′] 1 largest indices to all be designated as (ordered) output nodes. A
_≥_
threshold circuit with arity k and k[′] output nodes will thus be a function from 0, 1 to 0, 1 .
_{_ _}[k]_ _{_ _}[k][′]_
This will be convenient when simulating neural network components that output multiple bits.

We will find it useful to consider threshold circuits as a kind of compilation target for computation
graphs: in other words, we will be concerned with simulating computation graphs defined over more
complex functions and data types into threshold circuits.

**5.2** **Computation Graph Families**

A computation graph family over D and F is a mapping from n ∈ N to a computation graph Gn
for processing inputs of size n. Thus, G defines a function from D[∗] _→_ D, where G(x) = G|x|(x).

13Theorem 2 can also be extended to apply to log-precision transformers with log-uniform weights, i.e., where
_θT can grow in size and precision with n (see Appendix B)._
14By convention in computer science, we let computation graph nodes be zero-indexed.
15For more background on threshold circuits, see Merrill & Sabharwal (2023) and Merrill et al. (2022).


-----

Intuitively, computation graph families are useful because they generalize computation graphs to
define functions over unbounded-length strings as inputs.

**Size, depth, and arity. For computation graph families, the size, depth, and arity become functions**
of the input length n: sizeG(n) = size(Gn), depthG(n) = depth(Gn), arityG(n, i) = arity(Gn, i).

**Uniformity. The infinite set G can be alternatively represented by two functions:**

1. nodeG(n, i), which returns the type of node i in Gn if i ≤ size(Gn), and ∅ otherwise. For
example, if node i computes the logical AND of its inputs, then nodeG(n, i) = ∧.

2. edgeG(n, i, j), which returns the argument index of i into node j if Gn contains an edge
_i →_ _j and −1 otherwise. edgeG(n, i, j) only needs to be defined over i, j < size(Gn). For_
example, if Gn contains a node j with three incoming edges, the second of which comes
from node i, then edge (n, i, j) = 1.
_G_

A pair of algorithms implementing these two functions uniquely specifies a computation graph family,
as it enables building the computation graph Gn for any n. Uniform computation graph families
(generalizing uniform circuits; cf. Arora & Barak, 2009) are families where nodeG and edgeG can be
computed efficiently, i.e., under some constraints on space or time:

**Definition 5 (Uniformity). A computation graph family G is T** (n)-uniform iff nodeG(n, i) and
edge (n, i, j) can be computed by a deterministic Turing machine in time T (n). We focus on
_G_
_log-uniform computation graph families: i.e., where T_ (n) = O(log n).[16]

**Threshold circuit families. These are simply families of threshold circuits. We will be simulating**
computation graph families with threshold circuit families. Log-uniform TC[0] is the class of languages
recognized by log-uniform constant-depth, poly-size threshold circuit families. See Merrill &
Sabharwal (2023); Liu et al. (2023); Arora & Barak (2009) for more background on TC[0] and circuits.

#### 6 Proof of Theorem 2

The idea is to simulate a transformer with a log-uniform TC[0] circuit family. Since log-uniform
TC[0] = FO(M), this would imply any transformer can be expressed in FO(M). First, we note that
transformers are log-uniform computation graphs:

**Lemma 1 (Proof in Appendix B.1). A transformer T is a log-uniform computation graph family**
_where F contains embedding, self-attention, feedforward, and output components._

Further, each core module of the transformer can be simulated by a log-uniform TC[0] circuit family:

**Lemma 2 (Proof in Appendix B.2). Let T be a log-precision transformer with fixed parameters θT .**
_Then each component in F is computable in log-uniform TC[0]._

Intuitively, we can now simulate a transformer in log-uniform TC[0] by just simulating each of its
components with a threshold circuit and routing their inputs and outputs appropriately. However, we
will need two more technical conditions to verify that this construction is indeed log-uniform:

**Lemma 3 (Proof in Appendix B.3). Let T be a log-precision transformer with fixed parameters**
_θT . There exists a function bsize(n) that is a power of 2 and computable in O(log n) time s.t._
sizeF (n) ≤ bsize(n) for all F ∈ F.

**Lemma 4 (Proof in Appendix B.4). If F is a log-uniform TC[0]** _family and sizeF_ (n) ≤ bsize(n),
_there exists a log-uniform TC[0]_ _family F_ _[′]_ _s.t. F(x) = F_ _[′](x) for all x and sizeF_ _′_ (n) = bsize(n).

Combined, Lemmas 3 and 4 show that each F ∈ F is computable by a log-uniform TC[0] family with
size bsize(n) that is a power of 2 and computable in time O(log n). We will show these conditions
imply a transformer can be simulated by a TC[0] family (Theorem 3) and moreover that is
_T_ _C_ _C_
log-uniform (Corollary 3.2). By the equivalence of log-uniform TC[0] and FO(M) (Barrington et al.,
1990), we then conclude that any log-precision transformer can be expressed in FO(M).

16Past work (Merrill & Sabharwal, 2023) analyzes transformers with a similarly named but weaker notion of
uniformity, namely log-space (rather than log-time) uniformity.


-----

**Algorithm 1 nodeC(n, i)**
_Return the type of gate i in circuit Cn._

1: F ← nodeG(n, bnode(n, i))
2: if F ̸= ∅ **then**
3: **return nodeF** (n, i − bstart(n, i[′]))

4: else return ∅


**Algorithm 2 edge** (n, i, j)
_C_
_If Cn contains an edge i →_ _j, return the argument_
_number of that edge. Otherwise, return_ 1.
_−_

1: i[′] bnode(n, i)
_←_
2: j[′] bnode(n, j)
_←_
3: si ← bstart(n, i[′])
4: sj ← bstart(n, j[′])
5: if i[′] = j[′] **then**
6: _F ←_ nodeG(n, i[′])

7: **return edgeF** (n, i − _si, j −_ _sj)_

8: else if edgeG(n, i[′], j[′]) ≥ 0 then
9: _bi ←_ _i −_ (si + bsize(n, i[′]) − _p(n))_

10: _bj ←_ _j −_ (sj + p(n) · edgeG(n, i[′], j[′]))

11: **if bi = bj < p(n) then return j −** _sj_

12: **else return −1**

13: else return −1


**6.1** **Simulating Computation Graph Families with Circuit Families**

We give algorithms that take a computation graph family and define a circuit family simulating it.
Intuitively, the algorithms creates contiguous blocks of circuit gates simulating each node in the
computation graph and route inputs and outputs between blocks appropriately.

**Block mapping.** This algorithm depends on a block mapping, which is an implementation of the
following three functions:

1. The block node bnode(n, i): the index of the node that gate i’s block is simulating.
2. The block start bstart(n, i[′]): the smallest gate index in the block simulating node i[′].
3. The block size bsize(n, i[′]): the number of gates in the block simulating node i[′].

Further, we enforce that a valid block mapping must satisfy that, for all i, with i[′] = bnode(n, i),
bstart(n, i[′]) _i < bstart(n, i[′]) + bsize(n, i[′])._
_≤_
Let be a computation graph whose primitive functions are computable by log-uniform threshold
_G_
circuits. We can identify each primitive function with a log-uniform threshold circuit family that
_F_
computes it, where the first arity (n) gates are IDENTITY gates reserved for taking input. For such
_F_
a graph, nodeG can be taken to return a symbol identifying a circuit family F. In this case, our
algorithm requires that, for all i[′], the block size of i[′] must match the size of the circuit for the type of
block i[′], i.e., bsize(n, i[′]) = sizenodeG (n,i′)(n). These properties let us meaningfully identify a graph
node i[′] with a block of nodes that will simulate it. This intuition enables us to develop Algorithms 1
and 2 for constructing a uniform threshold circuit family from a uniform computation graph family.
**Theorem 3. Let G be a computation graph over a finite set of node types F, where each F ∈** F is
_specified by a log-uniform circuit family. Let bnode, bstart, and bsize be a valid block mapping in_
_the sense above. Then Algorithms 1 and 2 define a circuit family_ _such that_
_C_

_1. C and G compute the same D[∗]p_ _[→]_ [D][p] _[function (let the final][ p][ gates of each][ C][i]_ _[be its output).]_

_2. depthC(n) ≤_ depthG(n) · maxF depthF (n).
_3. sizeC(n) ≤_ sizeG(n) · maxF sizeF (n).

_Proof. Assume w.l.o.g. that the gates of_ are topologically ordered. We show by induction over
_C_
circuit gates j (with j[′] = bnode(n, j)) that:

1. For all i[′] _< j[′], the last p nodes of block i[′]_ store the value of node i[′].
2. For all i such that bstart(n, j[′]) ≤ _i ≤_ _j, gate i of C (as a function of the input nodes of j[′]_ )
computes gate i − bstart(n, j[′]) of nodeG(n, j[′]).

Base case. We have two circuits with no gates, so the premises are trivially satisfied.

Inductive case. Assume the premises hold up to j. We will show they hold for j + 1. Let =
_T_
nodeG(n, j[′]). By Premise 1, we know that the last p nodes of block i[′] store the output of node i[′], for


-----

_i[′]_ _< j[′]. By Algorithm 2, for each i[′]_ such that edge (n, i[′], j[′]) = a with 0 _k < arity_ (n), gates kp
_G_ _≤_ _F_
through k(p + 1) 1 of block j[′] will copy the final p gates of block i[′]. Thus, the first k arity (n)
_−_ _·_ _F_
gates of block j[′] store the inputs to node j[′].

At this point, we use Premise 2 to conclude that the first j − bstart(n, j[′]) gates of block j[′] compute
the same function as the first j bstart(n, j[′]) gates of with respect to this input. Thus, we just
_−_ _F_
need to show that gate j + 1 is also correct. Within Algorithm 2, we fall in case i[′] = j[′], meaning
that gate j + 1 of block j[′] gates the same inputs as gate j + 1 of . By Algorithm 1, the type of
_F_
gate j + 1 in block j[′] is the type of gate j + 1 of F. Thus, gate j + 1 in block j[′] computes the same
function of the input gates as gate j + 1 in . If j + 1 = bsize(n, j[′]), we conclude that the final p
_F_
gates of block j[′] store the output of node j[′].

Let XC[0] denote any family of constant-depth, poly-size circuits, including AC[0] and TC[0].[17]

**Corollary 3.1. Let G be a constant-depth, poly-size computation graph family over a finite F. If**
_every node type in F can be computed by XC[0]_ _circuits, the function computed by G is in XC[0]._

Since a transformer has constant depth and polynomial size, Corollary 3.1 lets us easily recover
prior results about hard-attention transformers (Hao et al., 2022; Hahn, 2020) and saturated attention
transformers (Merrill et al., 2022) using a common framework. All one has to do is show that all
individual node types in such transformers can be computed by AC[0] and TC[0] circuits, respectively.

Corollary 3.1 established that Algorithms 1 and 2 construct a circuit family that simulates . With
_G_
the right block mapping, will be log-uniform as long as and its node types are log-uniform.
_C_ _G_
**Corollary 3.2. Let G be a log-uniform, constant-depth computation graph family over a finite F,**
_where each F ∈_ F is specified by a log-uniform TC[0] _family with sizeF_ (n) = bsize(n) that is a power
_of 2 computable in O(log n) time. Then_ _can be simulated by a log-uniform TC[0]_ _family_ _that obeys_
_G_ _C_
_the size and depth properties of Theorem 3._

_Proof. Let_ be the circuit family defined by Algorithms 1 and 2 given and the following block
_C_ _G_
mapping: bnode(n, i) = _i/bsize(n)_ _, bstart(n, i[′]) = i[′]_ bsize(n), bsize(n, i[′]) = bsize(n). Since
_⌊_ _⌋_ _·_
bsize(n) is a power of 2, bnode and bstart are reducible to left and right shifting over O(log n)bit integers, which can be implemented in O(log n) time. Thus, each block mapping function is
computable in time O(log n). Since nodeG and edgeG are just calling functions computable in time
O(log n) with constant overhead, we conclude that, the circuit family they define, is log-uniform,
_C_
and it is already known to simulate with constant depth and polynomial size by Theorem 3.
_G_

#### 7 Conclusion

We proved that any log-precision transformer classifier can be translated to an FO(M) sentence that
computes the same function (on all inputs of any length). This result comes by first simulating
a transformer with a highly uniform threshold circuit family, and then leveraging the established
equivalence of log-uniform circuits and FO(M). Transformers and other neural nets are often
discussed in contrast with symbolic models based on logical formalisms (Garnelo & Shanahan,
2019)—an immediate implication of our result is that it is possible to express the inner workings of
transformers also in a simple logic, challenging the premise of a rigid division between symbolic and
neural models. Our results also provide the tightest known upper bound on log-precision transformers.

While it is striking that a full transformer can be translated to a sentence in a logic as simple as
FO(M), we believe the bound is not tight. In particular, we conjecture that it is possible to simulate
any transformer with an FO(M) sentence of quantifier depth of at most 2, which could be proven by
establishing a hierarchy theorem describing the FO(M) quantifier depth needed to simulate a TC[0]

family of a certain size. It would also be an interesting extension to translate real transformers to
FO(M) sentences. In this sense, we believe our results provide a theoretical foundation to guide
mechanistic interpretability work (cf. Weiss et al., 2021; Lindner et al., 2023).

Our findings provide a novel view into transformer classifiers and their limits. It would be exciting
for future research to extend our results to account for other common practical uses of transformers,
such as for long-form generation, chain-of-thought reasoning, and in-context learning.

17Formally, F just needs to contain ∧ and ∨.


-----

**Acknowledgments**

We thank Paul Beame, David Chiang, anonymous reviewers, and researchers at the Allen Institute for
AI for feedback. Thanks to Noa Nabeshima for pointing out a minor notational inconsistency. WM
was supported by an NSF graduate research fellowship and in part by NSF award 1922658.

#### References

Aaronson, S., Kuperberg, G., and Habryka, O. TC[0]: Constant depth threshold circuits, 2022. URL
```
 https://complexityzoo.net/Complexity_Zoo:T#tc0.

```
Arora, S. and Barak, B. Computational Complexity: A Modern Approach. Cambridge University
Press, 2009.

Barrington, D. A. M., Immerman, N., and Straubing, H. On uniformity within NC[1]. Journal of
_Computer and System Sciences, 41(3):274–306, 1990._

Bhattamishra, S., Ahuja, K., and Goyal, N. On the ability and limitations of transformers to recognize
formal languages. In EMNLP, 2020.

Brent, R. P. and Zimmermann, P. Modern computer arithmetic, volume 18. Cambridge University
Press, 2010.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam,
P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R.,
Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray,
S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D.
Language models are few-shot learners. In NeurIPS, 2020.

Chiang, D., Cholak, P., and Pillay, A. Tighter bounds on the expressivity of transformer encoders.
_ICML, 2023._

Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A.,
Conerly, T., DasSarma, N., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Jones, A.,
Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S.,
and Olah, C. A mathematical framework for transformer circuits. Transformer Circuits Thread,
2021. https://transformer-circuits.pub/2021/framework/index.html.

Furst, M. L., Saxe, J. B., and Sipser, M. Parity, circuits, and the polynomial-time hierarchy. Mathe_matical systems theory, 17:13–27, 1984._

Garnelo, M. and Shanahan, M. Reconciling deep learning with symbolic artificial intelligence:
representing objects and relations. Current Opinion in Behavioral Sciences, 29:17–23, 2019. ISSN
2352-1546.

Hahn, M. Theoretical limitations of self-attention in neural sequence models. TACL, 8:156–171,
2020.

Hao, Y., Angluin, D., and Frank, R. Formal language recognition by hard attention transformers:
Perspectives from circuit complexity. TACL, 10:800–810, 2022.

Hesse, W. Division is in uniform TC[0]. In International Colloquium on Automata, Languages, and
_Programming, pp. 104–114. Springer, 2001._

Hunter, P., Bouyer, P., Markey, N., Ouaknine, J., and Worrell, J. Computing rational radical sums in
uniform TC0. Foundations of Software Technology and Theoretical Computer Science, 2010.

Lindner, D., Kramár, J., Rahtz, M., McGrath, T., and Mikulik, V. Tracr: Compiled transformers as a
laboratory for interpretability. arXiv, abs/2301.05062, 2023.

Liu, B., Ash, J. T., Goel, S., Krishnamurthy, A., and Zhang, C. Transformers learn shortcuts to
automata. In ICLR, 2023.


-----

Merrill, W. and Sabharwal, A. The parallelism tradeoff: Limitations of log-precision transformers.
_TACL, 11:531–545, 2023._

Merrill, W., Ramanujan, V., Goldberg, Y., Schwartz, R., and Smith, N. A. Effects of parameter norm
growth during transformer training: Inductive bias from gradient descent. In EMNLP, 2021.

Merrill, W., Sabharwal, A., and Smith, N. A. Saturated transformers are constant-depth threshold
circuits. TACL, 10, 2022.

Pérez, J., Marinkovi´c, J., and Barceló, P. On the Turing completeness of modern neural network
architectures. In ICLR, 2019.

Thoppilan, R., Freitas, D. D., Hall, J., Shazeer, N. M., Kulshreshtha, A., Cheng, H.-T., Jin, A., Bos,
T., Baker, L., Du, Y., Li, Y., Lee, H., Zheng, H., Ghafouri, A., Menegali, M., Huang, Y., Krikun,
M., Lepikhin, D., Qin, J., Chen, D., Xu, Y., Chen, Z., Roberts, A., Bosma, M., Zhou, Y., Chang,
C.-C., Krivokon, I. A., Rusch, W. J., Pickett, M., Meier-Hellstern, K. S., Morris, M. R., Doshi, T.,
Santos, R. D., Duke, T., Søraker, J. H., Zevenbergen, B., Prabhakaran, V., Díaz, M., Hutchinson,
B., Olson, K., Molina, A., Hoffman-John, E., Lee, J., Aroyo, L., Rajakumar, R., Butryna, A.,
Lamm, M., Kuzmina, V. O., Fenton, J., Cohen, A., Bernstein, R., Kurzweil, R., Aguera-Arcas, B.,
Cui, C., Croak, M., Chi, E., and Le, Q. LaMDA: Language models for dialog applications. ArXiv,
abs/2201.08239, 2022.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., and
Polosukhin, I. Attention is all you need. In NeurIPS, 2017.

Weiss, G., Goldberg, Y., and Yahav, E. On the practical computational power of finite precision
RNNs for language recognition. In ACL, 2018.

Weiss, G., Goldberg, Y., and Yahav, E. Thinking like transformers. ICML, 2021.

Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., and Liu,
T.-Y. On layer normalization in the transformer architecture. In ICML, 2020.


-----

#### A Conditional Majority

Given formulas ϕ, ψ, Mi : ϕ. ψ is a sentence that is true iff ψ is true for at least half the values of i
that make ϕ true.

**Proposition 2. For any two predicates ϕ(i) and ψ(i), Mi : ϕ(i). ψ(i) can be expressed in FO(M).**

_Proof. Mi : ϕ. ψ can be rewritten using a counting quantifier and a threshold quantifier:_

� �
_k, k[′]._ 2k[′] = k _i : ϕ(i)_ _j : (ϕ(j)_ _ψ(j))_ _._
_∃_ _∧∃[k]_ _∧∃[≥][k][′]_ _∧_

The formula 2k[′] = k can be defined using bit. We then use the fact that counting and threshold
quantifiers can be expressed in terms of majority quantifiers (Barrington et al., 1990) to conclude that
Mi : ϕ. ψ can be expressed in FO(M).

#### B Omitted Proofs

Table 1 summarizes the notation we use in the following proofs when describing computation graphs
and circuit families.

Table 1: Summary of common notation for computation graph and circuit families.

Graph Circuit Output Range Description

_i[′]_ _i_ Z index of node or gate

nodeG(n, i[′]) nodeC(n, i) F[18] type of node or gate

edgeG(n, i[′], j[′]) edgeC(n, i, j) Z argument # of edge i → _j_

sizeG(n) sizeC(n) Z # of nodes or gates

depth (n) depth (n) Z longest path length
_G_ _C_

bnode(n, i) [0, sizeG(n)] block containing i

bstart(n, i[′]) [0, sizeC(n)] first gate in block i[′]

bsize(n, i[′]) Z size of block i[′]

**B.1** **Transformers are Log-Uniform Computation Graph Families**

We now justify that the computation graph family defining a transformer is log-uniform. To do this,
we introduce a stronger notion of uniformity called column uniformity that captures the highly regular
structure of the transformer.

Let node(G, i) be the i-th node of computation graph G. Let a mod b be the remainder when a is
divided by b.

**Definition 6 (Column uniformity). A computation graph family G is T** (n)-column-uniform iff there
exists a computation graph K (with fixed size w.r.t n) such that, for all i, j such that 0 _i, j <_
_≤_
sizeG(n):

1. nodeG(n, i) = node (K, i mod size(K)).
2. If _i/size(K)_ = _j/size(K)_, then
_⌊_ _⌋_ _⌊_ _⌋_

edge (n, i, j) = edge (K, i mod size(K), j mod size(K)) .
_G_

Otherwise, edge (n, i, j) can be computed by a deterministic Turing machine in time T (n).
_G_

We define log-column-uniform analogously to log-uniform: i.e., we let T (n) = O(log n). logcolumn-uniform implies log-uniform because our implementations of nodeG and edgeG can store K
in a finite lookup table and compute the quotient and remainder of i and j by size(K) in O(log n)
time using Lemma 12. The edges outside of K are computable in O(log n) time by construction.

**Lemma 1 (Proof in Appendix B.1). A transformer T is a log-uniform computation graph family**
_where F contains embedding, self-attention, feedforward, and output components._


-----

_Proof. We show the stronger condition that any transformer_ is a log-column-uniform computation
_T_
graph family, which implies it is log-uniform.

We have the column K by Definition 2: all that remains to show is that edgeGT can be computed
in time O(log n) for edges outside the column. These edges route from the layer ℓ output to the
self-attention heads of layer ℓ + 1. Following from the column structure, there exists kℓ such that a
node i is an output vector of layer ℓ iff kℓ = i mod size(K). In a finite lookup table, we can store
_kℓ_ for each ℓ + 1, and use this for self-attention routing. For an unmasked self-attention head j, we
compute:

�⌊i/size(K)⌋ if kℓ = i mod size(K)
edgeGT (n, i, j) = 1 otherwise.
_−_

For causally masked attention, we extend the first case to check that _i/size(K)_ _j/size(K)_ .
_⌊_ _⌋≤⌊_ _⌋_
Either way, this logic can be implemented in time O(log n) via Lemma 12. Thus, we conclude that
_GT is column-uniform._

**B.2** **Transformer Components are Computable by Log-Uniform Threshold Circuits**

**Lemma 2 (Proof in Appendix B.2). Let T be a log-precision transformer with fixed parameters θT .**
_Then each component in F is computable in log-uniform TC[0]._

We prove a more general version of Lemma 2 that handles some cases with weights growing with n.
The weights θT are just a special case of a computation graph (that do not depend on the input); we
can thus apply our definition of log-uniform to them. Lemma 2 follows from a more general result
with log-uniform θT :

**Lemma 5. Let T be a log-uniform transformer with log-uniform θT . Then each component in F is**
_computable in log-uniform TC[0]._

_Proof. In Appendix C, we show that log-uniform θT implies:_

1. The embedding component is computable in log-uniform TC[0] (Lemma 6).
2. The self attention mechanism is computable in log-uniform TC[0] (Lemma 7).
3. The activation block is computable in log-uniform TC[0] (Lemma 8).
4. The output classifier head is computable in log-uniform TC[0] (Lemma 9).

We have shown that each F ∈ F is computable in log-uniform TC[0].

**B.3** **Transformer Component Size Has a Log-Time Upper Bound**

**Lemma 3 (Proof in Appendix B.3). Let T be a log-precision transformer with fixed parameters**
_θT . There exists a function bsize(n) that is a power of 2 and computable in O(log n) time s.t._
sizeF (n) ≤ bsize(n) for all F ∈ F.

_Proof. Let 2[b][(][n][)]_ be the least power of 2 at least as large as sizeF (n) for all F. We observe that 2[b][(][n][)]
is at most 2 · maxF sizeF (n) for all n. Because each F has poly size, there is a fixed k such that, for
large enough n,[19]

2[b][(][n][)] _n[k]_
_≤_

_b(n)_ _k_ log n _._
_⇒_ _≤_ _⌈_ _⌉_

Define b[′](n) = k log n and bsize(n) = 2[b][′][(][n][)]. bsize(n) is both a power of 2 and an upper bound
_⌈_ _⌉_
on 2[b][(][n][)]; what remains to be shown is that it can be computed in time O(log n). We can first
compute log n in time O(log n) by finding the greatest nonzero index of n. Next, we can compute
_⌈_ _⌉_
_b[′](n) = k_ log n in time O(log log n) since k is fixed size and log n has size at most O(log log n)

_·⌈_ _⌉_ _⌈_ _⌉_
(Brent & Zimmermann, 2010). Finally, we compute bsize(n) = 2[b][′][(][n][)] by simply left-shifting 1 at
most O(log n) times.

19We can compute bsize(n) for small n using finite lookup.


-----

**B.4** **Circuit Families Can Be Padded to Log-Time Size Upper Bounds**

Recall that the last p bits of our circuits represent the circuit’s output (cf. Section 5.1). In Lemma 4,
we consider (x) = (x) if and only if the last p bits of and agree for all x.
_F_ _F_ _[′]_ _F_ _F_ _[′]_

**Lemma 4 (Proof in Appendix B.4). If F is a log-uniform TC[0]** _family and sizeF_ (n) ≤ bsize(n),
_there exists a log-uniform TC[0]_ _family F_ _[′]_ _s.t. F(x) = F_ _[′](x) for all x and sizeF_ _′_ (n) = bsize(n).

_Proof. The high level idea is that we can pad_ to a circuit that has size bsize(n) and simply
_F_ _F_ _[′]_
copies over the p output bits of to its own last p bits using identity gates.
_F_

We first set nodeF _′ to copy over the existing circuit and append identity nodes. Let Id denote an_
identity node. Then nodeF _′ is defined as:_


nodeF (n, i) if nodeF (n, i) ̸= ∅
Id if nodeF (n, i) = ∅∧ _i < bsize(n)_
otherwise.
_∅_


nodeF _′_ (n, i) =








We see that the size of will thus be of size bsize(n).
_F_ _[′]_

Next, we extend edgeF _′_ (n, i, j) to route the original output bits to the new output bits. Recall that
an edge value of 0 means i is the first argument of gate j, and an edge value of 1 means there is
_−_
no edge i → _j. Let kj = p(n) −_ (bsize(n) − _j) be the index of node j as an output gate in F_ _[′]. For_
example, k = 0 for the first output bit. Now let output (n, i, k) represent whether node i is the k-th
_F_
output of Fn. We can compute outputF (n, i, k) in terms of nodeF as follows:

outputF (n, i, k) ⇐⇒ nodeF (n, i + p(n) − _k −_ 1) ̸= ∅∧ nodeF (n, i + p(n) − _k) = ∅._

Then edgeF _′ is defined:_


edge (n, i, j) if edge (n, i, j) = 1
_F_ _F_ _̸_ _−_
0 if output (n, i, kj)
_F_
1 otherwise.
_−_


edgeF _′_ (n, i, j) =








The first condition simply copies over the original edges. The second condition adds p(n) new edges
(for the different values of k) that route the final p(n) nodes of to the final p(n) nodes of,
_F_ _F_ _[′]_
guaranteeing that the two circuits will compute the same function.

Because both nodeF _′ and edgeF_ _′ just rely on addition, conditional branching, and a finite number of_
calls to functions computable in time O(log n), they are both computable in time O(log n).

#### C Transformer Column Components

In this section, we generally omit layer subscripts for clarity. We assume a pre-norm (Xiong et al.,
2020) parameterization of the transformer for concreteness and because this is more standard in
newer transformers. However, the results would also hold with the original post-norm (Vaswani et al.,
2017).

As mentioned in the main text, we view θT as a concatenation of the parameters for the transformer
functions. Thus, if m and w are computable in time O(log n) and θT is log-uniform, it follows that
the parameter vector for each ϕ, s, v, f, and κ is itself log-uniform because we can map indices in the
smaller parameter vectors to indices in θT in time O(log n).

**C.1** **Transformer Embeddings**

For each position 1 ≤ _i ≤_ _n, the transformer embedding function represents token σi ∈_ Σ and its
position i with a vector. Let V be an embedding matrix of size Σ _m where each row represents_
_|_ _| ×_
the embedding for some σ. Let f : N → D[m]p [be computable in time][ O(log][ n][)][. Then,]

_ϕ(σi, i) = vσi + f_ (i).

**Lemma 6. If θT is log-uniform, then ϕ is computable in log-uniform TC[0].**


-----

_Proof. The embedding block can be expressed as a constant-size computation graph that constructs_
**V, computes vσi using an affine transformation, computes f** (i), and then, finally, sums vσi and
_f_ (i). The first step is computable by a log-uniform constant-depth, poly-size threshold circuit family
since θT is log-uniform. We can compute an affine transformation via a log-uniform constant-depth
poly-size threshold circuit family via Lemma 10. f (i) can be directly computed by the Turing
machine constructing the circuit by construction. The sum of the two terms can then be computed by
a log-uniform constant-depth threshold circuit of size polynomial in m, which is also polynomial
in n. Since we have a computation graph where all node types are computable by log-uniform,
constant-depth, poly-size threshold circuit families, we conclude by Corollary 3.2 that ϕ can also be
computed by log-uniform, constant-depth, poly-size threshold circuit family.

**C.2** **Self Attention**

The two components of the self attention block are s, the similarity function, and v, the value function.
Let hi be the hidden state at the previous layer and **h[¯]i = lnorm(hi). Then, the similarity function**
first computes queries and keys, and then takes the scaled dot-product between them:

**qi = Wqh[¯]i + bq**
**ki = Wkh[¯]i + bk**

� �
**q[⊤]i** **[k][i]**

_s(hi, hj) = exp_ � _._

_m/h_

Then the value function is defined v(hi) = Whh[¯]i + bh. We first show that the value function (and
also the keys and queries by symmetry) is computable in log-uniform TC[0]:

**Lemma 7. If θT is log-uniform, then the self-attention component is computable in log-uniform TC[0].**

_Proof. v is a composition of constructing the parameters (in log-uniform TC[0]_ since θT is loguniform), layer norm (in log-uniform TC[0] by Lemma 11), and an affine transformation (in loguniform TC[0] by Lemma 10). Thus, v is computable in log-uniform TC[0].

Computing s is a constant-depth computation graph. First, we compute qi and ki and then multiply
them, and all of these steps are in log-uniform TC[0]. Next, we can compute m and h in time O(log n)
and build a log-uniform TC[0] circuit that divides the product of the last step by �m/h. Finally, we

compute p-precision exp, which can be expressed in log-uniform TC[0] as multiplication followed by
left-shifting. Thus, by Corollary 3.2, s can be computed in log-uniform TC[0].

_s and v are log-uniform, so their size p is at most poly(n). Computing self attention reduces to_
binary multiplication and division over Dp, and performing iterated addition (summation) over n
numbers in Dp. Binary multiplication, binary division (Hesse, 2001), and iterated addition (Merrill &
Sabharwal, 2023) can all be computed in log-uniform TC[0], i.e., by a log-uniform, constant-depth
threshold circuit family of size at most poly(p) poly(n). Thus, self attention can also be computed
_≤_
in log-uniform TC[0].

**C.3** **Activation Block**

The activation function f encapsulates the aggregation of the attention head outputs and the feedforward subnetwork of the transformer. f takes as input attention head outputs ai,1, . . ., ai,h ∈ Dp[m/h]
and the previous layer value hi.

The first part of the activation block simulates the pooling part of the self-attention sublayer. The
head outputs are first concatenated to form a vector ai, which is then passed through an affine
transformation (Wo, bo) : D[m]p _p_ [followed by residual connections to form the sublayer output]

_[→]_ [D][m]
**oi ∈** D[m]p [:]
**oi = Woai + bo + hi.**

The second part of the activation block first applies layer-norm and then simulates the feedforward
subnetwork to compute the next layer vector h[′]i[. Let][ ¯][o][i][ = lnorm(][o][i][)][. Let][ σ][ be a nonlinearity]
computable in linear time on its input (in the most standard transformer, ReLU). Then, for affine


-----

transformations (W1, b1) : D[m]p _p_ [and][ (][W][2][,][ b][2][) :][ D]p[w] _p_ [, the feedforward subnetwork can]

_[→]_ [D][w] _[→]_ [D][m]
be defined:

**h[′]i** [=][ W][2][σ][(][W][1][o][¯][i] [+][ b][1][) +][ b][2] [+][ o][i][.]

**Lemma 8. If θT is log-uniform, then f is computable in log-uniform TC[0].**

_Proof. The activation block can be expressed as a constant-size computation graph where the nodes_
construct affine transformation parameters, apply affine transformations, compute layer-norm, and
compute elementwise nonlinearities. Since each of these nodes is computable by a log-uniform,
constant-depth, poly-size threshold circuit family, the activation block is as well.

**C.4** **Output Classifier Head**

We assume the output from the transformer is computed as follows. First, **h[¯]1 = lnorm(h1). Then,**
we use a parameter vector w ∈ D[m]p [and bias term][ b][ to compute:]

_κ(h1) = sgn(w[⊤]h[¯]1 + b)._

**Lemma 9. If θT is log-uniform, then κ is computable in log-uniform TC[0].**

_Proof. We can express computing κ as a composition of constructing the parameters w, b and_
computing the affine transformation. Both parts of this composition are computable by a log-uniform,
constant-depth, poly-size threshold circuit family, so computing κ is as well.

#### D Neural Net Building Blocks

In this section we analyze the uniformity of common neural net building blocks that are used within
the various high-level transformer components.

**D.1** **Affine Transformations**

Affine transformations are a core part of neural networks used in various parts of the transformer. An
affine transformation takes as input parameters (W, b) : D[a]p _p_ [and a vector][ x][ ∈] [D]p[a] [and returns]

_[→]_ [D][b]
**Wx + b.**

**Lemma 10. For p = O(log n), any p-precision affine transformation where W, b are log-uniform is**
_computable by a log-uniform, constant-size threshold circuit family of size polynomial in a and b._

_Proof. We first use the uniformity of W, b to construct them in O(log n) time. For the transformation_
**Wx + b, first compute each wi ⊙** **x in parallel, where ⊙** represents elementwise multiplication.
Since binary multiplication over polynomial-size numbers is in log-uniform TC[0], this can be done
in parallel with log-uniform TC[0] circuits. We then use b log-uniform, constant-depth, poly-size
threshold circuit families, each corresponding to an output index, that compute the sum over the a
entries of each wi ⊙ **x. The affine transformation corresponds to the composition of these two steps,**
and is thus computable by a log-uniform TC[0] circuit family.

**D.2** **Layer Norm**

The layer norm is applied between sublayers in the transformer. Let µ = (1/d) [�]i[d]=1 _[x][i][. The layer]_
norm y ∈ D[m]p [of a vector][ x][ ∈] [D]p[m] [is computed, for scalars][ a, b][ ∈] [D][p][,]


� **x** _µ_
_−_
**y = a**

**x** _µ_
_∥_ _−_ _∥_


�
+ b.


**Lemma 11. If a, b are log-uniform, the layer norm over a vector of size m can be computed by a**
_log-uniform threshold circuit family of constant depth and size polynomial in m._


-----

_Proof. First compute m using summation over the constant term 1 from 1 to m. This summation can_
be computed by a log-uniform constant-depth threshold circuit family of size polynomial in m. Then
compute the sum over x using a similar circuit, and divide them to get µ, using the fact that integer
division is in log-uniform TC[0] (Hesse, 2001). We can then compute x _µ in log-uniform TC[0]._
_−_

At this point, we can compute **x** _µ_ in log-uniform TC[0] (Hunter et al., 2010), then divide each
_∥_ _−_ _∥_
**x** _µ by the norm in log-uniform TC[0], and then apply the final affine transformation in log-uniform_
_−_
TC[0] (Lemma 10). Thus, computing layer norm is in log-uniform TC[0].

#### E Arithmetic Complexity

**Lemma 12. Given an m-bit integer a and n-bit integer b, we can compute the quotient ⌊a/b⌋** _and_
_remainder a mod b in time O(mn)._

_Proof. Let D(m, n) and M_ (m, n) denote, respectively, the time complexity of dividing and multiplying an m-bit integer by an n-bit integer. Brent & Zimmermann (2010) give the following fact:
_D(m + n, n)_ O(M (m, n)). With the goal of analyzing D(m, n), we apply this as follows:
_≤_

_D(m, n)_ _D(m + n, n)_
_≤_
O(M (m, n))
_≤_
O(mn).
_≤_

Applying Lemma 12 when a has size O(log n) and b has size O(1) says that we can do division in
time O(log n).


-----

