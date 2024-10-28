## Saturated Transformers are Constant-Depth Threshold Circuits

### William Merrill[∗†] Ashish Sabharwal [∗] Noah A. Smith[∗‡]

_∗_ Allen Institute for AI _† New York University_ _‡ University of Washington_
### willm@nyu.edu {ashishs,noah}@allenai.org


### Abstract


Transformers have become a standard neural network architecture for many NLP
problems, motivating theoretical analysis of
their power in terms of formal languages.
Recent work has shown that transformers
with hard attention are quite limited in
power (Hahn, 2020), as they can be simulated by constant-depth AND/OR circuits
(Hao et al., 2022). However, hard attention
is a strong assumption, which may complicate the relevance of these results in practice. In this work, we analyze the circuit complexity of transformers with satu_rated attention: a generalization of hard at-_
tention that more closely captures the attention patterns learnable in practical transformers. We first show that saturated transformers transcend the known limitations of
hard-attention transformers. We then prove
saturated transformers with floating-point
values can be simulated by constant-depth
threshold circuits, giving the class TC[0] as an
upper bound on the formal languages they
recognize.

### 1 Introduction


Saturated (ζ) TC[0] = ALL
_⊆_

Table 1: Summary of combined results from Hao et al.
(2022) and this paper. Each cell α, D characterizes
the languages recognizable by transformers with attention function α and datatype D (floats F or rationals
Q). AC[0] and TC[0] are circuit complexity classes, and
AC[0] TC[0]. ALL is the set of all formal languages
_⊂_
over alphabet 0, 1 . See §4 for formal definitions. Out
_{_ _}_
of these results, we view saturated attention with floats
as the best model of practical transformers.


Opening the “black box” (Alishahi et al., 2020)
of the representations within neural networks is an
important step towards building systems with robust and interpretable behavior. In NLP, one part
of this question is analyzing the languages networks can model, and the mechanisms they use
to represent linguistic structure and dependencies.
One path toward this goal is via formal analysis
of specific network architectures (Merrill, 2021);
for example, recurrent neural networks (RNNs).
Due to their autoregressive formulation, formal
linguistic analysis of RNNs has often characterized their power by relating them to automatatheoretic classes of formal languages (Weiss et al.,
2018; Peng et al., 2018; Merrill, 2019, inter alia).


Recently, however, RNNs have largely been overtaken in NLP by a new class of models: transformers (Vaswani et al., 2017). Transformers are
not autoregressive, and therefore less naturally resemble automata, posing challenges to characterizing their linguistic capacity or inductive biases
in the same terms as RNNs. Instead, some recent work has related them to circuit complexity classes, a direction that we continue to pursue in this paper. Drawing on classical circuit
lower bound results, Hao et al. (2022) and Hahn
(2020) derive theoretical limitations of transformers with hard attention, meaning the attention distributions focus all probability mass on one index.
Together, their results show that AC[0]—the class
of languages recognizable by constant-depth circuit families—upper bounds the formal languages
hard-attention transformers can recognize.
However, hard attention is a strong assumption, making it unclear how these results transfer to practical transformers. For example, Bhattamishra et al. (2020) showed how transformers
can solve synthetic counting tasks by using uniform attention patterns, which hard attention does
not allow. Motivated by this potential disconnect
between theory and practice, we aim to extend
circuit-based analysis to transformers with saturated attention: a generalization of hard attention


-----

that has been argued to approximate attention patterns acquired through gradient descent (Merrill
et al., 2021). Broadly speaking, saturated attention goes beyond hard attention in that it can “tie”
across a subset of positions, rather than selecting
just one position. The tied positions are then aggregated by averaging. Qualitatively, saturated attention heads can “count”: a capability observed
in transformers in practice (Bhattamishra et al.,
2020). Further, Merrill et al. (2021) show that
transformer training dynamics lead attention heads
in several pretrained transformers to approximate
saturated attention. In summary, saturated attention strictly generalizes hard attention and should
more closely reflect the attention patterns acquired
in practical transformers.
Our main contributions are twofold. First, we
show that saturated transformers can recognize
languages outside AC[0]. Then, as depicted in Table 1, we prove that transformers with floating
point activations and saturated attention can only
recognize formal languages in the circuit complexity class TC[0], constituting an upper bound for
a more realistic model of transformers than past
results with hard attention.

### 2 Roadmap

In §3, we formally define our model of the transformer, including defining saturated attention in
contrast to hard attention. §4 introduces circuits
in theoretical computer science and relevant complexity measures and classes for them.
In §5, we first briefly analyze saturated transformers with rational values where the embedding,
scoring, and activation functions are allowed to
be any size-preserving function. We find such
transformers to be universally powerful. We also
observe that when the positional embeddings are
computed in time linear in the sequence length,
saturated rational-valued transformers are exactly
as powerful as the complexity class of their activation functions, because the full input sequence
can be pooled to a single position, and an activation function can be used as an oracle over the full
input sequence. However, this setup relies on the
use of unrealistic embedding functions. To move
to a more realistic model of computation, we then
focus on saturated transformers whose values are
restricted to be floats, which have a coarser granularity and, thus, cannot encode the full input sequence into a single position.


Building on results of Pérez et al. (2019), we
demonstrate in §6 that saturated transformers with
_float activations transcend the theoretical limita-_
tions of hard-attention transformers. In particular,
we will show that they can recognize the majority language, which lies outside AC[0]. We experimentally validate that transformers can learn to
recognize the majority language. Taken together,
these results suggest that the very weak characterization of hard-attention transformers does not hold
in practice for saturated or soft attention.
In §7, we show that, on input sequences of
length n, the size of each state vector in a transformer over floats is O(log n) bits, similar to
saturated LSTMs (cf. Merrill, 2019). Thus, the
full transformer state at any given layer has size
O(n log n), although each feedforward block can
only locally access a small, O(log n) “piece”.
Thus, while hierarchical representations (e.g., to
process arbitrary-depth Dyck languages or reverse
strings) can be implemented in a transformer, our
result implies they must be distributed in some
way across n state vectors, rather than represented
compactly within a single vector.
Finally, in §8, we use the bounded size of transformer representations to upper bound that formal languages that can be recognized by a saturated transformers with floating-point values. In
particular we show that they can be simulated by
constant-depth threshold circuits, i.e., fall in TC[0].
Informally, this suggests that moving from hard attention to saturated attention can be thought of as
extending the implicit class of circuit gates available in the network to include threshold gates.
Our results make progress in the analysis of
transformers by deriving upper bounds for a more
realistic model of transformers than has previously
been analyzed. RoBERTa, T5, and other pretrained transformers have been shown to be approximately saturated (Merrill et al., 2021), so
our results imply TC[0] may be a meaningful upper bound on the computation expressible within
such networks. Our analysis also motivates future
work further refining the characterization of saturated transformers, as well as comparing transformers with soft and saturated attention.

### 3 Definitions and Notation

We will often use w to refer to a string over any
generic alphabet Σ, i.e., w ∈ Σ[∗]. Semantically,
_w corresponds to the string a transformer receives_


-----

as input. In contrast, we use x and other symbols
to refer to binary strings in 0, 1 . These binary
_{_ _}[∗]_
strings will represent intermediate values within
the transformer computation, rather than the raw
input to the transformer.

**3.1** **Datatypes**

Under our model, all values in the transformer are
binary strings. In order to compute self attention and other operations over binary strings, we
need to define datatypes describing the semantics
of these binary strings as numbers. We will describe a semantics for binary strings as integers, as
often comes up in circuit complexity. We then extend this to rational numbers and floats, which are
necessary for representing the division operations
that occur in attention heads within transformers.

**Unsigned Integers** We can interpret binary
strings x 0, 1 as unsigned integers in the
_∈{_ _}[∗]_
standard way, i.e., the numerical value of x
_∈_
0, 1 is
_{_ _}[n]_

_n−1_
�

_x_ Z = 2[i][−][1]xi
� �

_i=1_

We allow standard integer operations like
+Z, ·Z, <Z. For example, 101 +Z 1 = 110.

**Rationals** To interpret r 0, 1 as a rational
_∈{_ _}[∗]_
number, we first view it as a sign bit s along with
a tuple of two unsigned integer substrings _p, q_ .[1]
_⟨_ _⟩_

The numerical value represented by r is

�r�Q = (2s − 1)�p�Z/�q�Z.

Let red(p, q) return _s, t_ where s = p/gcd(p, q)
_⟨_ _⟩_
and t = q/gcd(p, q). Then, we can define arithmetic operations over two rationals r = _p, q_ and
_⟨_ _⟩_
_r[′]_ = ⟨p[′], q[′]⟩ in the standard way:

_r +Q r[′]_ = red(p ·Z q[′] + p[′] _·Z q, q ·Z q[′])_

_r ·Q r[′]_ = red(p ·Z p[′], q ·Z q[′]).

**Floats** We define floats F as the subset of the rationals where the denominator is constrained to be
a power of 2.[2] Multiplication and addition are defined as for Q, and are guaranteed to produce an
1Under the hood, we imagine the pair ⟨p, q⟩ is encoded by
padding p and q to the same length with 0’s and interweaving
bits from each.
2More generally, the denominator may be taken to have
a prime factorization of bounded length, although we work
with the power of 2 definition, which is both simpler and
closely resembles conventional floating point datatypes.


other float. Notably, division for floats is implemented by multiplying by an approximate multiplicative inverse, so it may be that (x/Fy)·Qy ̸= x.
See §A for a more formal discussion.
In §5, we will study transformers over rational
values. From §6 onwards, we will then take the
values in transformers to be floats unless otherwise
stated. Going forward, we will generally omit
datatype subscripts from operations where they
are clear from context. We will sometimes write D
as a set in function signatures, e.g., f : D[k] _→_ D[k].
In this usage, it refers to the set 0, 1, but it is of_{_ _}[∗]_
ten more intuitive to write the datatype shorthand
(rather than 0, 1 ) to hint at the intended seman_{_ _}[∗]_
tics of the functional arguments.

**Size of Binary Strings** Under our model, integers, rationals, and floats are all abstractions built
out of binary strings. For any x 0, 1 (which
_∈{_ _}[∗]_
can be interpreted semantically as an integer, float,
or rational), we define its size _x_ as the total
_|_ _|_
length of x measured in bits. We imagine a tuple _p, q_ is encoded by padding p, q to the same
_⟨_ _⟩_
length with leading 0’s, and interleaving bits from
each sequence. This means the size of a rational
is 2 max( _p_ _,_ _q_ ) + 1. For example, the integer 2
_|_ _|_ _|_ _|_
takes 2 bits to specify, while the float [1]

2 [takes][ 5]
bits (1 for the sign, 2 for the numerator, 2 for the
denominator).

**Size Preservation** We say that a function f :
0, 1 0, 1 is size-preserving iff there ex_{_ _}[∗]_ _→{_ _}[∗]_
ist constants c, n such that for all inputs x with
_n_ _x_, _f_ (x) _c_ _x_ . Let be the set of
_≤|_ _|_ _|_ _| ≤_ _· |_ _|_ _P_
size-preserving functions. While size-preserving
functions are defined here over binary strings, they
can be equivalently applied over integers, rationals, and floats, since these datatypes, as we have
defined them, are just binary strings.

**3.2** **Transformers**

We define the following general transformer
model, which can be parameterized to use different types of attention patterns and whose internal
functions (e.g., feedforward blocks) can be computed by different function classes.

**Definition 1 (Transformer) A transformer is a tu-**
ple ⟨Σ, D, α, L, H, φ, {sℓ,h}ℓ,h[L,H]=1[,][ {][f][ℓ][}]ℓ[L]=1[⟩] [where]

1. Σ is a finite input alphabet, i.e., the set of token types in a formal language.

2. D is a scalar datatype, i.e., a semantics for
interpreting binary strings as numbers. We


-----

will generally consider D = F.

3. α is an attention function that maps a vector
of attention scores in D[n] (for any n) to a normalized probability distribution, also in D[n].
In this paper we take α to be either hard (η)
or saturated (ζ) attention; see §3.3.

4. L ∈ N is the number of layers.
5. H ∈ N is the number of heads.
6. φ : Σ × N → D[m] is a position-aware embedding function that maps a token and position
to a vector, where m is a multiple of H.

7. For each ℓ, h, the function sℓ,h : D[m] _×D[m]_ _→_
D assigns attention scores to pairs of values.

8. For each ℓ, the function f : D[m] _×D[m]_ _→_ D[m],
maps a previous layer value and attention
head output to a new value vector.

On an input string w Σ[n], a transformer com_∈_
putes L layers of output sequences vℓ,1, · · ·, vℓ,n
(for ℓ _≤_ _L), where each vℓ,i ∈_ D[m]. In the 0th
layer, each token wi and its position i are embedded into a value v0,i. Subsequent layers aggregate
information from the previous value sequence vℓ
using a multi-head attention mechanism, and output a new value sequence vℓ+1. More formally,
these layers are structured as follows:

1. Embedding Layer: v0,i = φ(wi, i).
2. Attention Head: Each of the H attention
heads in layer ℓ maps the full previous sequence into a new value via sℓ,h and then applies the attention function α:

_aℓ,h,i,j = sℓ,h(vℓ,i, vℓ,j)_


Hard attention collapses the attention scores to
a one-hot distribution with all mass concentrated
at one index. Let M(a) = {i | ai = maxj aj}.

**Definition 2 (Hard attention) Define hard atten-**
tion η(a) as


_η(a)j =_


�
1 if j = minm∈M(a) m
0 otherwise.


In contrast, saturated attention spreads probability mass evenly across “tied” scores.

**Definition 3 (Strong saturated attention; Merrill**
et al. 2021) Define saturated attention ζ(a) as


1
_ζ(a)j =_

(a)
_|M_ _| [·]_


�
1 if j (a)
_∈M_
0 otherwise.


_bℓ+1,h,i =_


_n_
�

_α(aℓ,h,i,:)j · vℓ,j._
_j=1_


Crucially, the semantics for addition and multiplication here (as well as in the computation
of α) come from the datatype D.

3. Activation Block:[3]

_vℓ+1,i = fℓ+1(vℓ,i, bℓ,:,i)._

**3.3** **Attention Functions**

An attention function α maps a vector of scores
_a ∈_ D[n] to a probability distribution over 1, · · ·, n.
Specifically, we consider two attention functions:
_hard attention η(a) and soft attention ζ(a)._

3Let Vℓ,h be a head’s value matrix in the standard transformer parameterization. Then fℓ is computed by first multiplying each bℓ,h,i by Vℓ,h, aggregating the multiple attention
heads, and applying the feedforward subnetwork.


Merrill (2019) shows how this form of attention can be derived by taking a large-norm limit
of the network weights; a derivation can be found
there. Saturated attention reduces to hard attention
when (a) = 1, and attends uniformly when
_|M_ _|_
(a) = n. Both hard and uniform attention can
_|M_ _|_
be implemented with numerical stability, motivating weak saturated (or, “uniform”) attention:

**Definition 4 (Weak saturated attention) Each head**
implements either hard attention (Def. 2) or the
uniform pattern υ(a)j = _n[1]_ [.]

In general, we will use “saturated attention” to
refer to strong saturated attention and provide upper bounds for this setting. On the other hand, our
lower bounds only use weak saturated attention,
thereby showing that even weak saturated attention is more powerful than hard attention.

**3.4** **Language Recognition**

Finally, we define language recognition for transformers.

**Definition** **5** (Language recognition) Write
_vℓ,i(w) for the value of vℓ,i on input string w._
A transformer recognizes a language L ⊆ Σ[∗] if
there exists a D-valued affine transformation W, b
such that, for all w ∈ Σ[∗],

_W · vL,1(w) + b > 0 ⇐⇒_ _w ∈L._

This says the decision problem of recognizing
must be linearly separable using the first value
_L_
in the last layer of the transformer. In practice, the
first token in a transformer is often set to CLS, and
its output can be passed to a classifier during finetuning (Devlin et al., 2019). This inspires Def. 5.


-----

There are other potential ways to define language
recognition and generation for transformers (Hewitt et al., 2020; Yao et al., 2021), but they do not
lead to meaningful differences for our purposes.
Finally, we define AHAT(D) as the set of
languages recognizable by some saturated transformer over D, where the internal functions can be
any size-preserving function.[4]

**Definition 6 Let AHAT(D) be the set of lan-**
guages such that there exists a transformer
_L_
_⟨Σ, D, ζ, L, H, φ, sℓ,h, fℓ⟩_ that recognizes L where
each φ, sℓ,h, fℓ _∈P.[5]_

We note that size preservation is a weak
condition to assume about the internal functions in practical transformers: since any lineartime-computable function is size-preserving, it is
strictly weaker than assuming the internal functions can be computed in linear time. To further
justify this condition, we explicitly show in §B
that the component functions within transformers
are size-preserving.

### 4 Circuit Complexity

Circuit complexity is a branch of computational
complexity theory that studies circuit families as
a model of computation.[6] Intuitively, circuits are
useful for formally studying the types of computational problems that can be efficiently solved with
parallelism, as the depth of a circuit corresponds to
the runtime of a program on an idealized, fully parallel computer. We review background on circuits,
circuit families, and relevant complexity measures
and classes.

**Circuits** For a fixed n, a circuit is a computation graph, where leaves correspond to input bits
_xi and their negations ¬xi, and the internal nodes_
are logic gates (typically and ), with one la_∧_ _∨_
beled as the output node. The gates can conventionally be taken to have either binary or unbounded fan-in. The circuit computes a function
_f :_ 0, 1 0, 1 by substituting the input val_{_ _}[n]_ _→{_ _}_
ues into the leaf nodes, propagating the computation through the graph, and returning the value of

4The name AHAT standards for “averaging hard attention
transformer”, and is taken from Hao et al. (2022).
5To apply size preservation to the embedding function φ,
we consider the size of a token to be log(|Σ|).
6For more reference material on circuit complexity, we refer the reader to chapters 6 and 14 of Arora and Barak (2009)
or chapters 1 and 2 of the Handbook of Theoretical Computer
_Science, Volume A (van Emde Boas, 1991; Johnson, 1991)._


Figure 1: A circuit that takes a string 0, 1 and
_∈{_ _}[5]_
returns whether it contains the bigram 11.

the output node. Fig. 1 shows an example circuit
that takes inputs of length 5, and returns whether
they contain the bigram 11.

**Circuit Families** A circuit family is an ordered
set of circuits {Cn}n∈N where each circuit is identified with a particular input size n. We say a
circuit family recognizes a formal language
_L ⊆_
0, 1 iff, for all w,[7]
_{_ _}[∗]_ _∈L_

_C|w|(w) = 1 ⇐⇒_ _w ∈L._

**Circuit Complexity** Two important notions of
complexity for a circuit are its size and depth. The
size of a circuit is the number of gates. The depth
is the longest path from an input node to the output
node. For a circuit family, both quantities can be
expressed as functions of the input size n. A cir_cuit complexity class is a set of formal languages_
that can be recognized by circuit families of a certain size, depth, and set of gates. In particular, we
will discuss the classes AC[0] and TC[0].

**Definition 7 AC[0]** is the set of languages
_L ⊆_
0, 1 such that there exists a circuit family rec_{_ _}[∗]_
ognizing with unbounded arity _,_ gates,
_L_ _{∧_ _∨}_
poly(n) size, and O(1) depth.

Intuitively, AC[0] represents the class of problems that are highly parallelizable when the computational primitives are standard logic gates. In
contrast, TC[0] will also represent highly parallelizable computation, but when the gates are expanded
to include threshold gates. For a bitstring x
_∈_
_{0, 1}[∗], define the threshold gate θ≥k(x) to return_
1 iff _k bits in x are 1, and equivalently for_ _k._
_≥_ _≤_
For example, θ≥3(110011) = 1.

**Definition 8 TC[0]** is the set of languages
_L ⊆_
0, 1 such that there exists a circuit family rec_{_ _}[∗]_
ognizing with unbounded arity _,_ _, θ_ gates,
_L_ _{∧_ _∨_ _}_
poly(n) size, and O(1) depth.

7Similarly, for any alphabet Σ and L ⊆ Σ∗, we interpret
_wi as a one-hot vector over Σ and define the family to recog-_
nize L iff, for all w ∈L, C|w|·|Σ|(w) = 1 ⇐⇒ _w ∈L._


-----

It is known that AC[0] TC[0] NC[1],
_⊂_ _⊆_
where NC[1] denotes the languages recognizable
by O(log n)-depth circuits with bounded gate arity. Whether or not the latter containment between TC[0] and NC[1] is strict is an open question.
Whereas parity and other basic regular languages
are outside AC[0] (Furst et al., 1981), TC[0] properly
contains parity, although it is unknown whether it
contains all the regular languages. Between AC[0]

and TC[0] lies the class ACC[0] (Yao, 1990).

**Uniformity** The circuit classes defined above
(and which we will use in this paper) are non_uniform, meaning circuits for different input sizes_
are not constrained to have any relation to each
other. Non-uniform circuit families can recognize some uncomputable languages, such as the
language of strings 1[k] such that Turing machine
_k does not halt on the null input (cf. Arora and_
Barak, 2009). In contrast, the uniform variants
of circuit families are constrained such that a logspace Turing machine must output a string encoding of circuit Cn on the input string 1[n], forcing
any language the circuit family can recognize to
be computable. For these uniform classes (which
we write with a u prefix), it is known that

_uTC[0]_ _uNC[1]_ L P,
_⊆_ _⊆_ _⊆_

where L and P denote the conventional complexity classes of log-space and polynomial-time decision problems. Thus, it is unknown whether uTC[0]

is restricted compared to general polynomial-time
computation, but if we accept the common conjecture that one (if not all) of the above containments
are strict, then uTC[0] forms a restricted family of
problems compared to P, which, intuitively, are
more parallelizable than other problems in P.

### 5 Aren’t Transformers Universal?

We now begin our analysis of saturated transformers. Hao et al. (2022) and Hahn (2020) were able
to give upper bounds on the power of hard attention without imposing any constraints on the embedding, scoring, and activation functions. The
same will not be the case with saturated attention:
any bounds on transformers will require leveraging some properties constraining their internal
functions. One property we use will be size preservation. We will first show though that size preservation is not enough on its own: deriving a nontrivial upper bound will depend on subtle assumptions about the transformer’s datatype.


With rational values and size-preserving internal functions, we will show saturated transformers
can recognize any formal language, i.e., the class
ALL = 0, 1 . Our construction re_{L | L ⊆{_ _}[∗]}_
sembles the universal approximation construction
of Yun et al. (2020), which relies on the ability of
the transformer to uniquely encode the full input
string into a single value vector. After the full sequence is encoded locally into a single vector, the
activation block can be used as a black box to recognize any language.

**Theorem 1 AHAT(Q) = ALL.**

_Proof. We construct a 1-layer rational-valued_
transformer with a single head to recognize every
string w in any formal language ALL. We will
_L ∈_
omit ℓ, h subscripts. Let pi denote the ith prime
number. The embedding layer encodes each input
token according to


_φ(wi, i) =_


�
1/pi if wi = 1
0 otherwise.


Since pi ∼ _i log i for large i by the prime number_
theorem (cf. Goldstein, 1973), the number of bits
needed to represent φ(wi, i) is

_c log(i log i)_ _c log(i[2]) = 2c log i._
_≤_ _≤_

Since i had size log i, this implies φ is sizepreserving.
Now, we define a single uniform attention head
that sums across all i, outputting [�]wi=1 _p1i_ [. The]

denominator q of this sum is the product [�]i=1 _[p][i][.]_
Observe that wi = 1 if and only if pi divides q.
Thus, we can define a function f that extracts the
input sequence w from q by checking whether, for
each i, pi divides q. We let g be a function recognizing L, and set f = g _f_ . The output of the
_◦_
transformer will now compute whether w,
_∈L_
since f outputs an encoding of the original input sequence w, and g decides whether w .
_∈L_
Note that any function solving a decision problem
is size-preserving, hence f .
_∈P_

Thm. 1 says that our transformer architecture
parameterized with a rational datatype can recognize any formal language. But a construction of
this form feels unrealistic for two reasons. First,
it requires the embedding layer to implement an
unconventional prime encoding scheme in the embedding layer. Second, we are using the activation


-----

layer as a black box to recognize any language—
even uncomputable ones! On the other hand, the
feedforward subnetworks used in practice in transformers cannot even implement all computable
functions when the weights are fixed independent
of the sequence length n. We can get around both
these issues by instead restricting the datatype to
floats, which is the direction we will pursue in the
remaining sections.[8]

**5.1** **Resource-Bounded Transformers**

In §C, we develop an alternate perspective on the
universality of transformers, showing that, if the
embedding function is allowed to be computed in
time linear in the sequence length, then the transformer’s complexity is equivalent to its activation
functions’ complexity.

**Theorem 2 (Informal) If φ can be any function**
_computable in time linear in n, and the scor-_
_ing and activation functions can be computed in_
_T_ (m) time on inputs of size m with T (m) ⩾ _m,_
_then languages recognizable by the transformer_
_are TIME(T_ (m)).

§C contains a formal statement and proof. For
example, allowing polynomial-time functions inside the transformer implies that the transformer
will recognize exactly the complexity class P. A
major unrealism about this setup is the assumption
that φ can be an arbitrary function computable in
time linear in n, motivating our main results in a
more constrained setting in §8.

**5.2** **Discussion**

We are not stating the results in this section as
evidence that practical transformers are capable
of universal or arbitrary polynomial computation.
Rather, the unnaturalness of these constructions
(specifically, the prime numbers based position
encoding) motivates us to slightly constrain our
model of the transformer in a realistic way: we
will switch the datatype from rationals to floats,
since even using only simple uniform attention,
a model with rationals and unconstrained internal
functions is universal. We will soon see that this
realistic constraint prevents universal simulation,

8It may also be possible to derive tighter bounds for
rational-valued transformers by imposing stronger constraints on the internal functions. However, with floats, we
will see that size preservation is sufficient to derive a tighter
characterization of transformers’ power. We leave this alternate direction to future work.


frac0 = aggregate(
select_all,
indicator(tokens == 0));
frac1 = aggregate(
select_all,
indicator(tokens == 1));
maj = frac1 > frac0;

Figure 2: A program recognizing MAJ in RASP, a
programming language designed to abstract away details of transformer computation (Weiss et al., 2021).
frac{0,1} measure the fraction of inputs that are 0
or 1. Then maj checks whether frac1 > frac0.

and in fact bounds the capacity of the saturated
transformer within TC[0].

### 6 Beyond Hard Attention, with Floats

We now move to the setting of saturated transformers over floats. Hao et al. (2022) identified
that hard-attention transformers can only recognize languages within AC[0]. In contrast, saturated
transformers over floats can recognize the “majority” language MAJ, which is known to lie outside AC[0] (Furst et al., 1981). Pérez et al. (2019,
Prop. 3.3) show how MAJ can be recognized by
transformers. In Thm. 3, we offer a simpler construction that leverages only a single uniform attention head, as opposed to the model of transformers they were considering. Thus, this construction is achievable with saturated attention.

**Theorem 3 AHAT(F) ̸⊆** AC[0].

_Proof. Let #σ(w) ∈_ N denote the number of σ
tokens in string w 0, 1 . Let #(w) denote
_∈{_ _}[∗]_
a count vector where each element corresponds to
some σ 0, 1 . We define MAJ as follows:
_∈{_ _}_

MAJ = �w ∈{0, 1}[+] _| #1(w) > #0(w)�._

We will construct a 1-layer transformer with a single head to recognize MAJ, omitting ℓ, h subscripts
from s, f, x, b. Fig. 2 gives the same construction
in RASP (Weiss et al., 2021).
Let xi = φ(wi, i) be a 1-hot encoding of wi.
For all i, j, set s(xi, xj) = 1, resulting in a single
head attending everywhere:

_bi = [#(][w][)]_ _._

_n_

Finally, set f (bi) to return whether #1(w)/n >
#0(w)/n, which, for n > 0, is true iff w ∈ MAJ.


-----

Generalization accuracy on majority language


dings. It appears that while MAJ is in the capacity of the transformer, the standard sinusoidal
positional embedding scheme provides the wrong
inductive bias for learning it. This recalls the
finding of Yao et al. (2021) that the choice of
positional encodings seems to greatly impact the
transformer’s ability to generalize formal language
tasks to longer sequences.

### 7 Size of Transformer Values


1.0

0.9


0.8

0.7


0.6

0.5

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
||||||||||
||||||||||
||||||||||
||||||||||
||||none||||||
||||||||||
||||learne sinuso|d idal|||||
||||||||||
||||||||||
||||||||||


50 100 150 200 250

Mean length


Figure 3: In practice, transformers can learn the majority language (which lies outside AC[0]). We train 1-layer
transformers on majority, where each line represents a
different positional encoding scheme. Training string
length was binomial with n = 100. Trained models
were then evaluated on generalization sets with n ranging from 100 to 500. Mean length (x axis) is n/2.

Notably, the construction in Thm. 3 is not
just possible within our generalized transformer
framework, but can also be implemented by the
standard parameterization of φ, s, and f in real
transformers (Vaswani et al., 2017). The uniform
attention pattern can be implemented by setting all
query and key attention parameters to 0. Then, we
can use the affine transformation that aggregates
the head outputs to compute the tuple:

� #1(w) − #0(w) �

_, 0_ _._
_n_

This tuple is then passed through layer normalization (Ba et al., 2016), resulting in a new tuple
_⟨t1, t2⟩. Crucially, t1 > t2 if and only if the same_
applies to the quantities in the original tuple. Thus,
a linear classifier can decide whether t1 > t2 to
successfully recognize the language, as per Def. 5.


The theoretical limits on hard-attention transformers were derived by Hao et al. (2022) by bounding
the size in bits of the representation vℓ,i at each
layer ℓ and position i. Specifically, they show that
the value vℓ,i is representable in O(log n) bits on
input sequences of length n. Thus, each value can
only contain limited information about the input
sequence, intuitively explaining their upper bound
on hard-attention transformers. Inspired by their
analysis, this section will show that, in a saturated
transformer, each vℓ,i also has a size of O(log n)
bits. Later in §8, we will use this property to show
that saturated attention transformers are limited in
the formal languages they can recognize.

**7.1** **Size of Float Sums**


**6.1** **Empirical Validation**

In Fig. 3, we show empirically that a 1-layer
transformer can learn and generalize MAJ. This
supports our argument that the theoretical limitations of hard-attention transformers do not apply
to practical transformers. We train with three different types of positional encoding: none, meaning no positional information; learned, where each
position gets a trainable embedding vector, and
the sinusoidal scheme of Vaswani et al. (2017).
The model with no positional embeddings generalizes the best, followed by the learned embed

How many bits does it take to represent the value
of an attention head within a saturated transformer? As a naive bound, the output of a saturated attention head is specified by a float for
each of n values attended over from the last layer,
which would take at least linearly many bits in n.
However, this upper bound on its size is not tight.
Instead, we will show that all head and activation
values can be represented in O(log n) bits. Our
analysis will rely heavily on the following lemma:

**Lemma 1 Let v1, · · ·, vn be a sequence of floats,**
_each with size at most z. Then there exists c such_
_that_ [�]i[n]=1 _[v][i][ has size at most][ 4][cz][ + 2 log][ n][ + 1][.]_

_Proof. Let pi, qi denote the numerator and denom-_
inator of the floating point vi, respectively. Similarly, let ps, qs be the numerator and denominator of the float s. By assumption, there exists c
such that each pi, qi both have size ≤ _cz for large_
enough n. We let pmax = maxi pi and analogously for qmax. Since all qi’s are powers of 2,
the numerator ps is


_n_
�

_pi ·_ _[q][max]qi_ _≤_ _npmaxqmax,_
_i=1_


-----

which, represented as an integer, has size:

log n + cz + cz = 2cz + log n.
_≤_

On the other hand, the denominator qs = qmax,
which has size _z. Therefore, the float represent-_
_≤_
ing the sum has size

1 + 2 max(2cz + log n, z)
_≤_

= 4cz + 2 log n + 1,

which completes the proof.

In particular, we will use Lem. 1 to show that,
when each of a sequence of n values has size
O(log n), the sum will also have size O(log n).

**7.2** **Size of Transformer Values**

We will now leverage Lem. 1 to show that the values are of bounded size in any transformer over
floats with an elementwise-size-preserving attention function.

**Definition 9 A function α** : D[n] _→_ D[n] is
elementwise-size-preserving if, for 1 _i_ _n,_
_≤_ _≤_
the function xi _α(x)i is size-preserving (where_
_�→_
_x ∈_ D[n]).

Note that saturated attention satisfies this definition. We are ready to prove a theorem bounding
the size of the representations in transformers with
elementwise-size-preserving attention.

**Theorem 4 For any transformer over F with**
_φ, sℓ,h, fℓ_ _∈_ _P_ _and_ _α_ _elementwise-size-_
_preserving, for all ℓ_ _≤_ _L and i ≤_ _n, vℓ,i has size_
O(log n).

_Proof. By induction over ℓ. The proof follows the_
definition of transformer computation in §3.2.

**Base Case** _wi ∈_ Σ has size O(1), and i ∈ [n]
has size O(log n). Since φ ∈P, v0,i = φ(wi, i)
has size O(log n) for all i.

**Inductive Case** Assume vℓ,i has size O(log n).
Since sℓ+1,h, aℓ+1,h,i,j = sℓ+1,h(vℓ,i, vℓ,j)
_∈P_
has size O(log n) for all i, j. Since α is
elementwise-size-preserving, we can conclude
that α(aℓ+1,h,i,:)j also has size O(log n) for all
_h, i, j. Multiplying two floats is size-preserving_
(cf. §B), so α(aℓ+1,h,i,:)j _vℓ,j has size O(log n)_
_·_
for all h, i, j. We then apply Lem. 1 to conclude
that bℓ+1,h,i has size O(log n), where, recall,


Finally, computing vℓ+1,i = fℓ+1(vℓ,i, bℓ,:,i), we
conclude that vℓ+1,i has size O(log n) for all i by
size preservation.

**Corollary 4.1 For any saturated transformer over**
F with size-preserving internal functions, for all
_ℓ_ _≤_ _L and i ≤_ _n, vℓ,i has size O(log n)._

Cor. 4.1 follows because saturated attention is
elementwise-size-preserving. Softmax attention,
on the other hand, is not guaranteed to fulfill this
property, since it requires computing the exponential function. This technical challenge prevents
generalizing our technique to soft attention.

**7.3** **Discussion**

Similar to hard-attention transformers (Hao et al.,
2022), the size of each vector representation in
a saturated transformer over floats is O(log n).
This is enough memory for individual vectors to
“count”, a behavior that has been observed in
both LSTMs (Weiss et al., 2018) and transformers (Bhattamishra et al., 2020). On the other hand,
O(log n) space is not enough memory for individual vectors (for example, the CLS output) to
encode arbitrarily large combinatorial objects like
trees. However, transformers are not limited to
computing in an “online” fashion where tokens
are consumed sequentially, meaning their effective
state is n values of size O(log n). Notably, trees
with n leaves can be encoded in a distributed fashion across n values of size O(log n). One construction for this is, at index i, to store wi and i,
along with a pointer j to the parent. Since i, j can
both be represented in log n bits, each vector uses
only O(log n) bits.
Additionally, the O(log n) space bound has implications from the perspective of circuit complexity. While saturated attention cannot be simulated
in AC[0], we will show in §8 that saturated transformers over F can be simulated by TC[0] circuits.

### 8 Threshold Circuit Simulation

We have proved that each value vector in a saturated transformer over floats has O(log n) size.
Now, we show how this implies saturated transformers can be simulated by TC[0] circuits. Our results heavily leverage the following lemmas:

**Lemma 2 (Hao et al. 2022) Any function f :**
0, 1 0, 1 _can be computed by a boolean_
_{_ _}[c]_ _→{_ _}[d]_
_circuit of depth 3 and size at most (2[c]_ + c + 1)d.


_bℓ+1,h,i =_


_n_
�

_α(aℓ,h,i,:)j · vℓ,j._
_j=1_


-----

So that our results are self-contained, we reproduce a proof of this lemma in §D. Applying

Lem. 2 to a size-preserving function with at most
_c log n input bits immediately yields:_

**Corollary 2.1 Any size-preserving function with**
_at most c log n input bits can be computed by a_
_boolean circuit of depth 3 and polynomial size._

In other words, such functions can be computed
with AC[0] circuits. In addition, we will show that
the sum of n floats of size at most c log n can be
computed by TC[0] circuits.

**Lemma 3 Let v1, · · ·, vn be a sequence of floats,**
_each with size at most c log n for some c. Then the_
_sum_ [�]i[n]=1 _[v][i][ is computable by a threshold circuit]_
_of constant depth and polynomial size._

_Proof. Let the integers pi, qi be the numerator and_
denominator of vi. We first compute qmax, the
maximum qi, using an AC[0] circuit that compares
all pairs qi, qj, and returns the first qi such that
_qj_ _qj for all j. We then use the fact that multi-_
_≥_
plication and right shift (qi is a power of 2) are in
TC[0], in order to compute

_qmax_
_ri = pi_

_qi_

in parallel for all i. Note that qmax and qi are both
powers of 2, so the division will be exact. Next,
we leverage the fact that the sum of n integers of
size O(log n) is in TC[0] (Kayal, 2015), in order to
compute the numerator of the sum p[′] = [�]i _[r][i][.]_
We select the denominator as q[′] = qmax. Finally,
we can add an AC[0] circuit that “reduces” the fraction by removing shared trailing zeros from p[′], q[′],
which is possible by Corollary 2.1. Thus, we have
constructed a TC[0] circuit to compute the sum of n
floats with size O(log n).

We now construct a TC[0] circuit that simulates a
saturated transformer over floats.

**Theorem 5 AHAT(F) ⊆** TC[0].

_Proof. For each n, we construct a TC[0]_ circuit that
simulates a saturated transformer on inputs of size
_n. We construct the circuit modularly, with one_
subcircuit for the attention mechanism, and another for the feedforward subnetwork.

**Attention Head** Fix a single head in some layer.
We will construct a TC[0] subcircuit that simulates
the attention mechanism at position i. The head
attends over vectors v1, · · ·, vn. For all j, vj has


size O(log n) by Thm. 4. In parallel for each j,
we compute the scores ai,j = s(vi, vj) with an
AC[0] circuit by Corollary 2.1. We then compute
_ai,max ≜_ maxj ai,j with an AC[0] circuit by comparing all vj pairwise, and selecting the first vk
such that vk _vj for all j. We then compute_
_≥_
“masked” values ui,j for each j via an AC[0] circuit
by Lem. 2:


_ui,j ≜_


�
_vj_ if ai,j ≥ _ai,max_
0 otherwise.


We then compute the sum si ≜ [�][n]j=1 _[u][i,j][ by]_

Lem. 3. By Lem. 1, si has size O(log n). Now,
we similarly define


_zi,j ≜_


�
1 if ai,j ≥ _ai,max_
0 otherwise.


Using an analogous sum construction with zi,j instead of ui,j, we can use a TC[0] circuit to compute
_|M(a)|: the number of j such that ai,j ≥_ _ai,max._
Finally, since dividing floats is in TC[0] (cf. §A), we
can compute the head output as si/|M(a)|, which
has size O(log n) by size preservation of division.

**Feedforward** As input, f receives vi as well as
_H head outputs, all of which have size O(log n)._
As the total size of the input is O(log n), we can
use Corollary 2.1 to compute the output of f with
an AC[0] circuit. The size of the output is O(log n)
by size preservation of f . The same idea holds for
_φ as well as the linear classification head._
We have simulated each transformer component
with a TC[0] subcircuit, completing the proof.

**8.1** **Discussion**

Recall that, over rationals, we found that sizepreserving saturated transformers could recognize
any language. In contrast, we have now shown
that using floating-point representations places
such transformers within TC[0]. In this paper,
we have only considered non-uniform AC[0] and
TC[0], as opposed to the uniform variants of these
classes, which are more closely connected to familiar formal language classes like the regular and
context-free languages (cf. Cojocaru, 2016; Mahajan, 2007). As transformers satisfy some intuitive
notion of uniformity, an open question is whether
saturated transformers also fall into uniform TC[0].


-----

### 9 Conclusion

Compared to hard attention, saturated attention
adds theoretical power to transformers. We
showed that saturated attention lets transformers recognize languages outside AC[0], which is
the upper bound with hard attention. Further,
while saturated transformers with rational values
and size-preserving internal functions can recognize any language, we characterize the limits of
size-preserving saturated transformers with floats.
Specifically, saturated transformers with float values fall in TC[0], a more powerful circuit class than
AC[0]. Thus, going from hard to saturated attention can be understood as augmenting the model
with threshold gates. This illustrates one way that
the circuit complexity paradigm characterizes the
power of transformers. Going forward, there are
many interesting open questions that circuit analysis can answer, such as comparing the power of
saturated and soft attention, and refining existing
upper bounds for transformers in terms of uniform
circuit families.

### Acknowledgments

Thanks to Yiding Hao, Dana Angluin, and Robert
Frank for sharing an early draft of their work.
We also appreciate helpful feedback from Dana
Angluin, Matt Gardner, Yoav Goldberg, Michael
Hahn, Kyle Richardson, and Roy Schwartz.

### A Float Division

Let / be truncated division between integers. We
divide a float by an integer p by defining an approximate multiplicative inverse p[−][1]. The numerator is 2[|][p][|]/p and the denominator is 2[|][p][|]. For division by a float p, q, we simply apply the integer
approach and then multiply by q. This yields numerator 2[|][p][|]/p · q and denominator 2[|][p][|].
The fact that float division is defined in terms
of integer multiplication and division implies it
is size-preserving and can be simulated in TC[0],
which we use in §8.

### B Justifying Size Preservation

We justify that feedforward neural networks are
size-preserving over floats. Feedforward neural
networks are made up of a fixed (with respect to
_n) number of addition, multiplication, division,_
ReLU, and square root (for layer norm) opera

tions. Therefore, it suffices to show that these operations are all in S(F).
For addition, the numerator is

_≤_ _p1q2 + p2q1 ≤_ 2pmaxqmax,

which has size ≤ log 2 + |pmax| + |qmax| ≤
2(|pmax| + |qmax|) for large enough input size.
For multiplication, the numerator is just p1 · p2,
which has size ≤ 2|pmax|. Let the denominators
be q1 = 2[k][1] and q2 = 2[k][2]. Then the denominator
is 2[k][1][+][k][2], which has size ≤ 2|qmax|.
Division can be analyzed in terms of the approximate multiplicative inverse (§A).[9] Its numerator has size _p_ + 1 + _q_ 2( _p_ + _q_ ) for
_≤|_ _|_ _|_ _| ≤_ _|_ _|_ _|_ _|_
large enough input size. The denominator has size
_p_ + 1 2 _p_ for large enough input size.
_≤|_ _|_ _≤_ _|_ _|_
Size preservation is trivially satisfied for ReLU,
which cannot expand the size of the input.
To make layer norm work, we just need to analyze square root, which we will define in a truncated fashion over integers. The square root of a
rational, then, simple takes the square root of p and
_√_
_q. We have that_ _p_ _p_ and analogously for q.
�� �� _≤|_ _|_

### C Resource-Bounded Transformers

Size preservation is one way to characterize the
constraints on transformers’ internal functions; a
slightly different perspective is to fix φ and analyze how the language recognition abilities of the
transformer change depending on the computational resources allotted to each sℓ,h and fℓ. In this
section, we derive an alternate universality theorem in terms of time complexity classes. We will
show that as long as φ is powerful enough, such
transformers have equivalent time complexity to
their activation functions.
Recall that a transformer is a tuple
_⟨Σ, D, α, L, H, φ, sℓ,h, fℓ⟩._ In contrast to
AHAT(D) (cf. Def. 6), we will now work
with a different class of transformer languages
AHAT(D, T (m)) We will allow the embedding
functions to be linear in the sequence length, and
explore the effect of varying the complexity of
the other internal functions. Let FTIME(T (m))
be the set of functions computable by a Turing
machine in T (m) time.[10]

9The exact multiplicative inverse ⟨p, q⟩�→⟨q, p⟩ over
unconstrained rationals is also size-preserving. Thus, neural
networks are size preserving over both floats and rationals.
10We write FTIME(m) instead of the conventional
FTIME(n) to avoid confusion with the sequence length n.


-----

**Definition 10 Let AHAT(D, T** (n)) be the class of
languages L ⊆ Σ[∗] such that there exists a transformer ⟨Σ, D, α, L, H, φ, sℓ,h, fℓ⟩ that recognizes
, where φ runs in time linear in the sequence
_L_
length n, and sℓ,h, fℓ _∈_ FTIME(T (m)).

For any T (m) ⩾ _m, we will show transform-_
ers AHAT(D, T (m)) have the complexity of their
activation functions. Formally:

**Theorem 2** (Formal version of Thm. 2) _For_
D ∈{F, Q} and T (m) ⩾ _m,_

AHAT(D, T (m)) ⊆ TIME(T (m)).

_Proof. First, observe that AHAT(D, T_ (m)) _⊆_
TIME(T (m)), since the embedding function and
saturated attention can be computed in time linear
in the input sequence length, and the other internal functions can be computed in FTIME(T (m))
by construction.
We now show TIME(m) ⊆ AHAT(D, T (m)).
We adopt a 1-layer transformer construction, and
thus omit ℓ, h subscripts.We define three components of the embedding function φ : Σ _×_ N → D[3]:


1. Rationals: If D = Q, then u is the binary
string w. Any TIME(T (m)) has an in_L ∈_
dicator function δ FTIME(T (m)), which
_∈_
we now apply to recognize whether w .
_∈L_

2. Floats: If D = F, then u = 2[|][n][|]/n · w as in

§A. Therefore, in linear time, we compute


_φ(wi, i)1 =_


�
2[i][−][1] if wi = 1
0 otherwise


_φ(wi, i)2 = i_

_φ(wi, i)3 = 2[|][i][|]._

Each of these components is computable in time
linear in n. Define three heads b1,i, b2,i, b3,i.
Without loss of generality, consider bh,i to act on
_φ(wi, i)h alone, rather than the full embeding vec-_
tor. b1,i is defined as a uniform head, while b2,i
and b3,i are computed with sh(vi, vj) = vj. Thus,


_b1,i = [1]_

_n_

_b2,i = n_


�

2[j][−][1]
_wj_ =1


_b3,i = 2[|][n][|]._

Finally, we discuss how to set f to compute
whether w . Let p be the function that ex_∈L_
tracts the numerator of a float or rational number,
which is computable in O(m) time on float of size
_m. Within f_, we compute u = p(b1,i). At this
point, we proceed in two cases depending on the
datatype D:


_b2,i_ _n_
_u =_ = w,

_·_
_b3,i_ 2[|][n][|][ ·][ 2][|][n]n[|][w]

and feed w through δ as in the D = Q case.
So, TIME(T (m)) ⊆ AHAT(D, T (m)).

### D Proof from Hao et al. (2022)

The proof for Lem. 2 largely follows the proof of
a core lemma of Hao et al. (2022). We reproduce a
slightly adapted version of their proof here, since
their manuscript is not yet publicly available, and
we wish for our paper to be self-contained.

**Lemma 2** Any function f : 0, 1 0, 1
_{_ _}[c]_ _→{_ _}[d]_

can be computed by a boolean circuit of depth 3
and size at most d(2[c] + c + 1).

_Proof. The idea of the proof is to define d subcir-_
cuits of size at most 2[c] + c + 1 that compute the d
output bits of f in parallel. We will build a circuit
that computes each output bit of f according to its
representation in disjunctive normal form (DNF).
We define a first layer of the circuit that computes
the negation of each input, which takes c gates.
The second layer then computes the value of each
DNF term by computing a conjunction ( gate)
_∧_
over the corresponding literals or negated literals.
Note that a formula of c variables has at most 2[c]

DNF terms. Finally, the third layer of the circuit
computes a disjunction ( gate) over the values
_∨_
of all terms, yielding the output of f, and adding
a single gate. In summary, we have shown how
to compute each output bit with a circuit of size
at most 2[c] + c + 1, which implies the full function f can be computed by a circuit of size at most
_d(2[c]_ + c + 1).

### References

Afra Alishahi, Yonatan Belinkov, Grzegorz Chrupała, Dieuwke Hupkes, Yuval Pinter, and Has[san Sajjad, editors. 2020. Proceedings of the](https://www.aclweb.org/anthology/2020.blackboxnlp-1.0)
_[Third BlackboxNLP Workshop on Analyzing](https://www.aclweb.org/anthology/2020.blackboxnlp-1.0)_
_[and Interpreting Neural Networks for NLP. As-](https://www.aclweb.org/anthology/2020.blackboxnlp-1.0)_
sociation for Computational Linguistics, Online.


-----

[Sanjeev Arora and Boaz Barak. 2009. Computa-](https://books.google.com/books/about/Computational_Complexity.html?id=8Wjqvsoo48MC)
_[tional Complexity: A Modern Approach. Cam-](https://books.google.com/books/about/Computational_Complexity.html?id=8Wjqvsoo48MC)_
bridge University Press.

Jimmy Ba, Jamie Ryan Kiros, and Geoffrey E.
Hinton. 2016. Layer normalization. _ArXiv,_
abs/1607.06450.

Satwik Bhattamishra, Kabir Ahuja, and Navin
[Goyal. 2020. On the ability and limitations of](https://doi.org/10.18653/v1/2020.emnlp-main.576)
[transformers to recognize formal languages. In](https://doi.org/10.18653/v1/2020.emnlp-main.576)
_Proceedings of the 2020 Conference on Empir-_
_ical Methods in Natural Language Processing_
_(EMNLP), pages 7096–7116, Online. Associa-_
tion for Computational Linguistics.

[Liliana Cojocaru. 2016. Advanced Studies on the](https://trepo.tuni.fi/bitstream/handle/10024/99577/978-952-03-0184-2.pdf)
_[Complexity of Formal Languages. Ph.D. thesis,](https://trepo.tuni.fi/bitstream/handle/10024/99577/978-952-03-0184-2.pdf)_
University of Tampere.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
[Kristina Toutanova. 2019. BERT: Pre-training](https://doi.org/10.18653/v1/N19-1423)
[of deep bidirectional transformers for language](https://doi.org/10.18653/v1/N19-1423)
[understanding.](https://doi.org/10.18653/v1/N19-1423) In Proceedings of the 2019
_Conference of the North American Chapter_
_of the Association for Computational Linguis-_
_tics: Human Language Technologies, Volume_
_1 (Long and Short Papers), pages 4171–4186,_
Minneapolis, Minnesota. Association for Computational Linguistics.

Peter van Emde Boas. 1991. Machine Models and
_Simulations, chapter 1. MIT Press, Cambridge,_
MA, USA.

Merrick Furst, James B. Saxe, and Michael Sipser.
[1981. Parity, circuits, and the polynomial-time](https://doi.org/10.1109/SFCS.1981.35)
[hierarchy. In Proceedings of the 22nd Annual](https://doi.org/10.1109/SFCS.1981.35)
_Symposium on Foundations of Computer Sci-_
_ence, SFCS ’81, page 260–270, USA. IEEE_
Computer Society.

Larry J Goldstein. 1973. A history of the prime
number theorem. The American Mathematical
_Monthly, 80(6):599–615._

Michael Hahn. 2020. [Theoretical limitations](https://www.aclweb.org/anthology/2020.tacl-1.11)
[of self-attention in neural sequence models.](https://www.aclweb.org/anthology/2020.tacl-1.11)
_Transactions of the Association for Computa-_
_tional Linguistics, 8:156–171._

Yiding Hao, Dana Angluin, and Robert Frank.
2022. Hard attention transformers and constant
depth circuits. Unpublished manuscript.


John Hewitt, Michael Hahn, Surya Ganguli,
Percy Liang, and Christopher D. Manning.
[2020. RNNs can generate bounded hierarchi-](https://doi.org/10.18653/v1/2020.emnlp-main.156)
[cal languages with optimal memory.](https://doi.org/10.18653/v1/2020.emnlp-main.156) In Pro_ceedings of the 2020 Conference on Empiri-_
_cal Methods in Natural Language Processing_
_(EMNLP), pages 1978–2010, Online. Associa-_
tion for Computational Linguistics.

David S. Johnson. 1991. A Catalog of Complexity
_Classes, chapter 2. MIT Press, Cambridge, MA,_
USA.

[Neeraj Kayal. 2015. Lecture notes for topics in](https://www.csa.iisc.ac.in/~chandan/courses/arithmetic_circuits/notes/lec5.pdf)
[complexity theory.](https://www.csa.iisc.ac.in/~chandan/courses/arithmetic_circuits/notes/lec5.pdf)

[Meena Mahajan. 2007. Polynomial size log depth](https://www.uni-ulm.de/fileadmin/website_uni_ulm/iui.inst.190/Mitarbeiter/toran/beatcs/column91.ps)
[circuits: Between NC[1]](https://www.uni-ulm.de/fileadmin/website_uni_ulm/iui.inst.190/Mitarbeiter/toran/beatcs/column91.ps) and AC[1]. Bulletin of the
_EATCS, 91:42–56._

William Merrill. 2019. [Sequential neural net-](https://doi.org/10.18653/v1/W19-3901)
[works as automata. In Proceedings of the Work-](https://doi.org/10.18653/v1/W19-3901)
_shop on Deep Learning and Formal Languages:_
_Building Bridges, pages 1–13, Florence. Asso-_
ciation for Computational Linguistics.

William Merrill. 2021. [Formal language theory](https://arxiv.org/pdf/2102.10094.pdf)
[meets modern NLP. ArXiv, abs/2102.10094.](https://arxiv.org/pdf/2102.10094.pdf)

William Merrill, Vivek Ramanujan, Yoav Goldberg, Roy Schwartz, and Noah A. Smith. 2021.
[Effects of parameter norm growth during trans-](https://doi.org/10.18653/v1/2021.emnlp-main.133)
[former training: Inductive bias from gradient](https://doi.org/10.18653/v1/2021.emnlp-main.133)
[descent.](https://doi.org/10.18653/v1/2021.emnlp-main.133) In Proceedings of the 2021 Confer_ence on Empirical Methods in Natural Lan-_
_guage Processing, pages 1766–1781, Online_
and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Hao Peng, Roy Schwartz, Sam Thomson, and
[Noah A. Smith. 2018. Rational recurrences. In](https://doi.org/10.18653/v1/D18-1152)
_Proceedings of the 2018 Conference on Empir-_
_ical Methods in Natural Language Processing,_
pages 1203–1214, Brussels, Belgium. Association for Computational Linguistics.

Jorge Pérez, Javier Marinkovi´c, and Pablo
[Barceló. 2019. On the Turing completeness of](https://openreview.net/forum?id=HyGBdo0qFm)
[modern neural network architectures. In Inter-](https://openreview.net/forum?id=HyGBdo0qFm)
_national Conference on Learning Representa-_
_tions._

Ashish Vaswani, Noam Shazeer, Niki Parmar,
Jakob Uszkoreit, Llion Jones, Aidan N Gomez,


-----

[Ł ukasz Kaiser, and Illia Polosukhin. 2017. At-](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
[tention is all you need.](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) In Advances in Neu_ral Information Processing Systems, volume 30._
Curran Associates, Inc.

Gail Weiss, Yoav Goldberg, and Eran Yahav. 2018.

[On the practical computational power of finite](https://doi.org/10.18653/v1/P18-2117)
[precision RNNs for language recognition.](https://doi.org/10.18653/v1/P18-2117) In
_Proceedings of the 56th Annual Meeting of the_
_Association for Computational Linguistics (Vol-_
_ume 2: Short Papers), pages 740–745, Mel-_
bourne, Australia. Association for Computational Linguistics.

Gail Weiss, Yoav Goldberg, and Eran Yahav.
2021. [Thinking like transformers.](https://arxiv.org/pdf/2106.06981.pdf) _ArXiv,_
abs/2106.06981.

[Andrew C.-C. Yao. 1990. On ACC and thresh-](https://doi.org/10.1109/FSCS.1990.89583)
[old circuits. In Proceedings of the 31st Annual](https://doi.org/10.1109/FSCS.1990.89583)
_Symposium on Foundations of Computer Sci-_
_ence, pages 619–627 vol.2._

Shunyu Yao, Binghui Peng, Christos Papadimitriou, and Karthik Narasimhan. 2021. [Self-](https://doi.org/10.18653/v1/2021.acl-long.292)
[attention networks can process bounded hierar-](https://doi.org/10.18653/v1/2021.acl-long.292)
[chical languages.](https://doi.org/10.18653/v1/2021.acl-long.292) In Proceedings of the 59th
_Annual Meeting of the Association for Compu-_
_tational Linguistics and the 11th International_
_Joint Conference on Natural Language Pro-_
_cessing (Volume 1: Long Papers), pages 3770–_
3785, Online. Association for Computational
Linguistics.

Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh
Rawat, Sashank Reddi, and Sanjiv Kumar.
2020. [Are transformers universal approxima-](https://openreview.net/forum?id=ByxRM0Ntvr)
[tors of sequence-to-sequence functions? In In-](https://openreview.net/forum?id=ByxRM0Ntvr)
_ternational Conference on Learning Represen-_
_tations._


-----

