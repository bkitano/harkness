## On the Linguistic Capacity of Real-Time Counter Automata

William Merrill
willm@allenai.org


Allen Institute for AI, Seattle WA 98103


Abstract. Counter machines have achieved a newfound relevance to
the field of natural language processing (NLP): recent work suggests
some strong-performing recurrent neural networks utilize their memory
as counters. Thus, one potential way to understand the success of these
networks is to revisit the theory of counter computation. Therefore, we
study the abilities of real-time counter machines as formal grammars,
focusing on formal properties that are relevant for NLP models. We first
show that several variants of the counter machine converge to express the
same class of formal languages. We also prove that counter languages are
closed under complement, union, intersection, and many other common
set operations. Next, we show that counter machines cannot evaluate
boolean expressions, even though they can weakly validate their syntax.
This has implications for the interpretability and evaluation of neural
network systems: successfully matching syntactic patterns does not guarantee that counter memory accurately encodes compositional semantics.
Finally, we consider whether counter languages are semilinear. This work
makes general contributions to the theory of formal languages that are
of potential interest for understanding recurrent neural networks.

### 1 Introduction


It is often taken for granted that modeling natural language syntax well requires
a grammar formalism sensitive to compositional structure. Early work in linguistics established that finite-state models are insufficient for describing the
dependencies in natural language data [2]. Instead, a formalism capable of expressing the relations in terms of hierarchical constituents ought to be necessary.
Recent advances in deep learning and NLP, however, challenge this longheld belief. Neural network formalisms like the long short-term memory network
(LSTM) [5] perform fairly well on tasks requiring structure sensitivity [7], even
though it is not obvious that they have the capacity or bias to represent hierarchy.
This mismatch raises interesting questions for both linguists and practitioners of
NLP. It is unclear what about the LSTM architecture might lend itself towards
good linguistic representations, and under what conditions these representations
might fall short of grasping the structure and meaning of language.
Recent work has suggested that the practical learnable capacity of LSTMs
resembles that of counter machines [8][10] [12]. Theoretically, this connection is


-----

2 William Merrill willm@allenai.org

motivated by studying the “saturated” version [8] of the LSTM network, i.e.
replacing each continuous activation function with a step function. Under these
conditions, the LSTM reduces to a discrete automaton that uses its memory cell
as integer-valued counter registers. [12] define a simplified class of counter languages that falls within the expressive capacity of this saturated LSTM model.
On the other hand, a more general class of counter languages is an upper bound
on the expressive capacity of saturated LSTMs [8]. Thus, there is a strong theoretical connection between LSTMs and counter automata.
Furthermore, these theoretical results for saturated LSTMs seem to predict
what classes of formal languages LSTMs can empirically learn. [12] show how
LSTMs learn to model languages like a[n]b[n] by using their memory to count n,
whereas other recurrent neural network architectures without saturated counting
abilities fail. Similarly, [8] shows how LSTMs cannot reverse strings, just like realtime counter automata [4]. Further, LSTMs can flawlessly model 1-Dyck strings
by using their memory to count [10], but, like counter automata, they cannot
model 2-Dyck [11]. It seems that, where LSTMs succeed at algorithmic tasks,
they do so by counting, and where they fail, their failure might be explained by
their inability to reliably implement more complex types of memory.
Inspired by the connection of LSTMs to counter automata, we study the formal properties of counter machines as language recognizers. We do this with the
hope of understanding the abilities of counter-structured memory, and to what
degree it has computational properties well-suited for representing compositional
structure. The contributions of this paper are as follows:

– We prove that several interesting counter machine variants converge to the
same linguistic capacity, whereas simplified counter machines [12] are strictly
weaker than classical counter machines.
– We demonstrate that counter languages are closed under complement, union,
intersection, and many other common operations.
– We show counter machines cannot evaluate compositional boolean expressions, even though they can check whether such expressions are well-formed.
– We prove that a certain subclass of the counter languages are semilinear,
and conjecture that this result holds for all counter languages.

### 2 Definitions

Informally, we can think of counter automata as finite-state automata that have
been augmented by a finite number of integer-valued counters. While processing
a string, the machine can update the values of the counters, and the counters
can in turn inform the machine’s state transitions.
Early results in theoretical computer science established that a 2-counter
machine with unbounded computation time is Turing-complete [3]. However,
restricting computation to be real-time (i.e. one iteration of computation per
input) severely limits the counter machine’s computational capacity [4]. A similar
fact holds for recurrent neural networks like LSTMs [12]. We study the language
recognition abilities of several types of real-time counter automata.


-----

On the Linguistic Capacity of Real-Time Counter Automata 3

2.1 General Counter Machines

The first counter automaton we introduce is the general counter machine. This
machine manipulates its counters by adding or subtracting from them. Later,
we define other variants of this general automaton. For m ∈ Z, let ±m denote
the function λx.x ± m. Let ×0 denote the constant zero function λx.0.

Definition 1 (General counter machine [4]). A k-counter machine is a
tuple ⟨Σ, Q, q0, u, δ, F ⟩ with

1. A finite alphabet Σ
2. A finite set of states Q[1]

3. An initial state q0
4. A counter update function

k
u : Σ × Q × {0, 1}[k] → �{+m : m ∈ Z} ∪{×0}�

5. A state transition function

δ : Σ × Q × {0, 1}[k] → Q

6. An acceptance mask
F ⊆ Q × {0, 1}[k]

A machine processes an input string x one token at a time. For each token,
we use u to update the counters and δ to update the state according to the
current input token, the current state, and a finite mask of the current counter
values. We formalize this in Definition 2.
For a vector v, let z(v) to denote the broadcasted “zero-check” function, i.e.


z(v)i =


�0 if vi = 0
(1)
1 otherwise.


Definition 2 (Counter machine computation). Let ⟨q, c⟩∈ Q × Z[k] be a
configuration of machine M . Upon reading input xt ∈ Σ, we define the transition

⟨q, c⟩→xt ⟨δ(xt, q, z(c)), u(xt, q, z(c))(c)⟩.

Definition 3 (Real-time acceptance). For any string x ∈ Σ[∗] with length n,
a counter machine accepts x if there exist states q1, .., qn and counter configurations c1, .., cn such that

⟨q0, 0⟩→x1 ⟨q1, c1⟩→x2 .. →xn ⟨qn, cn⟩∈ F.

Definition 4 (Real-time language acceptance). A counter machines accepts a language L if, for each x ∈ Σ[∗], it accepts x iff x ∈ L.


-----

4 William Merrill willm@allenai.org

a/+1


b/−1


a, b/+0


start q0 q1 q2

Fig. 1. A graphical representation of a 1-counter machine that accepts {a[n]b[n] | n ∈ N}
if we set F to verify that the counter is 0 and we are in either q0 or q1.

⟨0, q0⟩→a ⟨1, q0⟩→a ⟨2, q0⟩→b ⟨1, q1⟩→b ⟨0, q0⟩∈ F

⟨0, q0⟩→a ⟨1, q0⟩→a ⟨2, q0⟩→b ⟨1, q1⟩→a ⟨1, q2⟩ ∈/ F

Fig. 2. Behavior of the counter machine in Figure 1 on aabb (top) and aaba (bottom).

We denote the set of languages acceptable in real time by a general counter
machine as CL. We will use the terms “accept” and “decide” interchangeably,
as accepting and deciding a language are equivalent for real-time automata.
Unlike context-free (CF) grammars, general counter machines cannot accept
palindromes [4]. However, they can accept non-CF languages like a[n]b[n]c[n]d[n] [4].
Thus, CL does not fall neatly into the classical Chomsky hierarchy.

2.2 Restricted Counter Machines

Now, we can can consider various restrictions of the general counter machine,
and the corresponding classes of languages acceptable by such automata.
First, we present the simplified counter machine [12]. The counter update
function in the simplified counter machine has two important constraints compared to the general machine. First, it can only be conditioned by the input
symbol at each time step. Second, it can only increment or decrement its counters instead of being able to add or subtract arbitrary constants.

Definition 5 (Simplified counter machine). A counter machine is simplified if u has the form

u : Σ →{−1, +0, +1, ×0}[k].

Another variant that we consider is the incremental counter machine. The
arguments to the update function of this machine are not restricted, but the
additive operations are constrained to ±1.

Definition 6 (Incremental counter machine). An counter machine is incremental if u has the form

u : Σ × Q × {0, 1}[k] →{−1, +0, +1, ×0}[k].

1 The original definition [4] distinguishes between “autonomous” and “polling” states,
a distinction that is vacuous in the real-time case we are studying.


-----

On the Linguistic Capacity of Real-Time Counter Automata 5

Finally, we define a stateless variant of the counter machine. Removing state
from the counter machine is equivalent to allowing it to only have one state q0.

Definition 7 (Stateless counter machine). A counter machine is stateless
if Q = {q0}.

2.3 Saturated LSTMs

The LSTM is a recurrent neural network resembling a counter machine. At each
step, a vector encoding of the input xt is used to update the state vectors ct, ht
and produce an acceptance decision yt. Let 1+ denote the function that returns 1
for positive reals and 0 otherwise. Similarly, let sgn return 1 for positive reals and
−1 otherwise. Let ⊙ be elementwise multiplication over vectors. The saturated
LSTM’s recurrent update [8], parameterized by weight tensors W and b, is:

ft = 1+(W[f] xt + U[f] ht−1) (2)

it = 1+(W[i]xt + U[i]ht−1) (3)
ot = 1+(W[o]xt + U[o]ht−1) (4)

˜ct = sgn(W[c]xt + U[c]ht−1) (5)
ct = ft ⊙ ct−1 + it ⊙ ˜ct (6)

ht = ot ⊙ ct (7)
yt = 1+(w[y]             - ht + b[y]). (8)

We say the LSTM accepts iff yt = 1. In practice, (7) is often ot ⊙ tanh(ct).
We remove the tanh for clarity, as its monotonicity does not change the expressiveness of the saturated network. These equations specify a discrete automaton
that is highly similar to a counter machine [12][8].
The major difference between the saturated LSTM and the classical counter
machines is that the LSTM partitions the counter values by passing them through
a linear map and applying a thresholding function, whereas the classical counter
machines probes whether or not the counters are zero. For example, for a counter
c, the saturated LSTM could test c ≤ 5, whereas the general counter machine
can only test c = 0. Motivated by this, we define the threshold counter machine,
which views its counters by thresholding them instead of testing equality to 0.

Definition 8 (Threshold counter machine). A threshold counter machine
is a general counter machine where all occurrences of the zero-check function z
are redefined as predicates of the form λm.c ≤ m for m ∈ Z. We refer to such a
function by the shorthand ≤m.

### 3 Counter Language Hierarchy

3.1 Simplified Counter Languages

Our first result relating counter classes is to show that the simplified counter
languages are a proper subset of the general counter languages. The weakness


-----

6 William Merrill willm@allenai.org

of the simplified machine is that the update function is conditioned only by
the input symbol. Thus, languages like a[m]b[2][m], which require switching counting
behavior, cannot be decided correctly. We formalize this in Theorem 1.

Theorem 1 (Weakness of SCL). Let SCL be the set of languages acceptable
in real time by a simplified counter machine. Then SCL ⊊ CL.

Proof. Consider the language a[m]b[2][m]. This is trivially acceptable by a 1-counter
machine that adds 2 for each a and subtracts 1 for each b. On the other hand, we
shall show that it cannot be accepted by any simplified machine. Assume by way
of contradiction that such a simplified machine M exists. We assume without
loss of generality that M does not apply a ×0 update, as doing so would erase
all information about the prefix.
Tracking the ratio between a’s and b’s requires infinite state. Thus, the counters of M, as opposed to the finite state, must encode whether 2m = l for strings
of the form a[m]b[l]. Let c be the value of some counter in M . We can decompose
c into the update contributed by a’s and the the update contributed by b’s:

c = mua + lub, (9)
ua, ub ∈{−1, 0, 1}. (10)

Exhausting all the possible functions that c can compute, we get

c ∈{0, ±m, ±l, ±(m + l), ±(m − l)} (11)
z(c) ∈{0, 1m>0, 1l>0, 1m+l>0, 1m−l̸=0}. (12)

We ignore the first four options for z(c), as they do not relate m to l. The final
option tests m/l = 1, not 2. Thus, z(c) cannot test whether 2m = l.

Note that this argument breaks down if the counter update can depend on
the state. In that case, we can build a machine that has two counters and two
states: q0 adds 1 to the first counter while it reads a, and then decrements the
first counter and increments the second counter when it reads b. When the first
counter is empty and the second counter is not empty, q0 transitions to q1, which
decrements the second counter. We accept iff both counters are 0 after xn.

3.2 Incremental Counter Languages

Unlike the simplified counter machine, the incremental machine has the same
linguistic capacity as the general machine. We can simulate each counter on a
general machine with a finite amount of overhead. This provides a reduction
from general to incremental machines.

Theorem 2 (Generality of ICL). Let ICL be the set of languages acceptable
in real time by an incremental counter machine. Then ICL = CL.


-----

On the Linguistic Capacity of Real-Time Counter Automata 7

Proof. Let d be the maximum that is ever added or subtracted from a counter c
in M . We simulate c in M [′] using a counter c[′] and a value q ∈ Z mod d encoded
in finite state. We will implement a “ring-counter” encoding of c such that

c[′] = ⌊c/d⌋
q = c mod d.

To simulate a ×0 update on c, we apply ×0 to c[′], and transition state such
that q := 0. To simulate a +m update on c for some m ∈ Z, we first change
state such that q := (q + m) mod d. Next, we apply the following update to c[′]:

+1 if q + m ≥ d


−1 if q + m < 0 (13)

+0 otherwise.


We can compute z(c) by checking whether z(c[′]) = 0 and q = 0.

3.3 Stateless Counter Languages

Similarly, restricting a counter machine to be stateless does not weaken its expressive capacity. We show how to reduce an arbitrary stateful machine to a
stateless machine that has been augmented with additional counters. The key
idea here is that we can use the additional counters as a one-hot vector that
tracks the state of the original machine.

Theorem 3 (Generality of QCL[˜] ). Let QCL[˜] be the set of languages acceptable
in real time by a stateless counter machine. Then QCL = CL[˜] .

Proof. We define a new stateless machine M [′] to simulate M by adding a |Q|length vector of counters called q[′]. Let ω(i) denote the |Q|-length one-hot vector
encoding i, i.e. [ω(i)]i = 1, and all other indices are 0. We consider ω(0) = 0.
At initialization, q[′] encodes the initial state since q[′] = 0 = ω(0). Furthermore, we define the invariant that, at any given time, q[′] = ω(i) for some state
i. Thus, the additional counters now encode the current state.
Let x∥y denote the concatenation of vectors x and y. We define the new
acceptance mask in M [′] as

F [′] = {⟨q0, b∥ω(i)⟩| ⟨qi, b⟩∈ F }. (14)

We can update the counters inherited from M analogously to (14). The last step
is to properly update the state counters q[′]. For each transition δ(xt, qi, b) = qj
in M, we update q[′] by adding −ω(i) + ω(j). This ensures q[′] is correct since

ω(i) + � − ω(i) + ω(j)� = ω(j). (15)


-----

8 William Merrill willm@allenai.org

3.4 Threshold Counter Languages

We show that the threshold counter languages are equivalent to the general
counter languages. As thresholding is a key capability of the saturated LSTM
formalism, this suggests that much of the LSTM capacity falls within the general
counter languages, although it does not provably establish containment.

Theorem 4 (Generality of ΘCL). Let ΘCL be the languages acceptable in
real time by a threshold counter machine. Then ΘCL = CL.

Proof. Given the ability to check ≤m on the counters for any m, we can simulate =0 by checking both ≤−1 and ≤0. Thus, CL ⊆ ΘCL. To prove the other
direction, we show how to simulate applying ≤m to the counters using only =0.
Assume without loss of generality that only one threshold check m applies
to each counter c (we can create copies of a counter and distribute the threshold
checks over them if this is not the case), and that m > 0. We implement a ringcounter construction similar to the one used in Theorem 2, representing c with
a new counter c[′] = ⌊c/m⌋ and finite-state component q = c mod m. We also
store the sign of c in finite state by recording whenever both c[′] and q pass zero.
Having all this information, we conclude c ≤ m iff the sign is negative or c[′] = 0.

The construction in Theorem 4 can be directly adapted to show that a general
counter machine can simulate checking =m in addition to =0.

3.5 Summary

The general counter machine, incremental counter machine, stateless counter
machine, and threshold counter machine all converge to the same linguistic capacity, which we call CL. The simplified counter machine [12], however, has a
linguistic capacity SCL that is strictly weaker than CL.

### 4 Closure Properties

Another way to understand the counter languages is through their closure properties. It turns out that the real-time counter languages are closed under a wide
array of common operations, including complement, intersection, union, set difference, and symmetric set difference. The general result in Theorem 5 implies
these closure properties, as well as many others.

Theorem 5 (General set operation closure). Let P be an m-ary operation
over languages. If there exists an m-ary boolean function p such that

1P (L1,..,Lm)(x) = p�1L1(x), .., 1Lm (x)�,

then CL and SCL are both closed under P .

Proof. First, we construct counter machines M1, .., Mm that decide the counter
languages L1, .., Lm. We define a new machine M [′] that, on input x, simulates
M1, .., Mm in parallel, and accepts if and only if

p(M1(x), .., Mm(x)) = 1. (16)


-----

On the Linguistic Capacity of Real-Time Counter Automata 9

Corollaries. Let Λ be a placeholder for either CL or SCL. Let L1, L2 ∈ Λ. By
Theorem 5, Λ is closed under the following operations:

Σ[∗] \ L1 (17)

L1 ∩ L2 (18)
L1 ∪ L2 (19)

L1 \ L2 (20)
(L1 \ L2) ∪ (L2 \ L1). (21)

### 5 Compositional Expressions

We now study the abilities of counter machines on the language Lm (Definition 9).
Like natural language, Lm has a deep structure consisting of recursively nested
hierarchical constituents.

Definition 9 (Lm [4]). For any m, let Lm be the language generated by:

<exp> -> <VALUE>
<exp> -> <UNARY> <exp>
<exp> -> <BINARY> <exp> <exp>
..
<exp> -> <m-ARY> <exp> .. <exp>

Surprisingly, even a 1-counter machines can decide Lm in real time by implementing Algorithm 1 [4]. Algorithm 1 uses a counter to keep track of the depth
at any given index. If the depth counter reaches −1 at the end of the string,
the machine has verified that the string is well-formed. We define the arity of a
<VALUE> as 0, and the arity of an <m-ARY> operation as m.

Algorithm 1 Deciding Lm [4]

1: procedure Decide(x)
2: c ← 0
3: for each xt ∈ x do
4: c ← c + Arity(xt) − 1
5: return c = −1

5.1 Semantic Evaluation as Structure Sensitivity

While Algorithm 1 decides Lm, it is agnostic to the deep structure of the input
in that it does not represent the hierarchical dependencies between tokens. This
means that it could not be used to evaluate these expressions. Based on this
observation, we prove that no counter machine can evaluate boolean expressions


-----

10 William Merrill willm@allenai.org

due to the deep structural sensitivity that semantic evaluation (as opposed to
syntactic acceptance) requires. We view boolean evaluation as a simpler formal
analogy to evaluating the compositional semantics of natural language.
To be more formal, consider an instance of L2 with values {0, 1} and binary
operations {∧, ∨}. We assign the following semantics to the terminals:

[[0]] = 0 [[1]] = 1 (22)

[[∧]] = λpq. p ∧ q (23)

[[∨]] = λpq. p ∨ q. (24)

Our semantics evaluates each nonterminal by applying the denotation of each
syntactic argument to the semantic arguments of the operation. For example:

[[∨01]] = [[∨]]( [[0]], [[1]]) = 0 ∨ 1 = 1. (25)

We also define semantics for non-constituent prefixes via function composition:

[[∨∨]] = [[∨]] ◦ [[∨]] = λpqr. p ∨ q ∨ r. (26)

We define the language B as the set of valid strings x where [[x]] = 1.

Theorem 6 (Weak evaluation). B /∈ CL.

Proof. Assume by way of contradiction that there exists a counter machine deciding B. We consider an input x that contains a prefix of p operators followed by
a suffix of p +1 values. For the machine to evaluate x correctly, the configuration
after xp must encode which boolean function xp specifies.
However, a counter machine with k counters only has O(p[k]) configurations
after reading p characters. We show by induction over p that an p-length prefix
of operators can encode ≥ 2[p] boolean functions. Since the machine does not have
enough configurations to encode all the possibilities, we reach a contradiction.

Base Case. With p = 0, we have a null prefix followed by one value that determines [[x]]. We can represent exactly 1 (2[0]) function, which is the identity.

Inductive Case. The expression has a prefix of operators x1:p+1 followed by
values xp+2:2p+3. We decompose the semantics of the full expression to

[[x]] = [[x1]]( [[x2:2p+2]], [[x2p+3]]). (27)

Since [[x2:2p+2]] has a prefix of p operators, we apply the inductive assumption
to show it can represent ≥ 2[p] boolean functions. Define f as the composition
of [[x1]] with [[x2:2p+2]]. There are two possible values for f : f∧, obtained when
x1 = ∧, and f∨, obtained when x1 = ∨. We complete the proof by verifying that
f∧ and f∨ are necessarily different functions.
To do this, consider the minimal sequence of values that will satisfy them
according to a right-to-left ordering of the sequences. For f∧, this minimal sequence ends in 1, whereas for f∨ it must end in a 0. Therefore, f represents
at least two unique functions for each value of [[x2:2p+2]]. Thus, a p + 1-length
sequence of prefixes can encode ≥ 2 · 2[p] = 2[p][+1] boolean functions.


-----

On the Linguistic Capacity of Real-Time Counter Automata 11

Theorem 6 shows how counter machines cannot represent certain hierarchical
dependencies, even when the generated language is within the counter machine’s
weak expressive capacity. This is analogous to how CFGs can weakly generate
Dutch center embedding [9], even though they cannot assign the correct crossserial dependencies between subjects and verbs [1]. Thus, while counter memory
can track certain formal properties of compositional languages, it cannot represent the underlying hierarchical structure in a deep way.

### 6 Semilinearity

Semilinearity is a condition that has been proposed as a desired property for
any formalism of natural language syntax [6]. Intuitively, semilinearity ensures
that the set of string lengths in a language is not unnaturally sparse. Regular,
context-free, and a variety of mildly context-sensitive languages are known to
be semilinear [6]. The semilinearity of CL is an interesting open question for
understanding the abilities of counter machines as grammars.

6.1 Definition

We first define semilinearity over sets of vectors before considering languages.
To start, we introduce the notion of a linear set:

Definition 10 (Linear set). A set S ⊆ N[k] is linear if there exist W ∈ N[k][×][m]

and b ∈ N[k] such that

S = {n ∈ N[m] | Wn + b = 0} .

Semilinearity, then, is a weaker condition that specifies that a set is made up
of a finite number of linear components:

Definition 11 (Semilinear set). A set S ⊆ N[k] is semilinear if it is the finite
union of linear sets.

To apply this definition to a language L, we translate each string x ∈ L into
a vector by taking Ψ (x), the Parikh mapping of x. The Parikh mapping of a
sentence is its “bag of tokens” representation. For example, the Parikh mapping
of abaa with respect to Σ = {a, b} is ⟨3, 1⟩. We say that a language L is semilinear
if its image under Ψ, i.e. {Ψ (x) | x ∈ L}, is semilinear.

6.2 Semilinearity of Counter Languages

We do not prove that the general counter languages are semilinear, but we do
prove it for a dramatically restricted subclass of the counter languages. Define
˜QSCL as the set of language acceptable by a counter machine that is both simplified (Definition 5) and stateless (Definition 7). QSCL is indeed semilinear.[˜]

Theorem 7 (Semilinearity of QSCL[˜] ). For all L ∈ QSCL[˜], L is semilinear.


-----

12 William Merrill willm@allenai.org

Proof. Applying the definition of counter machine acceptance, we express L as


L = � {x | c(x) = b} = �

b∈F b∈F


k
�{x | ci(x) = bi}. (28)

i=1


Semilinear languages are closed under finite union and intersection, so we
just need to show {x | ci(x) = bi} is semilinear. We apply the following trick:

{x | ci(x) = bi} = Σ[∗]∥Zi∥Li (29)

where Zi is the set of all tokens that set counter i to 0, and Li is the set of
suffixes after the last occurence of some token in Zi. Since semilinear languages
are closed under concatenation, and Σ[∗] and the finite language Zi are trivially
semilinear, we just need to show that Li is semilinear. Counter i cannot be set
to zero on strings of Li, so we can write


bi = ci(x) =


n
� ui(xt) = � ui(σ)#σ(x) = ui · Ψ (x) (30)

t=1 σ∈Σ


where ui denotes the vector of possible updates to counter i where each index
corresponds to a different σ ∈ Σ. So, Li is the linear language

Li = {x ∈ Σ[∗] | ui · Ψ (x) − bi = 0}. (31)

Although the proof of Theorem 7 is nontrivial, QSCL is a weak class. Such[˜]
languages have limited ability to even detect the relative order of tokens in a
string. We hope the proof might be extended to show SCL or CL is semilinear.

### 7 Conclusion

We have shown that many variants of the counter machine converge to express
the same class of formal languages, which supports that CL is a robustly defined
class. The variations we explored move the classical general counter machine

[4] closer to the LSTM in form without changing its expressive power. We also
proved real-time counter languages are closed under a large number of common
set operations, providing tools for future work investigating counter automata.
We also showed that counter automata are incapable of evaluating boolean
expressions, even though they are capable of verifying that boolean expressions
are syntactically well-formed. This result has a clear parallel in the domain
of natural language: deciding whether a sentence is grammatical is different
than building a sentence’s correct compositional meaning. A general take-away
from our results is that just because a counter machine (or LSTM) is sensitive
to surface patterns in language does not mean it can build correct semantic
representations. Counter memory can be exploited to weakly match patterns
in linguistic data, which might provide the wrong kinds of inductive bias for
achieving sophisticated natural language understanding.
Finally, we asked whether counter languages are semilinear as another way
of studying their power. We concluded that a weak subclass of the counter languages are semilinear, and encourage future work to address the general case.


-----

On the Linguistic Capacity of Real-Time Counter Automata 13

### Acknowledgments

Thanks to Dana Angluin, Robert Frank, Yiding Hao, Roy Schwartz, and Yoav
Goldberg, as well as other members of Computational Linguistics at Yale and
the Allen Institute for AI, for their suggestions on various versions of this work.
Additional thanks to several anonymous reviewers for their exceptional feedback.

### References

1. Bresnan, J., Kaplan, R.M., Peters, S., Zaenen, A.: Cross-serial dependencies in dutch. Linguistic Inquiry 13(4), 613–635 (1982),

[http://www.jstor.org/stable/4178298](http://www.jstor.org/stable/4178298)

2. Chomsky, N.: Three models for the description of language. IRE Transactions on
information theory 2(3), 113–124 (1956)
3. Fischer, P.C.: Turing machines with restricted memory access. Information and
Control 9(4), 364–379 (1966)
4. Fischer, P.C., Meyer, A.R., Rosenberg, A.L.: Counter machines and
counter languages. Mathematical systems theory 2(3), 265–283 (Sep 1968).
[https://doi.org/10.1007/BF01694011, https://doi.org/10.1007/BF01694011](https://doi.org/10.1007/BF01694011)

5. Hochreiter, S., Schmidhuber, J.: Long short-term memory. Neural Computation 9(8), 1735–1780 (1997). [https://doi.org/10.1162/neco.1997.9.8.1735,](https://doi.org/10.1162/neco.1997.9.8.1735)

[https://doi.org/10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)

6. Joshi, A.K., Shanker, K.V., Weir, D.: The convergence of mildly context-sensitive
grammar formalisms. Technical Reports (CIS) p. 539 (1990)
7. Linzen, T., Dupoux, E., Goldberg, Y.: Assessing the ability of LSTMs to
learn syntax-sensitive dependencies. Transactions of the Association for Com[putational Linguistics 4, 521–535 (2016). https://doi.org/10.1162/tacl a 00115,](https://doi.org/10.1162/tacl_a_00115)
[https://www.aclweb.org/anthology/Q16-1037](https://www.aclweb.org/anthology/Q16-1037)

8. Merrill, W.: Sequential neural networks as automata. In: Proceedings of
the Workshop on Deep Learning and Formal Languages: Building Bridges.
pp. 1–13. Association for Computational Linguistics, Florence (Aug 2019),
[https://www.aclweb.org/anthology/W19-3901](https://www.aclweb.org/anthology/W19-3901)

9. Pullum, G.K., Gazdar, G.: Natural languages and context-free languages. Linguis[tics and Philosophy 4(4), 471–504 (1980). https://doi.org/10.1007/BF00360802](https://doi.org/10.1007/BF00360802)
10. Suzgun, M., Belinkov, Y., Shieber, S., Gehrmann, S.: LSTM networks can perform dynamic counting. In: Proceedings of the Workshop on Deep Learning and
Formal Languages: Building Bridges. pp. 44–54. Association for Computational
[Linguistics, Florence (Aug 2019), https://www.aclweb.org/anthology/W19-3905](https://www.aclweb.org/anthology/W19-3905)

11. Suzgun, M., Gehrmann, S., Belinkov, Y., Shieber, S.M.: Memory-augmented recurrent neural networks can learn generalized dyck languages (2019)
12. Weiss, G., Goldberg, Y., Yahav, E.: On the practical computational power of finite precision RNNs for language recognition. CoRR abs/1805.04908 (2018),
[http://arxiv.org/abs/1805.04908](http://arxiv.org/abs/1805.04908)


-----

