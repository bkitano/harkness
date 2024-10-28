## Formal Language Theory Meets Modern NLP

#### William Merrill
```
         willm@allenai.org

```

#### July 28, 2021


### Contents

**1** **Introduction** **2**
1.1 Basic definitions . . . . . . . . . . . . . . . . . . . . . . . . . . . 2

**2** **Regular languages** **3**
2.1 Learning regular languages . . . . . . . . . . . . . . . . . . . . . 5

**3** **Data structures** **5**
3.1 Stacks and counters . . . . . . . . . . . . . . . . . . . . . . . . . 7

3.2 Power of unbounded memory . . . . . . . . . . . . . . . . . . . . 8

3.3 Power of counters . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

3.4 Power of a stack . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

3.5 Nondeterminism . . . . . . . . . . . . . . . . . . . . . . . . . . . 10

3.6 Learning memory-augmented automata . . . . . . . . . . . . . . 10

**4** **Chomsky hierarchy** **11**

**5** **Weighted finite-state automata** **13**
5.1 Hankel matrices . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

5.2 Learning WFAs . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15

**6** **Foundational neural networks results** **16**
6.1 Hypercomputation . . . . . . . . . . . . . . . . . . . . . . . . . . 16

6.2 Turing completeness . . . . . . . . . . . . . . . . . . . . . . . . . 16

6.3 Binarized neurons . . . . . . . . . . . . . . . . . . . . . . . . . . 17

**7** **Saturated networks** **18**
7.1 Saturated networks as automata . . . . . . . . . . . . . . . . . . 19

7.2 Space complexity . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

7.3 Connection to learning . . . . . . . . . . . . . . . . . . . . . . . . 21

**8** **Rational recurrences** **21**


1


-----

**9** **Transformers** **22**
9.1 Random feature attention . . . . . . . . . . . . . . . . . . . . . . 23

**10 Open questions** **23**

### 1 Introduction

NLP is deeply intertwined with the formal study of language, both conceptually
and historically. Arguably, this connection goes all the way back to Chomsky’s
_Syntactic Structures in 1957. It also still holds true today, with a strand of_
recent works building formal analysis of modern neural networks methods in
terms of formal languages.[1] In this document, I aim to explain background
about formal languages as they relate to this recent work. I will by necessity
ignore large parts of the rich history of this field, instead focusing on concepts
connecting to modern deep learning-based NLP. Where gaps are apparent, I
will allude to interesting topics that are being glossed over in footnotes. As
a general disclaimer, I encourage the reader to seek out resources on topics I
ignore, written by authors who know those areas much better than me.[2]

The intended audience of this document is someone with a background in
NLP research, and some knowledge of theory, who is interested in learning more
about how the classical theory of formal languages and automata relates to modern deep learning methods in NLP. For the sake of readability, formal proofs will
be omitted in favor of proof sketches or bare statements of results. Hopefully,
it will be a useful starting point that leads the reader to other resources and
questions.[3]

#### 1.1 Basic definitions

**Formal language** Whereas arithmetic studies numbers and the relations over
them, formal language theory is based around the study of sets of strings. Familiar to computer scientists, a string is simply a finite sequence of tokens from
some finite alphabet Σ. For example, say Σ = _a, b_ . Then ab, aabb, and bbaba
_{_ _}_
are examples of strings over Σ. Each string has a finite length, but there are in
fact countably infinite strings over Σ. We denote this infinite set of all strings
over Σ as Σ[∗]. A formal language is a (usually infinite) set of strings, i.e. a
subset of Σ[∗]. The set of all languages over Σ is uncountably infinite.[4]

**Grammar** Intuitively, a “grammar” is the finite set of rules that describes
the structure of an infinite language. In formal language theory, a grammar
can be seen as a computational system (automaton) describing a language. In
particular, we will concern ourselves with three types of grammars:

1See Ackerman and Cybenko (2020) for a survey of recent results.
2Naturally, this document is informed by the works I am familiar with. If you notice topics
or questions that are missing, please let me know.
3One classic textbook for automata theory is Minsky (1967).
4This follows from Cantor’s Theorem, since the power set of Σ∗ has cardinality 2ℵ0 .

2


-----

1. A recognizer R is an automaton that maps from Σ[∗] 0, 1 . R implicitly
_→{_ _}_
defines a language L where x _L iff R(x) = 1._
_∈_

2. A transducer T is an automaton that maps strings over one alphabet to
strings over another: i.e., Σ[∗]1 2[.] _T can be seen as implementing_

_[→]_ [Σ][∗]
any sequence transduction task, such as translation or mapping linguistic
utterances to logical forms.

3. An encoder E is an automaton that maps from Σ[∗] _→_ Q[k], where Q[k] is the
_k-dimensional vector space of rational numbers. An encoder can be seen_
as a device building a latent representation of some linguistic input.

A grammar formalism is a system for specifying grammars implementing these
tasks. Some examples that may be familiar to an NLP audience are finite-state
automata, context-free grammars, and combinatory category grammar. As you
continue through this document, it may help to keep in mind two different types
of questions for each formalism:

  - Declarative How are “rules” of a grammar specified in this formalism?

  - Procedural What is the algorithm by which a grammar written in this
formalism can be used to recognize, transduce, or encode a string?

With these preliminaries in place, we shall discuss a variety of formal language classes and their accompanying grammar formalisms.

### 2 Regular languages

The most basic class of languages we will discuss is the regular languages.[5] On a
historical note, the genesis of theory around regular languages is intimately tied
to the study of artificial neural networks. Inspired by neural circuitry, Mcculloch
and Pitts (1943) were the first to propose a model of artificial neural networks
that resembled something like a finite-state automaton. Subsequently, these
ideas were further developed by Kleene et al. (1956) and Minsky (1956), forming
the basis of the modern theory of regular languages that will be presented here.[6]

To a modern reader, the most natural way to define the regular languages is
likely as the languages that can be recognized by regular expressions. Formally,
regular expressions are an algebraic system for describing languages built around
the union, concatenation, and Kleene star ([∗]) operations:[7]

**Definition 1 (Regular languages) For σ** Σ, define L(σ) = _σ_ . The regular
_∈_ _{_ _}_
language L(e) for an expression e is defined recursively:

5It should be noted that subregular languages are also well studied, for example as models
of natural-language phonology (Heinz et al., 2011).
6I highly recommend consulting Forcada (2000) for about the connections of artificial neural
networks and early automata theory.
7We can also write parentheses in regular expressions to disambiguate the order of operations.

3


-----

_a_
start _q0_ _q1_ _b_

Figure 1: DFA that recognizes the regular language ab[∗]. Accepting states ( _F_ )
_∈_
are indicated with a double circle border. To simplify the graph, we assume
undefined transitions terminate the computation, returning 0.

1. Union: L(e _d) = L(e)_ _L(d)._
_|_ _∪_

2. Concatenation: L(ed) = {xexd | xe ∈ _L(e), xd ∈_ _L(d)}._

3. Kleene star: L(e[∗]) = [�]i[∞]=0 _[L][(][e][)][i][, where the Cartesian power][ L][(][e][)][i][ yields]_
strings, e.g. _a, b_ = _aa, ab, ba, bb_ .
_{_ _}[2]_ _{_ _}_

For example, ab[∗] is a regular expression defining the regular language of one
_a followed by 0 or more b’s, i.e.,_ _a, ab, abb,_ . Notice that, in Figure 1, we
_{_ _· · · }_
draw a finite-state automaton that recognizes ab[∗]. This is an intentional choice,
since it turns out that finite-state automata and regular expressions are actually
two sides of the same coin. We can formally definite a deterministic finite-state
automaton (DFA) as follows:

**Definition 2 (DFA) A DFA is a tuple ⟨Σ, Q, q0, δ, F** _⟩_ where Σ is the vocabulary,
_Q is a finite set of states, q0 ∈_ _Q is an initial state, δ : Σ×Q →_ _Q is the transition_
function, and F _Q is a set of accepting states._
_⊆_

Viewing a DFA A as a language recognizer, we will now discuss the algorithm
for accepting or rejecting an input string, assuming some level of familiarity. The
DFA starts in the state q0. Given an input string, it reads one token σ at a
time. Letting q be the current state, each token causes A to transition to a new
_σ_
state q[′] = δ(σ, q), which we write as q = q′. The machine accepts a string x of
length n if it ends in an accepting state after the final token, i.e., if there exists
a sequence of states {qi}i[n]=1 [such that]

_q0_ _x=1 q1_ _x=2 · · ·_ _x=n qn ∈_ _F._ (1)

The set of languages recognized by A is the set of strings that it accepts. We
will refer to the number of states _Q_ as the size of A.
_|_ _|_
An important classical result is that the regular languages are equivalent to
the languages that can be accepted by finite-state automata:

**Kleene’s Theorem. For all languages L, L is recognizable by a DFA iff L is**
_recognizable by a regular expression.[8]_

8In fact, this applies for both deterministic and nondeterministic finite-state acceptors, as
these two classes are equivalent (Rabin and Scott, 1959). Nondeterministic automata, while
not as computationally efficent, can be used to concisely describe regular languages.

4


-----

An important generalization of the DFA is the nondeterministic finite automaton (NFA). At a high level, NFAs relax the transition function δ to be a
relation, meaning that more than one transition can be possible for any token.
A string is accepted if any sequence of transitions leads to an accepting state.[9]

Extending Kleene’s Theorem, the set of languages recognizable by NFAs is also
the regular languages. Thus, in the finite-state setting, nondeterminism does
not introduce any additional expressive power. However, the size of a “determinized” DFA generated from an arbitrary NFA can be exponentially larger
than the original NFA.
In summary, regular expressions, DFAs, and NFAs all converge to the same
recognition capacity: the regular languages. There is a lot more to be said
about the theory and practice of regular languages. For a nice overview of the
theory from first principles, including the Myhill-Nerode Theorem and language
derivatives, see Wingbrant (2019).

#### 2.1 Learning regular languages

As our focus here is on the relevance of formal language theory for machine
learning, we will discuss foundational work on learning regular languages from
data. The most basic question we might want to know about this is how difficult
it is to learn a DFA recognizer from positive and negative data. By this, we
mean a finite sample of strings in L as well as strings in Σ[∗] _L. It is always_
_\_
possible to construct a DFA to fit such a sample: we can build a trie (tree of
prefix states) mapping each full string to its label. Since such a trie is a special
case of a DFA, we have a construction for a DFA recognizer whose size is linear
in the size of the sample. However, this large DFA will not generalize to new
samples. Inspired by Occam’s razor, we instead might hope to find the smallest
DFA matching our data set, hoping that this lower complexity automaton would
generalize well to other strings in L. This turns out to be a computationally
intractable problem: Gold (1978) proved that, in fact, it is NP-hard.
However, all is not lost. Considering a slightly different formal setup, Angluin (1987) introduced the L[∗] algorithm, which finds minimal DFAs for regular
languages from a polynomial number of membership and equivalence queries.
Let L be a “ground truth” regular language. A membership query takes as
input a string, and returns whether it is in L. An equivalence query takes as input a hypothesis L[′], represented by a finite-state machine. If L = L[′], the query
returns ; otherwise, it returns a “counterexample”: a string accepted by one
_∅_
machine, but not the other. The L[∗] algorithm uses these queries to converge to
a minimal finite-state machine for L in polynomial time.

### 3 Data structures

Finite-state automata represent the limits of computation with only finite memory available. With additional memory, a recognizer is capable of specifying

9We will discuss nondeterministic automata in greater detail in Section 3.

5


-----

more complex languages. One way to view more complicated formal grammars,
then, is as finite-state automata augmented with various kinds of data structures, which introduce different kinds of unbounded memory. This is the view
we will take for introducing deterministic pushdown and counter automata, two
useful abstractions for understanding the internal computation of RNN architectures.

**Definition 3 (Data structure) A data structure D is a tuple ⟨c0, U, r⟩.**

1. The null data structure c0 represents the base case for a newly constructed
object. For example, this could be an empty stack.

2. The update operations U are a finite set of computable functions u : C
_→_
_C, where C is the space of data structure states. We can interpret each u_
as taking an old state c, and returning an updated one c[′]. For example,
for a binary stack, U could be {noop, pop, push0, push1}. The update
pop would remove the top element, e.g. 1001 001.
_�→_

3. The readout operation r : C _R is a computable function that produces_
_→_
finite summaries r(c) from data structure states c. Importantly, the range
_R must be finite. For a stack, we can take r to return the top k elements_
of the stack, for some finite k.

This definition of a data structure provides a generic way to formalize unbounded memory that can interface with a finite-state machine. We now define
an automaton that “controls” the data structure, i.e. updates it and reads
from it according to the input string. Notice how the following generalizes the
finite-state recognizer from Definition 2:

**Definition 4 (D-automaton) Let D be a data structure ⟨c0, U, r⟩** as in Definition 3. A D-automaton is a tuple ⟨Σ, Q, q0, υ, δ, F _⟩_ with

1. A finite alphabet Σ

2. A finite set of states Q

3. An initial state q0 ∈ _Q_

4. A memory update function υ : Σ _R_ _Q_ _U_
_×_ _×_ _→_

5. A state transition function δ : Σ _R_ _Q_ _Q_
_×_ _×_ _→_

6. An acceptance mask F _R_ _Q_
_⊆_ _×_

What does it mean to “compute” with a -automaton? The definition of
_D_
language recognition is quite similar to that of a DFA, except that the steps
can update and read from the data structure. Let’s make this precise. A full
configuration for a -automaton is a data structure configuration c as well as a
_D_

6


-----

finite state q, Given _c, q_ _C_ _Q and an input symbol σ_ Σ[∗], we define the
_⟨_ _⟩∈_ _×_ _∈_
transition of the finite state q and data structure configuration c:

_c[′][ σ]= υ(σ, r(c), q)(c)_ (2)

_q[′][ σ]= δ(σ, r(c), q)._ (3)

Note that υ(σ, r(c), q) returns a function in (2), which explains the iterated
_σ_
function application. We will overload = to full machine configurations, i.e.,
_σ_
_⟨c, q⟩_ = ⟨c′, q′⟩ as defined above. Recall that co is the null data structure, and q0
is the initial state. We say that the machine accepts a string x Σ[∗] of length
_∈_
_n if there exists a sequence of configurations {⟨ci, qi⟩}i[n]=1_ [such that]

_⟨c0, q0⟩_ _x=1 ⟨c1, q1⟩_ _x=2 · · ·_ _x=n ⟨cn, qn⟩,_ (4)

and ⟨r(cn), qn⟩∈ _F_ . Informally, this says that the machine accepts x if its
deterministic computation sequence ends in an accepting configuration.

#### 3.1 Stacks and counters

This general framework allows us to formalize recognizers using two types of
data structures relevant to recent theoretical work on RNNs: stack and counter
automata. At a high level, we get these machines by letting be either a stack
_D_
or a vector of integer-valued counter variables.

**Definition 5 (Stack automaton) Let C = {0, 1}[∗]** and c0 = ϵ (the empty string).
For a string c, let c2: denote the string with the first token deleted. We define
the set U = {noop, pop, push0, push1} where

noop(c) = c (5)

pop(c) = c2: (6)
push0(c) = 0c (7)

push1(c) = 1c. (8)

We define r to return the k topmost elements of the stack: r(c) = c:k.

Another name for a stack automaton is a deterministic pushdown automaton.
Intuitively, this machine works by pushing and popping different symbols onto
a stack. The topmost contents of the stack can in turn inform the behavior of
the automaton.

**Definition 6 (Counter automaton) Let C = R[k], and c0 be the k-dimensional**
zero vector. We set U = 1, +0, +1, 0, where each element is a function,
_{−_ _×_ _}[k]_
e.g. +1(x) = x + 1. We let r(c) be the broadcasted “zero-check” function, i.e.,
for 1 _i_ _n:_
_≤_ _≤_

�
0 if ci = 0

_r(c)i =_ (9)

1 otherwise.

7


-----

This definition is equivalent to the real-time, deterministic counter automata
referred to in other works (Merrill, 2020; Fischer et al., 1968). Intuitively, these
automata resemble stack automata, except that the stack data structure has
been replaced by a finite number of “integer” variables. At each computational
step, these variables can be added to or subtracted from. The automaton can
condition its behavior on whether or not the counters are 0.[10]

#### 3.2 Power of unbounded memory

The unbounded memory introduced by counters or a stack allows these automata to handle more languages than finite-state recognizers. For example,
consider the language a[n]b[n], representing the set of strings with a certain number of a’s followed by a matching number of b’s. A finite-state recognizer cannot
accept a[n]b[n] because doing so requires “remembering” the number of a’s, which
is unbounded as n gets arbitrarily large.
With a counter automaton, however, we can simply add 1 to a counter when
we see a, and subtract 1 when we see b. At the end of the string, we accept if
the counter is back at 0.[11] Thus, a counter-based strategy provides one way to
handle this language.
With a stack automaton, we have a different way to create a recognizer for
_a[n]b[n]. We push each a onto the stack. For a b, we pop if the stack contains a,_
and raise an error if it contains a b or is empty. Finally, at the end of the string,
we accept if the stack is empty.
Thus, both stack and counter memory provide ways to accept the language
_a[n]b[n]. We will see, though, that other languages differentiate the abilities of_
these modes of computation.

#### 3.3 Power of counters

Parentheses languages pose a problem for counter machines, whereas they are
easy to accept for a stack automaton. We define the Dyck (parentheses) languages by the following context-free grammar:

**Definition 7 (k-Dyck language) For any k, define the k-Dyck language Dk as**
the language generated by the following context-free grammar:

S SS _ϵ_ (10)
_→_ _|_

S → (iS)i (for all 1 ≤ _i ≤_ _k),_ (11)

where ϵ is the empty string. Note that (i and )i refer to tokens in Σ.

Intuitively, Dk represents the language of well-balanced parentheses, with k
types of parenthetical brackets. Both counters and a stack can be used to accept

10Merrill (2020) shows various generalized operations over counters, such as adding or
thresholding at arbitary integers, that can be introduced without increasing the expressive
power of the automaton.
11Using finite state, we also track whether we have seen ba at any point. This lets us
constrain acceptance to strings of the form a[∗]b[∗].

8


-----

_D1, as D1 is quite similar to a[n]b[n]. However, for k ≥_ 2, counter automata cannot
recognize the language. The root of the problem is that a recognizer must model
not just the number of open parentheses at any point, but also remember the
full sequence of parentheses. A stack is naturally suited to do this—just push
each open bracket—but a counter machine will quickly run out of memory. To
be more precise, the number of memory configurations of a counter automaton
is bounded polynomially in the input length, whereas k-Dyck for k 2 requires
_≥_
_exponentially many memory configurations._
An interesting aside here is that, in some sense, Dyck structure is representative of all hierarchical structure. This is formalized by the following theorem:

**Theorem 1 (Chomsky and Sch¨utzenberger 1963) Let L be any context-free**
_language. Then L is the homomorphic image of a Dyck language intersected with_
_a regular language, i.e. for some k, regular language R, and homomorphism h,_

_L = h(Dk ∩_ _R)._

Thus, not being able to accept Dyck languages is truly a fundamental weakness of counter automata as models of hierarchical structure, which is widely
believed to be an important feature of natural language. We will say more
about the relationship between context-free languages and natural language in
Section 4.

#### 3.4 Power of a stack

The set of languages recognizable by a stack automaton is called the determinis_tic context-free languages (DCFLs), and forms a strict subset of the context-free_
languages (CFLs) proper.[12] Compared to CFLs, DCFLs always permit an un_ambiguous grammar: meaning a parsing system where each string is assigned_
exactly one tree.[13] On the other hand, the Dyck languages are DCFLs, and we
have already seen that, in some sense, they capture the fundamental structure
of all CFLs according to Theorem 1.
Despite this power, stack automata suffer from their own weaknesses relative
to counter automata. There is in fact no strict containment relation between
counter languages and DCFLs, or CFLs for that matter. An illustrative example
is the language a[n]b[n]c[n], which can be accepted by a machine with two counters.
The first one counts up for a’s and down for b’s, and the second one counts
up for b’s and down for c’s. A stack automaton, however, cannot do this. The
intuition is that each a can only be closed by one item, so once we have matched
it to some b, we cannot match it to any c.[14]

12An example of a CFL that is not deterministic is {anbn | n ≥ 0} ∪{anb2n | n ≥ 0}. See
Power (2002) for a nice proof of this.
13The converse is not always true: some unambiguous CFLs are not deterministic. Intuitively, this is related to the concept of local ambiguity, where a full string can have an
unambiguous parse, but ambiguity exists for building intermediate parses when reading the
[string left-to-right. In natural language, these kind of ambiguities can lead to garden-path](http://www-personal.umich.edu/~jlawler/gardenpath.pdf)
[sentences like The old man the boat.](http://www-personal.umich.edu/~jlawler/gardenpath.pdf)
14This can be formalized by using the pumping lemma to prove that anbncn is not a
context-free language (and thus not a deterministic context-free language).

9


-----

#### 3.5 Nondeterminism

In Section 2, we remarked that NFAs are equivalent in capacity to DFAs. Interestingly, this correspondence between nondeterministic and determistic automata does not hold for memory augmented automata. Informally, nondeterminism means that that transitions allow many possibilities instead of just
one. A string is accepted if any choice of valid transitions ends in an accepting
state. Formally, we relax the transition and state update functions to relations
_υ_ Σ _R_ _Q_ _U and δ_ Σ _R_ _Q_ _Q._
_⊆_ _×_ _×_ _×_ _⊆_ _×_ _×_ _×_
From a procedural point of view, we generalize the automaton transition
_σ_
rule as follows. We say that _c, q_ = _c′, q′_ if there exists u _U such that:_
_⟨_ _⟩_ _⟨_ _⟩_ _∈_

_c[′]_ = u(c) (12)

_σ, r(c), q, u_ _υ_ (13)
_⟨_ _⟩∈_

_σ, r(c), q, q[′]_ _δ._ (14)
_⟨_ _⟩∈_

We say that a nondeterministic automaton accepts a string if there exists some
sequence of valid transitions that reaches an accepting state. The language
recognition capacity of the machine is defined as the set of strings that are
accepted according to this overloaded definition.
This relaxation means that our automaton no longer must proceed in a linear sequence of transitions, but instead can “backtrack” across many possible
transition sequences. This essentially allows the automaton to traverse an exponential space of possible computational paths while deciding to accept a string,
increasing expressive power while making the machine less practical to implement. In the case of stack automata, the recognition capacity is expanded from
the DCFLs to the CFLs proper.[15]

#### 3.6 Learning memory-augmented automata

One interesting question is how methods for learning finite-state automata can
be extended to more complex automata with counter or stack memory. The
field of grammar induction consists of a family of problems related to this idea,
attempting to infer context-free grammars or other models of grammar from
data using algorithms similar to L[∗]. See de la Higuera (2010) for a textbook
providing a thorough overview of this extensive field.
More recently, a thread of work in deep learning has attempted to re-frame
data structures like stacks as differentiable objects that can be seamlessly integrated with standard gradient-based optimization (Graves et al., 2014; Grefenstette et al., 2015; Suzgun et al., 2019b). The data structure can be directly
controlled by an RNN in a way that resembles a -automaton. Since the data
_D_
structure is differentiable, gradients can be backpropagated through it to update
the network’s parameters. This approach combines the structural biases and interpretability of data structures with the practical utility of high-dimensional

15At the time of writing, I am not immediately sure what the capacity of the nondeterministic counter automaton is. Let me know if you have thoughts about it!

10


-----

optimization for learning. When done well, it produces models that learn how
to use data structures to solve a particular problem, perhaps similarly to a human programmer. However, there are challenges in defining the differentiable
data structures in ways that are amenable to learning and efficient at train time
(Hao et al., 2018). Empirically, stack-augmented networks have been shown
to perform well on synthetic tasks requiring strong hierarchical representations
(Grefenstette et al., 2015; Suzgun et al., 2019b) as well as on some natural language tasks (Yogatama et al., 2018; Merrill et al., 2019). Theoretically, Stogin
et al. (2020) proved that stack-augmented RNNs are stable, meaning that the
stack state remains close to that of some stack automaton. Although most
work incorporating stack memory into RNNs has used deterministic controllers,
DuSell and Chiang (2020) recently developed an algorithm for training a neural network simulating a nondeterministic stack automaton. In principle, this
approach can also be applied to other data structures beyond stacks, such as
Turing machines (Graves et al., 2014). In Section 7, we will develop the perspective that the LSTM gating mechanism can be viewed similarly to a differentiable
approximation of a counter data structure.

### 4 Chomsky hierarchy

The reader may be familiar with the Chomsky hierarchy (Chomsky, 1956,
1959): a classical framework for arranging the syntactic complexity of formal
languages.[16] The Chomsky hierarchy arranges classes of formal languages in
concentric sets based on the complexity of the grammars needed to parse them.
As shown in Figure 2, the results we have so far explored all lie in the lower
levels of the classes hierarchy. The regular languages from Section 2 form Type
3: the least complex level of the classical Chomsky hierarchy. The memoryaugmented automata discussed in Section 3, denoted in Figure 2 with dashed
lines, fall withing Type 2 and Type 1. The Dyck languages fall within the deterministic context free languages, intersecting the counter-acceptable ones (D1),
but not fully contained (Dk for k > 1).
What goes on in the higher levels of the hierarchy? Type 2 consists of the
context-free languages, which are acceptable by a nondeterministic stack automaton. In this context, nondeterministic means that the the transitions of
the machine are not functions, but relations. The machine accepts a string if
any choice of transitions over the string ends in an accepting configuration. As
discussed in Theorem 1, context-free languages can be thought of as languages
with nested hierarchical structure. Type 1 consists of the context-sensitive languages, corresponding to the languages acceptable by a nondeterministic Turing
machine with O(n) space. Finally, at the top of the hierarchy, we have Type 0,
which corresponds to the recursively enumerable functions. The acceptors for
this class of languages are arbitrary Turing machines, establishing them at the
top of the hierarchy.

16A good reference on the Chomsky hierarchy is Hopcroft et al. (2006).

11


-----

Figure 2: The Chomsky hierarchy superimposed with some additional classes
of languages we have discussed. Recall that the capacity of a stack automaton is the deterministic context-free languages, and the capacity of a counter
automaton is the counter(-acceptable) languages.

A big question in mathematical linguistics is the place of natural language
within this hierarchy. Context-free formalisms describe much of the hierarchical
structure of natural languages. However, the presence of various types of cross
serial dependencies (which do not easily fall into projective dependency trees)
lead many linguistics to believe that natural language is mildly context-sensitive
(Shieber, 1985).[17] In terms of the Chomsky hierarchy, this means that it falls in
a class between the context-free and context-sensitive classes, but close enough
to the context-free class such that parsing remains tractable. There is much
debate about what the right formalism is: two contenders are Tree Adjoining Grammar (TAG; Joshi et al., 1975) and Combinatory Category Grammar
(CCG; Steedman, 1996).
Another interesting extension of the Chomsky hierarchy is hypercomputation (Siegelmann, 1995). Hypercomputation refers to models of computation
that are more powerful than Turing machines. While this may seem to violate
the Church-Turing thesis that Turing machines represent a universal model of
effective computation, it does not, because all known models of hypercomputation are infinite machines. The standard theory of computability, on the other
hand, is concerned with the ability of finite machines to model infinite sets of

17The canonical example of this based purely on syntax is Swiss German, where centerembedded noun verb phrases embed serially rather than nesting. However, a mildly contextsensitive analysis also makes recovering deep structural (i.e. semantic) dependencies easier in
_any language._

12


-----

strings. Various models of hypercomputation exist, and can model languages
outside of Type 0. In fact, Type 0 only contains countably infinite languages
(since there are countably infinite Turing machines), whereas the set of all formal
languages is uncountably infinite. Thus, in some sense, “most” formal languages
are not acceptable by any computable grammar. We will return to the notion
of hypercomputation in Subsection 6.1, where we explore interesting theoretical
connections to the capacity of real-valued neural networks.

### 5 Weighted finite-state automata

So far, we have mostly discussed different types of recognizers for formal languages. We now turn to discussing a type of encoder, a weighted finite-state
automaton (WFA), which generalizes a finite-state recognizer. The WFA uses
a finite-state machine to encode a string into a number, rather than producing
a boolean decision for language membership.
Let ⟨K, ⊕, ⊗⟩ be a semiring.[18] We use this semiring to define a WFA as
a finite-state machine augmented with weighting functions for each transition.
To score a string, we consider each path that the string induces through the
automaton. We score a path by “multiplying” ( ) the transitions along it. We
_⊗_
then score the full string by “summing” ( ) the scores for all possible paths.
_⊕_
The notion of initial and accepting states are generalized to allow special weights
that can be multiplied for starting and ending in each state. Let’s go through
this a little more formally.
Let ⟨K, ⊕, ⊗⟩ be some semiring. A WFA W consists of an alphabet Σ, set
of states Q, and weighting functions λ, τ, ρ defined as follows:

1. Initial state weights λ : Q → K

2. Transition weights τ : Q × Σ × Q → K

3. Final state weights ρ : Q → K

Let q →σ q[′] denote a transition from q to q[′] licensed by token σ. The weighting
functions of a WFA are used to encode any string x Σ[∗] as follows:
_∈_

**Definition 8 (Path score) Let π be a path of the form q0 →x1 q1 →x2 · · · →xt qt**
through WFA A. The score of π is given by


�


_W_ (π) = λ(q0) ⊗


� _t_
�

_τ_ (qi−1, xi, qi)
_i=1_


_⊗_ _ρ(qt)._


By Π(x), denote the set of paths producing x.

18For concreteness, you can imagine the rational field ⟨Q, +, ·⟩, but WFAs are often defined
over other semirings, such as ⟨Q, max, +⟩.

13


-----

_σ/1_
_∀_

_σ/σ_
_∀_

start _q0_ _q1_


_σ/2_
_∀_


Figure 3: A WFA over⟨Q, +, ·⟩ mapping binary strings to their numeric value.
This can be extended for any base > 2. Let _σ/w(σ) denote a set of transitions_
_∀_
consuming each token σ with weight w(σ). We use standard DFA notation
to show initial weights λ(q0) = 1, λ(q1) = 0 and accepting weights ρ(q0) =
0, ρ(q1) = 1.

**Definition 9 (String encoding) The encoding computed by a WFA A on string**
_x is_
�
_W_ (x) = _A(π)._

_π∈Π(x)_

If we restrict the K to ⟨{0, 1}, ∨, ∧⟩, then we get nondeterministic finite
recognizers, whose recognition capacity is the regular languages. If we use the
rational field as our semiring, the WFAs compute a class of function called
the rational series. In this sense, the rational series are the class of encoding
functions that generalize the regular languages.
The rational series are counterintuitively powerful, capable of producing encodings that are “stateful”, i.e. depend on more than finite context. For example, consider the function mapping a string 0, 1 to the value that it represents
_{_ _}_
in binary. For example, 101 5. This function takes exponentially many out_�→_
put values, in the input string length. Nevertheless, Figure 3 shows a WFA that
computes it.
Another useful WFA to keep in mind is the n-counter. This machine uses n+
1 states to count all the occurrences of a specific n-gram. A 1-counter is shown
in Figure 4. Thus, there is a natural connection between WFAs and counter
machines: WFAs can simulate counting where the update to the counters cannot
be conditioned by their current values. As discussed in Merrill et al. (2020b), this
relationship is useful for developing a typology of RNN architectures. Merrill
et al. (2020b) also explore the power of a WFA encoder as a recognizer where
various types of decoding functions are used to accept or reject based on the
latent encoding.

#### 5.1 Hankel matrices

The Hankel matrix is an infinite matrix that can be used to represent any formal
language. It has many appealing theoretical properties that relate directly to
WFAs.

**Definition 10 Given a function f : Σ[∗]** _→_ K and two enumerations α, ω of Σ[∗],

14


-----

_σ/1_
_∀_

_α/1_

start _q0_ _q1_


_σ/1_
_∀_


Figure 4: A WFA over ⟨Q, +, ·⟩ that counts occurrences of the unigram α. This
can be generalized with n + 1 states to count all occurences of an n-gram. Each
occurence of α produces a path with score 1; thus, the string score is the number
of occurences of α.

the Hankel matrix Hf is defined at each coordinate i, j ∈ N as:

[Hf ]ij = f (αiωj).

This matrix has an elegant theoretical connection to the theory of WFAs:

**Theorem 2 Consider f : Σ[∗]** _→_ Q. Then Hf has finite rank iff there exists
_a WFA over the rational field computing f_ _. Further, rank(Hf_ ) is equal to the
_number of states of this WFA._

This is a perhaps surprising connection between rank, a notion in linear
algebra, and the discrete states of a finite-state machine. It justifies using a
finite state-machine to describe an infinite matrix, or vice versa. If A is a subblock of rank(Hf ), rank(A) ≤ rank(Hf ). Thus, an empirical Hankel sub-block
_Hˆf suffices to get a lower bound on the size of the finite-state machine computing_
_f_ .

#### 5.2 Learning WFAs

The correspondence between Hankel matrices and WFAs also suggests a natural
_spectral algorithm for learning WFAs from data (Hsu et al., 2012). The algo-_
rithm involves building a large empirical sublock H[ˆ]f of the Hankel matrix Hf .
It then uses a singular value decomposition (SVD) to solve for the parameters
of a WFA in terms of H[ˆ]f . If H[ˆ]f is large enough, i.e. has the same number of
states as the underlying Hf, then this approach will reconstruct the underlying
WFA. Variants of the algorithm exist, differing in their practical performance
with low or noisy data. Since H[ˆ]f can be built from a “black box” function f,
this approach can be applied to derive WFAs approximating arbitrary behavior,
such as the representations or outputs from a neural network (Rabusseau et al.,
2018). A related theoretical work (Marzouk and de la Higuera, 2020) considers
the problem of deciding equivalence between a WFA and an RNN, showing it
to be undecidable in general and EXP-hard over a finite support of data.

15


-----

### 6 Foundational neural networks results

Having reviewed core concepts about formal languages and automata, we shift
focus towards theory that centers neural networks as an object of study. We
first discuss foundational results about neural networks as language recognizers.
We assume some level of familiarity with modern NLP architectures like RNNs
and transformers.
It is perhaps useful to have some historical perspective on the development
of this area. The earliest expressiveness results about neural networks predate
the “deep learning revolution” of the early 2010s. In these days, neural networks
were viewed in some research as an abstract model of computation rather than
a practical method for learning behavior from data. Neural networks were an
interesting alternative model of computation because of their distributed nature
and potential resemblance to the human brain.

#### 6.1 Hypercomputation

One distinctive feature of neural networks is that their state can be viewed as
real-valued, rather than discrete like in a Turing machine. Siegelmann (1995,
2004, inter alia) explore the implications of having real-valued state in a computational system, showing how it leads real-valued RNNs to be capable of hypercomputation. In other words, they can compute functions over Σ[∗] 0, 1
_→{_ _}_
that a Turing machine can never compute in finite time, such as the halting
problem.[19] Intuitively, the reason why this is possible is that real numbers take
infinite bits to specify, and thus a description of a real-valued neural network
is actually an infinitely sized object, whereas conventional computational systems are finite. In computational complexity theory, a similar intuition explains
why infinite circuit families are capable of hypercomputation (Arora and Barak,
2009, Chapter 6). One interpretation of Siegelmann’s results is that “brainlike”
systems whose state can be specified with infinite precision may be qualitatively
different than computational devices like Turing machines.

#### 6.2 Turing completeness

By restricting the network weights to rational numbers instead of real numbers,
we make it a well-defined computational system by normal standards of computability. As shown by Siegelmann and Sontag (1992), these rational RNNs
are in fact Turing-complete, assuming arbitrary (but finite) precision of the rational weights, and unbounded computation time, i.e. recurrent steps.[20] The
trick behind the construction is that, since a single neuron has arbitrarily precision, we can use a small, finite number of neurons to remember a full Turing

19Specifically, their capacity is the complexity class P/poly.
20Siegelmann and Sontag (1992) talk about “linear time”, by which they mean the runtime
of the neural network is linear in the runtime of the simulated Turing machine. However,
the absolute number of recurrent steps is still unbounded, which is more powerful than the
real-time (one step per input) use of neural networks in deep learning.

16


-----

machine tape, no matter how big it gets. Siegelmann and Sontag (1992) define
a network graph that simulates running steps of a two-stack Turing machine.
Recursively running this network until the simulated machine halts allows us to
compute any Turing-computable function.

**Caveats of this result** This setting and construction differ substantially from
how we think about neural networks in modern deep learning. Generally, the
computation graph is unrolled proportionally to the input length, restricting us
to real-time computation. Furthermore, the precision of the network state is
bounded in practice, and attempting to encode large objects like Turing machine tapes within individual neurons will quickly run into hardware-imposed
limits. Finally, while they can be hand-constructed, it seems unlikely that constructions exploiting arbitrary precision are easily learnable by gradient-based
optimization. The encoding will be highly sensitive to slight perturbations in the
network parameters,[21] meaning that the solution corresponds to steep valleys in
the parameter space. Intuitively, gradient descent in a high dimensional space
is unlikely to find steep minima, an idea that various works in deep learning
theory (e.g. Keskar et al., 2016; He et al., 2019) have explored.

**Hierarchy** Another interesting thing to note is that there is an infinite hierarchy of complexity classes between the rational-valued networks and real-valued
networks, differentiated by how many real-valued neurons are granted (Balcazar
et al., 1997). For example, a network with 1 real-valued neuron is more powerful
than a network with 0, and a network with 2 is even more powerful than the network with 1. The increase in power can be described in terms of a generalization
of Kolmogorov complexity.

#### 6.3 Binarized neurons

Having discussed real and rational-valued neural networks, we now move to the
weakest variant: networks whose weights are binarized. We can formalize this
by assuming the network weights are rational or integer-valued, but that the
activation function of each neuron thresholds the activation value into 0, 1 .
_{_ _}_
This model of computation is much weaker than Turing-complete. In fact,
binarized RNNs are equivalent to finite-state machines, which we have seen can
only compute regular languages. More recently, S´ıma (2020) introduced the
analog neuron hierarchy, which explores the hierarchy between having a fully
binarized RNN and a rational-valued RNN based on the number of rational
neurons. Adding more neurons increases capacity, until, at 3 rational neurons,
the hierarchy collapses to the full Turing-complete capacity of rational-valued
RNNs. A recent extension (S´ıma, 2021) proves further separations between[ˇ]
intermediate classes in this hierarchy. This strand of work relates to saturated
RNNs, which we will explore in the next section.

21In Section 7, we will see that constructions to simulate Turing machines can actually only
exist at small ℓ2 norm within an RNN’s parameter space.

17


-----

### 7 Saturated networks

As mentioned in Subsection 6.2, RNNs with unbounded time and infinitely precise states are capable of impractical behavior like simulating a whole Turing
machine tape in a small number of neurons. In this section, we discuss the
framework of saturated networks.[22] Saturation is a simplifying assumption that
can be applied to a neural network that aims to exclude these degenerate representations that we might take to be impractical or unlearnable. This is similar in
spirit to Subsection 6.3, where we discussed how RNNs with binarized neurons
are equivalent to finite-state automata, except that that we will build a theory of low-precision networks that allows us to compare different architectures.
Merrill (2020) define the saturated network as follows:

**Definition 11 (Saturated network) Let f** (x; θ) be a neural network parameterized by θ. We define the saturated network sf (x; θ) as

sf (x; θ) = lim
_ρ→∞_ _[f]_ [(][x][;][ ρθ][)][.]

We can view s as a saturation operator mapping network functions to saturated network functions. By taking the norm of the affine transformations in
the network to infinity, saturation converts squashing functions like σ and tanh
to step functions, effectively bounding the precision of the network representations. This turns the network into a discrete automaton-like device that can be
analyzed in formal-language-theoretic terms.
For example, consider a single neuron σ(wx), for input and weight vectors
_x, w ∈_ Q[m]. The saturated version of this neuron is:

lim (15)
_ρ→∞_ _[σ][(][ρwx][)][.]_

If wx > 0, then this limit goes to 1. If wx < 0, then it goes to 0.[23] Thus, the
neuron in (15) reduces to a discrete cell whose output is either 0 or 1 according
to a step function activation. The decision boundary is a hyperplane specified
by the vector w. In general, saturation applies a similar analysis to every part
of the network. While simple neurons will reduce in the way described here,
more complex components, such as LSTM gates, reduce to different types of
discrete structures. This allows us to view the saturated versions of networks
as automata, i.e., discrete machines consuming strings. Past work has explored
the capabilities of these saturated networks as both language acceptors and
encoders producing latent states. It has also tested how the formal properties
of saturated models relates to the capabilities of unsaturated networks.

22In Merrill (2019), this was called the asymptotic network, but followup works use the
term saturated to avoid confusion with asymptotic notation, and because it was previously
introduced in a similar sense by Karpathy et al. (2015).
23When wx = 0, the limit is undefined. This is not a problem, since that case is measure-0.
We can simply restrict the set of saturated networks to the cases where the limit is defined.

18


-----

#### 7.1 Saturated networks as automata

Merrill (2020) conjectures that the saturated approximation of the network
somehow captures the network behavior that is most stable or learnable. Empirically, this is supported by the fact that the ability of trained networks to
learn formal language tasks is often predicted by the capacity of the saturated
version of that network. We now go through theoretical results for various architectures, and cite relevant experimental results testing whether unsaturated
networks agree with these predictions. We will denote a saturated architecture
by prefixing it with s, e.g., sLSTM.

**RNNs** Elman sRNNs, as well as sGRUs, are equivalent to finite-state automata (Merrill, 2019). Experimentally, these models cannot reliably recognize
_a[n]b[n]_ (Weiss et al., 2018b), Dyck languages (Suzgun et al., 2019a), or reverse
strings (Hao et al., 2018; Suzgun et al., 2019a). This agrees with the saturated
capabilities, since all of these tasks require more than finite memory.

**LSTMs** sLSTMs closely resemble counter automata (Merrill, 2019). Experimentally, this is reflected by the fact that trained LSTMs can consistently learn
counter languages like a[n]b[n] (Weiss et al., 2018a) or D1 (1-Dyck; Suzgun et al.,
2019a), unlike RNNs or GRUs. LSTMs cannot learn to model k-Dyck languages for k > 1 (Suzgun et al., 2019b), which matches the fact that sLSTMs
(and counter machines) do not have enough memory to model these languages.
Similarly, they cannot reliably reverse strings (Grefenstette et al., 2015; Hao
et al., 2018). Shibata et al. (2020) show that LSTM language models trained on
natural language acquire semi-saturated representations where the gates tightly
cluster around discrete values. Thus, sLSTMs appear to be a promising formal
model of the counting behavior of LSTMs on both synthetic and natural tasks.

**Stack RNNs** A stack-augmented sRNN (Suzgun et al., 2019b) becomes the
deterministic stack automaton we studied in Section 3 (Merrill, 2019). Experimentally, trained stack-augmented sRNNs can reverse strings (Hao et al.,
2018) and model higher-order Dyck languages (Suzgun et al., 2019b), unlike the
vanilla sRNN, sGRU, and sLSTM. This suggests that the expanded capacity of
the saturated versions of these models translates to an expanded set of formal
tasks that are learnable.

**CNNs** On the other hand, convolutional networks (saturated or unsaturated)
are less expressive than general finite-state acceptors (Merrill, 2019). For example, no sCNN can recognize the language a[∗]ba[∗]. On the other hand, the sCNN
capacity is a superset of the strictly local languages (Merrill, 2019), which are
of interest for modeling the phonology of natural language (Heinz et al., 2011).

**Transformers** Saturation imposes interesting restrictions on the attention
patterns in an sTransformer (Merrill, 2019; Merrill et al., 2020a). Recall that self

19


-----

attention is parameterized by queries Q, keys K, and values V . The computation
of the attention heads in an sTransformer reduces to:

1. The head at position i first selects the subsequence of keys K _[∗]_ that maximize the similarity qi · kj.

2. The output of the head is computed as the mean of the value vectors V _[∗]_

associated with K _[∗]._

If there is only one maximal key, then this reduces to hard attention. However,
if there is a “tie” between several keys, then saturated attention can attend
across unboundedly many positions. To see this, imagine a saturated transformer where kj is constant for all j. Then, attention is distributed uniformly
over all positions, reducing the attention computation to an average over all
value vectors. This enables the sTransformer to implement a restricted form
of counting that can be used to recognize languages like a[n]b[n]. For a[n]b[n], two
heads can be used to count the number of a’s and b’s, and then the feedforward
layer can verify that the quantities are the same. Bhattamishra et al. (2020)
show experimentally that unsaturated transformers learn to solve simple counting tasks using variants of this strategy. Merrill et al. (2020a) also find evidence
of counter heads in transformer language models. In summary, like the sLSTM,
the sTransformer can count, predicting the fact that unsaturated transformers
learn to count to solve synthetic tasks in practice.

#### 7.2 Space complexity

One of the useful consequences of saturation is that it allow us to compare
the memory available to different RNN representations in a formally precise
framework. By memory, we mean the amount of information that can be stored
in the hidden state of the sRNN at any given step. This analysis shifts the focus
away from viewing saturated networks as acceptors, instead asking what kinds
of representations they can encode.
Merrill (2019) introduce the notion of state complexity: the number of configurations the saturated hidden state can take ranging over all inputs of length
_n. It is useful to write it in asymptotic “big-_ ” notation, e.g., (n). Merrill
_O_ _O_
(2020) reframe this as space complexity: the number of bits needed to represent the hidden state, which (still expressed in big- notation) is essentially the
_O_
base-2 log of the previous quantity.[24] For consistency, we will discuss everything
in terms of space complexity here. Space complexity provides a useful metric
for describing the computational differences between different sRNNs.
The finite-state models like the sCNN, sRNN, and sGRU have (1) space
_O_
complexity. Due to a its counter memory, the sLSTM is (log n). This is enough
_O_
to count, but not enough to encode representations like trees. In this sense,
sLSTMs can be viewed as learned streaming algorithms: models that process a
sequence in one pass with limited memory. On the other hand, saturated stackaugmented RNNs have (n) space: enough to build unbounded hierarchical
_O_

24There are some technical edge cases that separate the definitions in these two papers.

20


-----

**Architecture** sCNN sRNN sGRU sLSTM Stack sRNN
**Complexity** (1) (1) (1) (log n) (n)
_O_ _O_ _O_ _O_ _O_

Table 1: Space complexities of some saturated architectures, measured in bits.
Results are taken from Merrill (2019).

representations over the input sequence. For transformers, space complexity
results may vary depending on assumptions about positional embeddings and
other architectural properties. Crystalizing the space complexity of transformers
is an interesting open question.[25]

#### 7.3 Connection to learning

Why should the capacity of saturated networks be meaningful at all for what
is learnable by a network? One hypothesis is that optimization methods like
stochastic gradient descent (SGD) have an inductive bias towards saturated networks. Merrill et al. (2020a) argues that such an inductive bias should result
from the continued norm growth that is a part of training. This idea is inspired
by recent works in deep learning theory (e.g., Li and Arora, 2019) exploring the
divergence of network parameters during training. In line with this view, Merrill
et al. (2020a) show that the parameter norm of T5 (Raffel et al., 2020) naturally
_√_
drifts away from the origin over training at a rate _t.[26]_ Following from Def_∼_

inition 11, the saturated capacity corresponds roughly to the class of networks
that exist stably as the norm grows proportional to a scalar. Thus, continued
norm growth during a long training process should guide networks towards saturated networks.[27] To test this, Merrill et al. (2020a) measures the difference in
saturation between randomly initialized transformers and pretrained transformers. Whereas the randomly initialized transformer exhibit 0 cosine similarity to
saturated transformers, the representations inside the pretrained transformers
are highly similar (though the similarity is < 1). Merrill et al. (2020a) also find
that smaller transformer language models converge to a saturation level of 1.00
early in training, meaning that their attention patterns essentially implement
the specific variant of hard attention that exists in saturated transformers.

### 8 Rational recurrences

An alternative automata-theoretic perspective on RNNs is the framework of
rational recurrences (Peng et al., 2018; Schwartz et al., 2018). Rather than
analyzing the language expressiveness and space complexity of RNN architectures, Peng et al. (2018) focus on the gating of the memory mechanism, i.e.,

25See Merrill (2019) and Merrill (2020) for basic analysis of transformer space complexity.
26This is the same growth rate one would expect for a random walk.
27This intuition is complicated by trends in step size. For example, if step size decays
exponentially over time, then the parameter norm found by training will converge, even if the
norm is monotonically increasing.

21


-----

the types of functions that can be computed by the hidden unit ht as a function of x:t and ht−1. In particular, they define rational recurrences as RNN
gating functions ht where each element [ht]i can be simulated by a WFA over
_x:t.[28]_ Crucially, they show that QRNNs, CNNs, and several other RNN variants
are rational recurrences. Aside from being illuminating on a theoretical level,
one practical implication of the rational recurrences framework is for efficiency,
as the restricted form of a WFA makes the recurrent part of the computation
potentially cheaper than for an arbitrary RNN.
One simple example of a rational recurrence is the unigram RNN. Interestingly, the equations for the unigram RNN can be “compiled” from a unigram
WFA similar to the one shown in Figure 4. Transitions in the WFA correspond
to parameterized gates in the WFA. This illustrates the intimate connection between WFAs and rational recurrences, and how the rational recurrences framework can motivate developing new RNNs from automata models. As an exercise
to the reader: say we want to generalize the unigram RNN/WFA to condition its
updates on bigrams. How would the WFA graph and RNN equations change?
While Peng et al. (2018) conjectured that more complicated RNNs like
LSTMs and GRUs were not rationally recurrent, this question was left open
in their original papers. However, Merrill et al. (2020b) proved sLSTMs are not
rationally recurrent by constructing a function computable by the sLSTM whose
Hankel matrix has infinite rank. In contrast, sRNNs and sGRUs are rationally
recurrent, following from their equivalence to finite-state machines. However,
they need not be efficiently rationally recurrent, in the sense that the number
of states required in the WFA simulating them can be very large.

### 9 Transformers

Starting with Vaswani et al. (2017), NLP as a field has gradually moved towards using transformers instead of LSTMs. While attention mechanisms were
originally viewed as interpretable, the highly distributed nature of transformer
networks makes their inner workings mysterious. Thus, formal analysis of transformer networks is a very interesting question that could shed light on their
capacity, inductive biases, or suggest principled methods of interpretability.
We have already discussed saturated transformers in Subsection 7.1. Here,
we will mention analysis of several other transformer variants. Hopefully the
incomplete discussion in this section will serve as an inspiration for future investigations.
From a formal perspective, a key component of the transformer architecture
is its positional encodings. Without positional encodings, the transformer is
barely sensitive to position at all. Even simple regular languages like a[∗]b would
be impossible to recognize, since the transformer would have no way to detect
relative positional dependencies (Merrill, 2019).
Whereas saturated transformers can spread attention equally over a subsequence of positions (Merrill et al., 2020a), Hahn (2020) explores the limitations

28Recall that we discussed theory about WFAs in Section 5.

22


-----

of transformer networks where attention can only target a bounded number of
positions. Adapting arguments from circuit complexity theory, Hahn (2020)
shows that these hard attention transformers cannot recognize parity or Dyck
languages. A similar probabilistic result holds for the soft transformer.

#### 9.1 Random feature attention

One of the challenges with transformers is the scalability of attention for large
sequence lengths. Say we have N queries, each attending over a sequence of M
keys and values. Then the time complexity of computing attention for a fixed
query is (M ), and (MN ) across all queries. In contrast, Peng et al. (2021)
_O_ _O_
develop random feature attention (RFA): a drop-in replacement for attention
that can be computed in time (M +N ). The name comes the fact that they use
_O_
a random feature approximation, which is derived from the unbiased estimator
in (17):

**Theorem 3 (Rahimi and Recht 2008) For 1 ≤** _i ≤_ _D, let wi ∼N_ (0, σ[2]Id), let
_φ : R[d]_ _→_ R[2][D] _be a (random) nonlinear transformation of x ∈_ R[d]:

_φ(x) =_ �1/D� sin(w1 · x), · · ·, sin(wD · x), cos(w1 · x), · · ·, cos(wD · x)�. (16)

_Then, the following holds:_


�
Ewi �φ(x) · φ(y)� = exp _−_ _[∥][x][ −]_ _[y][∥][2]_

2σ[2]


�
_._ (17)


At a high level, softmax is computed in terms of various exponential quantities resembling the right-hand side in (17). Peng et al. (2021) derive RFA by
applying this approximation to simplify the attention computation graph.

### 10 Open questions

Finally, I will leave a non-exhaustive list of interesting guiding questions for
ongoing formal methods research in modern NLP. There are a lot of potential
questions here—these are things that I at least have found interesting and have
maybe worked on a bit. In many cases, they relate to important issues in core
NLP, such as interpretability and dataset artifacts.

1. As we saw, there are many unknowns about the transformer architecture.
Can we get some more complete theory about its representational capacity
or inductive bias?

2. Can modern deep learning motivate alternative theories of linguistic complexity to the Chomsky hierarchy? In particular, can we build a non-trivial
theory of complexity not just for large n, but for bounded-length strings?

3. How can we leverage formal models of neural network architectures for
more interpretable machine learning?

23


-----

4. What are the theoretical limits on learning latent structure (syntactic or
semantic) from sets of strings?

5. Many NLP researchers believe that a major obstacle for building and
evaluating systems is that models sometimes exploit spurious correlations
and artifacts in datasets to generalize in “shallow” ways. Can we use
formal language theory to distinguish spurious linguistic patterns from
valid ones?

I hope you have found this document interesting! Please reach out if you
have thoughts, feedback, or questions.

### Acknowledgments

This document heavily benefited from discussions with Gail Weiss, Michael
Hahn, Kyle Richardson, and my past and present advisors: Dana Angluin, Bob
Frank, Yoav Goldberg, Noah Smith, and Roy Schwartz. Further thanks to members of Computational Linguistics at Yale, researchers at the Allen Institute for
AI, and attendees and organizers of the Deep Learning and Formal Languages
workshop at ACL 2019, as well as the Formalism in NLP meetup at ACL 2020.

### References

[Joshua Ackerman and George Cybenko. 2020. A survey of neural networks and](http://arxiv.org/abs/2006.01338)
[formal languages.](http://arxiv.org/abs/2006.01338)

Dana Angluin. 1987. Learning regular sets from queries and counterexamples.
_Inf. Comput., 75:87–106._

Sanjeev Arora and Boaz Barak. 2009. Computational Complexity: A Modern
_Approach, 1st edition. Cambridge University Press, USA._

[J. L. Balcazar, R. Gavalda, and H. T. Siegelmann. 1997. Computational power](https://doi.org/10.1109/18.605580)
[of neural networks: a characterization in terms of Kolmogorov complexity.](https://doi.org/10.1109/18.605580)
_IEEE Transactions on Information Theory, 43(4):1175–1183._

[Satwik Bhattamishra, Kabir Ahuja, and Navin Goyal. 2020. On the ability of](http://arxiv.org/abs/2009.11264)
[self-attention networks to recognize counter languages.](http://arxiv.org/abs/2009.11264)

Noam Chomsky. 1956. Three models for the description of language. IRE Trans.
_Inf. Theory, 2:113–124._

Noam Chomsky. 1957. Syntactic structures. Walter de Gruyter.

Noam Chomsky. 1959. On certain formal properties of grammars. Inf. Control.,
2:137–167.

24


-----

Noam Chomsky and M. P. Sch¨utzenberger. 1963. The algebraic theory of
context-free languages*. Studies in logic and the foundations of mathematics,
35:118–161.

[Brian DuSell and David Chiang. 2020. Learning context-free languages with](http://arxiv.org/abs/2010.04674)
[nondeterministic stack RNNs.](http://arxiv.org/abs/2010.04674)

[Patrick C. Fischer, Albert R. Meyer, and Arnold L. Rosenberg. 1968. Counter](https://doi.org/10.1007/BF01694011)
[machines and counter languages. Mathematical systems theory, 2(3):265–283.](https://doi.org/10.1007/BF01694011)

Mikel L Forcada. 2000. Neural networks: Automata and formal models of com_putation. Universitat d’Alacant._

[E Mark Gold. 1978. Complexity of automaton identification from given data.](https://doi.org/10.1016/s0019-9958(78)90562-4)
_Information and Control, 37(3):302–320._

[Alex Graves, Greg Wayne, and Ivo Danihelka. 2014. Neural Turing machines.](http://arxiv.org/abs/1410.5401)

Edward Grefenstette, Karl Moritz Hermann, Mustafa Suleyman, and Phil Blun[som. 2015. Learning to transduce with unbounded memory.](http://arxiv.org/abs/1506.02516)

[Michael Hahn. 2020. Theoretical limitations of self-attention in neural sequence](https://doi.org/10.1162/tacl_a_00306)
[models. Transactions of the Association for Computational Linguistics, 8:156–](https://doi.org/10.1162/tacl_a_00306)
171.

Yiding Hao, William Merrill, Dana Angluin, Robert Frank, Noah Amsel, An[drew Benz, and Simon Mendelsohn. 2018. Context-free transductions with](https://doi.org/10.18653/v1/w18-5433)
[neural stacks. Proceedings of the 2018 EMNLP Workshop BlackboxNLP: An-](https://doi.org/10.18653/v1/w18-5433)
_alyzing and Interpreting Neural Networks for NLP._

[Haowei He, Gao Huang, and Yang Yuan. 2019. Asymmetric valleys: Beyond](http://arxiv.org/abs/1902.00744)
[sharp and flat local minima. CoRR, abs/1902.00744.](http://arxiv.org/abs/1902.00744)

J. Heinz, C. Rawal, and H. Tanner. 2011. Tier-based strictly local constraints
for phonology. In ACL.

Colin de la Higuera. 2010. _Grammatical Inference: Learning Automata and_
_Grammars. Cambridge University Press, USA._

John E. Hopcroft, Rajeev Motwani, and Jeffrey D. Ullman. 2006. Introduction
_to Automata Theory, Languages, and Computation (3rd Edition). Addison-_
Wesley Longman Publishing Co., Inc., USA.

[Daniel Hsu, Sham M. Kakade, and Tong Zhang. 2012. A spectral algorithm for](https://doi.org/10.1016/j.jcss.2011.12.025)
[learning hidden markov models. Journal of Computer and System Sciences,](https://doi.org/10.1016/j.jcss.2011.12.025)
78(5):1460–1480.

Aravind K Joshi, Leon S Levy, and Masako Takahashi. 1975. Tree adjunct
grammars. Journal of computer and system sciences, 10(1):136–163.

25


-----

[Andrej Karpathy, Justin Johnson, and Li Fei-Fei. 2015. Visualizing and under-](http://arxiv.org/abs/1506.02078)
[standing recurrent networks.](http://arxiv.org/abs/1506.02078)

Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyan[skiy, and Ping Tak Peter Tang. 2016. On large-batch training for deep learn-](http://arxiv.org/abs/1609.04836)
[ing: Generalization gap and sharp minima. CoRR, abs/1609.04836.](http://arxiv.org/abs/1609.04836)

Stephen C Kleene, Claude E Shannon, and John McCarthy. 1956. Automata
studies. Princeton, NJ.

[Zhiyuan Li and Sanjeev Arora. 2019. An exponential learning rate schedule for](http://arxiv.org/abs/1910.07454)
[deep learning.](http://arxiv.org/abs/1910.07454)

[Reda Marzouk and Colin de la Higuera. 2020. Distance and equivalence between](http://arxiv.org/abs/2004.00478)
[finite state machines and recurrent neural networks: Computational results.](http://arxiv.org/abs/2004.00478)

Warren Mcculloch and Walter Pitts. 1943. A logical calculus of ideas immanent
in nervous activity. Bulletin of Mathematical Biophysics, 5:127–147.

[William Merrill. 2019. Sequential neural networks as automata. In Proceedings](https://www.aclweb.org/anthology/W19-3901)
_of the Workshop on Deep Learning and Formal Languages: Building Bridges,_
pages 1–13.

[William Merrill. 2020. On the linguistic capacity of real-time counter automata.](http://arxiv.org/abs/2004.06866)

William Merrill, Lenny Khazan, Noah Amsel, Yiding Hao, Simon Mendelsohn,
[and Robert Frank. 2019. Finding hierarchical structure in neural stacks us-](https://doi.org/10.18653/v1/W19-4823)
[ing unsupervised parsing. In Proceedings of the 2019 ACL Workshop Black-](https://doi.org/10.18653/v1/W19-4823)
_boxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 224–_
232, Florence, Italy. Association for Computational Linguistics.

William Merrill, Vivek Ramanujan, Yoav Goldberg, Roy Schwartz, and Noah
[Smith. 2020a. Parameter norm growth during training of transformers.](http://arxiv.org/abs/2010.09697)

William Merrill, Gail Weiss, Yoav Goldberg, Roy Schwartz, Noah A. Smith, and
[Eran Yahav. 2020b. A formal hierarchy of RNN architectures. In Proceedings](https://doi.org/10.18653/v1/2020.acl-main.43)
_of the 58th Annual Meeting of the Association for Computational Linguistics,_
pages 443–459, Online. Association for Computational Linguistics.

Marvin Minsky. 1967. Computation: Finite and infinite machines. Inc., Engel_wood Cliffs, NJ._

Marvin L Minsky. 1956. Some universal elements for finite automata. Annals
_of Mathematics studies, 35._

Hao Peng, Nikolaos Pappas, Dani Yogatama, Roy Schwartz, Noah A. Smith,
[and Lingpeng Kong. 2021. Random feature attention.](http://arxiv.org/abs/2103.02143)

[Hao Peng, Roy Schwartz, Sam Thomson, and Noah A. Smith. 2018. Rational](https://doi.org/10.18653/v1/D18-1152)
[recurrences. In Proceedings of the 2018 Conference on Empirical Methods in](https://doi.org/10.18653/v1/D18-1152)
_Natural Language Processing, pages 1203–1214, Brussels, Belgium. Associa-_
tion for Computational Linguistics.

26


-----

James Power. 2002. Non-deterministic CFLs. `http://www.cs.nuim.ie/`
```
 ~[jpower/Courses/Previous/parsing/node38.html][.]

```
M. Rabin and D. Scott. 1959. Finite automata and their decision problems.
_IBM J. Res. Dev., 3:114–125._

[Guillaume Rabusseau, Tianyu Li, and Doina Precup. 2018. Connecting weighted](http://arxiv.org/abs/1807.01406)
[automata and recurrent neural networks through spectral learning.](http://arxiv.org/abs/1807.01406)

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang,
[Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the](http://arxiv.org/abs/1910.10683)
[limits of transfer learning with a unified text-to-text transformer.](http://arxiv.org/abs/1910.10683)

[Ali Rahimi and Benjamin Recht. 2008. Random features for large-scale kernel](https://proceedings.neurips.cc/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf)
[machines. In Advances in Neural Information Processing Systems, volume 20.](https://proceedings.neurips.cc/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf)
Curran Associates, Inc.

[Roy Schwartz, Sam Thomson, and Noah A. Smith. 2018. Bridging CNNs, RNNs,](https://doi.org/10.18653/v1/P18-1028)
[and weighted finite-state machines. In Proceedings of the 56th Annual Meeting](https://doi.org/10.18653/v1/P18-1028)
_of the Association for Computational Linguistics (Volume 1: Long Papers),_
pages 295–305, Melbourne, Australia. Association for Computational Linguistics.

[Chihiro Shibata, Kei Uchiumi, and Daichi Mochihashi. 2020. How lstm encodes](http://arxiv.org/abs/2010.00363)
[syntax: Exploring context vectors and semi-quantization on natural text.](http://arxiv.org/abs/2010.00363)

Stuart M Shieber. 1985. Evidence against the context-freeness of natural language. In Philosophy, Language, and Artificial Intelligence, pages 79–89.
Springer.

Hava T. Siegelmann. 1995. Computation beyond the Turing limit. Science,
268(5210):545–548.

Hava T. Siegelmann. 2004. Neural and super-Turing computing. Minds and
_Machines, 13:103–114._

Hava T. Siegelmann and Eduardo Sontag. 1992. On the computational power
of neural nets. In COLT ’92.

J. S´ıma. 2020. Analog neuron hierarchy. Neural networks : the official journal
_of the International Neural Network Society, 128:199–215._

[Mark Steedman. 1996. A very short introduction to CCG. Unpublished paper.](http://cs.brown.edu/courses/csci2952d/readings/lecture5-steedman.pdf)

[John Stogin, Ankur Mali, and C Lee Giles. 2020. Provably stable interpretable](http://arxiv.org/abs/2006.03651)
[encodings of context free grammars in rnns with a differentiable stack.](http://arxiv.org/abs/2006.03651)

Mirac Suzgun, Yonatan Belinkov, Stuart Shieber, and Sebastian Gehrmann.
[2019a. LSTM networks can perform dynamic counting. Proceedings of the](https://doi.org/10.18653/v1/w19-3905)
_Workshop on Deep Learning and Formal Languages: Building Bridges._

27


-----

Mirac Suzgun, Sebastian Gehrmann, Yonatan Belinkov, and Stuart M. Shieber.
[2019b. Memory-augmented recurrent neural networks can learn generalized](http://arxiv.org/abs/1911.03329)
[dyck languages.](http://arxiv.org/abs/1911.03329)

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
[Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all](http://arxiv.org/abs/1706.03762)
[you need.](http://arxiv.org/abs/1706.03762)

Gail Weiss, Y. Goldberg, and Eran Yahav. 2018a. On the practical computational power of finite precision rnns for language recognition. _ArXiv,_
abs/1805.04908.

[Gail Weiss, Yoav Goldberg, and Eran Yahav. 2018b. On the practical compu-](http://arxiv.org/abs/1805.04908)
[tational power of finite precision RNNs for language recognition.](http://arxiv.org/abs/1805.04908)

[Ola Wingbrant. 2019. Regular languages, derivatives and finite automata.](http://arxiv.org/abs/1907.13577)

Dani Yogatama, Yishu Miao, Gabor Melis, Wang Ling, Adhiguna Kuncoro,
Chris Dyer, and Phil Blunsom. 2018. [Memory architectures in recurrent](https://openreview.net/forum?id=SkFqf0lAZ)
[neural network language models. In International Conference on Learning](https://openreview.net/forum?id=SkFqf0lAZ)
_Representations._

[Jiˇr´ı S´ıma. 2021. Stronger separation of analog neuron hierarchy by deterministic[ˇ]](http://arxiv.org/abs/2102.01633)
[context-free languages.](http://arxiv.org/abs/2102.01633)

28


-----

