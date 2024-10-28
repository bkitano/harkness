## Provable Limitations of Acquiring Meaning from Ungrounded Form: What Will Future Language Models Understand?

#### William Merrill[∗] Yoav Goldberg[∗†] Roy Schwartz[‡] Noah A. Smith[∗§]

_∗_ Allen Institute for AI _† Bar Ilan University_
_‡ Hebrew University of Jerusalem_ _§ University of Washington_
#### {willm,yoavg,roys,noah}@allenai.org


#### Abstract


Language models trained on billions of tokens have recently led to unprecedented results on many NLP tasks. This success
raises the question of whether, in principle, a system can ever “understand” raw text
without access to some form of grounding.
We formally investigate the abilities of ungrounded systems to acquire meaning. Our
analysis focuses on the role of “assertions”:
textual contexts that provide indirect clues
about the underlying semantics. We study
whether assertions enable a system to emulate representations preserving semantic relations like equivalence. We find that assertions enable semantic emulation of languages that satisfy a strong notion of semantic transparency. However, for classes
of languages where the same expression can
take different values in different contexts,
we show that emulation can become uncomputable. Finally, we discuss differences between our formal model and natural language, exploring how our results generalize
to a modal setting and other semantic relations. Together, our results suggest that
assertions in code or language do not provide sufficient signal to fully emulate semantic representations. We formalize ways
in which ungrounded language models appear to be fundamentally limited in their
ability to “understand”.

#### 1 Introduction


has revealed that, to some degree, knowledge of
syntactic and semantic dependencies can emerge
_without explicit supervision (Rogers et al., 2020;_
Tenney et al., 2019). This knowledge can then be
transferred to a variety of downstream NLP tasks.
Yet, today’s NLP systems built on large language models still fall short of human-level general understanding (Yogatama et al., 2019; Zhang
et al., 2020). Brown et al. (2020) discuss the limitations of their GPT-3 language model compared
to humans, suggesting that:

Scaling up any LM-like model ... may
eventually run into (or could already be
running into) the limits of the pretraining objective.


Recently, language models trained on huge
datasets of raw text have pushed the limits of
natural language processing (Devlin et al., 2019;
Raffel et al., 2019; Brown et al., 2020, among
others). Such systems transcend the expert sys_tem paradigm, where rules about language and_
meaning are hardcoded into a system, as well as
the supervised learning paradigm, where a notion
of meaning is provided through ground-truth labels. Rather, analysis of massive language models


This possibility raises an interesting theoretical
question. What are the fundamental limits of
learning meaning from language modeling, even
assuming a perfect learner with access to unlimited data? Recently, Bender and Koller (2020) argued that achieving true natural language understanding from text alone is impossible, and that,
to really get at meaning, some type of semantic
grounding is necessary.[1] Their style of argumentation largely focused on developing thought experiments, rather than making formal arguments.
One thought experiment featuring prominently
in Bender and Koller (2020) was the task of learning to understand a programming language’s semantics from raw code. Here, understanding was
defined as fully emulating a compiler. This setup
has clear parallels to learning to understand natural
language, although the more well-defined nature
of programming languages makes them easier to
reason about. Bender and Koller (2020) argue that
emulation is difficult in this setting, and perhaps

1See Michael (2020) for a summary of the informal discussion around Bender and Koller (2020), much of which
took place on social media.


-----

impossible, because the source code alone contains no information about how it should be interpreted to create outputs. One counterpoint raised
by the paper, as well as others (Michael, 2020;
Potts, 2020), is the existence of unit tests, with as_sertions encoding examples of input/output pairs_
for blocks of code.[2] For example, systematically
observing blocks like x = 3; assert x == 3 could let
a system bootstrap the semantics of variable assignment, because a programmer is likely to write
assertions that will pass. These assertions constitute a form of implicit grounding embedded within
language modeling by the pragmatic concerns of
programmers, and they could potentially be leveraged to emulate a compiler.[3] However, it is not
immediately clear if unit tests provide “enough”
supervision to do this, even with unlimited data.
Viewing the debate about the power of assertions as central to the larger philosophical question, we aim to clarify it in more formal terms.
In this paper, we formally study whether observing a generalized notion of assertions can allow
a system to “understand” strings. An assertion is
a query about whether two strings evaluate to the
same value within a fixed context. This is motivated by the role of assertions in unit tests, where
asserting two expressions are equal suggests that
they have the same value within the test.
While assertions are directly motivated by the
compiler thought experiment, they also have
analogs in natural language, where sentences
make assertions about the world, and it is reasonable to expect some form of bias towards true
statements (Potts, 2020). Indeed, this is one of
Grice’s Maxims (Grice, 1975): a set of basic principles proposed to govern the pragmatics of natural language. For example, the truth conditions of
_This cat is the cat that Mary owns verify that two_
cats in the world identified in distinct ways are the
same entity. In general, we might expect a sentence to appear with higher frequency if its truth
conditions hold within its context, similar to an assertion in code, although of course there will also
be other factors governing sentence frequency besides this. In this sense, the example sentence resembles the Python statement assert cat1 == cat2,
where cat1 and cat2 are two Cat objects. See

Section 6 for more discussion of how assertions

2Unit tests are blocks of code in a software project that are
designed to test whether the core code is behaving correctly.
3Contexts like assertions can be seen as an argument in
favor of the distributional hypothesis (Harris, 1954).


and other formal concepts translate to natural language. We will generalize assertions to an abstract
formal language context, allowing us to study how
they can be used to emulate semantic relations.
Our findings are as follows. If every expression
in a language has the same value in every valid
context, then the language can be emulated using
a finite number of assertion queries (Section 4).
However, we construct a class of languages where
expressions can take different values in different
contexts, and where assertions do not enable emulation, i.e., infinite queries would be required (Section 5). Intuitively, this means that assertions do
not provide enough signal for a Turing-complete
emulator to fully “understand” languages from
this class. We go on to discuss differences between
our formal model and the less well-defined context
of natural language (Section 6). These results provide a formal way to characterize upper bounds on
whether it is possible to emulate the semantics of a
language from distributional properties of strings.
Within our framework, in certain settings, we find
that meaning cannot be learned from text alone.
We strengthen claims made by Bender and Koller
(2020) that assertions in code do not necessarily
provide sufficient signal for a language model to
emulate understanding. We do not make strong
claims about how these results transfer to natural
language, although we expect that the added complexity of natural language would make it, if anything, more difficult to “understand” than code.[4]

#### 2 Preliminaries

Let L Σ[⋆] denote a formal language over alpha_⊆_
bet Σ. We will use λ to denote the empty string.
Let (Σ[⋆])[2] denote the Cartesian product of Σ[⋆]

with itself; i.e., the set of all pairs of strings. Resembling Clark (2010), we refer to a tuple _l, r_
_⟨_ _⟩∈_
(Σ[⋆])[2] as a syntactic context. We also use other
symbols to refer to a context, e.g., κ = _l, r_ . We
_⟨_ _⟩_
denote by λ[2] the empty context _λ, λ_ .
_⟨_ _⟩_

**2.1** **Meaning**

We will model formal languages not just as sets
of strings, but as having an associated semantics.[5] Specifically, we assume the existence of a
_denotational semantics over every substring of L,_

4Appendix C documents and motivates conceptual
changes since the original arXiv version of the paper.
5We slightly abuse notation by using L to refer to both
a set of strings, and a set of strings paired with a denotation
function, which could be written more verbosely as ⟨L, �·�L⟩.


-----

which we now elaborate on. Let Y be a countable set of referents. First, we will say that some
_e_ Σ[⋆] is a valid expression within the context
_∈_
_κ =_ _l, r_ if there exists some contextual deno_⟨_ _⟩_
tation �e | κ�L ∈ _Y . Intuitively, this represents_
the value of e when it occurs in the larger context
_ler ∈_ _L. We will also use the notation �e | l, r�L_
where convenient. We will reserve _Y as a spe-_
_∅∈_
cial null symbol, defining �e | κ�L = ∅ iff e is not
a valid expression in the context κ.[6]

Each context κ (Σ[⋆])[2] also has a support, or
_∈_
set of expressions that are valid within it:

suppL(κ) = {e ∈ Σ[⋆] _| �e | κ�L ̸= ∅}._

**Example** Let L be a language of integers along
with the + operator, e.g., 2 + 2. Y is simply the
integers. We take �e | κ�L to map e to its standard
arithmetic interpretation, i.e., �2 + 6 | λ, + 4�L =
8. We take expressions that are not conventionally
well-formed to be invalid: e.g., �+ | λ, +�L = ∅.
Finally, let κ = ⟨λ, + 4⟩. Then suppL(κ) = L,
since any valid expression can occur within κ.

**2.2** **Strong Transparency**

As defined above, we make very few assumptions
about denotations. They are not necessarily compositional, and expressions may take different referents in different contexts. However, we saw in
the integer expression language that the meanings
of an expression did not depend on its context. We
now define a property formalizing this idea.

**Definition 1 (Strong transparency) L is strongly**
_transparent iff, for all e_ Σ[⋆], κ (Σ[⋆])[2], either
_∈_ _∈_
�e | κ�L = �e | λ[2]�L ̸= ∅, or �e | κ�L = ∅.

Informally, strong transparency says each e has
a well-defined denotation that exists independent
of context, and that this simple denotation can be
“plugged into” any context. Our previous example
expression 2 + 6 is strongly transparent because it
can be said to have a well-defined value 8 independent of its context. We could break strong

6Our simple model of denotations does not reflect the full
range of semantic theories that have been proposed for natural language. In particular, our denotations �e | κ�L depend
only on the linguistic context κ rather than any external world
state. This differs substantially from how truth conditions are
traditionally conceptualized in formal semantics (Heim and
Kratzer, 1998). For example, in our framework, the referent
of English �the dog | κ�L must be fixed with no regard for the
extralinguistic context. Section 6 further contrasts our setup
with the richer semantics of natural language.


transparency by adding bound variables to the language, e.g. `x = 2; x + 6 in Python. In this case,`
�x | κ�L non-vacuously depends on κ.
Strong transparency resembles referential transparency (Whitehead and Russell, 1925–1927), but
is a stronger condition, in that it does not allow the same name to ever refer to different values. For example, for a Python program, strong
transparency does not allow assigning local variables within a function, even if the function output
would remain completely specified by its inputs.

**2.3** **Assertion Queries**

We now define an oracle function providing assertion information about expressions in L, resembling assert e1 == e2 for two Python expressions
```
e1, e2. A system is granted access to this func
```
tion, and it can make assertion queries to it in order to learn about the semantics of L.[7] An assertion query tells us whether two expressions e, e[′]

are equivalent within the context κ.

**Definition 2 (Assertion oracle) For e, e[′],** Σ[⋆] and
_∈_
_κ_ (Σ[⋆])[2], define the assertion oracle
_∈_


Recall that we defined �e | κ�L = ∅ if e is not
valid in the context κ. In our example language of
integer expressions, for all κ, ℵL(4, 2 + 2 | κ) = 1,
since 4 = 2 + 2. The computational power of this
oracle depends on the complexity of the underlying semantics: for arbitrary semantics, it can become uncomputable. In this paper, though, we focus on classes of languages for which the denotation function and assertion oracle are computable.
The ℵL oracle is motivated by assertion statements in programming languages, which occur
naturally in environments like unit tests. The distribution of strings in a corpus of code should capture some notion of this oracle, since a programmer is more likely to assert two expressions are
equal if they are expected to have the same value.
Our goal is to study the limits of understanding
achievable from raw text, so we consider an “upper bound” setup by assuming a system has full
access to ℵL. Can the system use this powerful
oracle to emulate the underlying semantics?

7This resembles the role of queries in classical grammar
induction works (e.g., Angluin, 1987).


_ℵL(e, e[′]_ _| κ) =_


�
1 if �e | κ�L = �e[′] _| κ�L_
0 otherwise.


-----

**2.4** **Turing Machines**

Our notion of language understanding will be
based around the idea of emulation, which in turn
requires a model of computational realizability.
We will use Turing machines (Turing, 1936) as a
model of universal computation. We write µ(e)
for the output of Turing machine µ evaluated on
input e Σ[⋆]. We will also define an oracle Tur_∈_
ing machine as a standard Turing machine that can
compute a blackbox “oracle” function f as a subroutine. We imagine the machine has a special
_query instruction and tape. After writing x to the_
query tape and executing the query instruction, the
query tape will contain f (x). We will write µf (e)
for the Turing machine µ evaluated on input e with
oracle access to f . In the case where f = ℵL, we
will simply write µL(e). Whereas, in computability theory, oracle Turing machines are generally
leveraged to make reductions from uncomputable
problems, here we will use them to formalize the
ability of an emulator to make assertion queries
about L. This oracle provides additional power
because these queries contain additional information beyond that encoded in the input expression.

#### 3 Research Question: Do Assertions Enable Emulation?

There is a long history in AI of trying to define
and measure understanding. Turing (1950) constitutes an early behaviorist perspective; more recent approaches tend to emphasize not just an external view of a system’s behavior, but also “how
it is achieved” (Levesque, 2014). Understanding
can be behaviorally diagnosed in neural models
by evaluating them on benchmarks (Wang et al.,
2018). An alternate approach is probing (Adi
et al., 2017; Conneau et al., 2018; Hupkes and
Zuidema, 2018; Hewitt and Liang, 2019; Belinkov
and Glass, 2019), which investigates how directly
a model’s representations encode semantic relations by measuring if they can be easily decoded
from them. Similarly, we take the position that
systems are capable of understanding if they emu_late representations that are isomorphic to under-_
lying meaning under important semantic relations
like equivalence. We will formalize this in Question 1, which asks whether such emulation is possible using assertions.

**Definition 3 (** -emulation) A class of languages
_ℵ_
over Σ is _-emulatable if there exists an oracle_
_L_ _ℵ_
Turing machine µ and standard Turing machine δ


Figure 1: An illustration of Definition 3. _µ emu-_
lates a representation of each expression using assertion queries. Then, δ compares the emulated representations to determine equivalence.

such that, for all L ∈L, κ ∈ (Σ[⋆])[2], and e, e[′] _∈_
suppL(κ),

�e | κ�L = �e[′] _| κ�L ⇐⇒_ _δ�µL(e), µL(e[′]) | κ�._

_µ can be thought of as an emulator that eval-_
uates expressions, whereas δ receives two values
and decides whether they are equal. Crucially,
only µ has direct access to ℵL. δ can only use
information from the oracle to the extent that it is
encoded in the representations µL(e) and µL(e[′]).

Definition 3 formulates emulation as a decision
problem, as is typical in theoretical computer science. Equivalently, δ can be replaced by a computable function ρ such that ρ(µL(e) | κ) eval_uates µL(e) in context κ, i.e., its output string is_
isomorphic to �e | κ�L under =. The functions δ
and ρ are Turing-reducible to each other, implying
that if one definition is satisfied, so is the other.
With our definition of emulation in place, we
can formally state the research question:

**Question 1 For a class of languages** _, is_ _-_
_L_ _L ℵ_
_emulatable?_

How does Question 1 relate to understanding
in large language models? We imagine that, with
sufficiently large amounts of data, the frequencies
of strings in L carry enough signal such that the
language model objective “supervises” access to
_ℵL. Thus, µL(e) can be thought of as the lan-_
guage model representation of an expression e.
We then hope to recover underlying semantic relations from the representations produced by the


-----

```
from typing import Callable
AssertType = Callable[[str, str, str, str], bool]
def emulate(expr: str, asserteq: AssertType) −> int:
  for idx, cand in enumerate (all_strings()):
       if asserteq(expr, cand, "", ""):
          return idx

```
Figure 2: emulate implements an emulator µ. Let all_strings be an iterable enumerating all strings in Σ[⋆]. We
provide a concrete implementation of all_strings in Figure 5.


language model via some function δ. The class
_L_
corresponds to a set of hypothesis languages over
which the language model must search for the true
_L. We will see that whether emulation is possible_
will depend on the properties of .
_L_
Stepping back, Question 1 bears on the
role of assertions raised by Bender and Koller
(2020). Does observing assertions allow a Turingcomplete system to emulate a compiler? In more
general terms, are assertions powerful enough implicit grounding to achieve representations that encode the denotational semantics of a language?

#### 4 Strong Transparency

We first consider the case where the language being learned is known to be strongly transparent.
Let TRANSPARENT denote the class of strongly
transparent languages. We will show that TRANS
PARENT is -emulatable. The core idea of the
_ℵ_
proof is to construct a canonical form for each expression. The canonical form is the first expression in a lexicographic ordering that the assertion
oracle deems equivalent to the target expression.
For technical reasons, the emulator returns the index of this string under the lexicographic order.

**Theorem 1 TRANSPARENT is** _-emulatable._
_ℵ_

_Proof. As Python is Turing-complete, we write_
_µ : Σ[⋆]_ _→_ N as a Python function emulate in Figure 2. The function receives as input an expression expr and a callback function asserteq to an oracle computing ℵL. For each e ∈ Σ[⋆], there exists
_e[⋆]_ _∈_ Σ[⋆] such that ℵL(e, e[⋆] _| λ[2]) = 1. In the_
“worst case”, this holds when e[⋆] = e by symmetry. By construction, all_strings reaches all strings
in finite time. Therefore, the number of loop iterations before reaching e[⋆] is finite. We can conclude
that emulate halts on every e Σ[⋆], establishing that
_∈_
it is computable.


Now, we move towards justifying that the emulation is correct for every κ (Σ[⋆])[2]. We note that
_∈_
_δ is simply the indicator function for equality over_
the natural numbers:


_δ(m, m[′]_ _κ) =_
_|_


�
1 if m = m[′]

0 otherwise.


The function emulate outputs i ∈ N, the index of
the first string e[⋆] such that �e | λ[2]�L = �e[⋆] _| λ[2]�L._
Now, let e, e[′] _∈_ suppL(κ) be different inputs to µ.
Because the enumeration order of the for loop is
fixed across computation of µL(e) and µL(e[′]):

_µL(e) = µL(e[′]) ⇐⇒_ �e | λ[2]�L = �e[⋆] _| λ[2]�L_
_∧_ �e[′] _| λ[2]�L = �e[⋆]_ _| λ[2]�L_
_⇐⇒_ �e | λ[2]�L = �e[′] _| λ[2]�L_
_⇐⇒_ �e | κ�L = �e[′] _| κ�L,_

where the last step follows by strong transparency.
We conclude that the conditions for emulation
(Definition 3) are fully satisfied.

Through a simple construction, we have shown
it is possible to emulate meaning from assertion
queries for languages with strongly transparent semantics. The number of bits in the emulated representation µL(e) is linear in the size of e. In the
next section, we consider what happens without
strong transparency, where, among other complexities, values can be bound to variables, complicating the construction used in Theorem 1.

#### 5 General Case

Requiring strong transparency precludes a broad
class of linguistic patterns allowing an expression
to refer to different values in different contexts.
For example, this includes assigning variable or
function names in Python, or binding pronouns in


-----

```
def leq() −> bool:
  return n < M
print(leq())

```
```
def leq() −> bool:
  return n < M
print (True)

```

Figure 3: Templates for strings in Lm, for m ∈ N ∪{∞}. M evaluates to m in all strings, while other expressions
are evaluated according to Python 3.8 semantics. The metavariable n ranges over N to form different strings in
_Lm, and is serialized as a decimal string._


natural language. These constructions can make
emulation impossible to achieve from assertions.
We will construct a class of languages based on
Python where emulation is uncomputable.

**Definition 4 Let LEQ = {Lm | m ∈** N ∪{∞}},
where strings in Lm are defined according to Figure 3. For semantics, we first define �M | κ�Lm =
_m. For any other ler ∈_ _Lm that is a well-formed_
Python 3.8 expression, we define �e | l, r�Lm as
the value of e assigned by the Python interpreter
in the context _l, r_ . For strings that are not valid
_⟨_ _⟩_
Python expressions, define �e | l, r�Lm = ∅.

What does it take to emulate the expressions
```
leq() and True in Lm? If we knew m, then we

```
could emulate them by simply comparing n < m.
However, it turns out that recovering m for any
_Lm ∈_ LEQ is not possible with a fixed number of
assertion queries. Formalizing this, we will show
that LEQ is not -emulatable.[8]
_ℵ_

**Theorem 2 LEQ is not** _-emulatable._
_ℵ_

_Proof. Without loss of generality, we focus on the_
contexts for leq()[9] and True within print( `), each of`

_·_
which is parameterized by some value of n. Notationally, we identify each Lm with m, and each
context with its parameter n. This enables shorthand like �e | n�m for the denotation of the expression e in the context parameterized by n in Lm.
When m =, it holds for all n that
_∞_
(leq(), True _n) = 1. To satisfy emulation of_
_ℵ∞_ _|_
_e_ `leq(), True`, µ makes a finite number of
_∈{_ _}_ _∞_
assertion queries

_ℵ∞(leq(), True | ni)._

for some sequence of contexts n1, · · ·, nq, which
we assume without loss of generality is sorted in
increasing order. We can adversarially construct
_m[′]_ = such that all these queries are the same,
_̸_ _∞_

8Another example of a non-ℵ-emulatable language takes
```
M to be a finite list of integers and replaces n < M with n in M.

```
9The only “valid” context for leq() is within print(·).
The denotation of leq() when it occurs next to def is ∅.


and thus µ∞(e) = µm′(e) for both e. To implement this, we simply set m[′] = nq + 1. Since
_µ∞(e) = µm′(e), we conclude that, for all n,_

_δ(µm′(leq()), µm′(True) | n) =_

_δ(µ_ (leq()), µ (True) _n)._
_∞_ _∞_ _|_

On the other hand, consider n > nq. In this case,

�leq() | n�m′ = False

`leq()` _n_ = True,
� _|_ �∞

which can be rewritten as


�leq() | n�m′ ̸= �True | n�m′

`leq()` _n_ = `True` _n_ _._
� _|_ �∞ � _|_ �∞

Therefore, the conditions of -emulation (Defini_ℵ_
tion 3) cannot be satisfied for both Lm′ and L∞.
This implies that LEQ is not -emulatable.
_ℵ_

**5.1** **Discussion**

We briefly summarize this result in less formal
terms. LEQ contains languages Lm defined by

Figure 3. Every program in each Lm is easily
computable. With knowledge of the Python interpreter and m, any agent could execute all of
these programs. This can be formalized by observing that, for a fixed m, the class {Lm} is ℵemulatable. Rather, what we have shown is that,
with finite time, it is impossible for an ungrounded
agent to emulate Lm using assertion queries when
_m is unknown in advance. In other words, without_
prior knowledge of m, no algorithm can use assertions to disambiguate which notion of = is used by
_Lm from the infinite other possibilities. In a rough_
sense, m can be thought of as a cryptographic
key enabling linguistic understanding: agents that
know m can directly emulate Lm, but agents without it cannot, at least using assertions.[10]

10Alternatively, we can take a more complexity-theoretic
perspective by measuring the number of queries needed to
emulate up to a bounded context size. Fix a maximum n.


-----

There i s a number .

_n i s_ l e s s than i t .


There i s a number .
Zero equals one .


Figure 4: An informal construction adapting the program templates in Figure 3 to English. Under our framework,
two sentences are considered equivalent if they are true in exactly the same set of contexts. If the number is allowed
to be, this cannot be done in general for the final lines of each template.
_∞_


Theorem 2 does not use the fact that δ must be
computable, as opposed to an arbitrary function.
Even if δ is an arbitrary function, it could not disambiguate whether m halts based on queries.
It is more precise to state Theorem 2 in a formal
language, but an argument similar to Theorem 2
can be adapted to a natural language like English.
An example is shown in Figure 4, where we define
the meaning of a sentence as its truth conditions,
and we imagine the class of candidate languages is
formed by varying the unspecified number, which
can potentially be . Deciding if n is less than it
_∞_
has the same truth conditions as Zero equals one is
equivalent to comparing leq() and True. A system
must necessarily fail to emulate the semantics of
these expressions in some context, for some secret
number. The rest of the paper further explores the
implications and limitations of applying our formal model to natural language.

#### 6 Towards Natural Language

As discussed in Section 1, our results are inspired
by the thought experiment of whether a language
model can use raw code to learn a compiler. A
goal of this, of course, is to examine whether understanding can be acquired from natural language
text in a simplified setting. In principle, our formal
results can bear on this broader question about natural language, although some differences emerge
when extending the results to a less well-defined
setting. In many cases, these differences appear
to make the task of learning meaning harder, suggesting that our negative claim in a simpler setting
(Theorem 2) may still hold as an impossibility result. We now discuss some points of difference
between our formal model and natural language.

Then we can use binary search with O(log n) queries to find
the value of m. Since the number of context bits is O(log n),
the numbers of queries is O(|κ|), beating the O(|Σ|[|][κ][|]) query
complexity achievable by brute force. This perspective somewhat resembles Pratt-Hartmann and Third (2006) and other
work in semantic complexity theory on the computational
complexity of evaluating fragments of natural language.


**Truth Conditions** There are connections between our framework and the concepts of truth
values and truth conditions in linguistic semantics.
For a boolean-valued expression e, a truth value
corresponds to computing �e | κ�L in a fixed context. On the other hand, truth conditions correspond roughly to a function computing �e | κ�L
for any κ. A crucial difference, though, is that
these conditions cannot be intensional (Von Fintel and Heim, 2011), i.e., they are not functions
of the world state, but rather of the linguistic context only. In this sense, emulation corresponds to
recovering the ability to resolve non-intensional
truth conditions of sentences. This model is natural for formalizing a closed programming language environment, e.g., with no environment
variables or user input, since in this case the program state is specified completely by the linguistic
context. On the other hand, English has common
elements like that whose meaning can change depending on world state external to language. Perhaps allowing such elements would only make understanding more difficult; or, arguably, generally
impossible, since there is no way for the model
to observe the grounding world state using only
an assertion oracle. We are inclined to believe
that, since such changes would make understanding more difficult, Theorem 2 would still hold
as an impossibility result. However, future work
would be needed to make this idea precise.

**Possible Worlds** In the last paragraph, we discussed how mutable world state is an additional complexity of natural language compared
to our setup. Similarly, speakers of natural
languages have imperfect information about the
world around them, which can be captured by
modeling the referent of an expression over a set
of possible worlds, rather than within a specific
evaluation context. In Appendix A, we explore to
what degree this setting makes the task of learning to understand more difficult. In adapting our
model to this context, the assertion oracle must become “modal” in the sense that it quantifies over
sets of worlds. We explore two different models of


-----

modality for the oracle, corresponding to different
physical interpretations. In one case, Theorem 1
and Theorem 2 apply analogously, while, in the
other, emulation becomes an ill-defined problem.

**Denotation vs. Intent** Bender and Koller (2020)
distinguish between standing meaning and com_municative intent, reflecting a distinction between_
denotational semantics and other pragmatic intentions that a speaker has in producing an utterance.
In this paper, it is most straightforward to take
�e | κ�L to reflect standing meaning. In principle,
we could imagine that it represents the speaker’s
communicative intent, and that an omniscient oracle ℵL can reveal information about the speaker’s
intents to the system. Even with this unrealistically powerful oracle, Theorem 2 says that the system cannot emulate the speaker’s intents.

**Competence vs. Performance** Chomsky (1965)
differentiates competence and performance in linguistic theory, where competence corresponds
roughly to the correct algorithmic modeling of
a linguistic process, and performance describes
its implementation subject to resource constraints
like memory. Arguably, agents might be said to
understand language if they are competent in this
sense, even if they sometimes make performance
errors. In contrast, our definition of emulation
(Definition 3) permits no performance errors. In
future work, it would be interesting to adapt an
approximate notion of emulation that tolerates performance errors in order to more closely target understanding in a sense reflecting competence.

**Other Relations** Theorem 1 and Theorem 2

investigate whether ℵL can be used to emulate
meaning representations that preserve an equivalence relation. While equivalence is an important
part of semantics, other semantic relations like entailment are also necessary for language understanding. In Appendix B, we show a generalization of Theorem 5 extends to any semantic relation. In other words, referential transparency also
enables emulation of relations besides =.

**Other Oracles** We believe assertions are a fairly
general model of the types of semantics encoded
in unsupervised learning resulting from a pragmatic bias for truth; however, it is possible other
information is also represented, resulting from
other pragmatic biases governing language usage
and dataset creation. This additional information


could be formalized as access to additional oracles. It would be exciting to formalize the power
of multimodal setups by analyzing the interactions
of oracles enabled by different input modalities.

#### 7 Stepping Back

In this work, we formalized an argument that was
raised by Bender and Koller (2020) as a thought
experiment. Bender and Koller (2020) question
whether unsupervised training objectives are the
right goal to target for achieving natural language
understanding. If meaning is defined as identifying which object in the real world, or which set of
situations, a linguistic element refers to, then, in a
direct sense, an ungrounded system cannot understand meaning. But Bender and Koller (2020) go
farther than this, claiming that an ungrounded system cannot even emulate understanding because it
is not clear how a system should learn to interpret
strings, even if it can model their distribution. We
formalize this idea of emulation as -emulation.
_ℵ_
One counterargument mentioned by Bender and
Koller (2020) is that indirect forms of grounding do exist in programming and natural language,
which we formalize as assertions. The syntactic
distributions of statements like assert allow us to
indirectly observe semantic relations over the denotations. Assertions are one way that the distribution of strings in a corpus is not blind to their semantics. By studying them, we study whether this
indirect grounding enables a computational system to emulate the underlying semantic relations.

**Key Takeaways** While assertions allow a system to emulate semantic relations in simple cases
where the semantics are referentially transparent, we find that linguistic constructs like variable
binding bring this task in conflict with the fundamental laws of computability. In other words,
under our formal model of meaning and emulation, it is not just intractable for an ungrounded
system to emulate understanding of a formal language, but, in some cases, impossible. We provide
constructive examples where understanding must
necessarily break down. We present these results
in a well-defined framework building off formal
approaches in logic, linguistics, and computer science. While we do not prove anything about natural languages, we do show that ungrounded models must fail to emulate equivalence in a very simple setting. A similar result likely extends to natural language understanding as well, which among


-----

other things, requires modeling referential identity
(e.g., for sentences like Manny is the cat). Further,
we believe much of our framework can be readily
adopted in other works formalizing understanding
in Turing-complete systems.

**Open Questions** In this work, we have focused
on utterances, by default, as opposed to dialogues.
An exciting extension would be to formalize a dialogue between two speakers, interrupted by the
“octopus” of Bender and Koller (2020).[11] Existing
theories of discourse could potentially be synthesized with this framework. What linguistic properties besides referential transparency relate to emulatability? Can this framework be extended to formalize multimodal setups, where multiple oracles
from different domains can potentially be combined to gain additional power? Finally, is there
a natural way to relax our standard of emulation
towards a probabilistic definition, and how would
this change the results?

#### Acknowledgments

We thank Mark-Jan Nederhof for his excellent suggestions. We also thank Dana Angluin, Matt Gardner, Eran Yahav, Zachary Tatlock,
Kyle Richardson, Ruiqi Zhong, Samuel Bowman,
Christopher Potts, Thomas Icard, and Zhaofeng
Wu for their feedback on various versions of this
work. Further thanks to our anonymous reviewers
and researchers at the Allen Institute for AI and
UW NLP. Finally, we appreciate the lively online
discussion of the paper, which informed updates
to the camera-ready version.

#### References

Yossi Adi, Einat Kermany, Yonatan Belinkov, Ofer
[Lavi, and Yoav Goldberg. 2017. Fine-grained](https://openreview.net/forum?id=BJh6Ztuxl)
[analysis of sentence embeddings using auxiliary](https://openreview.net/forum?id=BJh6Ztuxl)
[prediction tasks. In 5th International Confer-](https://openreview.net/forum?id=BJh6Ztuxl)
_ence on Learning Representations, ICLR 2017,_
_Toulon, France, April 24-26, 2017, Conference_
_Track Proceedings. OpenReview.net._

[Dana Angluin. 1987. Learning regular sets from](https://doi.org/10.1016/0890-5401(87)90052-6)
[queries and counterexamples.](https://doi.org/10.1016/0890-5401(87)90052-6) _Inf. Comput.,_
75(2):87–106.

11The octopus thought experiment imagines a deep-sea octopus O observes a dialogue between two humans by intercepting an underwater cable. Could O learn to emulate the
role of one of the speakers without exposure to life on land?


[Yonatan Belinkov and James Glass. 2019. Analy-](https://doi.org/10.1162/tacl_a_00254)
[sis methods in neural language processing: A](https://doi.org/10.1162/tacl_a_00254)
[survey.](https://doi.org/10.1162/tacl_a_00254) _Transactions of the Association for_
_Computational Linguistics, 7:49–72._

Emily M. Bender and Alexander Koller. 2020.

[Climbing towards NLU: On meaning, form, and](https://doi.org/10.18653/v1/2020.acl-main.463)
[understanding in the age of data. In Proceed-](https://doi.org/10.18653/v1/2020.acl-main.463)
_ings of the 58th Annual Meeting of the As-_
_sociation for Computational Linguistics, pages_
5185–5198, Online. Association for Computational Linguistics.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla
Dhariwal, Arvind Neelakantan, Pranav Shyam,
Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger,
Tom Henighan, Rewon Child, Aditya Ramesh,
Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler,
Mateusz Litwin, Scott Gray, Benjamin Chess,
Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario
[Amodei. 2020. Language models are few-shot](http://arxiv.org/abs/2005.14165)
[learners.](http://arxiv.org/abs/2005.14165)

[Noam Chomsky. 1965. Aspects of the Theory of](http://www.colinphillips.net/wp-content/uploads/2015/09/chomsky1965-ch1.pdf)
_[Syntax, volume 11. MIT press.](http://www.colinphillips.net/wp-content/uploads/2015/09/chomsky1965-ch1.pdf)_

[Alexander Clark. 2010. Three learnable models](https://link.springer.com/chapter/10.1007/978-3-642-13089-2_2)
[for the description of language. In Language](https://link.springer.com/chapter/10.1007/978-3-642-13089-2_2)
_and Automata Theory and Applications, pages_
16–31, Berlin, Heidelberg. Springer Berlin Heidelberg.

Alexis Conneau, German Kruszewski, Guillaume
Lample, Loïc Barrault, and Marco Baroni.
[2018. What you can cram into a single $&!#*](https://doi.org/10.18653/v1/P18-1198)
[vector: Probing sentence embeddings for lin-](https://doi.org/10.18653/v1/P18-1198)
[guistic properties. In Proceedings of the 56th](https://doi.org/10.18653/v1/P18-1198)
_Annual Meeting of the Association for Compu-_
_tational Linguistics (Volume 1: Long Papers),_
pages 2126–2136, Melbourne, Australia. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
[Kristina Toutanova. 2019. BERT: Pre-training](https://doi.org/10.18653/v1/N19-1423)
[of deep bidirectional transformers for language](https://doi.org/10.18653/v1/N19-1423)
[understanding.](https://doi.org/10.18653/v1/N19-1423) In Proceedings of the 2019
_Conference of the North American Chapter_
_of the Association for Computational Linguis-_
_tics: Human Language Technologies, Volume_
_1 (Long and Short Papers), pages 4171–4186,_


-----

Minneapolis, Minnesota. Association for Computational Linguistics.

[Herbert P Grice. 1975. Logic and conversation. In](https://www.ucl.ac.uk/ls/studypacks/Grice-Logic.pdf)
_Speech acts, pages 41–58. Brill._

Zellig S. Harris. 1954. [Distributional structure.](https://doi.org/10.1080/00437956.1954.11659520)
_WORD, 10(2-3):146–162._

[Irene Heim and Angelika Kratzer. 1998. Seman-](http://users.uoa.gr/~wlechner/Creteling2017/Textbooks/Heim%20and%20Kratzer%201998.pdf)
_[tics in Generative Grammar. Blackwell.](http://users.uoa.gr/~wlechner/Creteling2017/Textbooks/Heim%20and%20Kratzer%201998.pdf)_

John Hewitt and Percy Liang. 2019. [De-](https://doi.org/10.18653/v1/D19-1275)
[signing and interpreting probes with control](https://doi.org/10.18653/v1/D19-1275)
[tasks.](https://doi.org/10.18653/v1/D19-1275) In Proceedings of the 2019 Confer_ence on Empirical Methods in Natural Lan-_
_guage Processing and the 9th International_
_Joint Conference on Natural Language Pro-_
_cessing (EMNLP-IJCNLP), pages 2733–2743,_
Hong Kong, China. Association for Computational Linguistics.

Laurence R. Horn and Heinrich Wansing. 2020.

[Negation. In Edward N. Zalta, editor, The Stan-](https://plato.stanford.edu/entries/negation/)
_ford Encyclopedia of Philosophy, spring 2020_
edition. Metaphysics Research Lab, Stanford
University.

[Dieuwke Hupkes and Willem Zuidema. 2018. Vi-](https://doi.org/10.24963/ijcai.2018/796)
[sualisation and ’diagnostic classifiers’ reveal](https://doi.org/10.24963/ijcai.2018/796)
[how recurrent and recursive neural networks](https://doi.org/10.24963/ijcai.2018/796)
[process hierarchical structure (extended ab-](https://doi.org/10.24963/ijcai.2018/796)
[stract). In Proceedings of the Twenty-Seventh](https://doi.org/10.24963/ijcai.2018/796)
_International Joint Conference on Artificial In-_
_telligence, IJCAI-18, pages 5617–5621. Inter-_
national Joint Conferences on Artificial Intelligence Organization.

[Hector Levesque. 2014. On our best behaviour.](https://doi.org/10.1016/j.artint.2014.03.007)
_Artificial Intelligence, 212._

Julian Michael. 2020. [To dissect an octopus:](https://blog.julianmichael.org/2020/07/23/to-dissect-an-octopus.html)
[Making sense of the form/meaning debate.](https://blog.julianmichael.org/2020/07/23/to-dissect-an-octopus.html)

Christopher Potts. 2020. [Is it possible for lan-](https://chrisgpotts.medium.com/is-it-possible-for-language-models-to-achieve-language-understanding-81df45082ee2)
[guage models to achieve understanding?](https://chrisgpotts.medium.com/is-it-possible-for-language-models-to-achieve-language-understanding-81df45082ee2)

[Ian Pratt-Hartmann and Allan Third. 2006. More](https://doi.org/10.1305/ndjfl/1153858644)
[fragments of language. Notre Dame J. Formal](https://doi.org/10.1305/ndjfl/1153858644)
_Logic, 47(2):151–177._

Colin Raffel, Noam Shazeer, Adam Roberts,
Katherine Lee, Sharan Narang, Michael
Matena, Yanqi Zhou, Wei Li, and Peter J. Liu.
[2019. Exploring the limits of transfer learning](http://arxiv.org/abs/1910.10683)
[with a unified text-to-text transformer.](http://arxiv.org/abs/1910.10683)


Anna Rogers, Olga Kovaleva, and Anna
Rumshisky. 2020. [A primer in BERTol-](http://arxiv.org/abs/2002.12327)
[ogy: What we know about how bert works.](http://arxiv.org/abs/2002.12327)

Ian Tenney, Dipanjan Das, and Ellie Pavlick. 2019.

[BERT rediscovers the classical NLP pipeline.](https://doi.org/10.18653/v1/P19-1452)
In Proceedings of the 57th Annual Meeting of
_the Association for Computational Linguistics,_
pages 4593–4601, Florence, Italy. Association
for Computational Linguistics.

[Alan M. Turing. 1936. On computable numbers,](https://www.cambridge.org/core/journals/journal-of-symbolic-logic/article/abs/m-turing-on-computable-numbers-with-an-application-to-the-entscheidungs-problcm-proceedings-of-the-london-mathematical-society-2-s-vol-42-19361937-pp-230265/4DFCA89035F7F7C5BF4DB5129B8BB09E)
[with an application to the Entscheidungsprob-](https://www.cambridge.org/core/journals/journal-of-symbolic-logic/article/abs/m-turing-on-computable-numbers-with-an-application-to-the-entscheidungs-problcm-proceedings-of-the-london-mathematical-society-2-s-vol-42-19361937-pp-230265/4DFCA89035F7F7C5BF4DB5129B8BB09E)
[lem. J. of Math, 58(345-363):5.](https://www.cambridge.org/core/journals/journal-of-symbolic-logic/article/abs/m-turing-on-computable-numbers-with-an-application-to-the-entscheidungs-problcm-proceedings-of-the-london-mathematical-society-2-s-vol-42-19361937-pp-230265/4DFCA89035F7F7C5BF4DB5129B8BB09E)

[Alan M. Turing. 1950. Computing machinery and](https://doi.org/10.1093/mind/LIX.236.433)
[intelligence. Mind, LIX(236):433–460.](https://doi.org/10.1093/mind/LIX.236.433)

[Kai Von Fintel and Irene Heim. 2011. Intensional](https://github.com/fintelkai/fintel-heim-intensional-notes/blob/master/fintel-heim-2011-intensional.pdf)
[semantics. Unpublished Lecture Notes.](https://github.com/fintelkai/fintel-heim-intensional-notes/blob/master/fintel-heim-2011-intensional.pdf)

Alex Wang, Amanpreet Singh, Julian Michael,
Felix Hill, Omer Levy, and Samuel Bowman.
2018. [GLUE: A multi-task benchmark and](https://doi.org/10.18653/v1/W18-5446)
[analysis platform for natural language under-](https://doi.org/10.18653/v1/W18-5446)
[standing. In Proceedings of the 2018 EMNLP](https://doi.org/10.18653/v1/W18-5446)
_Workshop BlackboxNLP: Analyzing and Inter-_
_preting Neural Networks for NLP, pages 353–_
355, Brussels, Belgium. Association for Computational Linguistics.

Alfred North Whitehead and Bertrand Russell.
1925–1927. _[Principia Mathematica.](https://philpapers.org/rec/RUSPMV)_ Cambridge University Press.

Yoad Winter. 2016. Elements of Formal Seman_tics: An Introduction to the Mathematical The-_
_ory of Meaning in Natural Language._ Edinburgh University Press.

Dani Yogatama, Cyprien de Masson d’Autume,
Jerome Connor, Tomas Kocisky, Mike
Chrzanowski, Lingpeng Kong, Angeliki
Lazaridou, Wang Ling, Lei Yu, Chris Dyer, and
[Phil Blunsom. 2019. Learning and evaluating](http://arxiv.org/abs/1901.11373)
[general linguistic intelligence.](http://arxiv.org/abs/1901.11373)

Hongming Zhang, Xinran Zhao, and Yangqiu
[Song. 2020. WinoWhy: A deep diagnosis of es-](https://doi.org/10.18653/v1/2020.acl-main.508)
[sential commonsense knowledge for answering](https://doi.org/10.18653/v1/2020.acl-main.508)
[Winograd schema challenge. In Proceedings of](https://doi.org/10.18653/v1/2020.acl-main.508)
_the 58th Annual Meeting of the Association for_
_Computational Linguistics, pages 5736–5745,_
Online. Association for Computational Linguistics.


-----

```
from itertools import count, product
from typing import Iterable
def all_strings() −> Iterable[str ]:
  for length in count():
     iterable = product(*[SIGMA for _ in range(length)])
     yield from ("".join(x) for x in iterable)

```
Figure 5: An concrete implementation of all_strings, which is referenced in Figure 2 and Figure 6.


#### A Multiple Worlds

Programs execute in well-defined environments
with a clear state. Speakers of natural language,
on the other hand, have imperfect information and
beliefs about the world around them. Thus, it can
be more natural to model grounding context for
language as a set of possible worlds, rather than a
single world state. We formalize this in two different ways (with two different physical interpretations) and explore how it affects our results.
Let W be a set of all possible worlds. We redefine denotations to be intensionalized (Von Fintel and Heim, 2011), i.e., we write _e_ _κ_ as
� _|_ �[w]
the denotation of e in the context κ, evaluated
in world w _W_ . Assume for simplicity that
_∈_
_Y_ = 0, 1, . We will now introduce modal
_{_ _∅}_
denotations and assertions using a generic modal
_quantifier_, which reduces a sequence of worlds
_⊙_
to a boolean value according to some intensional
predicate. This quantifier controls how multiple
possible worlds are collapsed to form denotations
and query outputs.

**Definition 5 (Modal denotation) Let** be a modal
_⊙_
quantifier. For all e Σ[⋆], κ (Σ[⋆])[2], define
_∈_ _∈_

�
_⊙�e | κ�L =_ �e | κ�L[w][.]

_w∈W_

We will write the previously defined assertion
oracle to apply in a specific world w, i.e. ℵL[w][. We]
also extend it to quantify over multiple worlds:

**Definition 6 (Modal assertion) Let** be a modal
_⊙_
quantifier. For all e Σ[⋆], κ (Σ[⋆])[2], define
_∈_ _∈_

�
_⊙ℵL(e, e[′]_ _| κ) =_ _ℵL[w][(][e, e][′][ |][ κ][)][.]_

_w∈W_

Specifically, we consider = _2, 3_, corre_⊙_ _{_ _}_
sponding to universal and existential quantifiers
over worlds. Thus, 2 can be thought of as as
_∀_
over worlds, and 3 can be thought of as . For
_∃_


either quantifier, if any �e | κ�L[w] [=][ ∅][, we define]
_⊙�e | κ�L = ∅_ as well. Each quantifier will have
a different physical interpretation. With universal
quantification, we will find that results analogous
to Theorem 1 and Theorem 2 hold. With existential quantification, it turns out that the equivalence
class of µ is underspecified. In other words, not
only is it impossible to compute an emulator with
a finite number of assertion queries, but, even with
infinite assertions, there is no consistent way to
emulate the underlying modal semantics.

**A.1** **Universal Quantification**

In the first case we let = 2. Two expressions
_⊙_
are viewed as having the same meaning if they
are equivalent in every possible belief world. This
is interpretable as observing text L2 written by a
single author whose belief state is represented by
multiple possible worlds. The author only asserts a
statement is true if it is consistent across all worlds
that they believe are possible.

In this setting, we will show that the modal assertion oracle uniquely specifies a modal denotation for each expression, up to isomorphism. In
other words, as with the non-modal assertion oracle, each assertion query would let us decide some
relation between two expressions. Thus, the same
results for the non-modal setting discussed in the
main body of the paper will also hold here.

**Theorem 3 Consider e, e[′]** Σ[⋆] _and any context_
_∈_
_κ ∈_ (Σ[⋆])[2] _such that 2�e | κ�L ̸= ∅_ _and 2�e[′]_ _|_
_κ�L ̸= ∅. Then,_

_2�e | κ�L = 2�e[′]_ _| κ�L ⇐⇒_ _2ℵL(e, e[′]_ _| κ)._


-----

_Proof._

_2�e | κ�L = 2�e[′]_ _| κ�L_

� �
_⇐⇒_ �e | κ�L[w] [=] �e[′] _| κ�L[w]_

_w∈W_ _w∈W_

_⇐⇒_ � ��e | κ�L[w] [=][ �][e][′][ |][ κ][�]L[w]�

_w∈W_

�
_⇐⇒_ _ℵL[w][(][e, e][′][ |][ κ][)]_

_w∈W_

_⇐⇒_ _2ℵL(e, e[′]_ _| κ)._

Crucial to this simple proof is the fact that is
_∧_
distributive over =. This is specific to the quantifier being 2. Theorem 3 implies that 2�e | κ�L
can be recovered from modal assertion queries
analogously to the non-modal case. Thus, results
analogous to Theorem 1 and Theorem 2 apply for
emulating 2�e | κ�L using queries to 2ℵL.

**A.2** **Existential Quantification**

In the second case we let = 3. Two expres_⊙_
sions are viewed as having the same meaning if
they are equivalent in some world. This is interpretable as observing a large dataset of text L3
generated by many authors, each with a different
single belief world w. In the corpus, we imagine
two expressions can be asserted to be equivalent in
some context if any of the authors would consider
them to be equal in that context.
In this case, assertions do not even fully specify
equivalence between the modal denotations. This
is a stronger sense in which meaning cannot be
emulated from assertion queries. Emulation is not
just impossible with finite assertions, but mathematically underspecified.

**Theorem 4 There exist e, e[′]** _E(L) and κ_
_∈_ _∈_
(Σ[⋆])[2] _such that 3�e | κ�L ̸= ∅_ _and 3�e[′]_ _|_
_κ�L ̸= ∅, and also 3ℵL(e, e[′]_ _| κ) = 1 is con-_
_sistent with either 3�e | κ�L = 3�e[′]_ _| κ�L or_
_3�e | κ�L ̸= 3�e[′]_ _| κ�L._

_Proof. We construct an example with expressions_
_e1, e2 in a single context κ. Fix W = {w1, w2}._
Table 1 shows two versions of this modal setup. In
both versions of the universe, 3ℵL(e, e[′] _| κ) = 1._
However, on the left, 3�e | κ�L = 3�e[′] _| κ�L,_
while, on the right, the opposite holds. So, with 3,
modal assertions do not uniquely determine equivalence of modal denotations.

|Col1|e e 1 2|ℵ|e e 1 2|ℵ|
|---|---|---|---|---|
|w 1 w 2|0 0 0 0|1 1|0 0 0 1|1 0|
|3|0 0|1|0 1|1|


Table 1: Two tables (separated by a thick line) representing two different versions of W . Within each table, each cell i, j in the main 2-by-2 grid contains the
boolean value �ej | κ�L[w][i] [. The column to the right con-]
tains ℵL[w][i] [(][e][1][, e][2][ |][ κ][)][. The bottom row aggregates each]
column by quantifying 3.

As an equivalence class for µ is not even welldefined by 3ℵL, we cannot hope to compute it
from queries. This is an even stronger sense in
which emulation is impossible using assertions.
On some level, this may be a natural model for
language modeling corpora, which aggregate text
from potentially inconsistent sources.
In summary, if assertions uniquely determine
equivalence between denotations in a strongly
transparent language, then we can expect to emulate representations preserving equivalence using
assertions. Otherwise, there are various levels of
formal challenges to emulating equivalence.

#### B Other Semantic Relations

Sections 4, 5, and A investigate whether ℵL can
be used to emulate meaning representations that
preserve semantic equivalence. While equivalence
is an important part of semantics, other semantic relations are also necessary for language understanding. For example, the following feature
prominently in theories of linguistic semantics:

  - Entailment In general terms, an entailment
(Winter, 2016) relation is a partial order
_→_
over Y . Intuitively, if y _y[′], then y is a_
_→_
“special case” of y[′]. For example, one could
construct E, a semantic analysis of English,
where �fat cat | a, sits�E → �cat | a, sits�E.

  - Contrary negation Negation is a complex
topic in semantics. One sense of negation
is if two meaning representations are “contrary” (Horn and Wansing, 2020), meaning
both cannot be true at the same time.
Does Theorem 2 generalize to other relations besides =? To answer this, we first extend assertions and emulation to apply to a generic relation
: M [2]. The proof for Theorem 1 does not fully
_◦_
translate to this new setting, but we will show via
a new argument that emulation is still possible.

**Definition 7 For e, e[′],** Σ[⋆] and κ (Σ[⋆])[2], define
_∈_ _∈_


-----

```
from typing import Callable, Dict, Tuple
AssertType = Callable[[str, str, str, str], bool]
def emulate(expr: str, assertrel: AssertType) −> Dict[Tuple[str, str], bool ]:
  repres = {}
  for cand in all_strings():
     repres[expr, cand] = assertrel(expr, cand, "", "")
     repres[cand, expr] = assertrel(cand, expr, "", "")
     if expr == cand:
       return repres

```
Figure 6: emulate computes a structured representation of the input string expr that preserves any semantic relation
in terms of assertion queries. The iterable all_strings is defined in Figure 5.
_◦_


the assertion oracle

_ℵL,◦(e, e[′]_ _| κ) =_


�
1 if �e | κ�L ◦ �e[′] _| κ�L_
0 otherwise.


**Definition 8 A class of languages** over Σ is _L_ _ℵ_
emulatable w.r.t. if there exists an oracle Turing
_◦_
machine µ and standard Turing machine δ such
that, for all L ∈L, κ ∈ (Σ[⋆])[2], and e, e[′] _∈_
suppL(κ),

�e | κ�L ◦ �e[′] _| κ�L_ _⇐⇒_ _δ�µL(e), µL(e[′]) | κ�._

We now are ready to prove the extended form
of Theorem 1. The main idea of the proof will
be to memoize the value of the relation between
_◦_
�e | κ�L and the values of all expressions smaller
than e. This guarantees that δ will be able to “look
up” the correct output.

**Theorem** **5** TRANSPARENT _is_ _-emulatable_
_ℵ_
_w.r.t._ _._
_◦_

_Proof. Similarly to Theorem 1, we present the_
proof constructively as a Python program to compute µ. We then show how to define δ appropriately, completing the proof.

Figure 6 shows the algorithm to compute
_µL(e) ∈_ _M_ . In Python, µL(e) is a dictionary;
we interpret it as a function µL(e) : Σ[⋆] _× Σ[⋆]_ _→_
0, 1,, where represents values that are not
_{_ _∅}_ _∅_
set. We define δ as follows:


_δ(m, m[′]_ _κ)_
_|_ _⇐⇒_


�
_m(e, e[′])_ if m(e, e[′]) =
_̸_ _∅_
_m[′](e, e[′])_ otherwise.


Crucially, it must be that µL(e)(e, e[′]) ̸= ∅ or
_µL(e[′])(e, e[′]) ̸= ∅. In Figure 6, cand either reaches_


_e before e[′], or e[′]_ before e. By symmetry, assume
it reaches e before e[′]. Then µL(e[′])(e, e[′]) ̸= ∅, so

_δ(µL(e), µL(e[′]) | κ) ⇐⇒_ _µL(e[′])(e, e[′]) = 1_

_⇐⇒ℵL,◦(e, e[′]_ _| λ[2]) = 1_

_e_ _λ[2]_ _e[′]_ _λ[2]_
_⇐⇒_ � _|_ � _◦_ � _|_ �

_e_ _κ_ _e[′]_ _κ_ _._
_⇐⇒_ � _|_ � _◦_ � _|_ �

Therefore emulate satisfies Definition 3.

We needed to change the proof of Theorem 5

compared to Theorem 1 because is not an equiv_◦_
alence relation. In Theorem 1, the final steps relied on reflexivity, transitivity, and symmetry: the
three properties that constitute equivalence. The
new proof enlarges the size of the emulated representations. Rather than representing each e with
a number, µL(e) becomes a large dictionary of
strings. This represents an increase in space complexity from linear to exponential in the size of e.

#### C Old Emulation Definition

[A previous version of this paper defined emulation](https://arxiv.org/abs/2104.10809v1)
slightly differently. We discuss the differences and
explain the advantages of the new definition. First,
we defined a general denotation as

�e�L = {⟨κ, �e | κ�L⟩| κ ∈ (Σ[⋆])[2]}.

The general meaning represents the meaning of a
word across all contexts. Now, say that two functions f, g are isomorphic (with respect to =) over
a set X iff, for all x, x[′] _X,_
_∈_

_f_ (x) = f (x[′]) _g(x) = g(x[′])._
_⇐⇒_

We will write f =[∼]= g in this case. We will refer to a set of contexts S (Σ[⋆])[2] as a syntactic
_⊆_


-----

_role. Each syntactic role has a set of expressions_
supp[−]L [1][(][S][)][ whose][ support][ is that role:]

suppL(e) = {κ ∈ (Σ[⋆])[2] _| �e | κ�L ̸= ∅}_

supp[−]L [1][(][S][) =][ {][e][ ∈] [Σ][⋆] _[|][ supp][L][(][e][) =][ S][}][.]_

We can now give the old definition of emulation:


**Definition 9 (Old** -emulation) µ : Σ[⋆] _M em-_
_ℵ_ _→_
ulates �·�L w.r.t. = iff:

1. µ =[∼]= �·�L over supp[−]L [1][(][S][)][, for all][ S][ ⊆]
(Σ[⋆])[2]

2. There exists a Turing machine that computes
whether m = m[′] for each m, m[′] _M_
_∈_

3. There exists a Turing machine with oracle access to ℵL that computes µ

For a set of languages, this is equivalent to
_L_
saying is -emulatable iff, for all L, there
_L_ _ℵ_ _∈L_
exists an oracle Turing machine µ and normal
Turing machine δ such that, for all S (Σ[⋆])[2],
_∈_
_e, e[′]_ _∈_ supp[−]L [1][(][S][)][,]

_e_ _L =_ _e[′]_ _L_ _δ�µL(e), µL(e[′])�._
� � � � _⇐⇒_

This more closely resembles Definition 3, but we
will make two slight changes. First, we will
change the quantifier order, such that a single µ
must work for every L . Then, we will grant δ
_∈L_
access to a context κ, and rephrase the equation to
hold over all κ ∈ (Σ[⋆])[2] and e, e[′] _∈_ suppL(κ):

�e | κ�L = �e[′] _| κ�L ⇐⇒_ _δ�µL(e), µL(e[′]) | κ�._

This recovers Definition 3. This version more
faithfully reflects the intuitive notion of emulation.
The old version required µL(e) to determine how
_e should evaluate in every possible context. Em-_
ulation would not be possible in some cases even
with perfect knowledge of L. Now, it must just be
possible in any context κ to compute �e | κ�L from
_κ and µL(e), which is a weaker standard. Under_
the new definition, it is always possible to emulate
a class of languages with one element, assuming
�e | κ�L is computable. An additional improvement is that emulation now applies to all expressions that share a context, whereas before it only
targeted expressions with the same support.


-----

