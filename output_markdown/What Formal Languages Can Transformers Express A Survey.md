## What Formal Languages Can Transformers Express? A Survey


### Lena Strobl Umeå University, Sweden
 lena.strobl@umu.se


### William Merrill New York University, USA
 willm@nyu.edu


### Gail Weiss EPFL, Switzerland
 gail.weiss@epfl.ch


### David Chiang University of Notre Dame, USA
 dchiang@nd.edu

 Abstract


### Dana Angluin Yale University, USA
 dana.angluin@yale.edu

of formal languages – that is, the inputs or outputs are treated as sequences of discrete symbols
from a finite alphabet, and crucially as sequences
of unbounded length.
The core research question in this subarea is:
_How can we characterize the expressivity of trans-_
_formers in relation to various formal models, such_
_as automata, boolean circuits or formal logic? Ap-_
plications of this subarea, which are not addressed
by the papers surveyed here but could be by future
work, would hopefully answer questions like:


As transformers have gained prominence
in natural language processing, some researchers have investigated theoretically
what problems they can and cannot solve,
by treating problems as formal languages.
Exploring such questions can help clarify
the power of transformers relative to other
models of computation, their fundamental
capabilities and limits, and the impact of architectural choices. Work in this subarea has
made considerable progress in recent years.
Here, we undertake a comprehensive survey
of this work, documenting the diverse assumptions that underlie different results and
providing a unified framework for harmonizing seemingly contradictory findings.

### 1 Introduction


Transformers (Vaswani et al., 2017) have gained
prominence in natural language processing (NLP),
both in direct applications like machine translation and in pretrained models like BERT (Devlin
et al., 2019) and GPT (Radford et al., 2018; Brown
et al., 2020; OpenAI, 2023). Consequently, some
researchers have sought to investigate their theoretical properties. Such studies can broadly be divided
into studies of expressivity and trainability. While
trainability is very important and the focus of much
study (e.g., Bhattamishra et al., 2023; Allen-Zhu
and Li, 2023), here we focus on expressivity, which
is a prerequisite for trainability.
Studies of expressivity could be further divided
into those from the perspectives of approximation
theory and of formal language theory. The former
(e.g., Yun et al., 2020; Sanford et al., 2023), investigates transformers as approximators of various
classes of functions, along the lines of the universal
approximation theorem for feedforward neural networks (Hornik et al., 1989; Cybenko, 1989). The
latter, which is the subject of this survey, investigates transformers as recognizers or generators



  - What new transformer variants are suggested
by formal models?

  - Do failure cases anticipated from formal models occur in practice?

  - What insights into the complexity of human
language are offered by a characterization of
transformer expressivity?

This paper provides a comprehensive survey of
research in this subarea. Compared to the surveys of Ackerman and Cybenko (2020) and Merrill
(2021, 2023), which cover convolutional neural networks (CNNs), RNNs, and transformers, this is a
narrower, but deeper, survey on transformers only.
Interpreting theoretical transformer results is
complex due to diverse assumptions. Many variants of transformers exist in practice, and even
more have been proposed in theory. This diversity
leads to varied, even seemingly contradictory, results. We set up a unified framework for talking
about transformer variants (§4), and discuss how
some of these variants compare to one another in
expressivity.
We then provide background on various formal
models that transformers have been compared with
(§5). Then, in §6, we systematically survey current results in this literature, documenting their
assumptions and claims in terms of the definitions
of Sections 4 and 5.


1


-----

### 2 Overview

Table 1 summarizes the results surveyed here. One
way to classify them is into lower bounds (what
transformers can do) and upper bounds (what transformers can’t do).
Much work on lower bounds has looked at au_tomata like finite automata, counter machines, and_
Turing machines, all of which had been successfully related to RNNs before (Siegelmann and Sontag, 1995; Merrill, 2020). This wide diversity of
machines is due to different variants of transformers, especially whether a transformer decoder is allowed to take a number of intermediate steps before
outputting a decision (§4.3.4), which dramatically
increases its power (§6.1).
By contrast, investigation of upper bounds has
mainly focused on circuit complexity (§5.2), which
had been successfully related to feedforward networks before (Parberry, 1994; Siu et al., 1995; Beiu
and Taylor, 1996; Šíma and Orponen, 2003). This
line of research began with restricted models of
transformer encoders and progressed to increasingly realistic variants and tighter bounds. One
way to restrict transformers is by discretizing the
attention mechanism (§4.2.1); another is to limit
the precision of number representations (§4.4).
More recent work has turned to formal logic
(§5.3) as a way of characterizing the expressive
power of transformers. The finer control afforded
by logics opens the possibility for them to be used
as upper bounds, lower bounds, or both.

### 3 Preliminaries

**Sets** We denote by N0 = {0, 1, 2, . . .} and N =
N0\{0} the set of natural numbers with and without
0, respectively. We write _𝑛_ = 0, 1, 2, . . ., 𝑛 1
[ ] { − }
for any 𝑛 N. We write Σ for a finite alphabet,
∈
which, in NLP applications, is the set of words or
subwords known to the model.

**Vectors** We use 𝑑, 𝑑[′], etc., for dimensionalities
of vector spaces, lowercase bold letters (x, y, . . . )
for vectors, and uppercase bold letters (X, Y, . . . )
for matrices. For any vector x R[𝑑], we number its
∈
elements starting from 0. For 𝑖 ∈[𝑑], we write x𝑖
or [x]𝑖 (not 𝑥𝑖) for the 𝑖-th component of x.

**Sequences** For any set 𝐴, we write 𝐴[∗] for the set
of all finite sequences over 𝐴. We write the length
of a sequence 𝑠 _𝐴[∗]_ as _𝑠_ and number its elements
∈ | |
starting from 0; thus, 𝑠 = 𝑠0𝑠1 · · · 𝑠|𝑠 |−1. We use
the variable 𝑤 for a string in Σ[∗] and 𝑛 for the length


of 𝑤. For sequences in R[∗], we use lowercase bold
letters (x, y, . . .), and for sequences in R[𝑑], we
( )[∗]
use the variable 𝑋.
A function 𝑓 : 𝐴[∗] _𝐵[∗]_ is length-preserving if
→
_𝑓_ _𝑤_ = _𝑤_ for all 𝑤 _𝐴[∗]. For every function_
| ( )| | | ∈
_𝑔_ : 𝐴 _𝐵, we denote its extension to sequences_
→
by 𝑔 as well. That is, 𝑔 : 𝐴[∗] _𝐵[∗]_ is defined as
→
follows: for all 𝑠 ∈ _𝐴[∗]_ and 𝑖 ∈[|𝑠|], 𝑔(𝑠)𝑖 = 𝑔(𝑠𝑖).

**Neural networks** An affine transformation is
a function 𝐿 : R[𝑑][in] R[𝑑][out] parameterized by
→
weights W𝐿 ∈ R[𝑑][out] [×][𝑑][in] and bias b𝐿 ∈ R[𝑑][out] such
that for every x ∈ R[𝑑][in], 𝐿(x) = W𝐿x + b𝐿. We say
that 𝐿 is linear if b𝐿 = 0.
The activation functions we use are the recti_fied linear unit (ReLU)_ _𝑥_ = max _𝑥, 0_ and the
R( ) ( )
logistic sigmoid function 𝜎 _𝑥_ = 1 1 _𝑒[−]_ _[𝑥]_ .
( ) /( + )
The softmax function : R[∗] R[∗] converts any
S →
sequence of reals into a probability distribution:

_𝑒[x][𝑖]_
S(x)𝑖 = � ∀𝑖 ∈[|x|].

_𝑖_ ∈[|x|] _[𝑒][x][𝑖]_

### 4 Transformers

In this section, we define transformers and relevant
variants, and how transformers are used to describe
formal languages. For additional background on
transformers (not in relation to formal languages),
Huang et al. (2022) give a lucid commentary on
the original paper, Phuong and Hutter (2022) give
formal definitions and pseudocode, and Lin et al.
(2022) survey many variants of transformers.
Transformers are composed of an input layer
(§4.1), one or more hidden layers (§4.2), and an
output layer (§4.3). The inputs and outputs of the
layers are sequences of vectors, which we treat as
members of R[𝑑] .[1]
( )[∗]

**4.1** **Input layer**

Strings are initially mapped to sequences of vectors using a length-preserving function 𝑒 : Σ[∗]
→
R[𝑑], which is the sum of a word embedding
( )[∗]
WE : Σ R[𝑑] and a position(al) embedding or
→
_encoding PE𝑛_ : [𝑛] → R[𝑑] for 𝑛 ∈ N:

_𝑒(𝑤0 · · · 𝑤𝑛−1)𝑖_ = WE(𝑤𝑖) + PE𝑛 (𝑖).

In theoretical constructions, the word embedding
can be any computable function.

1This differs from the original paper (Vaswani et al., 2017),
which treats them as matrices in R[𝑛][×][𝑑]. Our notation aligns
better with notation for formal languages and emphasizes the
variability of the sequence length.


2


-----

The original transformer paper (Vaswani et al.,

2017) introduced the following position embedding:



[PE𝑛 (𝑖)] 𝑗 =


�
sin 10000[−] _[𝑗][/][𝑑]_ _𝑖_ if 𝑗 even
(   - )
cos 10000[−(][ 𝑗] [−][1][)/][𝑑] _𝑖_ if 𝑗 odd.
(    - )


Theoretical papers have explored other position
embeddings, including 𝑖 itself (Pérez et al., 2021),
_𝑖_ _𝑛_ (Yao et al., 2021; Chiang and Cholak, 2022),
/
and 1 _𝑖_ or 1 _𝑖[2]_ (Pérez et al., 2021).
/ /

**4.2** **Hidden layers**

A transformer layer is a length-preserving function
: R[𝑑] R[𝑑] . There are two variants. The
L ( )[∗] →( )[∗]
_post-norm variant (Vaswani et al., 2017) is_

_𝑋_ [′] = N1(𝑋 + A(𝑋))

L(𝑋) = N2(𝑋 [′] + F (𝑋 [′])) (1)

and the pre-norm variant (Wang et al., 2019) is

_𝑋_ [′] = 𝑋 + A(N1(𝑋))

L(𝑋) = 𝑋 [′] + F (N2(𝑋 [′])) (2)

where

  - is a multi-head self-attention with 𝑑 inA
put/output dimensions, 𝐻 heads, and 𝑑kv
key/value dimensions per head

  - is a feed-forward network (§4.2.2) with 𝑑
F
input/output dimensions and 𝑑ff hidden dimensions

  - N1 and N2 are layernorms with 𝑑 dimensions.

We define each of these components below.

**4.2.1** **Attention**

Attention was initially developed to facilitate retrieval of previously processed data from a variablelength history (Bahdanau et al., 2015). Transformers use a simple variant of attention known as
_scaled dot-product attention._

**Scaled dot-product attention with 𝑑** input/output
dimensions and 𝑑kv key/value dimensions is a function A: R[𝑑] R[𝑑] R[𝑑] parameterized by linear
× ( )[∗] →
transformations

_𝑊A[Q][, 𝑊]A[K][, 𝑊]A[V]_ [:][ R][𝑑] [→] [R][𝑑][kv] _𝑊A[O]_ [:][ R][𝑑][kv][ →] [R][𝑑]


and defined for every z R[𝑑], 𝑋 R[𝑑] (with
∈ ∈( )[∗]
_𝑋_ = 𝑛), and 𝑗 _𝑛_ as
| | ∈[ ]

_𝑊_ [Q]
**s(z, 𝑋) 𝑗** = A [(][z][) ·]√ _[ 𝑊]A[K][(][𝑋]_ _[𝑗][)]_ (3)

_𝑑kv_

_𝛼_ **z, 𝑋** = **s** **z, 𝑋** (4)
( ) S( ( ))

�∑︁ �
A(z, 𝑋) = 𝑊A[O] _𝛼(z, 𝑋) 𝑗_ _𝑊A[V][(][𝑋]_ _[𝑗][)]_ _._

_𝑗_ ∈[𝑛]


Typically, A is extended to a function A: R[𝑑]
( )[∗] ×
R[𝑑] R[𝑑] that is length-preserving in its first
( )[∗] →( )[∗]
argument. In cross-attention, z is computed by the
decoder while 𝑋 is computed by the encoder. In
_self_ -attention, the two arguments are identical:

SA: R[𝑑] ∗ R𝑑 ∗
( ) →( )

SA _𝑋_ = A _𝑋, 𝑋_ _._
( ) ( )

**Attention masking** In future masked (also
known as causally masked) self attention, a term
_𝑚_ _𝑖, 𝑗_ is added to Eq. (3) to force every position
( )
to attend only to preceding positions:


_𝑚_ _𝑖, 𝑗_ =
( )


�
0 if 𝑗 _𝑖_
≤
otherwise.
−∞


Some papers use strict future masking, that is,
_𝑚_ _𝑖, 𝑗_ = 0 iff 𝑗< 𝑖, and occasionally past mask( )
ing ( _𝑗_ _𝑖) and strict past masking (_ _𝑗> 𝑖)._
≥

**Multi-head attention with 𝑑kv key/value dimen-**
sions per head is the sum of 𝐻 attentions with 𝑑kv
key/value dimensions:

∑︁
A(z, 𝑋) = Aℎ (z, 𝑋).

_ℎ∈[_ _𝐻_ ]

Multi-head self attention is defined analogously.
This is equivalent to the original formulation,
which concatenated the outputs of the heads and
passed the result through a shared, larger, 𝑊 [O]
A [.]

**Hard attention** Some theoretical analyses simplify attention by replacing the softmax with variants that focus attention only on the position(s)
with the maximum value, breaking ties in various
ways. For any s R[∗], let 𝑀 **s** = _𝑖_ **s** _𝑗_
∈ ( ) { ∈[| |] | ∀ ∈

[|s|], s _𝑗_ ≤ **s𝑖** } be the set of indices of the maximal
elements of s. In leftmost-argmax, the leftmost
maximal element is used:

[Sh(s)]𝑖 = I[𝑖 = min 𝑀 (s)]


3


-----

whereas in average-argmax the maximal elements
share weight equally:

[Sa(s)]𝑖 = [I][[][𝑖] [∈] _[𝑀]_ [(][s][)]] _._

_𝑀_ **s**
| ( )|

If softmax is thought of as a Boltzmann distribution,
then average-argmax is its low-temperature limit.
By substituting Sh or Sa for S in Eq. (4), we
get leftmost-hard and average-hard attention, respectively. Leftmost-hard attention was previously
called hard attention by Hahn (2020) and unique
_hard attention by Hao et al. (2022). One may also_
consider rightmost-hard attention, in which the
rightmost maximal element is used. Average-hard
attention was also called hard attention by Pérez
et al. (2021) and saturated attention by Merrill
et al. (2022), and has been argued to be a realistic
approximation to how trained transformers behave
in practice (Merrill et al., 2021).

**4.2.2** **Feed-forward networks**
Although feed-forward networks can take many
forms, in the context of transformers, we use the following definition. A feed-forward network (FFN)
with 𝑑 input/output dimensions and 𝑑ff hidden dimensions is a function : R[𝑑] R[𝑑] parameterF →
ized by two affine transformations, 𝐿[1]F [:][ R][𝑑] [→] [R][𝑑][ff]

and 𝐿[2]F [:][ R][𝑑][ff][ →] [R][𝑑][, such that]

**x** = 𝐿[2]
F ( ) F [(R(][𝐿][1]F [(][x][)))]

where is applied component-wise.
R

**4.2.3** **Layer normalization**
A 𝑑-dimensional layer normalization (Ba et al.,
2016), or layernorm for short, is a function
: R[𝑑] R[𝑑] parameterized by vectors 𝛾 _, 𝛽_
N → N N ∈
R[𝑑] and scalar 𝜀 0:
N ≥

**x** **x¯**
−
**x** = 𝛾 _𝛽_
N ( ) N ⊙ √︁ + N

var **x** _𝜀_
( ) + N

where is component-wise multiplication and
⊙


**x¯ =** [1]

_𝑑_


∑︁

**x𝑖** var(x) = [1]

_𝑑_

_𝑖_ ∈[𝑑 ]


∑︁

(x𝑖 − **x¯)[2].**
_𝑖_ ∈[𝑑 ]


The original definition of layernorm (Ba et al.,

2016) sets 𝜀 N = 0, but, for numerical stability,
all implementations we are aware of set 𝜀 N > 0.
Observe that N is Lipschitz-continuous iff 𝜀 N > 0.
Some transformer analyses omit for simplicN
ity (Pérez et al., 2021), while others set 𝜀 N to
achieve various effects (Hahn, 2020; Chiang and
Cholak, 2022).


**4.3** **Networks and output layers**

We now define a complete transformer network.

**4.3.1** **Transformer encoders**
A transformer encoder is a length-preserving function : Σ[∗] R[𝑑] parameterized by the weights
T →( )[∗]
of an input layer 𝑒 and 𝐷 transformer layers
L1, . . ., L𝐷. A post-norm transformer encoder is:

T (𝑤) = L𝐷 ◦· · · ◦L2 ◦L1 ◦ _𝑒(𝑤)_

where each L𝑙 is a post-norm layer (1) and ◦ is
function composition. A pre-norm transformer encoder is additionally parameterized by the weights
of a final layernorm and is defined as:
N

T (𝑤) = N ◦L𝐷 ◦· · · ◦L2 ◦L1 ◦ _𝑒(𝑤)_

where each L𝑙 is a pre-norm layer (2).
The encoder’s output is a sequence of vectors in
R[𝑑] . To use it as a language recognizer, we add
( )[∗]
an output layer that converts _𝑤_ to a probability
T ( )

_𝑝ˆ_ = 𝜎(w · [T (𝑤)]𝑖 + 𝑏)

where w R[𝑑], 𝑏 R, and 𝑖 is a distinguished
∈ ∈
position. The encoder accepts iff ˆ𝑝 ≥ 2[1] [.]

Chiang and Cholak (2022) also consider a requirement that an encoder accepts/rejects strings
with bounded cross-entropy. That is, we say that
an encoder recognizes a language 𝐿 with crossentropy at most 𝜂 iff for all strings 𝑤, if 𝑤 _𝐿_ then
∈
log ˆ𝑝 _𝜂, and if 𝑤_ ∉ _𝐿_ then log 1 _𝑝ˆ_ _𝜂._
− ≤ − ( − ) ≤
We are aware of two choices for the distinguished position 𝑖. Most papers use the last position
(𝑖 = 𝑛 1), but some (Chiang and Cholak, 2022;
−
Chiang et al., 2023), inspired by binary classifiers
based on BERT (Devlin et al., 2019), prepend a
special symbol CLS at position 0 and use 𝑖 = 0.
While this is a minor difference, it should be noted
that the guarantee of exactly one occurrence of CLS
in the input can be useful in some constructions.

**4.3.2** **Transformer decoders**
A transformer decoder is a transformer encoder
with future masking in its attention, typically
T
used to generate rather than recognize strings. The
input is the prefix of previously-generated symbols,
_𝑤<𝑡_ = 𝑤0 · · · 𝑤𝑡 −1, and the output is a probability
distribution ˆ𝑝(𝑤𝑡 | 𝑤<𝑡 ) over the next symbol,

_𝑝ˆ(· | 𝑤<𝑡_ ) = S(W [T (𝑤<𝑡 )]𝑡 −1 + b)

where W ∈ R[|][Σ][|×][𝑑] and b ∈ R[|][Σ][|]. We assume 𝑤0 =
BOS and every string ends with EOS, where BOS and


4


-----

EOS are special symbols that do not occur anywhere
else. To sample a string, we first sample 𝑤1 from
_𝑝ˆ(𝑤1 | BOS), then, for each time step 𝑡> 1, sample_
_𝑤𝑡_ from ˆ𝑝(𝑤𝑡 | 𝑤<𝑡 ). The process stops when
_𝑤𝑡_ = EOS. Because each sampled output symbol
becomes part of the input at the next time step, this
kind of model is called autoregressive.
While a decoder can be used to recognize strings
similarly to an encoder, it can also be used to generate the entire string; at least two definitions have
been given for this.
First, Hahn (2020) considers a weighted language as a distribution over strings 𝑝 _𝑤_ . For any
( )
length 𝑡, the KL divergence (relative entropy) of the
model ˆ𝑝 _𝑤_ from the true distribution 𝑝 _𝑤_, for
( ) ( )
predicting 𝑤𝑡 conditioned on all previous words, is


∑︁
Δ𝑡 [ ˆ𝑝 ∥ _𝑝] =_

_𝑤<𝑡_


∑︁

_𝑝(𝑤<𝑡_ _𝑤𝑡_ ) log _[𝑝][(][𝑤][𝑡]_ [|][ 𝑤][<𝑡] [)]
_𝑤𝑡_ _𝑝ˆ(𝑤𝑡_ | 𝑤<𝑡 ) _[.]_


As Hahn’s results are negative, he does not spell
out a positive criterion, but he seems to implicitly
require that this divergence vanish at infinity:

lim (5)
_𝑡→∞_ [Δ][𝑡] [[][ ˆ][𝑝] [∥] _[𝑝][]][ =][ 0][.]_

Second, let us say that a transformer decoder
_𝜀-generates 𝐿_ iff

_𝐿_ = {𝑤 | ∀𝑡 ∈[|𝑤|], ˆ𝑝(𝑤𝑡 | 𝑤<𝑡 ) ≥ _𝜀}._

Then Yao et al. (2021), following Hewitt et al.
(2020), say that a transformer decoder 𝑇 generates a language 𝐿 iff there exists an 𝜀> 0 such that
_𝑇𝜀-generates 𝐿. (This means that a transformer_
decoder may generate more than one language, depending on the 𝜀 chosen.) They also show that any
_𝜀-generator can be converted into a recognizer._
While not focusing on transformers, Lin et al.

(2021) demonstrate limitations of autoregressive
models for generation; for example, that there is
a language 𝐿 P that cannot be 𝜀-generated in
∈
polynomial time for any 𝜀> 0 if P ≠ NP.

**4.3.3** **Transformer encoder–decoders**
A transformer encoder–decoder combines a transformer encoder and decoder, adding to each layer
of the decoder an additional attention sublayer,
known as cross attention, which attends to the output of the encoder. In the literature surveyed here,
only the construction of Pérez et al. (2021) and
related constructions (Bhattamishra et al., 2020b;

Wei et al., 2022a) employ an encoder–decoder.


**4.3.4** **Intermediate steps**
When a transformer decoder or encoder–decoder
is run as a language recognizer, it allows for the
possibility of inserting a number of intermediate
time steps between the end of the input string and
the decision. The encoder–decoder models above
do this, as do some decoder-only models (Feng
et al., 2023; Merrill and Sabharwal, 2024). As we
will see (§6.1), intermediate steps vastly increase
the model’s power, which has also been observed
in practice in the form of a “scratchpad” (Nye et al.,
2022) or “chain of thought” (Wei et al., 2022b).

**4.4** **Uniformity and precision**

Although meaningful theoretical claims can be
made about transformers for fixed-length strings
(e.g. Yun et al., 2020), it is crucial when examining transformers as language recognizers to allow
for unbounded string length. Fixing a maximum
length makes all languages finite, collapsing many
language classes into one.
It might be objected that considering unbounded
lengths is too abstract, because in practice one
can always fix a maximum length. But this maximum length, driven by practical needs, is growing
steadily: for example, GPT-4 Turbo uses 128,000
tokens of context. At the same time, some theoretical findings surveyed here seem to have practical
consequences for modest string lengths. For example, we will see that there are reasons to think that
in theory, transformers cannot recognize PARITY;
in practice, they fail to learn PARITY for strings
with lengths in 2, 50 (Bhattamishra et al., 2020a).
[ ]
Some theoretical studies of transformers do allow them to depend on the input length 𝑛. To borrow a term from circuit complexity (§5.2), they
allow certain kinds of non-uniformity. As we have
seen, some position embeddings (§4.1) depend on
_𝑛. We discuss some other instances below._

**Numeric precision** Transformers operate, in
principle, on real numbers. While hard attention
transformers could be defined using only rational
numbers, even rational numbers can represent an arbitrary amount of information. With RNNs, the use
of real or rational numbers has led to results that
make them appear more powerful in theory than
in practice (Siegelmann and Sontag, 1994, 1995;

Weiss et al., 2018).
Consequently, many studies use limitedprecision numbers. Some studies limit number
representations to have 𝑂 1 bits, as floating-point
( )


5


-----

numbers do in practice (Chiang et al., 2023). But
Merrill and Sabharwal (2023b) argue that in 𝑂 1
( )
precision, attention cannot attend uniformly to a
string of sufficient length 𝑛, as the attention weights
(𝛼) would all round down to zero. So 𝑂 log 𝑛 bits
( )
of precision is a common choice (Yao et al., 2021;
Merrill and Sabharwal, 2023a,b). Other choices are
possible as well: Merrill and Sabharwal (2023a)
use the set F = _𝑎_ 2[𝑏] _𝑎_ Z, 𝑏 N .
{ / | ∈ ∈ }
Restricting intermediate activations to limited
precision introduces many decisions about when
and how rounding should take place, which can
potentially affect expressivity. For example, when
summing 𝑛 numbers, one could round after each
addition or only at the end of the summation. Better
formalizing these decisions and their impact on
expressivity is an area for future research.

**Parameters** A few constructions allow the parameters themselves to depend on 𝑛, which we
consider to be a stronger dependence, because if
these transformers were to be learned from data,
different transformers would have to be learned for
different maximum lengths. Finally, a few papers
construct transformers in which 𝑑, and therefore
the number of parameters, depends on 𝑛, which we
consider to be stronger still.

**4.5** **Summary**

In summary, transformers can vary in at least the
following ways, any of which could a priori impact
theoretical claims:

  - Architecture: encoder-only, decoder-only, or
encoder–decoder

  - For encoders: definition of recognition

  - For decoders and encoder–decoders: definition of generation and how many intermediate
steps

  - Position embedding (PE)

  - Attention pattern: leftmost-hard, rightmosthard, average-hard, or softmax

  - Attention masking: none, future, or past

  - Layernorm: inclusion or omission, value of
_𝜀_
N

  - Residual connections: pre-norm or post-norm

  - Precision: infinite, 𝑂 log 𝑛, 𝑂 1
( ) ( )



  - Uniformity: whether parameter values or number of parameters depend on 𝑛.

### 5 Languages and Language Classes

Next, we present various formal models that transformers are compared to in the literature surveyed.

**5.1** **Automata and classes L, NL, P**

We assume familiarity with finite automata and Turing machines; for definitions, please see the textbook by Sipser (2013). Counter machines are automata with integer-valued registers (Fischer et al.,

1968); they have been studied extensively in connection with LSTM RNNs (Weiss et al., 2018; Suzgun et al., 2019; Merrill, 2019, 2020).
The language classes L (languages decidable
in 𝑂 log 𝑛 space) and P (languages decidable
( )
in polynomial time) are defined using deterministic Turing machines (with a read-only input
tape and a read/write work tape). The class NL
(languages decidable in nondeterministic 𝑂 log 𝑛
( )
space) uses nondeterministic Turing machines. The
class DLOGTIME (languages decidable in 𝑂 log 𝑛
( )
time) uses random-access Turing machines (Barrington et al., 1990). It is known that

L ⊆ NL ⊆ P

but none of these inclusions are known to be strict.

**5.2** **Circuits and classes AC[0], ACC[0], TC[0], NC[1]**

Circuits are a model of parallel computation particularly relevant to transformers. For more details,
please see the textbook by Arora and Barak (2009).
Circuits operate on binary values. If we choose
a fixed-length encoding of the symbols of Σ as
strings of 𝑏 = ⌈log2 |Σ|⌉ bits, then a circuit can
simulate input alphabet Σ by encoding the value of
the 𝑖-th input symbol into positions 𝑖𝑏 to 𝑖𝑏 _𝑏_ 1 .
+( − )
For the rest of this section, we assume Σ = 0, 1 .
{ }

**Circuits** A circuit 𝐶 with input length 𝑛 is
a directed acyclic graph with 𝑛 _input vertices_
_𝑠1, . . ., 𝑠𝑛_ and zero or more gate vertices, each labeled with a type NOT, AND, or OR. Input vertices
have fan-in (in-degree) zero, NOT gates have fanin one, and the fan-in of AND and OR gates can
be either two or unbounded. One (input or gate)
vertex 𝑡 is designated the output of the circuit.
Given an input string 𝑤 0, 1, each input ver∈{ }[𝑛]
tex 𝑠𝑖 is assigned the value 𝑤𝑖, and each gate vertex
is assigned the value computed by applying the logical function corresponding to its type to the values


6


-----

assigned to its in-neighbors. The circuit computes
the boolean function 𝐶 : 0, 1 0, 1, map{ }[𝑛] →{ }
ping each input string to the value assigned to 𝑡.
The depth of 𝐶, denoted 𝐷 _𝐶_, is the length of the
( )
longest directed path from any 𝑠𝑖 to 𝑡. The size of
_𝐶, denoted_ _𝐶_, is the number of vertices in 𝐶.
| |

**Circuit families** A circuit family is a sequence
C = {𝐶𝑛}𝑛∈N such that for each 𝑛, 𝐶𝑛 is a circuit
with input length 𝑛. We treat as a function on
C
0, 1 as follows: for every 𝑤 0, 1, _𝑤_ =
{ }[∗] ∈{ }[∗] C( )
_𝐶|𝑤_ | (𝑤). Then C defines the language 𝐿 (C) =
_𝑤_ 0, 1 _𝑤_ = 1, and we say that
{ ∈{ }[∗] | C( ) } C
recognizes 𝐿 . The depth and size of are the
(C) C
functions 𝑛 ↦→ _𝐷_ (𝐶𝑛) and 𝑛 ↦→|𝐶𝑛|.

**Uniformity** As defined, a circuit family contains
a different circuit for each length 𝑛, with no constraint on the relationship between the circuits. For
example, let 𝐿 be any unary language: 𝐿 1 .
⊆{ }[∗]
For 𝑛 ∈ N, if 1[𝑛] ∉ _𝐿, define 𝐶𝑛_ to be a circuit
for the constant 0 function (an OR gate with fan-in
0), and if 1[𝑛] ∈ _𝐿, define 𝐶𝑛_ to be a circuit for the
AND of all the inputs. Thus, every unary language,
even an undecidable one, is recognized by a circuit
family of size 𝑂 _𝑛_ and depth 𝑂 1 .
( ) ( )
A uniformity restriction on a circuit family
{𝐶𝑛}𝑛∈N requires that the task of constructing a
description of the circuit 𝐶𝑛 given input 𝑛 be computable within some specified resource bound as
a function of 𝑛, potentially making it comparable
with classes defined by bounds on Turing machine
time or space. Two such uniformity bounds are
used in the work here: L and DLOGTIME. Because
these bounds are very restrictive, a special representation of the circuit 𝐶𝑛 is used, namely, the ability
to answer queries of the type of a gate and whether
the output of one gate is an input to another gate.
We assume that the vertices of the circuit 𝐶𝑛 are
numbered from 0 to |𝐶𝑛| − 1. The direct connec_tion language of a family of circuits_ is the set of
C
all tuples ⟨ _𝑓, 𝑖, 𝑗, 1[𝑛]⟩_ such that in 𝐶𝑛, vertex 𝑖 has
type 𝑓 and there is an edge from vertex 𝑖 to vertex
_𝑗_ (Barrington et al., 1990). Given a computable
function bounding the size of and access to a
C
membership oracle for the direct connection language, for any 𝑛 it is straightforward to write out
the list of vertices, edges, and types in 𝐶𝑛.
Then a circuit family is L-uniform (resp.,
C
DLOGTIME-uniform) if there is a Turing machine
that runs in logarithmic space (resp., deterministic logarithmic time) to decide membership in the


direct connection language of .
C

**Circuit complexity classes** Circuit complexity
classes classify circuit families and the languages
they recognize based on uniformity, depth, size,
fan-in bound, and the allowed gates. Since transformers have constant depth, circuit classes with
constant depth are of particular interest; the classes
that are used in the work we survey are:

  - AC[0] contains those languages that can be
recognized by families of circuits with unbounded fan-in, constant depth, and polynomial size.

  - ACC[0] is like AC[0], but also has gates that output 1 iff the inputs sum to 0 modulo some
constant.

  - TC[0] is like AC[0], but also allows MAJORITY
gates, which have unbounded fan-in and output 1 iff at least half of their inputs are 1.

  - NC[1] is like AC[0], but with fan-in at most 2 and
depth in 𝑂 log 𝑛 .
( )

The known relationships between these classes are:

AC[0] ⊊ ACC[0] TC[0] NC[1]
⊆ ⊆

in the DLOGTIME-uniform, L-uniform, and nonuniform settings; moreover, L-uniform NC[1] L.
⊆

**5.3** **Logic**

A formal language can also be defined as a set
of finite strings that satisfy a closed formula of a
logic. For more details, refer to Thomas (1997) or
Straubing (1994).
In the first-order logic of strings, or FO, the formulas are the smallest set containing:

  - Variables 𝑥, 𝑦, and so on.

  - Atomic formulas 𝑄 _𝑎_ (𝑥), 𝑥 = 𝑦, 𝑥< 𝑦, where
_𝑎_ Σ is a symbol and 𝑥, 𝑦 are variables.
∈

  - 𝜙1 ∧ _𝜙2, 𝜙1 ∨_ _𝜙2, 𝜙1 →_ _𝜙2, ¬𝜙1, where 𝜙1_
and 𝜙2 are formulas.

  - _𝑥.𝜙,_ _𝑥.𝜙, where 𝑥_ is a variable and 𝜙 is a
∀ ∃
formula.

Under the intended interpretation, variables stand
for positions of a finite string 𝑤, and 𝑄 _𝑎_ (𝑥) is
true iff 𝑤 _𝑥_ = 𝑎. For example, if Σ = {𝑎, 𝑏},
∀𝑥.∀𝑦.𝑄 _𝑎_ (𝑥) ∧ _𝑄𝑏_ (𝑦) → _𝑥< 𝑦_ defines the regular


7


-----

language 𝑎[∗]𝑏[∗]. The language defined by a closed
formula 𝜙 consists of those strings that satisfy 𝜙.
The languages definable in FO are exactly
the star-free languages (McNaughton and Papert,

1971). Other variants add more quantifiers:

  - FOC adds counting quantifiers _𝑦.𝜙, which_
∃[=][𝑥]
hold iff there are exactly 𝑥 values of 𝑦 that
make 𝜙 true (Barrington et al., 1990).

  - FOM adds majority quantifiers M𝑥.𝜙, which
hold iff at least half of the values of 𝑥 make 𝜙
true (Barrington et al., 1990).

We are also interested in various sets of predicates:

  - Modular predicates MOD[𝑟]𝑚[(][𝑥][)][, which hold iff]
_𝑥_ _𝑟_ mod 𝑚 (Barrington et al., 1992).
≡ ( )

  - BIT _𝑥, 𝑦_, which holds iff the 𝑦-th bit of 𝑥 is 1.
( )

  - Mon, the set of all predicates on one position,
possibly depending on 𝑛.[2]

  - ARB, the set of all predicates on one or more
positions.

A logic extended with predicates is conventionally
written with the predicates in square brackets; for
example, we write FO BIT for first-order logic

[ ]
with the BIT predicate.
In linear temporal logic or LTL (Kamp, 1968),
every formula implicitly depends on a single time
(or position). There are atomic formulas 𝑄 _𝑎_ for
every 𝑎 Σ, the connectives,, and, as well as
∈ ∧ ∨ ¬
operators since and until. The formula 𝛼 **since 𝛽**
is true iff 𝛼 was true at some past time 𝑖 and 𝛽 was
true from 𝑖 to now (exclusive). LTL is equivalent
to FO (Kamp, 1968).

**5.4** **Relationships**

Figure 1, which depicts the relationships between
the language classes defined above, shows that the
classes defined by circuits/logics cut across the
(perhaps more familiar) Chomsky hierarchy. In
this figure and in this section, all circuit classes
are understood to be DLOGTIME-uniform unless
specified otherwise.

2Although Barrington et al. (2005) define Mon to be the
collection of all monadic predicates without dependence on 𝑛,
Barceló et al. (2024) do allow them to depend on 𝑛.


free _LP_ FOM[BIT]

SHUFFLE-DYCK-2

BFVP

regular AC[0]

W(S5) MAJORITYDYCK-k _ww_ _a[2][n]_ FO[BIT]

PARITY _ww[R]_

DYCK-(k, D)

Figure 1: Relationship of some languages and language
classes discussed in this paper (right) to the Chomsky
hierarchy (left), assuming that TC[0] ⊊ NC[1] and L ⊊ NL.
Circuit classes are DLOGTIME-uniform.

**5.4.1** **Beyond AC[0]**

The classic examples of languages not in AC[0] are
PARITY and MAJORITY. The language PARITY
⊆
0, 1 contains all bit strings containing an odd
{ }[∗]
number of 1’s, and MAJORITY 0, 1 consists
⊆{ }[∗]
of all bit strings in which more than half of the
bits are 1’s. Other problems in TC[0] but not AC[0]

include sorting, integer multiplication (Chandra
et al., 1984), and integer division (Hesse, 2001).

**Dyck languages** The language DYCK-𝑘 for 𝑘>
0 is the language of strings over 𝑘 pairs of parentheses that are correctly balanced and nested. If
we write the 𝑖-th parenthesis pair as (𝑖 )𝑖 for each
_𝑖_ _𝑘_, then DYCK-𝑘 is generated by the context∈[ ]
free grammar {𝑆 → (𝑖𝑆)𝑖𝑆 | 𝑖 ∈[𝑘]} ∪{𝑆 →
_𝜀_ . These languages are of interest because any
}
context-free language can be obtained by applying a string homomorphism to the intersection of a
Dyck language with a regular language (Chomsky
and Schützenberger, 1963).
Some papers surveyed here consider variations
on Dyck languages. The language DYCK- _𝑘, 𝐷_
( )
for 𝐷> 0 is the subset of DYCK-𝑘 consisting of
strings with maximum nesting depth 𝐷; it is a starfree regular language (and therefore in AC[0]).
The language SHUFFLE-DYCK-𝑘 is the set of
strings over 𝑘 pairs of parentheses in which, for
each parenthesis pair, erasing the other types of
parentheses leaves a correctly balanced and nested
string. For example, is in SHUFFLE-DYCK[(()])
2. If 𝑘> 1, SHUFFLE-DYCK-𝑘 is not context free.

**5.4.2** **Beyond TC[0]**

As we will see (§6.3.2), some transformer variants
lie within TC[0]. What problems lie beyond?

|recursively enumerable|Col2|NC1 TC0 FOM[BIT] AC0 FO[BIT]|
|---|---|---|
|context sensitive context free regular|LP SHUFFLE-DYCK-2 BFVP W(S5) M DA YJO CR KI -T kY ww a2n PARITY wwR DYCK-(k, D)||


8


-----

**The word problem for permutation groups** A
permutation of _𝑘_ is a bijection 𝜋 : _𝑘_ _𝑘_,
[ ] [ ] →[ ]
and 𝑆𝑘 is the set of all permutations of [𝑘]. Treating 𝑆𝑘 as an alphabet and compositions of permutations as strings, we can define the language
W(𝑆𝑘) of compositions of permutations of [𝑘]
that equal the identity permutation. For example, in 𝑆3, the permutation (120) maps 0 ↦→ 1,
1 ↦→ 2, and 2 ↦→ 0, so that W(𝑆3) contains
120 120 120 but not 120 120 . These
( ) ◦( ) ◦( ) ( ) ◦( )
languages are easy for finite automata to recognize,
but difficult with only fixed computation depth. Indeed, W(𝑆5) is complete for NC[1] under AC[0] reductions (Barrington, 1989), so it is not in TC[0],
assuming that TC[0] ⊊ NC[1] (as is widely believed).
This makes it an example of a regular language that
transformer encoders probably cannot recognize.
The languages W(𝑆𝑘) have some relevance to
natural language: they resemble expressions like
_the child of the enemy of Ann where the interpre-_
tation of the child of is (roughly) a permutation of
possible referents (Paperno, 2022), and problems
that have been used to benchmark transformers’
state-tracking abilities (Kim and Schuster, 2023).

**Other languages that are widely believed to be**
not in TC[0] include:

  - The language of closed Boolean formulas that
are true (BFVP) is context-free but complete
for NC[1] under DLOGTIME reductions (Buss,

1987), so it is outside TC[0] if TC[0] ⊊ NC[1].

  - Undirected graph connectivity is L-complete
under L-uniform NC[1] reductions (Cook and
McKenzie, 1987; Reingold, 2008), so it is
outside L-uniform NC[1] (and therefore outside
TC[0]) if L-uniform NC[1] ⊊ L.

  - There is a context-free language 𝐿 _𝑃_ that is
NL-complete under L reductions (Sudborough,

1975), so it is outside L (and therefore outside
NC[1] and TC[0]) if L ⊊ NL.

  - Solving systems of linear equalities and universal context-free grammar recognition are
P-complete under L reductions (Jones and
Laaser, 1976; Greenlaw et al., 1995), so they
are outside TC[0] if L ⊊ P.

  - Matrix permanent is known to be outside of
TC[0] (Allender, 1999).


**5.4.3** **Circuits and logics**

DLOGTIME-uniform AC[0] and TC[0] are equivalent
to FO BIT and FOM BIT, respectively. There are

[ ] [ ]
many such equivalences between circuit classes
and logics. As a rule of thumb, adding unbounded
fan-in gates to a circuit family correlates with
adding quantifiers to the corresponding logic, and
increasing the degree of non-uniformity of a circuit family correlates with adding numerical predicates to the corresponding logic (Barrington and
Immerman, 1994). For example, making AC[0] and
TC[0] completely non-uniform corresponds to adding
arbitrary numerical predicates ARB to FO and
( )
FOM, respectively (Immerman, 1997; Barrington
et al., 1990).
As we will see below, circuits and logics have
their advantages and disadvantages for capturing
the expressivity of transformers. An advantage of
the circuit approach is that they have a more transparent resemblance to transformers. Transformers
are computations with bounded depth, so it’s not
hard to see that they should be computable by circuit families with bounded depth (AC[0] or TC[0]).
On the other hand, an advantage of the logical approach is that if we seek an exact characterization
of transformers, it can be easier in a logic to add or
remove quantifiers or predicates, to limit quantifier
depth or number of variables, to partition terms into
different sorts, and so on, than to make adjustments
to a circuit family.

### 6 Current Results

While this area of research still has many unresolved questions, the emerging picture has three
levels of expressivity. At the upper end are
decoders or encoder–decoders with intermediate
steps; these are equivalent to Turing machines
(§6.1). At the lower end are encoders with leftmosthard or rightmost-hard attention; these can recognize only languages in AC[0] (§6.2). In the middle
are encoders with average-hard or softmax attention, which are the least well-understood but appear
to lie between AC[0] and TC[0] (§6.3).
In this section, “transformer” refers to a transformer encoder unless otherwise indicated.

**6.1** **Decoders with intermediate steps**

Pérez et al. (2021) consider transformer encoder–
decoders with several modifications:

  - The PE includes components 𝑖, 1 _𝑖, and 1_ _𝑖[2]._
/ /


9


-----

Lower bound Source PE Attention Notes

∋ MAJORITY Pérez et al. 2019 none average-hard
∋ SHUFFLE-DYCK-𝑘 Bhattamishra et al. 2020a none softmax, future mask
⊇ SSCMs Bhattamishra et al. 2020a none softmax, future mask
∋ DYCK-k Yao et al. 2021 _𝑖/𝑛, 𝑖/𝑛[3], 𝑛_ softmax & leftmost-hard
⊇ P Pérez et al. 2021 _𝑖, 1/𝑖, 1/𝑖[2]_ average-hard poly(𝑛) steps
∋ PARITY Chiang and Cholak 2022 _𝑖/𝑛, (−1)[𝑖]_ softmax
⊇ FOC[MOD; +] Chiang et al. 2023 sinusoidal softmax
⊇ FO[Mon] Barceló et al. 2024 arbitrary leftmost-hard
⊇ LTL+C[Mon] Barceló et al. 2024 arbitrary average-hard

Upper bound Source Precision Attention Notes

∌ PARITY, DYCK-1 Hahn 2020 R leftmost-hard
∌ PARITY, DYCK-2 Hahn 2020 R softmax, future mask _𝜀_ N > 0, vanishing KL
⊆ AC[0] Hao et al. 2022 Q leftmost-hard
⊆ TC[0] Merrill et al. 2022 F average-hard
⊆ FOC[MOD; +] Chiang et al. 2023 _𝑂_ (1) softmax
⊆ L-uniform TC[0] Merrill & Sabharwal 2023a _𝑂_ (log 𝑛) softmax
⊆ FOM[BIT] Merrill & Sabharwal 2023b _𝑂_ (log 𝑛) softmax
⊆ L-uniform TC[0] Strobl 2023 F average-hard

Equivalent Source PE Attention Notes

= RE Pérez et al. 2021 _𝑖, 1/𝑖, 1/𝑖[2]_ average-hard unbounded steps
= FO Angluin et al. 2023 none rightmost-hard, strict future mask
= FO[MOD] Angluin et al. 2023 sinusoidal rightmost-hard, strict future mask
= FO[Mon] Angluin et al. 2023 arbitrary rightmost-hard, strict future mask
= P Merrill & Sabharwal 2024 none average-hard, future mask poly(𝑛) steps

Table 1: Surveyed claims and their assumptions. Please see the main text for full details of assumptions.



  - In self attention, Eq. (3) takes the negative
absolute value of the dot-product, and Eq. (4)
uses average-hard attention.

  - The FFNs use sigmoids instead of ReLUs.

As described above (§4.3.3), the decoder is allowed
to run for arbitrarily many time steps until an acceptance criterion is met. Under these assumptions,
transformer encoder–decoders can recognize any
recursively enumerable language.[3] This result uses
arbitrary precision, but as a corollary, they show
that a 𝑇 _𝑛_ -time-bounded Turing machine can be
( )
simulated in a transformer using 𝑂 log _𝑇_ _𝑛_ pre( ( ))
cision and 𝑂 _𝑇_ _𝑛_ intermediate steps.
( ( ))

Bhattamishra et al. (2020b) provide a simpler
proof of Pérez et al.’s result by reducing to an RNN
and appealing to the construction of Siegelmann
and Sontag (1995). They do this for two sets of
assumptions. First,

  - The PE includes only 𝑖.

  - The self attention sublayers are as above.

3Pérez et al. (2021) define both Turing machines and
encoder–decoders to halt only when accepting. The construction could easily be modified to capture decidable languages.



  - The FFNs use saturated linear activation functions: 𝜎 _𝑥_ = max 0, min 1, 𝑥 .
( ) ( ( ))

Second, they show the same with no PE and standard dot-product attention with future masking.

Wei et al. (2022a) define a notion of statistically_meaningful (SM) approximation and show that_
transformer encoder–decoders SM-approximate
Turing machines. Both the decoder and Turing
machine are limited to 𝑁 time steps; additionally,

  - The PE can be an arbitrary computable function on _𝑁_ .
[ ]

  - Attention is average-hard.

  - The FFNs have three ReLU layers.

Feng et al. (2023) observe that the problems of
evaluating arithmetic expressions or solving linear
equations over Z _𝑝_ are NC[1]-hard under DLOGTIME
reductions, so (if TC[0] ⊊ NC[1]) they cannot be
solved by 𝑂 log 𝑛 -precision transformer decoders
( )
without intermediate steps.[4] Similarly, the universal recognition problem for CFGs is P-complete, so

4This uses the result of Merrill and Sabharwal (2023b),
which would have to be adapted to transformer decoders, but
this should be straightforward.


10


-----

(if L ⊊ P) it cannot be solved by 𝑂 log 𝑛 -precision
( )
transformer decoders without intermediate steps.
However, these problems can be solved by a
transformer decoder using (a polynomial number
of) intermediate steps. The decoder has GELU
activations (Hendrycks and Gimpel, 2016) and
PE including 𝑖 and (for linear equation solving)
_𝑚[2]_ sin [2][𝑖𝜋]

_𝑚_ [and][ 𝑚][2][ cos][ 2]𝑚[𝑖𝜋] [where][ 𝑚] [is the number]

of variables. More generally, they define a class of
dynamic-programming algorithms that these transformers can solve using intermediate steps. All
these decoders have parameters that depend on 𝑛.

Merrill and Sabharwal (2024) show that a transformer decoder with 𝑂 log _𝑛_ _𝑇_ _𝑛_ precision
( ( + ( )))
and 𝑂 _𝑇_ _𝑛_ intermediate steps can simulate a Tur( ( ))
ing machine for 𝑇 _𝑛_ steps, and in particular, de( )
coders with a polynomial number of intermediate
steps recognize exactly the languages in P. The
proof is similar to that of Pérez et al. (2021), but
uses a standard definition of transformers without
PEs, relying only on the mild assumption that the
input string begins with BOS.

**6.2** **Leftmost-hard/rightmost-hard attention**

Hahn (2020) shows that leftmost-hard attention
transformers cannot recognize PARITY or DYCK-1,
using a variant of Furst et al.’s random restriction
method for proving that PARITY is outside of AC[0].

Hao et al. (2022) show more generally that
any language recognized by a transformer with
leftmost-hard attention is in AC[0]. The proof gives
a normal form for transformers with leftmost-hard
attention and uses it to construct an AC[0] circuit
family. It uses the fact that only 𝑂 log 𝑛 bits of
( )
information are needed per position.

Barceló et al. (2024) give a lower bound on
leftmost-hard-attention transformers with arbitrary
PEs depending on a single position 𝑖 and length
_𝑛, including 𝑖,_ _𝑖+11_ [,][ (−][1][)][𝑖][,][ cos][ 𝜋] [(][1]10[−][2][−][𝑖] [)], and

sin _[𝜋]_ [(][1]10[−][2][−][𝑖] [)] . They show that these transformers

can recognize any language definable in FO Mon .

[ ]
Their proof converts a FO Mon formula to LTL

[ ]
(§5.3), which is simulated in a transformer.

Angluin et al. (2023) exactly characterize
rightmost-hard-attention transformers with strict
future masking. Without PEs, these transformers
recognize exactly the class of star-free languages,
that is, languages definable in FO. With periodic
PEs, they are exactly equivalent to FO MOD, and

[ ]
with arbitrary PEs, they are exactly equivalent to
FO Mon . Strict masking is important, as non
[ ]


strict masking is less expressive. They give two
proofs of the star-free to transformer direction, one
which goes through LTL (§5.3) and one which uses
Krohn-Rhodes theory. These proofs use a Booleanvalued version of RASP (Weiss et al., 2021) as an
intermediate representation.

**6.3** **Average-hard and softmax attention**

Theoretical results on average-hard and softmax attention transformers have not yet clearly separated
the two, so we treat them together. Both kinds of
attention enable counting, which can be used to
solve problems like MAJORITY that are outside
AC[0]. But these transformers are no more powerful
than DLOGTIME-uniform TC[0], implying that they
likely cannot solve problems complete for NC[1], L,
and other classes believed to be above TC[0] (§5.4).

**6.3.1** **Lower bounds: particular languages**
The languages MAJORITY, DYCK-𝑘, and PARITY
are all not in AC[0], so are interesting test cases.

Pérez et al. (2019) prove that a transformer
encoder–decoder with a trivial decoder and without
any PE recognizes MAJORITY; Merrill et al. (2022)
prove the same for transformer encoders.

Bhattamishra et al. (2020a) prove that
SHUFFLE-DYCK-𝑘 (which equals DYCK-1 when
_𝑘_ = 1) is recognizable by a soft-attention transformer with future masking, no PE, no layernorm,
and no residual connections. Yao et al. (2021)
show that a transformer decoder can generate
DYCK-𝑘 using 𝑂 log 𝑛 precision, softmax and
( )
leftmost-hard attention, future masking, and a
PE including 𝑖 _𝑛, 𝑖_ _𝑛[3], and 𝑛._ They also give
/ /
constructions for DYCK- _𝑘, 𝐷_ .
( )

Chiang and Cholak (2022) show that transformers whose PE includes 𝑖 _𝑛_ and 1 = cos _𝑖𝜋_ can
/ (− )[𝑖]
recognize PARITY.
On the other hand, Hahn (2020) shows that softmax attention transformers cannot generate PAR
ITY or DYCK-2 under the following two conditions:

1. all position-wise functions are Lipschitzcontinuous, and

2. generation is defined using the KL divergence
criterion in Eq. (5).

The apparent contradiction is resolved by considering the different assumptions underlying each
result. Chiang and Cholak (2022) address this by
giving two constructions corresponding to Hahn’s
two conditions. The first has Lipschitz-continuous


11


-----

position-wise functions, but has high cross-entropy
(§4.3.1); as a generator, it would not meet criterion (5). The second construction uses layernorm
with 𝜀 N = 0, which is not Lipschitz-continuous,
but it has arbitrarily low cross-entropy.
A number of authors have tested empirically
whether transformers can learn the above languages. Ebrahimi et al. (2020) find that they are
competitive with LSTMs at learning DYCK-2 and
DYCK-4, and that prepending a BOS symbol helps.

Bhattamishra et al. (2020a) train transformers
with future masking and no PE on DYCK-1 and
SHUFFLE-DYCK-𝑘, finding near-perfect learning
and length generalization. For the languages
DYCK- 1, 𝐷 with learned or sinusoidal PEs, they
( )
find that the models do not generalize well for
_𝐷> 1. Yao et al. (2021) then investigate DYCK-_
_𝑘, 𝐷_ for several values of 𝑘 and 𝐷 and several
( )
PEs. They report strong generalization only when
using 𝑖 _𝑛_ for the PE, and posit that this is the key.
/
It is hard, however, to directly compare the two
results: Bhattamishra et al. (2020a) require correct prediction of the possible next symbols at each
string prefix, while Yao et al. (2021) average over
predictions of right brackets.

Delétang et al. (2023) study experimentally how
well transformers (and other networks) learn tasks
at various levels of the Chomsky hierarchy, including generalization to longer strings. They find that
transformers learn MAJORITY, but not PARITY.

**6.3.2** **Upper bounds: TC[0]**

Merrill et al. (2022) prove an upper bound analogous to that of Hao et al. (2022), but for averagehard-attention transformers. They show that an
average-hard-attention transformer with activations
in F can be simulated in TC[0]. Strobl (2023) tightens
this bound to L-uniform TC[0].
Furthermore, Merrill and Sabharwal (2023a)
show that softmax attention, 𝑂 log 𝑛 -precision
( )
transformers are in L-uniform TC[0], and then tighten
this bound to DLOGTIME-uniform TC[0] (Merrill
and Sabharwal, 2023b). The proof constructs subroutines to answer queries about the types of nodes
and connectivity of pairs of nodes in the computation graph of a transformer, and shows that these
queries can be translated to queries for a TC[0] circuit
family with 𝑂 log 𝑛 time overhead.
( )
An upper bound of DLOGTIME-uniform TC[0] immediately implies an upper bound of FOM BIT

[ ]
(Merrill and Sabharwal, 2023b). Chiang et al.

(2023) prove a tighter upper bound using a logic


called FOC MOD;, but on transformers with

[ +]
_𝑂_ 1 precision. This result is discussed below.
( )

**6.3.3** **Other lower bounds**
In addition to explicit constructions for particular
languages mentioned above, various lower bounds
have been proven, which are quite diverse.

**Counter machines** Bhattamishra et al. (2020a),
following Merrill et al. (2020), define a subclass
of counter machines called simplified and stateless
_𝑘-counter machines (SSCMs). These can update_
each counter based on the current input symbol,
but have no state and cannot read the counters until
the end of the string. They show that any SSCM
can be converted to an equivalent transformer with
future masking and no residual connections.

**Finite automata** Liu et al. (2023) study the ability of transformers with future masked attention
to simulate deterministic finite automata (DFAs),
in the sense of computing not only the same acceptance decision but also the same state sequence.
Although a transformer with depth 𝑁 can simulate
a DFA for 𝑁 timesteps, Liu et al. show how to construct lower-depth shortcuts for subclasses roughly
corresponding to classes of regular languages in
Fig. 1. Though the parameters of these constructions depend on 𝑁, in the context of this survey,
a noteworthy finding is that any regular language
in ACC[0] can be recognized up to length 𝑁 by a
transformer whose FFNs use sine activations and
whose number of parameters is independent of 𝑁.

**First-order logic** Chiang et al. (2023) obtain
both an upper and a lower bound by defining a
logic FOC MOD;, which is first-order logic with

[ +]
counting quantifiers, using two sorts for positions
and counts (Immerman, 1999, p. 185–187), where
positions have the MOD predicate (but not < or =),
and counts have <,, and =, capturing the fact that
+
transformers can add and compare activations, but
not positions. They show that this logic is intermediate in expressivity between 𝑂 1 -precision and
( )
infinite-precision transformers. The lower-bound
proof uses a normal form that eliminates quantifiers
over counts and makes quantifiers over positions
have depth 1; a perhaps surprising consequence is
that 𝑂 1 -precision transformers are no more pow( )
erful than 2-layer uniform-attention transformers.

**Temporal logic** Barceló et al. (2024) show
that average-hard-attention transformers with arbitrary PEs depending on a single position 𝑖 and


12


-----

length 𝑛, including 𝑖, _𝑖+11_ [,][ (−][1][)][𝑖][,][ cos][ 𝜋] [(][1]10[−][2][−][𝑖] [)], and

sin _[𝜋]_ [(][1]10[−][2][−][𝑖] [)], can recognize any language definable

in LTL with counting operators, Presburger arithmetic on counts, and predicates in Mon.

**Programming languages** Weiss et al. (2021) introduce the RASP (Restricted Access Sequence
Processing) language as an abstraction of transformers, discussing how its components relate to
the transformer architecture. However, they do not
prove any relationship.

Lindner et al. (2023) present Tracr, a compiler
from RASP programs to transformers. To do so,
they impose some restrictions: a maximum input
length, given at compile time; a mandatory BOS
token; and the removal of selector composition, a
RASP operation with no clear parallel in transformers. They rewrite several programs from Weiss
et al. (2021) without this operation. In the other
direction, Friedman et al. (2023) define a restricted
class of transformers that can be learned and decompiled into RASP. Finally, Angluin et al. (2023)
use a version of RASP restricted to Boolean values,
and Zhou et al. (2024) use a restricted version of
RASP to explore length generalization.

### 7 Conclusions

Out of the large body of research surveyed above,
we highlight several conclusions:

1. Transformer decoders can use intermediate
steps to simulate Turing machines; with unbounded steps, they are Turing-complete.

2. Regarding the expressivity of transformer encoders, circuit complexity and logic are especially promising frameworks.

3. Leftmost-hard-attention transformer encoders
are in AC[0] and cannot solve some intuitively
easy problems, like PARITY and MAJORITY.

4. Softmax and average-hard attention give transformer encoders the ability to count. Still, they
lie within TC[0] and likely cannot solve problems like evaluating closed Boolean formulas.

Some open questions that we think should be priorities for future research are:

5. Some variants (PEs, average-hard vs. softmax
attention, pre-norm vs. post-norm, the presence of BOS/EOS/CLS) appear to be instrumental in proofs reviewed here; can their effect on
expressivity be clarified?


6. Can the expressivity of softmax-attention
transformers be characterized more tightly or
even exactly in terms of some logic?

7. Given the current practical importance of
decoder-only transformers and chain-ofthought, what further insights can circuits or
logic provide into transformer decoders?

We hope this paper can serve as a valuable resource
for researchers pursuing these and other questions.

### Acknowledgements

We would like to thank Frank Drewes, Jon Rawski,
Ashish Sabharwal, and the anonymous reviewers
for their valuable comments on earlier versions of
this paper.

### References

[Joshua Ackerman and George Cybenko. 2020. A](http://arxiv.org/abs/2006.01338)
[survey of neural networks and formal languages.](http://arxiv.org/abs/2006.01338)
arXiv:2006.01338.

[Zeyuan Allen-Zhu and Yuanzhi Li. 2023. Physics](https://arxiv.org/abs/2305.13673)
[of language models: Part 1, context-free gram-](https://arxiv.org/abs/2305.13673)
[mar. arXiv:2305.13673.](https://arxiv.org/abs/2305.13673)

[Eric Allender. 1999. The permanent requires large](http://cjtcs.cs.uchicago.edu/articles/1999/7/contents.html)
[uniform threshold circuits. Chicago Journal of](http://cjtcs.cs.uchicago.edu/articles/1999/7/contents.html)
_Theoretical Computer Science, 1999(7)._

Dana Angluin, David Chiang, and Andy Yang.
[2023. Masked hard-attention transformers and](https://arxiv.org/abs/2310.13897)
[Boolean RASP recognize exactly the star-free](https://arxiv.org/abs/2310.13897)
[languages. arXiv:2310.13897.](https://arxiv.org/abs/2310.13897)

[Sanjeev Arora and Boaz Barak. 2009. Computa-](http://www.cambridge.org/catalogue/catalogue.asp?isbn=9780521424264)
_[tional Complexity: A Modern Approach. Cam-](http://www.cambridge.org/catalogue/catalogue.asp?isbn=9780521424264)_
bridge University Press.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E.
[Hinton. 2016. Layer normalization. In NIPS](https://arxiv.org/abs/1607.06450)
_2016 Deep Learning Symposium._

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua
[Bengio. 2015. Neural machine translation by](http://arxiv.org/abs/1409.0473)
[jointly learning to align and translate. In Pro-](http://arxiv.org/abs/1409.0473)
_ceedings of the Third International Conference_
_on Learning Representations (ICLR)._

Pablo Barceló, Alexander Kozachinskiy, Anthony Widjaja Lin, and Vladimir Podolskii. 2024.
[Logical languages accepted by transformer en-](https://openreview.net/forum?id=gbrHZq07mq)
[coders with hard attention. In Proceedings of the](https://openreview.net/forum?id=gbrHZq07mq)


13


-----

_Twelfth International Conference on Learning_
_Representations (ICLR)._

David A. Barrington. 1989. [Bounded-width](https://doi.org/10.1016/0022-0000(89)90037-8)
[polynomial-size branching programs recognize](https://doi.org/10.1016/0022-0000(89)90037-8)
[exactly those languages in NC[1]. Journal of Com-](https://doi.org/10.1016/0022-0000(89)90037-8)
_puter and System Sciences, 38(1):150–164._

David A. Barrington, Kevin Compton, Howard
[Straubing, and Denis Thérien. 1992. Regular](https://doi.org/https://doi.org/10.1016/0022-0000(92)90014-A)
[languages in NC[1]. Journal of Computer and](https://doi.org/https://doi.org/10.1016/0022-0000(92)90014-A)
_System Sciences, 44(3):478–499._

David A. Mix Barrington, Neil Immerman,
Clemens Lautemann, Nicole Schweikardt, and
[Denis Thérien. 2005. First-order expressibility](https://doi.org/10.1016/j.jcss.2004.07.004)
[of languages with neutral letters or: The Crane](https://doi.org/10.1016/j.jcss.2004.07.004)
[Beach conjecture. Journal of Computer and Sys-](https://doi.org/10.1016/j.jcss.2004.07.004)
_tem Sciences, 70(2):101–127._

David A. Mix Barrington, Neil Immerman, and
[Howard Straubing. 1990. On uniformity within](https://doi.org/https://doi.org/10.1016/0022-0000(90)90022-D)
_[NC[1]. Journal of Computer and System Sciences,](https://doi.org/https://doi.org/10.1016/0022-0000(90)90022-D)_
41(3):274–306.

David Mix Barrington and Neil Immerman. 1994.

[Time, hardware, and uniformity. In Proceedings](https://doi.org/10.1109/SCT.1994.315806)
_of the IEEE 9th Annual Conference on Structure_
_in Complexity Theory, pages 176–185._

[Valeriu Beiu and John G. Taylor. 1996. On the cir-](https://doi.org/10.1016/0893-6080(96)00130-X)
[cuit complexity of sigmoid feedforward neural](https://doi.org/10.1016/0893-6080(96)00130-X)
[networks. Neural Networks, 9(7):1155–1171.](https://doi.org/10.1016/0893-6080(96)00130-X)

Satwik Bhattamishra, Kabir Ahuja, and Navin
[Goyal. 2020a. On the ability and limitations](https://doi.org/10.18653/v1/2020.emnlp-main.576)
[of Transformers to recognize formal languages.](https://doi.org/10.18653/v1/2020.emnlp-main.576)
In Proceedings of the 2020 Conference on Em_pirical Methods in Natural Language Processing_
_(EMNLP), pages 7096–7116._

Satwik Bhattamishra, Arkil Patel, and Navin Goyal.
[2020b. On the computational power of Trans-](https://doi.org/10.18653/v1/2020.conll-1.37)
[formers and its implications in sequence mod-](https://doi.org/10.18653/v1/2020.conll-1.37)
[eling. In Proceedings of the 24th Conference](https://doi.org/10.18653/v1/2020.conll-1.37)
_on Computational Natural Language Learning_
_(CoNLL), pages 455–475._

Satwik Bhattamishra, Arkil Patel, Varun Kanade,
and Phil Blunsom. 2023. [Simplicity bias in](https://doi.org/10.18653/v1/2023.acl-long.317)
[Transformers and their ability to learn sparse](https://doi.org/10.18653/v1/2023.acl-long.317)
[Boolean functions. In Proceedings of the 61st](https://doi.org/10.18653/v1/2023.acl-long.317)
_Annual Meeting of the Association for Computa-_
_tional Linguistics (ACL), pages 5767–5791._


Tom B. Brown, Benjamin Mann, Nick Ryder,
Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish
Sastry, Amanda Askell, Sandhini Agarwal, Ariel
Herbert-Voss, Gretchen Krueger, Tom Henighan,
Rewon Child, Aditya Ramesh, Daniel M.
Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz
Litwin, Scott Gray, Benjamin Chess, Jack Clark,
Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020.
[Language models are few-shot learners. In Ad-](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
_vances in Neural Information Processing Sys-_
_tems 33 (NeurIPS), pages 1877–1901._

[Samuel R. Buss. 1987. The Boolean formula value](https://doi.org/10.1145/28395.28409)
[problem is in ALOGTIME. In Proceedings of](https://doi.org/10.1145/28395.28409)
_the Nineteenth Annual ACM Symposium on The-_
_ory of Computing (STOC), pages 123–131._

Ashok K. Chandra, Larry Stockmeyer, and Uzi
Vishkin. 1984. [Constant depth reducibility.](https://doi.org/10.1137/0213028)
_SIAM J. Computing, 13(2):423–439._

David Chiang and Peter Cholak. 2022. [Over-](https://doi.org/10.18653/v1/2022.acl-long.527)
[coming a theoretical limitation of self-attention.](https://doi.org/10.18653/v1/2022.acl-long.527)
In Proceedings of the 60th Annual Meeting of
_the Association for Computational Linguistics_
_(ACL), pages 7654–7664._

David Chiang, Peter Cholak, and Anand Pillay.
2023. [Tighter bounds on the expressivity of](https://proceedings.mlr.press/v202/chiang23a.html)
[transformer encoders. In Proceedings of the 40th](https://proceedings.mlr.press/v202/chiang23a.html)
_International Conference on Machine Learning_
_(ICML), volume 202 of Proceedings of Machine_
_Learning Research, pages 5544–5562._

[N. Chomsky and M. P. Schützenberger. 1963. The](https://doi.org/10.1016/S0049-237X(08)72023-8)
[algebraic theory of context-free languages. In](https://doi.org/10.1016/S0049-237X(08)72023-8)
P. Braffort and D. Hirschberg, editors, Computer
_Programming and Formal Systems, volume 35_
of Studies in Logic and the Foundations of Math_ematics, pages 118–161. Elsevier._

Stephen A. Cook and Pierre McKenzie. 1987.

[Problems complete for deterministic logarithmic](https://doi.org/10.1016/0196-6774(87)90018-6)
[space. Journal of Algorithms, 8(3):385–394.](https://doi.org/10.1016/0196-6774(87)90018-6)

[G. Cybenko. 1989. Approximation by superposi-](https://doi.org/10.1007/BF02551274)
[tions of a sigmoidal function. Mathematics of](https://doi.org/10.1007/BF02551274)
_Control, Signals, and Systems, 2(4):303–314._

Grégoire Delétang, Anian Ruoss, Jordi Grau-Moya,
Tim Genewein, Li Kevin Wenliang, Elliot Catt,


14


-----

Chris Cundy, Marcus Hutter, Shane Legg, Joel
[Veness, and Pedro A. Ortega. 2023. Neural net-](https://openreview.net/forum?id=WbxHAzkeQcn)
[works and the Chomsky hierarchy. In Proceed-](https://openreview.net/forum?id=WbxHAzkeQcn)
_ings of the Eleventh International Conference on_
_Learning Representations (ICLR)._

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
[Kristina Toutanova. 2019. BERT: Pre-training](https://aclanthology.org/N19-1423)
[of deep bidirectional Transformers for language](https://aclanthology.org/N19-1423)
[understanding. In Proceedings of the 2019 Con-](https://aclanthology.org/N19-1423)
_ference of the North American Chapter of the As-_
_sociation for Computational Linguistics: Human_
_Language Technologies (NAACL HLT), pages_
4171–4186.

Javid Ebrahimi, Dhruv Gelda, and Wei Zhang.
[2020. How can self-attention networks recog-](https://doi.org/10.18653/v1/2020.findings-emnlp.384)
[nize Dyck-n languages? In Findings of the Asso-](https://doi.org/10.18653/v1/2020.findings-emnlp.384)
_ciation for Computational Linguistics: EMNLP_
_2020, pages 4301–4306._

Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian
[Ye, Di He, and Liwei Wang. 2023. Towards](https://papers.nips.cc/paper_files/paper/2023/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html)
[revealing the mystery behind Chain of Thought:](https://papers.nips.cc/paper_files/paper/2023/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html)
[A theoretical perspective. In Advances in Neural](https://papers.nips.cc/paper_files/paper/2023/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html)
_Information Processing Systems 36 (NeurIPS),_
pages 70757–70798.

Patrick C. Fischer, Albert R. Meyer, and Arnold L.
[Rosenberg. 1968. Counter machines and counter](https://doi.org/10.1007/BF01694011)
[languages.](https://doi.org/10.1007/BF01694011) _Mathematical Systems Theory,_
2:265–283.

Dan Friedman, Alexander Wettig, and Danqi Chen.
[2023. Learning Transformer programs. In Ad-](https://papers.nips.cc/paper_files/paper/2023/hash/995f693b73050f90977ed2828202645c-Abstract-Conference.html)
_vances in Neural Information Processing Sys-_
_tems 36 (NeurIPS), pages 49044–49067._

Merrick Furst, James B. Saxe, and Michael Sipser.
[1984. Parity, circuits, and the polynomial-time](https://doi.org/10.1007/BF01744431)
[hierarchy. Mathematical Systems Theory, 17:13–](https://doi.org/10.1007/BF01744431)
27.

Raymond Greenlaw, H. James Hoover, and Walter L. Ruzzo. 1995. Limits to Parallel Computa_tion: P-Completeness Theory. Oxford Univer-_
sity Press. Preliminary version of Appendix A
[available as Technical Report TR91-11, Univer-](https://doi.org/10.7939/R39Z90F7X)
[sity of Alberta, Department of Computing Sci-](https://doi.org/10.7939/R39Z90F7X)
[ence.](https://doi.org/10.7939/R39Z90F7X)

[Michael Hahn. 2020. Theoretical limitations of](https://doi.org/10.1162/tacl_a_00306)
[self-attention in neural sequence models. Trans-](https://doi.org/10.1162/tacl_a_00306)
_actions of the Association for Computational_
_Linguistics, 8:156–171._


Yiding Hao, Dana Angluin, and Robert Frank.
[2022. Formal language recognition by hard at-](https://doi.org/10.1162/tacl_a_00490)
[tention Transformers: Perspectives from circuit](https://doi.org/10.1162/tacl_a_00490)
[complexity. Transactions of the Association for](https://doi.org/10.1162/tacl_a_00490)
_Computational Linguistics, 10:800–810._

[Dan Hendrycks and Kevin Gimpel. 2016. Gaussian](https://arxiv.org/abs/1606.08415)
[error linear units (GELUs). arXiv:1606.08415.](https://arxiv.org/abs/1606.08415)

William Hesse. 2001. [Division is in uniform](https://doi.org/10.1007/3-540-48224-5_9)
[TC[0]. In Automata, Languages and Program-](https://doi.org/10.1007/3-540-48224-5_9)
_ming (ICALP), pages 104–114. Springer._

John Hewitt, Michael Hahn, Surya Ganguli, Percy
Liang, and Christopher D. Manning. 2020.
[RNNs can generate bounded hierarchical lan-](https://doi.org/10.18653/v1/2020.emnlp-main.156)
[guages with optimal memory. In Proceedings of](https://doi.org/10.18653/v1/2020.emnlp-main.156)
_the 2020 Conference on Empirical Methods in_
_Natural Language Processing (EMNLP), pages_
1978–2010.

Kurt Hornik, Maxwell B. Stinchcombe, and Hal[bert White. 1989. Multilayer feedforward net-](https://doi.org/10.1016/0893-6080(89)90020-8)
[works are universal approximators. Neural Net-](https://doi.org/10.1016/0893-6080(89)90020-8)
_works, 2(5):359–366._

Austin Huang, Suraj Subramanian, Jonathan Sum,
Khalid Almubarak, and Stella Biderman. 2022.
[The annotated Transformer. Based on original](http://harvardnlp.github.io/annotated-transformer)
version by Sasha Rush.

[Neil Immerman. 1997. Languages that capture](https://doi.org/10.1137/0216051)
[complexity classes. SIAM Journal on Comput-](https://doi.org/10.1137/0216051)
_ing, 16(4):760–778._

Neil Immerman. 1999. Descriptive Complexity.
Springer.

[Neil D. Jones and William T. Laaser. 1976. Com-](https://doi.org/10.1016/0304-3975(76)90068-2)
[plete problems for deterministic polynomial](https://doi.org/10.1016/0304-3975(76)90068-2)
[time. Theoretical Computer Science, 3(1):105–](https://doi.org/10.1016/0304-3975(76)90068-2)
117.

[Johan Anthony Willem Kamp. 1968. Tense Logic](https://www.proquest.com/docview/302320357)
_[and the Theory of Linear Order. Ph.D. thesis,](https://www.proquest.com/docview/302320357)_
University of California, Los Angeles.

[Najoung Kim and Sebastian Schuster. 2023. Entity](https://doi.org/10.18653/v1/2023.acl-long.213)
[tracking in language models. In Proceedings](https://doi.org/10.18653/v1/2023.acl-long.213)
_of the 61st Annual Meeting of the Association_
_for Computational Linguistics (Volume 1: Long_
_Papers), pages 3835–3855._

Chu-Cheng Lin, Aaron Jaech, Xin Li, Matthew R.
[Gormley, and Jason Eisner. 2021. Limitations](https://doi.org/10.18653/v1/2021.naacl-main.405)


15


-----

[of autoregressive models and their alternatives.](https://doi.org/10.18653/v1/2021.naacl-main.405)
In Proceedings of the 2021 Conference of the
_North American Chapter of the Association for_
_Computational Linguistics: Human Language_
_Technologies (NAACL HLT), pages 5147–5173._

Tianyang Lin, Yuxin Wang, Xiangyang Liu, and
[Xipeng Qiu. 2022. A survey of transformers. AI](https://doi.org/10.1016/j.aiopen.2022.10.001)
_Open, 3:111–132._

David Lindner, János Kramár, Matthew Rahtz,
Thomas McGrath, and Vladimir Mikulik. 2023.
[Tracr: Compiled transformers as a laboratory](https://papers.nips.cc/paper_files/paper/2023/hash/771155abaae744e08576f1f3b4b7ac0d-Abstract-Conference.html)
[for interpretability. In Advances in Neural Infor-](https://papers.nips.cc/paper_files/paper/2023/hash/771155abaae744e08576f1f3b4b7ac0d-Abstract-Conference.html)
_mation Processing Systems 36 (NeurIPS), pages_
37876–37899.

Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay
[Krishnamurthy, and Cyril Zhang. 2023. Trans-](https://openreview.net/forum?id=De4FYqjFueZ)
[formers learn shortcuts to automata. In Proceed-](https://openreview.net/forum?id=De4FYqjFueZ)
_ings of the Eleventh International Conference on_
_Learning Representations (ICLR)._

Robert McNaughton and Seymour A. Papert. 1971.

_[Counter-Free Automata. MIT Press.](https://archive.org/details/CounterFre_00_McNa)_

[William Merrill. 2019. Sequential neural networks](https://doi.org/10.18653/v1/W19-3901)
[as automata. In Proceedings of the Workshop on](https://doi.org/10.18653/v1/W19-3901)
_Deep Learning and Formal Languages: Building_
_Bridges, pages 1–13._

William Merrill. 2020. On the linguis[tic capacity of real-time counter automata.](https://arxiv.org/abs/2004.06866)
arXiv:2004.06866.

[William Merrill. 2021. Formal language theory](https://arxiv.org/abs/2102.10094)
[meets modern NLP. arXiv:2102.10094.](https://arxiv.org/abs/2102.10094)

[William Merrill. 2023. Formal languages and the](https://doi.org/10.1007/978-3-031-33264-7_1)
[NLP black box. In Developments in Language](https://doi.org/10.1007/978-3-031-33264-7_1)
_Theory, pages 1–8._

William Merrill, Vivek Ramanujan, Yoav Goldberg, Roy Schwartz, and Noah A. Smith. 2021.
[Effects of parameter norm growth during trans-](https://doi.org/10.18653/v1/2021.emnlp-main.133)
[former training: Inductive bias from gradient](https://doi.org/10.18653/v1/2021.emnlp-main.133)
[descent. In Proceedings of the 2021 Conference](https://doi.org/10.18653/v1/2021.emnlp-main.133)
_on Empirical Methods in Natural Language Pro-_
_cessing (EMNLP), pages 1766–1781._

William Merrill and Ashish Sabharwal. 2023a.

[The parallelism tradeoff: Limitations of log-](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00562/116413/The-Parallelism-Tradeoff-Limitations-of-Log)
[precision transformers. Transactions of the As-](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00562/116413/The-Parallelism-Tradeoff-Limitations-of-Log)
_sociation for Computational Linguistics, 11:531–_
545.


[William Merrill and Ashish Sabharwal. 2023b. A](https://papers.nips.cc/paper_files/paper/2023/hash/a48e5877c7bf86a513950ab23b360498-Abstract-Conference.html)
[logic for expressing log-precision transformers.](https://papers.nips.cc/paper_files/paper/2023/hash/a48e5877c7bf86a513950ab23b360498-Abstract-Conference.html)
In Advances in Neural Information Processing
_Systems 36 (NeurIPS), pages 52453–52463._

[William Merrill and Ashish Sabharwal. 2024. The](https://openreview.net/forum?id=NjNGlPh8Wh)
[expressive power of transformers with chain of](https://openreview.net/forum?id=NjNGlPh8Wh)
[thought. In Proceedings of the Twelfth Interna-](https://openreview.net/forum?id=NjNGlPh8Wh)
_tional Conference on Learning Representations_
_(ICLR)._

William Merrill, Ashish Sabharwal, and Noah A.
Smith. 2022. [Saturated transformers are](https://doi.org/10.1162/tacl_a_00493)
[constant-depth threshold circuits. Transactions](https://doi.org/10.1162/tacl_a_00493)
_of the Association for Computational Linguistics,_
10:843–856.

William Merrill, Gail Weiss, Yoav Goldberg, Roy
Schwartz, Noah A. Smith, and Eran Yahav.
[2020. A formal hierarchy of RNN architectures.](https://doi.org/10.18653/v1/2020.acl-main.43)
In Proceedings of the 58th Annual Meeting of
_the Association for Computational Linguistics_
_(ACL), pages 443–459._

Maxwell Nye, Anders Andreassen, Guy GurAri, Henryk Michalewski, Jacob Austin, David
Bieber, David Dohan, Aitor Lewkowycz,
Maarten Bosma, David Luan, Charles Sutton,
[and Augustus Odena. 2022. Show your work:](https://openreview.net/forum?id=HBlx2idbkbq)
[Scratchpads for intermediate computation with](https://openreview.net/forum?id=HBlx2idbkbq)
[language models. In Proceedings of the Work-](https://openreview.net/forum?id=HBlx2idbkbq)
_shop on Deep Learning for Code (DL4C)._

OpenAI. 2023. GPT-4 [technical](https://arxiv.org/abs/2303.08774) report.
arXiv:2303.08774.

[Denis Paperno. 2022. On learning interpreted lan-](https://doi.org/10.1162/coli_a_00431)
[guages with recurrent models. Computational](https://doi.org/10.1162/coli_a_00431)
_Linguistics, 48(2):471–482._

Ian Parberry. 1994. Circuit Complexity and Neural
_Networks. MIT Press._

Jorge Pérez, Pablo Barceló, and Javier Marinkovic.
[2021. Attention is Turing-complete. Journal of](http://jmlr.org/papers/v22/20-302.html)
_Machine Learning Research, 22:75:1–75:35._

[Mary Phuong and Marcus Hutter. 2022. Formal](http://arxiv.org/abs/2207.09238)
[algorithms for transformers. arXiv:2207.09238.](http://arxiv.org/abs/2207.09238)

Jorge Pérez, Javier Marinkovi´c, and Pablo Barceló.
[2019. On the Turing completeness of modern](https://openreview.net/forum?id=HyGBdo0qFm)
[neural network architectures. In Proceedings of](https://openreview.net/forum?id=HyGBdo0qFm)
_the Seventh International Conference on Learn-_
_ing Representations (ICLR)._


16


-----

Alec Radford, Karthik Narasimhan, Tim Salimans,
[and Ilya Sutskever. 2018. Improving language](https://openai.com/research/language-unsupervised)
[understanding by generative pre-training.](https://openai.com/research/language-unsupervised)

[Omer Reingold. 2008. Undirected connectivity in](https://doi.org/10.1145/1391289.1391291)
[log-space. Journal of the ACM, 55(4):1–24.](https://doi.org/10.1145/1391289.1391291)

Clayton Sanford, Daniel Hsu, and Matus Telgarsky.
[2023. Representational strengths and limitations](https://papers.nips.cc/paper_files/paper/2023/hash/73bf692447f174984f30499ec9b20e04-Abstract-Conference.html)
[of transformers. In Advances in Neural Infor-](https://papers.nips.cc/paper_files/paper/2023/hash/73bf692447f174984f30499ec9b20e04-Abstract-Conference.html)
_mation Processing Systems 36 (NeurIPS), pages_
36677–36707.

Hava T. Siegelmann and Eduardo D. Sontag. 1994.

[Analog computation via neural networks. Theo-](https://doi.org/10.1016/0304-3975(94)90178-3)
_retical Computer Science, 131(2):331–360._

Hava T. Siegelmann and Eduardo D. Sontag. 1995.

[On the computational power of neural nets. Jour-](https://doi.org/10.1006/jcss.1995.1013)
_nal of Computer and System Sciences, 50(1):132–_
150.

Jiˇrí Šíma and Pekka Orponen. 2003. [General-](https://doi.org/10.1162/089976603322518731)
[purpose computation with neural networks: A](https://doi.org/10.1162/089976603322518731)
[survey of complexity theoretic results. Neural](https://doi.org/10.1162/089976603322518731)
_Computation, 15(12):2727–2778._

Michael Sipser. 2013. Introduction to the Theory
_of Computation, 3rd edition. Cengage Learning._

Kai-Yeung Siu, Vwani Roychowdhury, and
Thomas Kailath. 1995. Discrete Neural Compu_tation. Prentice Hall._

Howard Straubing. 1994. Finite Automata, Formal
_Logic, and Circuit Complexity. Springer._

[Lena Strobl. 2023. Average-hard attention trans-](https://arxiv.org/abs/2308.03212)
[formers are constant-depth uniform threshold](https://arxiv.org/abs/2308.03212)
[circuits. arXiv:2308.03212.](https://arxiv.org/abs/2308.03212)

[I. H. Sudborough. 1975. On tape-bounded com-](https://doi.org/10.1016/S0022-0000(75)80014-6)
[plexity classes and multihead finite automata.](https://doi.org/10.1016/S0022-0000(75)80014-6)
_Journal of Computer and System Sciences,_
10(1):62–76.

Mirac Suzgun, Yonatan Belinkov, Stuart Shieber,
and Sebastian Gehrmann. 2019. [LSTM net-](https://doi.org/10.18653/v1/W19-3905)
[works can perform dynamic counting. In Pro-](https://doi.org/10.18653/v1/W19-3905)
_ceedings of the Workshop on Deep Learning and_
_Formal Languages: Building Bridges, pages 44–_
54.

[Wolfgang Thomas. 1997. Languages, automata,](https://doi.org/10.1007/978-3-642-59126-6_7)
[and logic.](https://doi.org/10.1007/978-3-642-59126-6_7) In Grzegorz Rozenberg and Arto


Salomaa, editors, Handbook of Formal Lan_guages: Volume 3 Beyond Words, pages 389–_
455. Springer.

Ashish Vaswani, Noam Shazeer, Niki Parmar,
Jakob Uszkoreit, Llion Jones, Aidan N. Gomez,
[Lukasz Kaiser, and Illia Polosukhin. 2017. At-](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
[tention is all you need. In Advances in Neural](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
_Information Processing Systems 30 (NeurIPS)._

Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu,
Changliang Li, Derek F. Wong, and Lidia S.
[Chao. 2019. Learning deep Transformer mod-](https://doi.org/10.18653/v1/P19-1176)
[els for machine translation. In Proceedings of](https://doi.org/10.18653/v1/P19-1176)
_the 57th Annual Meeting of the Association for_
_Computational Linguistics (ACL)._

Colin Wei, Yining Chen, and Tengyu Ma. 2022a.

[Statistically meaningful approximation: a case](https://papers.nips.cc/paper_files/paper/2022/hash/4ebf1d74f53ece08512a23309d58df89-Abstract-Conference.html)
[study on approximating Turing machines with](https://papers.nips.cc/paper_files/paper/2022/hash/4ebf1d74f53ece08512a23309d58df89-Abstract-Conference.html)
[transformers. In Advances in Neural Informa-](https://papers.nips.cc/paper_files/paper/2022/hash/4ebf1d74f53ece08512a23309d58df89-Abstract-Conference.html)
_tion Processing Systems 35 (NeurIPS), pages_
12071–12083.

Jason Wei, Xuezhi Wang, Dale Schuurmans,
Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi,
[Quoc V. Le, and Denny Zhou. 2022b. Chain-](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html)
[of-thought prompting elicits reasoning in large](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html)
[language models. In Advances in Neural Infor-](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html)
_mation Processing Systems 35 (NeurIPS), pages_
24824–24837.

Gail Weiss, Yoav Goldberg, and Eran Yahav. 2018.

[On the practical computational power of finite](https://doi.org/10.18653/v1/P18-2117)
[precision RNNs for language recognition. In](https://doi.org/10.18653/v1/P18-2117)
_Proceedings of the 56th Annual Meeting of_
_the Association for Computational Linguistics_
_(ACL), pages 740–745._

Gail Weiss, Yoav Goldberg, and Eran Yahav. 2021.

[Thinking like Transformers. In Proceedings of](https://proceedings.mlr.press/v139/weiss21a.html)
_the 38th International Conference on Machine_
_Learning (ICML), volume 139 of Proceedings_
_of Machine Learning Research, pages 11080–_
11090.

Shunyu Yao, Binghui Peng, Christos Papadimitriou, and Karthik Narasimhan. 2021. [Self-](https://doi.org/10.18653/v1/2021.acl-long.292)
[attention networks can process bounded hier-](https://doi.org/10.18653/v1/2021.acl-long.292)
[archical languages. In Proceedings of the 59th](https://doi.org/10.18653/v1/2021.acl-long.292)
_Annual Meeting of the Association for Compu-_
_tational Linguistics and the 11th International_
_Joint Conference on Natural Language Process-_
_ing (ACL-IJCNLP), pages 3770–3785._


17


-----

Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh
Rawat, Sashank J. Reddi, and Sanjiv Kumar.
[2020. Are Transformers universal approxima-](https://openreview.net/forum?id=ByxRM0Ntvr)
[tors of sequence-to-sequence functions? In 8th](https://openreview.net/forum?id=ByxRM0Ntvr)
_International Conference on Learning Represen-_
_tations (ICLR)._

Hattie Zhou, Arwen Bradley, Etai Littwin, Noam
Razin, Omid Saremi, Josh Susskind, Samy Ben[gio, and Preetum Nakkiran. 2024. What algo-](https://openreview.net/forum?id=AssIuHnmHX)
[rithms can Transformers learn? A study in length](https://openreview.net/forum?id=AssIuHnmHX)
[generalization. In Proceedings of the Twelfth](https://openreview.net/forum?id=AssIuHnmHX)
_International Conference on Learning Represen-_
_tations (ICLR)._


18


-----

