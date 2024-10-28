## Finding Syntactic Representations in Neural Stacks

### William Merrill[∗] [†] Lenny Khazan[†] Noah Amsel[†]
 Yiding Hao[†] Simon Mendelsohn[†] Robert Frank[†]

_† Yale University, New Haven, CT, USA_
_∗_ Allen Institute for Artificial Intelligence, Seattle, WA, USA
### first.last@yale.edu


### Abstract


Neural network architectures have been augmented with differentiable stacks in order to
introduce a bias toward learning hierarchysensitive regularities. It has, however, proven
difficult to assess the degree to which such a
bias is effective, as the operation of the differentiable stack is not always interpretable. In
this paper, we attempt to detect the presence
of latent representations of hierarchical structure through an exploration of the unsupervised learning of constituency structure. Using
a technique due to Shen et al. (2018a,b), we
extract syntactic trees from the pushing behavior of stack RNNs trained on language modeling and classification objectives. We find that
our models produce parses that reflect natural
language syntactic constituencies, demonstrating that stack RNNs do indeed infer linguistically relevant hierarchical structure.

### 1 Introduction


Sequential models such as long short-term memory networks (LSTMs; Hochreiter and Schmidhuber, 1997) have been proven capable of exhibiting qualitative behavior that reflects a sensitivity to
regularities that are structurally conditioned, such
as subject–verb agreement (Linzen et al., 2016;
Gulordava et al., 2018). However, detailed analysis of such models has shown that this apparent
sensitivity to structure does not always generalize
to inputs with a high degree of syntactic complexity (Marvin and Linzen, 2018). These observations suggest that sequential models may not in
fact be representing sentences in the kind of hierarchically organized representations that we might
expect.
Stack-structured recurrent memory units (Joulin
and Mikolov, 2015; Grefenstette et al., 2015; Yo
_∗_ Work completed while the author was at Yale University.


gatama et al., 2018; and others) offer a possible method for explicitly biasing neural networks
to construct hierarchical representations and make
use of them in their computation. Since syntactic
structures can often be modeled in a context-free
manner (Chomsky, 1956, 1957), the correspondence between pushdown automata and contextfree grammars (Chomsky, 1962) makes stacks a
natural data structure for the computation of hierarchical relations. Recently, Hao et al. (2018)
have shown that stack-augmented RNNs (henceforth stack RNNs) have the ability to learn classical stack-based algorithms for computing contextfree transductions such as string reversal. However, they also find that such algorithms can be difficult for stack RNNs to learn. For many contextfree tasks such as parenthesis matching, the stack
RNN models they consider instead learn heuristic “push-only” strategies that essentially reduce
the stack to unstructured recurrent memory. Thus,
even if stacks allow hierarchical regularities to
be expressed, the bias that stack RNNs introduce
does not guarantee that the networks will detect
them.
The current paper aims to move beyond the
work of Hao et al. (2018) in two ways. While that
work was based on artificially generated formal
languages, this paper considers the ability of stack
RNNs to succeed on tasks over natural language
data. Specifically, we train such networks on two
objectives: language modeling and the number
_prediction task, a classification task proposed by_
Linzen et al. (2016) to determine whether or not a
model can capture structure-sensitive grammatical
dependencies. Further, in addition to using visualizations of the pushing and popping actions of the
stack RNN to assess its hierarchical sensitivity, we
use a technique proposed by Shen et al. (2018a,b)
to assess the presence of implicitly-represented
hierarchically-organized structure through the task


-----

of unsupervised parsing. We extract syntactic constituency trees from our models and find that they
produce parses that broadly reflect phrasal groupings of words in the input sentences, suggesting
that our models utilize the stack in a way that reflects the syntactic structures of input sentences.

This paper is organized as follows. Section 2

introduces the architecture of our stack models,
which extends the architecture of Grefenstette
et al. (2015) by allowing multiple items to be
pushed to, popped from, or read from the stack
at each computational step. Section 3 then describes our training procedure and reports results
on language modeling and agreement classification. Section 4 investigates the behavior of the
stack RNNs trained on these tasks by visualizing their pushing behavior. Building on this, Section 5 describes how we adapt Shen et al.’s (2018a;
2018b) unsupervised parsing algorithm to stack
RNNs and evaluates the degree to which the resulting parses reveal structural representations in
stack RNNs. Section 6 discusses our observations,
and Section 7 concludes.

### 2 Network Architecture

In a stack RNN (Grefenstette et al., 2015; Hao
et al., 2018), a neural network adhering to a standard recurrent architecture, known as a controller,
is enhanced with a non-parameterized stack. At
each time step, the controller network receives an
input vector xt and a recurrent state vector ht−1
provided by the controller architecture, along with
a read vector rt−1 summarizing the top elements
on the stack. The controller interfaces with the
stack by computing continuous values that serve
as instructions for how the stack should be modified. These instructions consist of vt, a vector that
is pushed to the top of the stack; dt, a number representing the strength of the newly pushed vector
**vt; ut, the number of items to pop from the stack;**
and rt, the number of items to read from the top of
the stack. The instructions ⟨vt, ut, dt, rt⟩ are produced by the controller as output and presented to
the stack. The stack then computes the next read
vector rt, which is given to the controller at the
next time step. This general architecture is portrayed in Figure 1. In the next two subsections,
we describe how the stack computes rt using the
instructions ⟨vt, ut, dt, rt⟩ and how the controller
computes the stack instructions.


**yt, ht** **rt, Vt, st**

Controller Stack

**vt, ut, dt, rt**

**xt, ht−1, rt−1** **Vt−1, st−1**

Figure 1: The neural stack architecture.

**2.1** **Stack Actions**

A stack at time t consists of a sequence of vectors
_⟨Vt[1], Vt[2], . . ., Vt[t]⟩, organized into a matrix_
**Vt whose ith row is Vt[i]. By convention, Vt[t]**
is the “top” element of the stack, while Vt[1] is
the “bottom” element. Each element Vt[i] of the
stack is associated with a strength st[i] ≥ 0. The
strength of a vector Vt[i] represents the degree to
which the vector is on the stack: a strength of 1
means that the vector is “fully” on the stack, while
a strength of 0 means that the vector has been
popped from the stack. The strengths are organized into a vector st = ⟨st[1], st[2], . . ., st[t]⟩.
At time t, the stack receives a set of instructions
_⟨vt, ut, dt, rt⟩_ and performs three operations: pop_ping, pushing, and reading, in that order._
The popping operation is implemented by reducing the strength of each item on the stack by
a number ut[i], ensuring that the strength of each
item can never fall below 0.

**st[i] = ReLU (st−1[i] −** **ut[i])**

The ut[i]s are computed as follows. The total amount of strength to be reduced is the pop
_strength ut. Popping begins by attempting to re-_
duce the strength st[t − 1] of the top item on
the stack by the full pop strength ut. Thus, as
shown below, ut[t − 1] = ut. For each i, if
**st−1[i] < ut[i], then the ith item has been fully**
popped from the stack, “consuming” a portion
of the pop strength of magnitude st−1[i]. The
strength of the next item is then reduced by an
amount ut[i − 1] given by the “remaining” pop
strength ut[i] − **st−1[i].**

**ut[i] =**
�
_ut,_ _i = t −_ 1
ReLU(ut[i + 1] − **st−1[i + 1]),** _i < t −_ 1

The pushing operation simply places the vector
**vt at the top of the stack with strength dt. Thus,**


-----

**Vt and st[t] are updated as follows.**


**st[t] = dt** **Vt[i] =**


�
**vt,** _i = t_
**Vt−1[i],** _i < t_


Note that st[1], st[2], . . ., st[t − 1] have already
been updated during the popping step.
Finally, the reading operation produces a “summary” of the top of the stack by computing a
weighted sum of all the vectors on the stack.


**rt =**


_t_
�

min (st[i], ρt[i]) · Vt[i]
_i=1_


The weights ρt[i] are computed in a manner similar to the ut[i]s. The sum should include the
top elements of the stack whose strengths add
up to the read strength rt. The weight ρt[t] assigned to the top item is initialized to the full
read strength rt, while the weights ρt[i] assigned
to lower items are based on the “remaining” read
strength ρt[i +1] _−_ **st[i** +1] after strength has been
assigned to higher items.


This paper departs from Grefenstette et al.’s architecture by allowing for push, pop, and read
operations to be executed with variable strength
greater than 1. We achieve this by using an enhanced control interface inspired by Yogatama
et al.’s (2018) Multipop Adaptive Computation
Stack architecture. In that model, the controller
determines how much weight to pop from the
stack at each time step by computing a distribution
P[u] describing the probability of popping u units
from the stack. The next stack state V is computed
as a superposition of the possible stack states V[u]

resulting from popping u units from the stack,
weighted by P[u]. Our model follows Yogatama
et al. in computing probability distributions over
possible values of ut, dt, and rt. However, instead
of superimposing stack states, which may hinder
interpretability, we simply set the value of each instruction to be the expected value of its associated
distribution. For a distribution vector p, define the
operator E[p] as follows:


_k_
�

_ip[i + 1]_

_i=0_


_ρt[i] =_


�
_rt,_ _i = t_
ReLU (ρt[i + 1] − **st[i + 1])** _i < t_


E[p] =


**2.2** **Stack Interface**

The architecture of Grefenstette et al. (2015) assumes that the controller is a neural network of the
form
_⟨ot, ht⟩_ = C(xt, ht−1, rt−1)

where ht is its state at time t, xt is its input, rt
is the vector read from the stack at the previous
step, and ot is an output vector used to produce
the network output yt and the stack instructions
_⟨vt, ut, dt, rt⟩._
The stack instructions ⟨vt, ut, dt, rt⟩ are computed as follows. The read strength rt is fixed to
1. The other values are determined by passing ot
to specialized layers. The vectors yt and vt are
computed using a tanh layer, while the scalar values ut and dt are obtained from a sigmoid layer.
Thus, the push and pop strengths are constrained
to values between 0 and 1.

**yt = softmax (W[y]ot + b[y])**

**vt = tanh (W[v]ot + b[v])**

_ut = σ (W[u]ot + bu)_

� �
_dt = σ_ **W[d]ot + bd** (1)

_rt = 1_


E[p] denotes the expected value of p if we treat it
as a distribution over 0, 1, . . ., k . The maximum
_{_ _}_
value k is fixed in advance as a hyperparameter of
our model. The output yt and instructions vt, ut,
_dt, and rt are then computed as follows:_

**yt = softmax (W[y]ot + b[y])**

**vt = tanh (W[v]ot + b[v])**

_ut = E [softmax (W[u]ot + b[u])]_

� �
_dt = E_ softmax **W[d]ot + b[u][��]**

_rt = E [softmax (W[r]ot + b[r])]_

The full architecture that we used for language
modeling and agreement classification is a controller network which, at time t, reads the word xt
as well as the previous stack summary rt−1. These
vectors are passed through an LSTM layer to produce the vector ot. Then, instructions for the stack
are computed from ot according to the equations
above. Finally, these instructions are executed to
modify the stack state and produce the next stack
summary vector rt. In our experiments, the size
of the LSTM layer was 100, and the size of each
stack vector was 16.


-----

### 3 Model Training

This paper considers models trained on a language
modeling objective and a classification objective.
On each objective, we train several neural stack
models along with an LSTM baseline.[1] This section describes the procedure used to train our models and presents the perplexity and classification
values they attain on their training objectives.

**3.1** **Data and Training**

Our models are trained using the Wikipedia cor_pus, a subset of the English Wikipedia used by_
Linzen et al. (2016) for their experiments. The
classification task we consider is the number pre_diction task, proposed by Linzen et al. (2016) as_
a diagnostic for assessing whether or not LSTMs
can infer grammatical dependencies sensitive to
syntactic structure. In this task, the network is
shown a sequence of words forming the beginning
of a sentence from the Wikipedia corpus. The
next word in the sentence is always a verb, and
the network must predict whether the verb is singular (SG) or plural (PL). For example, on input
_The cats on the boat, the network must predict PL_
to match cats. We train and evaluate our models
on the number prediction task using Linzen et al.’s
(2016) simple dependency dataset, which contains
141,948 training examples, 15,772 validation examples, and 1,419,491 testing examples.

We used a model with very few parameters and
basic setting of hyperparameters. The LSTM hidden state was fixed to a size of 100, while the
vectors placed on the stack had size 16. Including the embedding layer, the Wikipedia model had
1,584,255 parameters. We used the Adam optimizer (Kingma and Ba, 2015) with a learning rate
of 0.001. The language models were trained for
five epochs, while the agreement classifiers used
an early stopping criterion. In addition to the
LSTM baseline, for each task, we trained a stack
RNN in which ut is fixed to 1 and dt ranges from 0
to k = 4, as well as a stack RNN in which dt fixed
to 1 and ut ranges from 0 to k = 4. Additionally, for the classification task we trained a stack
RNN in which ut ranges from 0 to k = 4 and dt is
computed as in Equation 1.

[1Our code is available at https://github.com/](https://github.com/viking-sudo-rm/industrial-stacknns)
[viking-sudo-rm/industrial-stacknns.](https://github.com/viking-sudo-rm/industrial-stacknns)

|Col1|Stack Stack (u = 1) (d = 1) t t|LSTM|
|---|---|---|
|Perp Agree|92.81 128.28 93.59 92.28|91.69 93.95|


Table 1: Results for language models trained on the
Wikipedia dataset.

**3.2** **Evaluation**

Our language models are evaluated according to
two metrics. Firstly, we reserve 10% of the
Wikipedia corpus for evaluating test perplexity of
the trained language models. Secondly, as a simple diagnostic of sensitivity to syntactic structure,
we evaluate the performance of our Wikipediatrained language models on number agreement
_prediction (Linzen et al., 2016). Under this evalu-_
ation regime, we use our language model to simulate the number prediction task and compute the
resulting classification accuracy. We do this by
presenting the model with an input for the number
prediction task and comparing the probabilities assigned to the verb that follows the input in the
Wikipedia corpus. For example, if The cats on the
_boat purr appears in the Wikipedia corpus, then_
we present The cats on the boat to the language
model and compare the probabilities assigned to
the singular and plural forms purrs and purr, respectively. We consider the language model to
make a correct prediction if the form of the next
lexical item with the correct grammatical number
(SG or PL) is predicted with greater probability
than the alternative.
The number prediction classifiers we trained are
evaluated according to classification accuracy. For
each input sentence, we define the attractors of the
input to be the nouns intervening between the subject and the verb whose number is being classified.
For example, in the input The cat on the boat, cat
is the subject of the following verb, while boat is
an attractor. We compute the accuracy of our classifiers on the full testing set of the simple dependency data set as well as subsets of the testing set
consisting of sentences with a fixed number of attractors.

**3.3** **Training Results**

Table 1 shows the quantitative results for our language models. The stack RNN is comparable to
our LSTM baseline in terms of language modeling perplexity and agreement prediction accuracy when ut is fixed to 1, though the latter per

-----

|Number of Attractors|Stack Stack Stack (u = 1) (d = 1) t t|LSTM|
|---|---|---|
|Overall 0 1 2 3 4 5|98.89 98.88 98.88 99.29 99.23 99.24 94.75 95.43 95.18 89.85 91.70 91.86 83.42 86.59 87.47 79.50 84.14 85.56 71.07 71.70 77.99|98.89 99.26 95.27 90.15 84.30 78.61 74.21|


Table 2: Number prediction accuracies attained by the
three stack RNN classifiers and the LSTM baseline.

Figure 2: Push and read strengths computed by the
_ut = 1 language model. Values underneath each word_
show the total strength remaining on the stack at that
step.

forms slightly better according to both metrics.
The stack RNN attains a significantly worse perplexity when dt is fixed to 1, and its agreement
prediction accuracy is worse than both the LSTM
baseline and the stack RNN with ut = 1.

Table 2 shows test accuracies attained by classifiers trained on the number prediction task. While
the stack classifier with ut fixed to 1 and the
LSTM baseline achieve the best overall accuracy,
the stack with unrestricted ut and sigmoid dt and
the stack with dt fixed to 1 exceed the baseline
on sentences with at least 2 attractors. We take
this to suggest that the hierarchical bias provided
by the stack can improve performance on syntactically complex cases.

### 4 Interpreting Stack Usage

The results presented in Subsection 3.3 show that
the ut = 1 stack RNNs perform comparably to
LSTMs in terms of quantitative evaluation metrics. The goal of this section is to assess whether
or not stack RNNs achieve this level of perfor

mance in an interpretable manner. We do this by
visualizing the push and read strengths computed
by the ut = 1 language model when processing two example sentences. These visualizations
are shown in Figure 2 and Figure 3. Notice that
the push strength tends to spike following words
with subcategorization requirements. For example, the preposition in and the transitive verbs eat
and is both require NP objects, and accordingly the
model assigns a high push strength to these words.
This suggests that the model is using the stack
to capture hierarchical dependencies by keeping
track of words that predictably introduce various
kinds of phrases.

Figure 4 shows push strengths computed by the
_ut = 1 language model, aggregated across the en-_
tire Wikipedia corpus. We see that push strengths
differ systematically based on part of speech. The
distribution of push strengths computed by the network upon seeing a noun is tightly concentrated
around 0.5, whereas the push strength upon seeing a verb tends to be greater—usually more than
2.5. This phenomenon reflects the fact that verbs
typically take objects while nouns do not.
We also find that push strengths assigned to
verbs depend on their transitivity. The right panel
of Figure 4 shows push strength distributions for
a collection of common transitive and intransitive
verbs. The model distinguishes between these two
types of verbs by assigning high push strengths to
transitive verbs and low push strengths to intransitive verbs. We make similar observations for
other parts of speech: prepositions, which take
objects, typically receive higher push strengths,
while determiners and adjectives, which do not
take phrasal complements, receive lower push
strengths.

### 5 Inference of Syntactic Structure

Section 4 has shown that the push strengths dt
computed by the ut = 1 language model reflect
the subcategorization requirements of the words
encountered by the network. Based on this phenomenon, we may interpret the stack to be keeping
track of phrases that are “in progress.” A high push
strength induced by a transitive verb, for example,
may be thought to indicate that a verb phrase has
begun, and that this phrase ends when the object of
the verb is seen. We thus hypothesize that for each
time step t, dt represents the size of the phrase that
begins with the word read by the network at time


-----

Figure 3: Distributions for push and read strengths at each step of processing example sentences. For example, the
push strength chosen after processing the (0.46) is the expected value of the blue distribution in the far left plot.

All Words Nouns and Verbs Transitive vs. Intransitive Verbs

Figure 4: Distributions of dt for the ut = 1 language model over all test sentences. The center panel shows
the distributions of dt for nouns and verbs, and the right panel shows the distributions for selected transitive and
intransitive verbs.


_t. If dt is low, then this phrase consists of a single_
word; if it is high, then this is a complex phrase
consisting of multiple words.

A similar intuition underlies the unsupervised
parsing framework of Shen et al. (2018a,b). Under
this framework, constituency structure is induced
from a sequence of words by computing a syntac_tic distance between every two adjacent words. In-_
tuitively, the syntactic distance between two words
measures the distance from the lowest common
parent node of the two words to the bottom of the
tree. If two words have a low syntactic distance,
then they are likely siblings in a small constituent;
if they have a high syntactic distance, then they
probably belong to different phrases. Whereas

Figure 2 and Figure 3 allow us identify specific
time steps at which the stack recognizes the beginning of a phrase, the unsupervised parsing framework allows us to explicitly visualize the phrasal


organization of input sequences induced by our interpretation of the push strengths.

Given an input sequence x1, x2, . . ., xn, we define the syntactic distance between each xt and
**xt−1 for our ut = 1 model to be the push strength**
_dt computed by the controller during time t. If the_
current word does not open any new constituents,
then it belongs to the same constituent as the previous word, and therefore should be assigned a
low syntactic distance. On the other hand, if the
current word opens a complex constituent, then it
is lower in the parse tree than the previous word,
and therefore should be assigned a high syntactic distance. Similarly, for our dt = 1 model,
we let ut be the syntactic distance between xt and
**xt+1. Under this interpretation, the pop strength**
estimates the complexity of the constituents that
the current word closes. If the current word closes
many complex constituents, then the next word


-----

appears at a higher level in the parse tree, and
is therefore syntactically distant from the current
word.

**Algorithm 1 (Shen et al., 2018a,b)**

1: procedure MAKETREE(X, d)

2: **if X has at most one word then**

3: **return X**

4: **else**

5: _i ←_ argmaxj dj

6: _l_ MAKETREE(X[: i 1], d[: i 1])
_←_ _−_ _−_

7: _r_ MAKETREE(X[i+1 :], d[i+1 :])
_←_

8: **if l and r are not empty then**


9: **return Tree[l, Tree[X[i], r]]**

10: **else if l is empty then**

11: **return Tree[X[i], r]**

12: **else**

13: **return Tree[l, X[i]]**

Algorithm 1 shows our procedure for constructing trees. The algorithm takes as input a sequence
of words arranged into a matrix X and a vector
**d containing the syntactic distance between each**
word and the previous word. Following Shen et al.
(2018a,b), we recursively split X into binary constituents. At each recursion level, we greedily
choose the word with the highest syntactic distance as the split point. The final output is a binary
tree spanning the full sentence.

**5.1** **Evaluation**

We compute F1 scores for the parses obtained
from our Wikipedia language models by comparing against parses from Section 23 of the Penn
Treebank’s Wall Street Journal corpus (WSJ23,
Marcus et al., 1994). Since Algorithm 1 produces unlabeled binary trees, our evaluation uses
the gold standard of Htut et al. (2018), which
consists of unlabeled, binarized versions of the
WSJ23 trees. We also decapitalize the first word
of every sentence for compatibility with our training data.
As a baseline, we the F1 scores attained by our
models to those computed for purely right- and
left-branching trees. A right-branching parse is
equivalent to the output of Algorithm 1 on a sequence of equal syntactic distances. Thus, the difference between the right-branching F1 score and
our models’ scores is a measure of the amount
of syntactic information encoded by the push and
pop strength sequences. We also compare our


Table 3: Unsupervised parsing performance evaluated
on the WSJ23 dataset, attained by our stack models
(top), the right- and left-branching baselines (middle),
and the PRPN models (bottom).

F1 scores to the results of Htut et al.’s (2018)
replication study for the parsing–reading–predict
_network models (PRPN-LM and PRPN-UP), the_
two syntactic-distance-based unsupervised parsers
originally proposed by Shen et al. (2018a).

**5.2** **Results**

The F1 evaluation (see Table 3) shows that our
Wikipedia model with ut = 1 significantly outperforms the baseline on the Penn Treebank, while
our model with dt = 1 performs slightly better
than the baseline. This is evidence that the types
of hierarchical structures produced by Algorithm 1
resemble expert-annotated constituency parses.
Our results do not exceed those of Htut et al.’s
(2018) replication study. It is worth noting that
our right- and left-branching baseline scores are
somewhat lower than theirs. This suggests that
differences in data processing or implementation
might make our evaluation more difficult. Regardless, we consider our results to still be somewhat
competitive, given that our language models were
trained on out-of-domain data with few parameters
and minimal hyperparameter tuning.
We provide example parses extracted from the
stack RNN language models with ut = 1 in Figure 5. Overall, our unsupervised parses tend to resemble the gold-standard parses with some differences. Periods in our parses systematically attach
lower in the structure in our extracted parses than
in the gold-standard trees. High attachment would
require a high syntactic distance (i.e., high push
strength) between the period and the remainder of
the sentence. However, the period inherently does
not have any subcategorization requirements, so it
induces a low push strength. In contrast, prepositional phrases attach higher in our structures than
in the gold parses. This may be the result of
fixed subcategorization-associated push strengths
for prepositions that give rise to fairly high esti

-----

.
The finger-pointing has The finger-pointing

.


already begun


has already

assailed

by

Big

$ 2,400 to Paris


.


The

CONCORDE


The futures

CONCORDE


floor traders

to

$ 3,200


.

.
London


floor traders


.


Figure 5: Sample parses obtained from our stack RNN language model with ut = 1 (left), compared to Htut et al.’s
(2018) gold-standard parses (right).


mates of syntactic distance.

### 6 Discussion

Overall, our stack language models show no improvement over the LSTM baseline in terms of
perplexity and classification accuracy. Although
the ut = 1 language model is comparable to
the LSTM on these metrics, it ultimately achieves
worse scores than the baseline. However, we
have now seen that the pushing behavior of the
model reflects subcategorization properties of lexical items that play an important role in determining their syntactic behavior, and that these properties allow reasonable parses to be extracted from
this model. These observations show that the ut =
1 model has learned to encode structural representations using the stack. Quantitatively, the importance of this structural information for the training objectives can be seen in Table 2, where the
stack at least partially alleviates the difficulty experienced by the LSTM classifier in handling syntactically complex inputs.


While our stack language models do not exceed the LSTM baseline in terms of perplexity
and agreement accuracy, Yogatama et al. (2018)
find that their Multipop Adaptive Computation
Stack architecture substantially outperforms a bare
LSTM on these metrics. Compared to their models, we use fewer parameters and minimal hyperparameter tuning. Thus, it is possible that increasing the number of parameters in our controller
may lead to similar increases in performance in
addition to the structural interpretability that we
have observed.

### 7 Conclusion

The results reported here point to the conclusion
that stack RNNs trained on corpora of natural language text do in fact learn to encode sentences in
a hierarchically organized fashion. We show that
the sequence of stack operations used in the processing of a sentence lets us uncover a syntactic
structure that matches standardly assigned structure reasonably well, even if the addition of the


-----

stack does not improve the stack RNN’s performance over the LSTM baseline in terms of the language modeling objective. We also find that using
the stack RNN to predict the grammatical number of a verb results in better hierarchical generalizations in syntactically complex cases than is
possible with stackless models. Taken together,
these results suggest that the stack RNN model
yields comparable performance to other architectures, while producing structural representations
that are easier to interpret and that show signs of
being linguistically natural.

### References

[Noam Chomsky. 1956. Three models for the descrip-](https://doi.org/10.1109/TIT.1956.1056813)
[tion of language. IRE Transactions on Information](https://doi.org/10.1109/TIT.1956.1056813)
_Theory, 2(3):113–124._

Noam Chomsky. 1957. Syntactic Structures, 1 edition.
Mouton, The Hague, Netherlands.

Noam Chomsky. 1962. Context-free grammars and
pushdown storage. Technical Report 65, MIT Research Laboratory for Electronics, Cambridge, MA.

Edward Grefenstette, Karl Moritz Hermann, Mustafa
Suleyman, and Phil Blunsom. 2015. [Learning to](http://arxiv.org/abs/1506.02516)
[Transduce with Unbounded Memory.](http://arxiv.org/abs/1506.02516) _Computing_
_Research Repository, arXiv:1506.02516v3._

Kristina Gulordava, Piotr Bojanowski, Edouard Grave,
Tal Linzen, and Marco Baroni. 2018. [Colorless](https://doi.org/10.18653/v1/N18-1108)
[Green Recurrent Networks Dream Hierarchically.](https://doi.org/10.18653/v1/N18-1108)
In Proceedings of the 2018 Conference of the North
_American Chapter of the Association for Computa-_
_tional Linguistics: Human Language Technologies,_
volume 1, pages 1195–1205, New Orleans, LA. Association for Computational Linguistics.

Yiding Hao, William Merrill, Dana Angluin, Robert
Frank, Noah Amsel, Andrew Benz, and Simon
[Mendelsohn. 2018. Context-free transductions with](https://www.aclweb.org/anthology/W18-5433)
[neural stacks. In Proceedings of the 2018 EMNLP](https://www.aclweb.org/anthology/W18-5433)
_Workshop BlackboxNLP: Analyzing and Interpret-_
_ing Neural Networks for NLP, pages 306–315, Brus-_
sels, Belgium. Association for Computational Linguistics.

Sepp Hochreiter and J¨urgen Schmidhuber. 1997.

[Long short-term memory.](https://doi.org/10.1162/neco.1997.9.8.1735) _Neural Computation,_
9(8):1735–1780.

Phu Mon Htut, Kyunghyun Cho, and Samuel Bowman.
2018. [Grammar induction with neural language](https://www.aclweb.org/anthology/W18-5452)
[models: An unusual replication.](https://www.aclweb.org/anthology/W18-5452) In Proceedings
_of the 2018 EMNLP Workshop BlackboxNLP: An-_
_alyzing and Interpreting Neural Networks for NLP,_
pages 371–373, Brussels, Belgium. Association for
Computational Linguistics.


Armand Joulin and Tomas Mikolov. 2015. Inferring
Algorithmic Patterns with Stack-Augmented Recurrent Nets. In Advances in Neural Information
_Processing Systems 28, pages 190–198, Montreal,_
Canada. Curran Associates, Inc.

Diederik P. Kingma and Jimmy Lei Ba. 2015. Adam:
A Method for Stochastic Optimization. In ICLR
_2015 Conference Track, San Diego, CA. arXiv._

Tal Linzen, Emmanuel Dupoux, and Yoav Goldberg. 2016. Assessing the Ability of LSTMs to
Learn Syntax-Sensitive Dependencies. _Transac-_
_tions of the Association for Computational Linguis-_
_tics, 4(0):521–535._

Mitchell Marcus, Grace Kim, Mary Ann
Marcinkiewicz, Robert MacIntyre, Ann Bies,
Mark Ferguson, Karen Katz, and Britta Schasberger. 1994. [The Penn Treebank:](https://doi.org/10.3115/1075812.1075835) Annotating
[Predicate Argument Structure.](https://doi.org/10.3115/1075812.1075835) In Proceedings of
_the Workshop on Human Language Technology,_
pages 114–119, Plainsboro, NJ. Association for
Computational Linguistics.

Rebecca Marvin and Tal Linzen. 2018. Targeted Syntactic Evaluation of Language Models. In Proceed_ings of the 2018 Conference on Empirical Methods_
_in Natural Language Processing, pages 1192–1202,_
Brussels, Belgium. Association for Computational
Linguistics.

Yikang Shen, Zhouhan Lin, Chin-wei Huang, and
Aaron Courville. 2018a. Neural Language Modeling by Jointly Learning Syntax and Lexicon. In
_ICLR 2018 Conference Track, Vancouver, Canada._
OpenReview.

Yikang Shen, Zhouhan Lin, Athul Paul Jacob, Alessandro Sordoni, Aaron Courville, and Yoshua Bengio.
2018b. Straight to the Tree: Constituency Parsing with Neural Syntactic Distance. In Proceedings
_of the 56th Annual Meeting of the Association for_
_Computational Linguistics, volume 1: Long Papers,_
pages 1171–1180, Melbourne, Australia. Association for Computational Linguistics.

Dani Yogatama, Yishu Miao, Gabor Melis, Wang Ling,
Adhiguna Kuncoro, Chris Dyer, and Phil Blunsom.
2018. Memory Architectures in Recurrent Neural
Network Language Models. In ICLR 2018 Confer_ence Track, Vancouver, Canada. OpenReview._


-----

