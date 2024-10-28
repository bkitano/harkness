## Competency Problems: On Finding and Removing Artifacts in Language Data


### Matt Gardner*[∗][♠] William Merrill*[♠] Jesse Dodge[♠]
 Matthew E. Peters[♠] Alexis Ross[♠] Sameer Singh[♥] Noah A. Smith[♣♠]

_♠Allen Institute for Artificial Intelligence_
_♥University of California, Irvine_
_♣University of Washington_


### Abstract


Much recent work in NLP has documented
dataset artifacts, bias, and spurious correlations between input features and output labels. However, how to tell which features
have “spurious” instead of legitimate correlations is typically left unspecified. In this
work we argue that for complex language understanding tasks, all simple feature correlations are spurious, and we formalize this notion into a class of problems which we call
_competency problems. For example, the word_
“amazing” on its own should not give information about a sentiment label independent of the
context in which it appears, which could include negation, metaphor, sarcasm, etc. We
theoretically analyze the difficulty of creating
data for competency problems when human
bias is taken into account, showing that realistic datasets will increasingly deviate from
competency problems as dataset size increases.
This analysis gives us a simple statistical test
for dataset artifacts, which we use to show
more subtle biases than were described in prior
work, including demonstrating that models are
inappropriately affected by these less extreme
biases. Our theoretical treatment of this problem also allows us to analyze proposed solutions, such as making local edits to dataset instances, and to give recommendations for future data collection and model design efforts
that target competency problems.

### 1 Introduction


Artifact statistics in SNLI

1.0 nobody[c] = 0.01/28k

neutral

vacationcats[c] [n] contradict

entailment

0.8 cat[c] sleeping[c]

first[n] no[c]

outdoors[e]

competition[n]

0.6 animal [e]

outside[e]

person[e]

0.4 people[e]

0.2

0.0

10[2] 10[3] 10[4] 10[5]

n

Figure 1: A statistical test for deviation from a competency problem, where no individual feature (here
words) should give information about the class label, plotting the number of occurrences of each word
against the conditional probability of the label given
the presence of the word. The label associated with
each point is marked by color and superscript. All features above the blue line have detectable correlation
with class labels, using a very conservative Bonferronicorrected statistical test.


Attempts by the natural language processing community to get machines to understand language
or read text are often stymied in part by issues in
our datasets (Chen et al., 2016; Sugawara et al.,
2018). Many recent papers have shown that popular datasets are prone to shortcuts, dataset artifacts, bias, and spurious correlations (Jia and Liang,
2017; Rudinger et al., 2018; Costa-jussà et al.,
2019). While these empirical demonstrations of

_∗Equal contribution_


deficiencies in the data are useful, they often leave
unanswered fundamental questions of what exactly
makes a correlation “spurious”, instead of a feature
that is legitimately predictive of some target label.
In this work we attempt to address this question
theoretically. We begin with the assumption that in
a language understanding problem, no single feature on its own should contain information about
the class label. That is, all simple correlations between input features and output labels are spurious:
_p(y|xi), for any feature xi, should be uniform over_
the class label. We call the class of problems that
meet this assumption competency problems (§2).[1]

This assumption places a very strong restriction


1Our use of the term “competency problems” is inspired
by, but not identical to, the term “competence” in linguistics.
We are referring to the notion that humans can understand
essentially any well-formed utterance in their native language.


-----

on the problems being studied, but we argue that it
is a reasonable description of complex language understanding problems. Consider, for example, the
problem of sentiment analysis on movie reviews.
A single feature might be the presence of the word
_“amazing”, which could be legitimately correlated_
with positive sentiment in some randomly-sampled
collection of actual movie reviews. However, that
correlation tells us more about word frequency in
movie reviews than it tells us about a machine’s
ability to understand the complexities of natural
language. A competent speaker of a natural language would know that “amazing” can appear in
many contexts that do not have positive sentiment
and would not base their prediction on the presence
of this feature alone. That is, the information about
the sentiment of a review, and indeed the mean_ing of natural language, is contained in complex_
feature interactions, not in isolated features. To
evaluate a machine’s understanding of language,
we must remove all simple feature correlations that
would allow the machine to predict the correct label
without considering how those features interact.

Collecting data that accurately reflects the assumptions of a competency problem is very challenging, especially when humans are involved in
creating it. Humans suffer from many different
kinds of bias and priming effects, which we collectively model in this work with rejection sampling
during data collection. We theoretically analyze
data collection under this biased sampling process,
showing that any amount of bias will result in increasing probability of statistically-significant spurious feature correlations as dataset size increases
(§3).

This theoretical treatment of bias in data collection gives us a new, simple measure of data artifacts
(§3.2), which we use to explore artifacts in several
existing datasets (§4). Figure 1 revisits prior analyses on the SNLI dataset (Bowman et al., 2015) with
our statistical test. An analysis based on pointwise
mutual information (e.g., Gururangan et al., 2018)
would correspond to a horizontal line in that figure, missing many features that have less extreme
but still significant correlations with class labels.
These less extreme correlations still lead models
to overweight simple features. The problem of
bias in data collection is pervasive and not easily
addressed with current learning techniques.

Our framework also allows us to examine the theoretical impact of proposed techniques to mitigate


bias, including performing local edits after data collection (§5) and filtering collected data (§6). We derive properties of any local edit procedure that must
hold for the procedure to effectively remove data
artifacts. These proofs give dataset builders tools to
monitor the data collection process to be sure that
resultant datasets are as artifact-free as possible.
Our analysis of local edits additionally suggests a
strong relationship to sensitivity in boolean functions (O’Donnell, 2014), and we identify gaps in
the theory of sensitivity that need to be filled to
properly account for bias in sampled datasets.
We believe our theoretical analysis of these problems provides a good starting point for future analyses of methods to improve NLP data collection, as
well as insights for inductive biases that could be
introduced to better model competency problems.

### 2 Competency Problems

We define a competency problem to be one where
the marginal distribution over labels given any single feature is uniform. For our analysis, we restrict
ourselves to boolean functions: we assume an input
vector x and an output value y, where x 0, 1
_∈{_ _}[d]_

and y 0, 1 .[2] In this setting, competency means
_∈{_ _}_
_p(y|xi) = 0.5 for all i. In other words, the infor-_
mation mapping x to y is found in complex feature
interactions, not in individual features.
Our core claim is that language understanding
requires composing together many pieces of meaning, each of which on its own is largely uninformative about the meaning of the whole. We do not
believe this claim is controversial or new, but its
implications for posing language understanding as
a machine learning problem are underappreciated
and somewhat counterintuitive. If a model picks
up on individual feature correlations in a dataset,
it has learned something extra-linguistic, such as
information about human biases, not about how
words come together to form meaning, which is
the heart of natural language understanding. To

2Boolean functions are quite general, and many machine
learning problems can be framed this way. For NLP, consider
that before the rise of embedding methods, language was often
represented in machine learning models as bags of features
in a very high-dimensional feature space, exactly as we are
modeling the problem here. The first (embedding) layer of
a modern transformer is still very similar to this, with the
addition of a position encoding. The choice of what counts
as a “simple feature” is admittedly somewhat arbitrary; we
believe that considering word types as simple features, as we
do in most of our analysis, is uncontroversial, but there are
other more complex features which one still might want to
control for in competency problems.


-----

push machines towards linguistic competence, we
must control for all sources of extra-linguistic information, ensuring that no simple features contain
information about class labels.
For some language understanding problems,
such as natural language inference, this intuition
is already widely held. We find it surprising and
problematic when the presence of the word “cat”,
_“sleeping” or even “not” in either the premise or_
the hypothesis gives a strong signal about an entailment decision (Gururangan et al., 2018; Poliak
et al., 2018). Competency problems are broader
than this, however. Consider the case of sentiment
analysis. It is true that a movie review containing
the word “amazing” is more likely than not to express positive sentiment about the movie. This is
because of distributional effects in how humans
choose to use phrases in movie reviews. These
distributional effects cause the lexical semantics
of “amazing” to carry over into the whole context,
essentially conflating lexical and contextual cues.
If our goal is to build a system that can accurately
classify the sentiment of movie reviews, exploiting
this conflation is useful. But if our goal is instead to
build a machine that understands how sentiment is
expressed in language, this feature is a red herring
that must be controlled for to truly test linguistic
competence.

### 3 Biased Sampling

To get machines to perform well on competency
problems, we need data that accurately reflects the
competency assumption, both to evaluate systems
and (presumably) to train them. However, humans
suffer from blind spots, social bias, priming, and
other psychological effects that make collecting
data for competency problems challenging. Examples of these effects include instructions in a
crowdsourcing task that prime workers to use particular language,[3] or distributional effects in source
material, such as the “amazing” examples above,
or racial bias in face recognition (Buolamwini
and Gebru, 2018) and abusive language detection
datasets (Davidson et al., 2019; Sap et al., 2019).
In order to formally analyze the impact of human bias on collecting data for competency problems, we need a plausible model of this bias. We
represent bias as rejection sampling from the tar
3This is ubiquitous in crowdsourcing; see, e.g., common
patterns in DROP (Dua et al., 2019) or ROPES (Lin et al.,
2019) that ultimately derive from annotator instructions.


get competency distribution based on single feature values. Specifically, we assume the following dataset collection procedure. First, a person
samples an instance from an unbiased distribution
_pu(x, y) where the competency assumption holds._
The person examines this instance, and if feature
_xi = 1 appears with label y = 0, the person rejects_
the instance and samples a new one, with probability ri. If y = 0 corresponds to negative sentiment
and xi indicates the presence of the word “amaz_ing”, a high value for ri would lead to “amazing”_
appearing more often with positive sentiment, as is
observed in typical sentiment analysis datasets.

We do not that claim rejection sampling is a plausible psychological model of dataset construction.
However, we do think it is a reasonable first-order
approximation of the outcome of human bias on
data creation, for a broad class of biases that have
empirically been found in existing datasets, and it
is relatively easy to analyze.

**3.1** **Emergence of Artifacts Under Rejection**
**Sampling**

Let pu(y|xi) be the conditional probability of y = 1
given xi = 1 under the unbiased distribution,
_pb(y|xi) be the same probability under the biased_
distribution, and ˆp(y|xi) denote the empirical probability within a biased dataset of n samples. Additionally, let fi be the marginal probability pu(xi).
Recall that pu(y|xi) is 0.5 by assumption.

We will say that dimension i has an artifact if the
empirical probability ˆp(y|xi) statistically differs
from 0.5. In this section, we will show that an
artifact emerges if there is a bias at dimension i
in the sampling procedure, which is inevitable for
some features in practice. We will formalize this
bias in terms of a rejection sampling probability ri.

For a single sample x, y, we first derive the joint
and marginal probabilities pb(y, xi) and pb(xi),
from which we can obtain pb(y|xi). These formulas use a recurrence relation obtained from the
rejection sampling procedure.

_pb(y, xi) = [1]_

2 _[f][i][ + 1]2_ _[f][i][r][i][p][b][(][y, x][i][)]_

_fi_
∴ _pb(y, xi) =_

2 − _firi_


-----

_pb(xi) = [1]_

2 _[f][i][ + 1]2_ _[f][i][(1][ −]_ _[r][i][) + 1]2_ _[f][i][r][i][p][b][(][x][i][)]_

∴ _pb(xi) = [2][f][i][ −]_ _[f][i][r][i]_

2 − _firi_

1

∴ _pb(y | xi) =_ _[p][b][(][y, x][i][)]_ =

_pb(xi)_ 2 − _ri_

With no bias (ri = 0), this probability is 0.5, as
expected, and it rises to 1 as ri increases to 1.
We define ˆp(y|xi) as the empirical expectation
of pb(y|xi) over n samples containing xi, with different samples indexed by superscript j. ˆp(y|xi) =
1 �n
_n_ _j=1_ _[y][j][. Note that][ ˆ][p][ is a conditional binomial]_
random variable. By the central limit theorem, ˆp is
approximately ∼N (µpˆ, σp[2]ˆ[)][ for large][ n][, where]


a z-statistic[4] using the standard formula:

_pˆ −_ _p0_
_z[∗]_ = (1)
�

_p0(1 −_ _p0)/n_


1
_µpˆ = pb(y | xi) =_

2 − _ri_

� 1 − _ri_ �2
_σp[2]ˆ_ [=] _·_ [1]

(2 − _ri)[2]_ _n_ _[.]_

This variance is inversely proportional to the
number of samples n. Thus, ˆp(y|xi) can be well
approximated by its expected value for a large
number of samples. As the rejection probability
_ri increases, the center of this distribution tends_
from 0.5 to 1. This formalizes the idea that bias
in the sampling procedure will cause the empirical
probability ˆp(y|xi) to deviate from 0.5, even if the
“true” probability is 0.5 by assumption. Increasing
the sample size n concentrates the distribution inversely proportional to _n, but the expected value_

_[√]_
is unchanged. Thus, artifacts created by rejection
sampling will not be combated by simply sampling
more data from the same biased procedure—the
empirical probability will still be biased by ri even
if n increases arbitrarily. These persistent artifacts
can be exploited at i.i.d. test time to achieve high
performance, but will necessarily fail if the learner
is evaluated under the competency setting.

**3.2** **Hypothesis Test**

Here we set up a hypothesis test to evaluate if there
is enough evidence to reject the hypothesis that ri is
0, i.e., that the data is unbiased. In this case, we can
use a one-sided binomial proportion hypothesis test,
as our rejection sampling can only lead to binomial
proportions for pb(y | xi) that are greater than [1]2 [.]

Our null hypothesis is that the binomial proportion
_pb(y | xi) = 0.5 = p0, or equivalently, that ri = 0._
Our alternative hypothesis is that pb(y | xi) ≥ 0.5.
Let ˆp be the observed probability. We can compute


Thus, if our observed proportion ˆp is far from p0 =
0.5, we will have enough evidence to reject the null
hypothesis that ri = 0. This depends on n as well,
and to explore this interaction, we solve for ˆp for a
given n and confidence level z[∗]: ˆp = _z[∗]_

2[√]n [+][ 1]2 [.]

### 4 Empirical Analysis

With a hypothesis test in hand, we can examine existing datasets for evidence of statisticallysignificant feature bias, and then explore the extent to which this bias impacts models supervised
with this data. Prior work has used pointwise mutual information (PMI) to find features that have
high correlation with labels (e.g., Gururangan et al.,
2018). This measure is useful for understanding
why certain features might get used as deterministic
decision rules by models (Ribeiro et al., 2018; Wallace et al., 2019). However, studies involving PMI
have also intuitively understood that PMI by itself
does not tell the whole story, as a strict ranking by
PMI would return features that only appear once in
the dataset. To account for this problem, they used
arbitrary cutoffs and included information about
feature occurrence in addition to their PMI ranking.
A benefit of our approach to defining and detecting artifacts is that we have a single statistical test
that takes into account both the number of times
a feature appears and how correlated it is with a
single label. We use this test to find features with
the strongest statistical evidence for artifacts (§4.1)
and then show empirically that models use these
features inappropriately when making predictions
(§4.2). This analysis goes beyond deterministic prediction rules, showing that the impact of sampling
bias on model behavior is subtle and pervasive.

**4.1** **Data Analysis**

We analyze two datasets with the hypothesis test
from §3.2: SNLI (Bowman et al., 2015) and
the Universal Dependencies English Web Treebank (Silveira et al., 2014).

**SNLI** Each feature xi represents the presence of
a word in a given example, counting each appearance in an instance as a separate occurrence[5] for

4The use of a z-statistic depends on the normal approximation to a binomial distribution, which holds for large n.
5We remove punctuation and tokenize on whitespace only.


-----

the purposes of computing n and ˆp in Equation 1.
We compute a z-statistic for every token that appears in the SNLI data, where p0 = 3[1] [, as SNLI has]

three labels. We then plot the z-statistic for each
token against the number of times the token appears in the data. We also plot a curve for the value
of the z-statistic at which the null hypothesis (that
_ri = 0) should be rejected, using a significance_
level of α = 0.01 and a conservative Bonferroni
correction (Bonferroni, 1936) for all 28,000 vocabulary items. This analysis is shown in Figure 1.
We label in Figure 1 several words that were also
found to be artifacts by Gururangan et al. (2018)
and Wallace et al. (2019), among others.
We find a very large number of deviations
from the competency assumption, many more than
would be suggested by a PMI-based analysis. PMI
equals log _[p][ˆ][(][y][|][x][i][)]_

_pˆ(y)_ [; because][ ˆ][p][(][y][)][ does not vary]
across features, and the data is balanced over labels,
a PMI analysis ranks features by ˆp(y|xi), looking
only at the y-axis in Figure 1.[6] But the threshold
for which a deviation in ˆp(y|xi) becomes a statistical artifact depends on the number of times the
feature is seen, so our statistical test gives a simpler
and more complete picture of data artifacts. Strong
statistical deviations with less extreme PMI values
still impact model behavior (§4.2 and §6).

**UD English Web Treebank** Next we turn to dependency parsing. In particular, we focus on the
classic problem of prepositional phrase (PP) attachment (Collins and Brooks, 1995), which involves
determining whether a PP attaches to a verb (e.g.,
_We ate spaghetti with forks) or a noun (e.g., We_
_ate spaghetti with meatballs). We heuristically ex-_
tract (verb, noun, prepositional phrase) constructions with ambiguous attachment from the UD English Web Treebank (EWT) training data.[7] We
treat (verb, preposition) tuples as features and attachment types (noun or verb) as labels, and we
compute a z-statistic for each tuple.
Figure 2 shows the z-statistic for each tuple that
appears 10 or more times in the data. We labeled
tuples that also appear in the locally edited samples
from the UD English contrast set created by Gardner et al. (2020). Many of these tuples fall either
above or close to the significance curve, suggesting
that the low contrast consistency reported by Gard
6In practice an arbitrary threshold is chosen on the x-axis
to avoid rare features. Again here a statistical test is a more
principled way to account for rare features.
7See Appendix A for how we extract these constructions.


The previous section reveals a large number of individual word artifacts in the SNLI dataset. Here, we
ask whether typical NLP models learn to bias their
predictions based on these artifacts for both the
SNLI and RTE (Dagan et al., 2005)[9] datasets. That
is, we will show that these single words noticeably
influence a model’s confidence in particular predictions, even when the PMI value is not extreme
enough to create a universal trigger (Wallace et al.,
2019). Importantly, this analysis focuses on words
with high z-statistics, which are often words that
show up very frequently with slight deviations from
_pu(y|xi). This includes words such as “for” and_
_“to” (the two words with highest z-statistic for the_
neutral class), and “there” and “near” (the highest and fifth-highest z-statistic for the entailment
class).
To measure the model bias learned from these
words, we employ RoBERTa-base (Liu et al., 2019)
fine-tuned on RTE, and ALBERT-base (Lan et al.,
2020) fine-tuned on SNLI.[10] Given a single type
such as xi = “nobody” and a target class such as
_y = “contradiction”, we estimate the model ˜p(y|xi)_
as follows. We first create two synthetic input examples, one with the premise containing only the
single token with an empty hypothesis, and one

8Some tuples in the plot with high p(y|xi) are not artifacts,
as the attachment decision is nearly deterministic. For instance, the top right blue dot corresponds to (have, of ); of can
only attach to have in archaic or idiosyncratic constructions.
9We use the RTE data from SuperGlue (Wang et al., 2019).
10Both models are from Morris et al. (2020), and are implemented in the Transformers library (Wolf et al., 2020).


1.0

0.8

0.6

0.4

0.2

0.0


Artifact statistics in UD English PP Attachment

([′]get[′], [′]in[′])[V]

make[′], [′]on[′])[V]

([′]go[′], [′]for[′])[V]

([′]use[′], [′](for[′]make[′])[V] [′], [′]in[′])([V][′]have[′], [′]on[′])[V]

([′]have[′], [′]in[′])[V]


10[1] 10[2]

n

Figure 2: Artifact statistics of (verb, preposition) tuples in PP attachment in the UD English Web Treebank
(EWT). Plotted are tuples that appear in the original
EWT training data, and labeled are tuples that also appear in the UD English locally edited contrast set.


ner et al. (2020) could potentially be explained by
models’ reliance on these artifacts.[8]

**4.2** **Model Analysis**


-----

Dataset Class ∆ˆpy

RTE entailment +2.2 %
SNLI entailment +14.7 %
SNLI neutral +7.9 %
SNLI contradiction +12.5 %

Table 1: Comparison of the average model predicted
class probability for a single token across two cohorts
of large z[∗] tokens and low z[∗] tokens using RoBERTa
(RTE) and ALBERT (SNLI).

with an empty premise and hypothesis containing
the single token. As each input contains only a
single token without additional context, this tests
whether the model will bias its output based on
the token. We run a forward pass with each input
and average the target class probabilities as an estimate of ˜p(y|xi). All of the words in each dataset
appearing at least 20 times are partitioned among
the classes based on their largest class conditional
_z[∗], and for each class we form two cohorts of 50_
words each with the highest and lowest z[∗]. Let
_Xy,z<[∗]_ [denote the set of][ x][i][ with the lowest][ z][∗] [for]
class y and similarly let Xy,z>[∗] [denote the set with]
the largest z[∗]. Finally, we compute the average
∆˜py = [�]xi∈Xy,z>∗ _[p][˜][(][y][|][x][i][)][ −]_ [�]xi∈Xy,z<∗ _[p][˜][(][y][|][x][i][)][.]_

The results are shown in Table 1. As can be seen,
these models exhibit non-trivial bias based on the
single token inputs, with ∆˜py exceeding 10% for
some classes. The bias is much more extreme for
SNLI versus RTE, likely due to the fact that RTE
has two orders of magnitude less data than SNLI.
A caveat about this experiment is in order: due to
the fact that automatically replacing high z[∗] words
with low z[∗] words will likely make most inputs
nonsensical, we chose to use very unnatural singleword inputs to the model instead. We believe this
is a reasonable estimate of the model’s marginal
prior on these tokens, measured in a way that introduces the fewest possible confounding variables
into the experiment, but it’s possible that it does not
completely reflect how a model treats these tokens
in context. Section 6 discusses some additional
empirical evidence for models’ reliance on these
artifacts.

### 5 Mitigating Artifacts with Local Edits

Many works have tried to remove data artifacts by
making minimal changes to existing data (Shekhar
et al., 2017; Sennrich, 2017; Zhao et al., 2018, inter
_alia). In this section we show that this kind of data_
augmentation can be effective with an appropri

ately sensitive edit model, where sensitivity refers
to how often a change to inputs results in the label
changing. However, because humans are involved
in making these changes, achieving appropriate
sensitivity is challenging, and bias in this process
can lead to the introduction of new artifacts. This
suggests that care must be taken when performing
edit-based data augmentation, as large edited training datasets are not likely to be artifact-free (cf.
Tafjord et al., 2019; Huang et al., 2020).
Imagine a new dataset De consisting of samples
**x[′], y[′]** generated by making local edits according to
the following repeated procedure:

1. Randomly sample an instance x from a dataset
_Db of n instances created under pb._

2. Make some changes to x to arrive at x[′].
3. Manually label y[′] and add ⟨x[′], y[′]⟩ to De.
We examine the expected probability pe(y[′]|x[′]i[)][ un-]
der this edit process. Informally, this probability
should depend on how often a change to x affects
_y. Formalizing this, we define the edit sensitivity si_
to be the probability that y changes during editing
given the occurrence of a particular feature in the
edited data, i.e.,

_si = pb(y[′]_ = ¬y | x[′]i[)][.]

The other quantity of interest for an edit model
is ei, the probability that dimension i gets flipped
when going from x to x[′]. In order to make theoretical progress, we also need to make strong independence assumptions on xi, ei and si; we will
examine these assumptions momentarily. We first
show that under these assumptions, si and ei control whether samples generated by local editing
debias first-order artifacts.

**Proposition 1 (Proof in §B). Assume xi, xj, ei,**
_ej, si, and sj are independent for all i, j. Then_
_pe(y[′]_ _| x[′]i[) =][ 1]2_ _[if and only if][ r][i][ = 0][ or][ s][i][ =][ 1]2_ _[or]_

_ei = 1._

This proposition shows that there are three ways
to achieve unbiased data from a local edit procedure
that edits dimensions independently: (1) start with
unbiased data, (2) always flip every feature, and (3)
flip the label half the time for each feature.
The first of these conditions is not under the
control of the edit procedure, and if we start with
unbiased data there is no need for debiasing.
The second condition is roughly analogous to the
approach taken by most prior work that performs
local edits: they aim to always change whichever
features led to a prediction, with si = 1 and ei = 1.


-----

However, ei just indicates whether a feature was
_flipped, which means that to achieve ei = 1, every_
non-zero feature must be added to every instance
where it is missing, which is somewhat nonsensical for language data, and thus this solution isn’t
practical.
This leaves the third condition as the only practical solution for local edit procedures to have a
hope of debiasing datasets. That is, these procedures should aim to flip the label on average half
of the time, for each feature that is changed.
We emphasize here that the assumptions we
made in deriving this result are very strong. If these
assumptions are violated, it is easy to construct an
adversarial edit procedure that will introduce bias
into an unbiased dataset (the ri = 0 solution). Similarly, if ei and si are correlated, one can construct
cases that break the si = 2[1] [solution as well.]

Furthermore, these independence assumptions
are not realistically achievable for any humanproduced local edits on language data. We thus
take the guarantees we derived with some skepticism, and view this result more as guidelines for
how to set up and monitor local edit procedures:
aim to flip the label roughly half the time, in a way
that is uncorrelated with which features are getting
edited, and monitor the resulting data for edit sensitivity and artifact statistics. In the next section we
show how to use the theoretical lens we have developed to analyze local edits that have been made in
prior work.

**5.1** **Local Edits in Practice**

We empirically investigate the effectiveness of local edits for reducing single feature artifacts using
locally edited samples generated from two datasets:
(1) the Boolean Questions dataset (BoolQ; Clark
et al., 2019a), which consists of pairs of paragraphs
(p) and questions (q), where each q has a binary
answer that can be found by reasoning over p; and
(2) IMDb (Maas et al., 2011), a sentiment classification dataset in the domain of movie reviews.
We define each feature xi as the occurrence of a
particular word within q for BoolQ, and within the
text of the review for IMDb. Gardner et al. (2020)
generated additional data for BoolQ and IMDb by
making local edits to the question or review text
and recording the updated binary label.

Figure 3 visualizes the effect of these changes
on single-feature artifacts by comparing the artifact
statistics for the original texts to the statistics for the


Artifact statistics for BoolQ before and after local edits

z = ± 2

0.8 i.i.d.

edits

0.7

0.6

0.5

0.4

0.3

10[2]

n

Artifact statistics for IMDb before and after local edits

1.0 z = ± 2

i.i.d.
edits

0.8

0.6

0.4

0.2

0.0

10[2]

n

Figure 3: The artifact statistics of the original BoolQ
(above) and IMDb (below) samples are plotted in red,
compared to the artifact statistics over the edited instances, plotted in green.

edited texts generated by Gardner et al. (2020). For
BoolQ, many tokens in the original data exhibit artifacts in the positive (> 0.5) direction, while, within
the edited data, almost all tokens fall within the
confidence region. In contrast, there is no apparent
distributional difference between artifact statistics
for the original vs. edited texts on IMDb. We find
that for BoolQ, the per-token edit sensitivity distribution has a median of ˜s = 0.429 (mean 0.348,
std 0.274), which, by Proposition 1, explains why
most of the ˆp(y|xi) values for the edited samples
are not significantly different from 0.5. For IMDb,
_s˜ = 1.00 (mean 1, std 0). This case study illus-_
trates the importance of leveraging our theory to
engineer better edit models.

**5.2** **Local Edits and Boolean Sensitivity**

In the above discussion we used the term sensitivity
in an informal way to describe the probability that
a local edit changes the label. This term also has
a related formal definition in the study of boolean
functions, where it is an implicit complexity measure (Wegener, 1987). Sensitivity in this sense has
been shown to correlate with generalization in neural networks (Franco, 2001), and has been extended
for use with practical NLP datasets (Hahn et al.,


-----

2021). In this section we discuss the intersection
of our theory with sensitivity analysis, highlighting limitations in sensitivity analysis for sampled
datasets that could be addressed in future work.
For a boolean vector x, let x[i] be the Hamming
_neighbor of x at dimension i: i.e., the vector where_
_xi has been flipped and all other bits remain the_
same. Consider f : 0, 1 0, 1 . The sensi_{_ _}[d]_ _→{_ _}_
_tivity set S(f, x) is the set of Hamming neighbors_
of x with different labels:

_S(f, x) =_ �i [0, . . ., d] _f_ (x) = f (x[i])� _._
_∈_ _|_ _̸_

The local sensitivity s(f, x) is the size of this set:
_s(f, x) =_ _S(f, x)_ . Finally, the global sensitivity
_|_ _|_
is defined as s(f ) = maxx s(f, x).

**Importance of sensitivity** In our case, the effect
of local editing on a dataset can be understood in
terms of sensitivity. Imagine a boolean function
_f :_ 0, 1 0, 1 from which we draw n sam_{_ _}[d]_ _→{_ _}_
ples **x, y** . If these samples are drawn uniformly
_⟨_ _⟩_
over 0, 1, then the probability of observing any
_{_ _}[d]_
Hamming neighbors goes to 0 rapidly with d.[11]

Thus, it is possible to pick a low sensitivity function that can perfectly fit the data. In this sense, the
true sensitivity of f is likely underspecified by the
dataset.
Imagine we give this data to a learner with inductive bias resembling some variant of Occam’s razor.
If the learner’s notion of complexity is correlated
with sensitivity (which many complexity measures
are), then the learner will favor low sensitivity decision boundaries. Thus, the fact that sensitivity
is underspecified in the training data is a problem
if the gold-standard function has high sensitivity,
as the inductive bias of the learning algorithm may
favor low-sensitivity alternatives.
Contrast this with a dataset where some local
neighborhoods in the input space have been filled
in with local edits. The set of observed neighbors
around a point x provide a lower bound on s(f, x),
which is a lower bound on s(f ). In this sense, s(f )
is no longer underspecified by the dataset.
In this discussion we have used underspecified
in an informal way; there is no precise measure of
the sensitivity of a sampled dataset (as opposed to
a fully-specified function), particularly when generalizing from finite boolean functions to natural language inputs. Attempts to generalize sensitivity to
natural language have done so by leveraging large

11This is essentially the curse of dimensionality.


language models to generate neighbors from which
sensitivity can be estimated (Hahn et al., 2021).
Resampling data in this way can give reasonable
estimates of the sensitivity of the underlying task,
but it is fundamentally incompatible with measuring dataset artifacts of the kind we discuss in this
paper, as the generative model can fill in parts of
the data distribution that are missing due to sampling bias, giving a higher estimate of sensitivity
than is warranted by the sampled dataset.

### 6 Other Mitigation Techniques

In this section we briefly discuss the implications of
our theoretical analysis for other artifact mitigation
techniques that have been proposed in the literature.
Our analysis in this section is not rigorous and is
meant only to give high-level intuition or potential
starting points for future work.

**More annotators** One suggested mitigation technique for dataset artifacts is to increase the number
of annotators (Geva et al., 2019). Especially when
people generate the text that is used in a dataset,
there can be substantial person-specific correlations
between features and labels. Having more annotators washes out those correlations in aggregate,
making the data less biased overall.
We briefly analyze this procedure using our rejection sampling framework. For simplicity, we
have so far only considered a single possible rejection probability, where an instance is rejected with
probability ri if xi = 1 and y = 0. If we introduce
additional rejection probabilities for the other three
possible combinations of values for xi and y, there
will be the possibility that some rejections balance
out other rejections. We can model multiple annotators by splitting a dataset into k different slices
that have their own bias vectors r. If the r vectors
are uncorrelated, it seems likely that as k increases,
the probability that ˆp(y|xi) deviates from pu(y|xi)
tends towards zero. Even in our simplistic model,
if we assume a sparse r, averaging more and more
of them will make the deviation tend toward zero,
if the non-zero dimensions are uncorrelated.
However, if the r vectors are correlated, increasing the number of annotators will not produce data
reflecting the competency assumption. When might
the r vectors be correlated? This could happen due
to societal biases, word usage frequencies, or priming effects from data collection instructions given
to all annotators. Surely across any pool of annotators there will be some dimensions along which r


-----

values are correlated, and other dimensions along
with they are not. Increasing the number of annotators thus helps mitigate the problem, but does not
solve it completely.

**Data filtering** A recent trend is to remove data
from a training set that is biased in some way in
order to get a model that generalizes better (Le
Bras et al., 2020; Swayamdipta et al., 2020; Oren
et al., 2020). While this method can be effective for
very biased datasets, it is somewhat unsatisfying to
remove entire instances because of bias in a single
feature. In the extreme case where ri ≈ 1, such as
with “nobody” in SNLI (Fig. 1), this process could
effectively remove xi from the observed feature
space.
To understand the effect of these automated
methods on dataset artifacts, we repeat the
analysis from §4.1 on data that was classified
as “ambiguous” according to Dataset Cartography (Swayamdipta et al., 2020). This data was
shown to provide better generalization when used
as training data compared to the original training
set. The ambiguous instances did not have a balanced label distribution, so we downsampled the
data to balance it, then downsampled the whole
training data to get the same number of instances
as the balanced ambiguous set.
The resulting artifact plots are shown in Figure 4.
As can be seen, the “ambiguous” instances have
many fewer deviations from the competency assumption, across the entire range of our hypothesis
test. It is not just high PMI values that are getting
corrected by finding ambiguous instances; all statistical deviations are impacted. This effect is striking,
and it further corroborates our arguments about the
importance of the competency assumption.[12]

### 7 Other Related Work

**Theoretical analysis of bias** Several recent
works explore sources and theoretical treatments
of bias or spurious correlations in NLP (Shah
et al., 2020a; Kaushik et al., 2020) or ML more
broadly (Shah et al., 2020b). Our work differs by
introducing a competency assumption and exploring its implications. The difference between our
biased and unbiased distributions is an instance of
covariate shift (Quionero-Candela et al., 2009).

12Comparing the lower part of Figure 4 to Figure 1 also
corroborates our derived result (§3.1) that larger datasets are
more likely to have artifacts. With 24% of the data there are
many fewer artifacts.


Figure 4: Statistical artifacts in ambiguous instances (Swayamdipta et al., 2020; above) versus a random (same-size) sample from the SNLI training set (below). The filtering done by ambiguous instance detection targets statistical artifacts across the whole range
of the statistical test, not just high PMI values.

**Competent models** An interesting question is
whether we can inject a “competency inductive
bias” into models, i.e., discourage relying on individual features. The closest works we are aware of
are methods that ensemble weak models together
with strong models during training (Clark et al.,
2020; Dagaev et al., 2021), or ensembles of models with unaligned gradients (Teney et al., 2021).
Other works use ensembles with models targeted
at known sources of data artifacts, but these are
less close to a competency assumption (Clark et al.,
2019b; Karimi Mahabadi et al., 2020).

### 8 Conclusion

The more NLP models advance, the better they are
at learning statistical patterns in datasets. This is
problematic for language understanding research
if some statistical patterns allow a model to bypass linguistic competence. We have formalized
this intuition with a class of problems called com_petency problems, arguing that, for any language_
understanding task, all correlations between simple
features and labels are spurious. Collecting data
meeting this assumption is challenging, but we


-----

have provided theoretical analysis that can inform
future data collection efforts for such tasks.
We conclude with some final thoughts on general
best practices for data collection, informed by the
analysis in this paper. If annotators are generating
text for some data collection task, find ways to
decrease priming effects. This could involve using
images as prompts instead of text (Novikova et al.,
2017; Weller et al., 2020), or randomly sampling
words to include in the generated text. If existing
text is being collected and annotated, make local
edits to the text while monitoring the sensitivity
of those edits according to the guidelines in §5,
perhaps using different processes between train
and test, to minimize correlations between train
features and test labels.

### Acknowledgements

We thank Sarthak Jain for pointing out an error in
our original proof of what was Proposition 1, which
led to the updated proposition and discussion in this
version of the paper.

### References

[C. E. Bonferroni. 1936. Teoria statistica delle classi](https://ci.nii.ac.jp/naid/20001561442/en/)
[e calcolo delle probabilita. Pubblicazioni del R Is-](https://ci.nii.ac.jp/naid/20001561442/en/)
_tituto Superiore di Scienze Economiche e Commeri-_
_ciali di Firenze, 8:3–62._

Samuel R. Bowman, Gabor Angeli, Christopher Potts,
[and Christopher D. Manning. 2015. A large anno-](https://doi.org/10.18653/v1/D15-1075)
[tated corpus for learning natural language inference.](https://doi.org/10.18653/v1/D15-1075)
In Proceedings of the 2015 Conference on Empiri_cal Methods in Natural Language Processing, pages_
632–642, Lisbon, Portugal. Association for Computational Linguistics.

Joy Buolamwini and Timnit Gebru. 2018. Gender
shades: Intersectional accuracy disparities in commercial gender classification. In FAT.

Danqi Chen, Jason Bolton, and Christopher D. Manning. 2016. [A thorough examination of the](https://doi.org/10.18653/v1/P16-1223)
[CNN/Daily Mail reading comprehension task.](https://doi.org/10.18653/v1/P16-1223) In
_Proceedings of the 54th Annual Meeting of the As-_
_sociation for Computational Linguistics (Volume 1:_
_Long Papers), pages 2358–2367, Berlin, Germany._
Association for Computational Linguistics.

Christopher Clark, Kenton Lee, Ming-Wei Chang,
Tom Kwiatkowski, Michael Collins, and Kristina
Toutanova. 2019a. Boolq: Exploring the surprising
difficulty of natural yes/no questions. In NAACL.

Christopher Clark, Mark Yatskar, and Luke Zettlemoyer. 2019b. [Don’t take the easy way out: En-](https://doi.org/10.18653/v1/D19-1418)
[semble based methods for avoiding known dataset](https://doi.org/10.18653/v1/D19-1418)


[biases. In Proceedings of the 2019 Conference on](https://doi.org/10.18653/v1/D19-1418)
_Empirical Methods in Natural Language Processing_
_and the 9th International Joint Conference on Natu-_
_ral Language Processing (EMNLP-IJCNLP), pages_
4069–4082, Hong Kong, China. Association for
Computational Linguistics.

Christopher Clark, Mark Yatskar, and Luke Zettle[moyer. 2020. Learning to model and ignore dataset](https://doi.org/10.18653/v1/2020.findings-emnlp.272)
[bias with mixed capacity ensembles.](https://doi.org/10.18653/v1/2020.findings-emnlp.272) In Findings
_of the Association for Computational Linguistics:_
_EMNLP 2020, pages 3031–3045, Online. Associa-_
tion for Computational Linguistics.

Michael Collins and James Brooks. 1995. [Prepo-](https://www.aclweb.org/anthology/W95-0103)
[sitional phrase attachment through a backed-off](https://www.aclweb.org/anthology/W95-0103)
[model. In Third Workshop on Very Large Corpora.](https://www.aclweb.org/anthology/W95-0103)

Marta R. Costa-jussà, Christian Hardmeier, Will Radford, and Kellie Webster, editors. 2019. _[Proceed-](https://www.aclweb.org/anthology/W19-3800)_
_[ings of the First Workshop on Gender Bias in Natu-](https://www.aclweb.org/anthology/W19-3800)_
_[ral Language Processing. Association for Computa-](https://www.aclweb.org/anthology/W19-3800)_
tional Linguistics, Florence, Italy.

Nikolay Dagaev, Brett D. Roads, Xiaoliang Luo,
Daniel N. Barry, Kaustubh R. Patil, and Bradley C.
[Love. 2021. A too-good-to-be-true prior to reduce](http://arxiv.org/abs/2102.06406)
[shortcut reliance.](http://arxiv.org/abs/2102.06406)

Ido Dagan, Oren Glickman, and B. Magnini. 2005.
The pascal recognising textual entailment challenge.
In MLCW.

Thomas Davidson, Debasmita Bhattacharya, and Ing[mar Weber. 2019. Racial bias in hate speech and](https://doi.org/10.18653/v1/W19-3504)
[abusive language detection datasets. In Proceedings](https://doi.org/10.18653/v1/W19-3504)
_of the Third Workshop on Abusive Language Online,_
pages 25–35, Florence, Italy. Association for Computational Linguistics.

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel
Stanovsky, Sameer Singh, and Matt Gardner. 2019.
[DROP: A reading comprehension benchmark requir-](https://doi.org/10.18653/v1/N19-1246)
[ing discrete reasoning over paragraphs. In Proceed-](https://doi.org/10.18653/v1/N19-1246)
_ings of the 2019 Conference of the North American_
_Chapter of the Association for Computational Lin-_
_guistics: Human Language Technologies, Volume 1_
_(Long and Short Papers), pages 2368–2378, Min-_
neapolis, Minnesota. Association for Computational
Linguistics.

Leonardo Franco. 2001. A measure for the complexity of boolean functions related to their implementation in neural networks. _arXiv preprint cond-_
_mat/0111169._

Matt Gardner, Yoav Artzi, Victoria Basmov, Jonathan
Berant, Ben Bogin, Sihao Chen, Pradeep Dasigi,
Dheeru Dua, Yanai Elazar, Ananth Gottumukkala,
Nitish Gupta, Hannaneh Hajishirzi, Gabriel Ilharco,
Daniel Khashabi, Kevin Lin, Jiangming Liu, Nelson F. Liu, Phoebe Mulcaire, Qiang Ning, Sameer
Singh, Noah A. Smith, Sanjay Subramanian, Reut
Tsarfaty, Eric Wallace, Ally Zhang, and Ben Zhou.
[2020. Evaluating models’ local decision boundaries](https://doi.org/10.18653/v1/2020.findings-emnlp.117)


-----

[via contrast sets.](https://doi.org/10.18653/v1/2020.findings-emnlp.117) In Findings of the Association
_for Computational Linguistics: EMNLP 2020, pages_
1307–1323, Online. Association for Computational
Linguistics.

Mor Geva, Yoav Goldberg, and Jonathan Berant. 2019.

[Are we modeling the task or the annotator? an inves-](https://doi.org/10.18653/v1/D19-1107)
[tigation of annotator bias in natural language under-](https://doi.org/10.18653/v1/D19-1107)
[standing datasets. In Proceedings of the 2019 Con-](https://doi.org/10.18653/v1/D19-1107)
_ference on Empirical Methods in Natural Language_
_Processing and the 9th International Joint Confer-_
_ence on Natural Language Processing (EMNLP-_
_IJCNLP), pages 1161–1166, Hong Kong, China. As-_
sociation for Computational Linguistics.

Suchin Gururangan, Swabha Swayamdipta, Omer
Levy, Roy Schwartz, Samuel Bowman, and Noah A.
Smith. 2018. [Annotation artifacts in natural lan-](https://doi.org/10.18653/v1/N18-2017)
[guage inference data. In Proceedings of the 2018](https://doi.org/10.18653/v1/N18-2017)
_Conference of the North American Chapter of the_
_Association for Computational Linguistics: Human_
_Language Technologies, Volume 2 (Short Papers),_
pages 107–112, New Orleans, Louisiana. Association for Computational Linguistics.

Michael Hahn, Dan Jurafsky, and Richard Futrell. 2021.

[Sensitivity as a complexity measure for sequence](https://arxiv.org/abs/2104.10343)
[classification tasks. Transactions of the Association](https://arxiv.org/abs/2104.10343)
_for Computational Linguistics._

William Huang, Haokun Liu, and Samuel R. Bowman.
2020. [Counterfactually-augmented SNLI training](https://doi.org/10.18653/v1/2020.insights-1.13)
[data does not yield better generalization than unaug-](https://doi.org/10.18653/v1/2020.insights-1.13)
[mented data. In Proceedings of the First Workshop](https://doi.org/10.18653/v1/2020.insights-1.13)
_on Insights from Negative Results in NLP, pages 82–_
87, Online. Association for Computational Linguistics.

[Robin Jia and Percy Liang. 2017. Adversarial exam-](https://doi.org/10.18653/v1/D17-1215)
[ples for evaluating reading comprehension systems.](https://doi.org/10.18653/v1/D17-1215)
In Proceedings of the 2017 Conference on Empiri_cal Methods in Natural Language Processing, pages_
2021–2031, Copenhagen, Denmark. Association for
Computational Linguistics.

Rabeeh Karimi Mahabadi, Yonatan Belinkov, and
James Henderson. 2020. [End-to-end bias mitiga-](https://doi.org/10.18653/v1/2020.acl-main.769)
[tion by modelling biases in corpora. In Proceedings](https://doi.org/10.18653/v1/2020.acl-main.769)
_of the 58th Annual Meeting of the Association for_
_Computational Linguistics, pages 8706–8716, On-_
line. Association for Computational Linguistics.

Divyansh Kaushik, Amrith Rajagopal Setlur, E. Hovy,
and Zachary Chase Lipton. 2020. Explaining the efficacy of counterfactually-augmented data. _ArXiv,_
abs/2010.02114.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman,
Kevin Gimpel, Piyush Sharma, and Radu Soricut. 2020. Albert: A lite bert for self-supervised
learning of language representations. _ArXiv,_
abs/1909.11942.

Ronan Le Bras, Swabha Swayamdipta, Chandra Bhagavatula, Rowan Zellers, Matthew E. Peters, Ashish


Sabharwal, and Yejin Choi. 2020. Adversarial filters
of dataset biases. In ICML.

Kevin Lin, Oyvind Tafjord, Peter Clark, and Matt Gard[ner. 2019. Reasoning over paragraph effects in situ-](https://doi.org/10.18653/v1/D19-5808)
[ations. In Proceedings of the 2nd Workshop on Ma-](https://doi.org/10.18653/v1/D19-5808)
_chine Reading for Question Answering, pages 58–_
62, Hong Kong, China. Association for Computational Linguistics.

Y. Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar
Joshi, Danqi Chen, Omer Levy, M. Lewis, Luke
Zettlemoyer, and Veselin Stoyanov. 2019. Roberta:
A robustly optimized bert pretraining approach.
_ArXiv, abs/1907.11692._

Andrew L. Maas, Raymond E. Daly, Peter T. Pham,
Dan Huang, Andrew Y. Ng, and Christopher Potts.
[2011. Learning word vectors for sentiment analy-](https://www.aclweb.org/anthology/P11-1015)
[sis. In Proceedings of the 49th Annual Meeting of](https://www.aclweb.org/anthology/P11-1015)
_the Association for Computational Linguistics: Hu-_
_man Language Technologies, pages 142–150, Port-_
land, Oregon, USA. Association for Computational
Linguistics.

John Morris, Eli Lifland, Jin Yong Yoo, Jake Grigsby,
Di Jin, and Yanjun Qi. 2020. Textattack: A framework for adversarial attacks, data augmentation, and
adversarial training in nlp. In Proceedings of the
_2020 Conference on Empirical Methods in Natu-_
_ral Language Processing: System Demonstrations,_
pages 119–126.

Jekaterina Novikova, Ondˇrej Dušek, and Verena Rieser.
2017. [The E2E dataset: New challenges for end-](https://doi.org/10.18653/v1/W17-5525)
[to-end generation. In Proceedings of the 18th An-](https://doi.org/10.18653/v1/W17-5525)
_nual SIGdial Meeting on Discourse and Dialogue,_
pages 201–206, Saarbrücken, Germany. Association
for Computational Linguistics.

Ryan O’Donnell. 2014. Analysis of boolean functions.
Cambridge University Press.

Inbar Oren, Jonathan Herzig, Nitish Gupta, Matt Gard[ner, and Jonathan Berant. 2020. Improving compo-](https://doi.org/10.18653/v1/2020.findings-emnlp.225)
[sitional generalization in semantic parsing. In Find-](https://doi.org/10.18653/v1/2020.findings-emnlp.225)
_ings of the Association for Computational Linguis-_
_tics: EMNLP 2020, pages 2482–2495, Online. As-_
sociation for Computational Linguistics.

Adam Poliak, Jason Naradowsky, Aparajita Haldar,
Rachel Rudinger, and Benjamin Van Durme. 2018.
[Hypothesis only baselines in natural language in-](https://doi.org/10.18653/v1/S18-2023)
[ference. In Proceedings of the Seventh Joint Con-](https://doi.org/10.18653/v1/S18-2023)
_ference on Lexical and Computational Semantics,_
pages 180–191, New Orleans, Louisiana. Association for Computational Linguistics.

Joaquin Quionero-Candela, Masashi Sugiyama, Anton
Schwaighofer, and N. Lawrence. 2009. _Dataset_
_Shift in Machine Learning. MIT Press._

Marco Tulio Ribeiro, Sameer Singh, and Carlos
Guestrin. 2018. Anchors: High-precision modelagnostic explanations. In AAAI.


-----

Rachel Rudinger, Jason Naradowsky, Brian Leonard,
and Benjamin Van Durme. 2018. [Gender bias in](https://doi.org/10.18653/v1/N18-2002)
[coreference resolution. In Proceedings of the 2018](https://doi.org/10.18653/v1/N18-2002)
_Conference of the North American Chapter of the_
_Association for Computational Linguistics: Human_
_Language Technologies, Volume 2 (Short Papers),_
pages 8–14, New Orleans, Louisiana. Association
for Computational Linguistics.

Maarten Sap, Dallas Card, Saadia Gabriel, Yejin Choi,
[and Noah A Smith. 2019. The risk of racial bias in](https://www.aclweb.org/anthology/P19-1163.pdf)
[hate speech detection. In ACL.](https://www.aclweb.org/anthology/P19-1163.pdf)

Rico Sennrich. 2017. How grammatical is characterlevel neural machine translation? Assessing MT
quality with contrastive translation pairs.

Deven Santosh Shah, H. Andrew Schwartz, and Dirk
[Hovy. 2020a. Predictive biases in natural language](https://doi.org/10.18653/v1/2020.acl-main.468)
[processing models: A conceptual framework and](https://doi.org/10.18653/v1/2020.acl-main.468)
[overview. In Proceedings of the 58th Annual Meet-](https://doi.org/10.18653/v1/2020.acl-main.468)
_ing of the Association for Computational Linguistics,_
pages 5248–5264, Online. Association for Computational Linguistics.

Harshay Shah, Kaustav Tamuly, Aditi Raghunathan,
[Prateek Jain, and Praneeth Netrapalli. 2020b. The](https://proceedings.neurips.cc/paper/2020/file/6cfe0e6127fa25df2a0ef2ae1067d915-Paper.pdf)
[pitfalls of simplicity bias in neural networks. In Ad-](https://proceedings.neurips.cc/paper/2020/file/6cfe0e6127fa25df2a0ef2ae1067d915-Paper.pdf)
_vances in Neural Information Processing Systems,_
volume 33, pages 9573–9585. Curran Associates,
Inc.

Ravi Shekhar, Sandro Pezzelle, Yauhen Klimovich, Aurélie Herbelot, Moin Nabi, Enver Sangineto, and
Raffaella Bernardi. 2017. Foil it! Find One mismatch between image and language caption.

Natalia Silveira, Timothy Dozat, Marie-Catherine
de Marneffe, Samuel Bowman, Miriam Connor,
John Bauer, and Christopher D. Manning. 2014. A
gold standard dependency corpus for English. In
_Proceedings of the Ninth International Conference_
_on Language Resources and Evaluation (LREC-_
_2014)._

Saku Sugawara, Kentaro Inui, Satoshi Sekine, and
Akiko Aizawa. 2018. [What makes reading com-](https://doi.org/10.18653/v1/D18-1453)
[prehension questions easier?](https://doi.org/10.18653/v1/D18-1453) In Proceedings of
_the 2018 Conference on Empirical Methods in Nat-_
_ural Language Processing, pages 4208–4219, Brus-_
sels, Belgium. Association for Computational Linguistics.

Swabha Swayamdipta, Roy Schwartz, Nicholas Lourie,
Yizhong Wang, Hannaneh Hajishirzi, Noah A.
[Smith, and Yejin Choi. 2020. Dataset cartography:](https://doi.org/10.18653/v1/2020.emnlp-main.746)
[Mapping and diagnosing datasets with training dy-](https://doi.org/10.18653/v1/2020.emnlp-main.746)
[namics. In Proceedings of the 2020 Conference on](https://doi.org/10.18653/v1/2020.emnlp-main.746)
_Empirical Methods in Natural Language Process-_
_ing (EMNLP), pages 9275–9293, Online. Associa-_
tion for Computational Linguistics.

Oyvind Tafjord, Matt Gardner, Kevin Lin, and Peter
[Clark. 2019. QuaRTz: An open-domain dataset of](https://doi.org/10.18653/v1/D19-1608)
[qualitative relationship questions. In Proceedings of](https://doi.org/10.18653/v1/D19-1608)


_the 2019 Conference on Empirical Methods in Nat-_
_ural Language Processing and the 9th International_
_Joint Conference on Natural Language Processing_
_(EMNLP-IJCNLP), pages 5941–5946, Hong Kong,_
China. Association for Computational Linguistics.

Damien Teney, Ehsan Abbasnejad, Simon Lucey, and
[Anton van den Hengel. 2021. Evading the simplic-](http://arxiv.org/abs/2105.05612)
[ity bias: Training a diverse set of models discovers](http://arxiv.org/abs/2105.05612)
[solutions with superior ood generalization.](http://arxiv.org/abs/2105.05612)

Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner,
[and Sameer Singh. 2019. Universal adversarial trig-](https://doi.org/10.18653/v1/D19-1221)
[gers for attacking and analyzing NLP. In Proceed-](https://doi.org/10.18653/v1/D19-1221)
_ings of the 2019 Conference on Empirical Methods_
_in Natural Language Processing and the 9th Inter-_
_national Joint Conference on Natural Language Pro-_
_cessing (EMNLP-IJCNLP), pages 2153–2162, Hong_
Kong, China. Association for Computational Linguistics.

Alex Wang, Yada Pruksachatkun, Nikita Nangia,
Amanpreet Singh, Julian Michael, Felix Hill, Omer
Levy, and Samuel R. Bowman. 2019. Superglue: A
stickier benchmark for general-purpose language understanding systems. In NeurIPS.

Ingo Wegener. 1987. The complexity of Boolean func_tions. Wiley._

Orion Weller, Nicholas Lourie, Matt Gardner, and
[Matthew Peters. 2020. Learning from task descrip-](https://doi.org/10.18653/v1/2020.emnlp-main.105)
[tions.](https://doi.org/10.18653/v1/2020.emnlp-main.105) In Proceedings of the 2020 Conference on
_Empirical Methods in Natural Language Process-_
_ing (EMNLP), pages 1361–1375, Online. Associa-_
tion for Computational Linguistics.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen,
Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu,
Teven Le Scao, Sylvain Gugger, Mariama Drame,
Quentin Lhoest, and Alexander M. Rush. 2020.
[Transformers: State-of-the-art natural language pro-](https://www.aclweb.org/anthology/2020.emnlp-demos.6)
[cessing. In Proceedings of the 2020 Conference on](https://www.aclweb.org/anthology/2020.emnlp-demos.6)
_Empirical Methods in Natural Language Processing:_
_System Demonstrations, pages 38–45, Online. Asso-_
ciation for Computational Linguistics.

Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Or[donez, and Kai-Wei Chang. 2018. Gender bias in](https://doi.org/10.18653/v1/N18-2003)
coreference resolution: [Evaluation and debiasing](https://doi.org/10.18653/v1/N18-2003)
[methods.](https://doi.org/10.18653/v1/N18-2003) In Proceedings of the 2018 Conference
_of the North American Chapter of the Association_
_for Computational Linguistics: Human Language_
_Technologies, Volume 2 (Short Papers), pages 15–20,_
New Orleans, Louisiana. Association for Computational Linguistics.


-----

Now, we compute pb(y | ¬xi):

_pb(y | ¬xi) =_ _[p][b][(][y,][ ¬][x][i][)]_ = [1]

_pb(¬xi)_ 2 _[.]_

|Head: Noun|Head: Verb|
|---|---|

|I think 2012 is going to be a great year for Fujairah as we have A LOT of projects to be done by 2012. V: going NP: year PP: for Fujairah|Went to the Willow Lounge this past weekend for dinner and drinks ... place is awe- some. V: Went NP: weekend PP: for dinner and drinks|
|---|---|


Table 2: Examples of constructions extracted from the
UD EWT training data using the heuristics described
in Appendix A. Shown are sentences with differing PP
attachment types for tuple (go, for).

### A Ambiguous PP Attachment Extraction

Here, we describe how we heuristically extract
(verb, noun, prepositional phrase) constructions
with ambiguous attachment from the UD English
Web Treebank training data (Section 4.1). Examples of such constructions are shown in Table 2. We
extract (V, N, PP) constructions from UD EWT inputs that meet the following criteria, which operate
over the dependency relation annotations:

1. V, NP, and PP are contained in same sentence

2. Either PP depends on NP or PP depends on V

3. NP depends on V and is not subject of V

4. PP follows both V and NP in the sentence

### B Proof of Proposition 1

This section will rely on the assumption of pairwise
independence between input features, i.e., xi, xj
are independent for all i.j.

**Lemma 1. Assume input features are pairwise in-**
_dependent. Then, pb(y | ¬xi) =_ [1]2 _[.]_

_Proof. Let fi = pu(xi). We first derive the joint_
distribution pb(y, ¬xi):

_pb(y, ¬xi) = [1]_

2 [(1][ −] _[f][i][) + 1]2_ _[f][i][r][i][p][b][(][y,][ ¬][x][i][)]_


We now turn to the main proof of Proposition 1.
Recall that we define the edit sensitivity si of feature i as p(y[′] = ¬y | x[′]i[)][. We also similarly define]
_ei = p(x[′]i_ [=][ ¬][x][i][ |][ x]i[′][)][.]

**Proposition 1.** _Assume xi, xj, ei, ej, si, and sj_
_are independent for all i, j. Then pe(y[′]_ _| x[′]i[) =][ 1]2_

_if and only if ri = 0 or si =_ [1]2 _[or][ e][i][ = 1][.]_

_Proof. We first consider the case where y[′]_ = y,
and derive p(y | x[′]i[)][. Let][ e][i][ =][ p][(][¬][x][i][ |][ x][′]i[)][.]

_p(y | x[′]i[) =][ p][(][y][ |][ x][i][)][p][(][x][i]_ _[|][ x]i[′][)]_

+ p(y | ¬xi)p(¬xi | x[′]i[)]

1
= (1 − _ei) + [1]_

2 − _ri_ 2 _[e][i]_

= [2(1][ −] _[e][i][) +][ e][i][(2][ −]_ _[r][i][)]_

2(2 − _ri)_

= [2][ −] _[e][i][r][i]_

2(2 − _ri)_ _[.]_

Now we let y[′] = ¬y and derive pb(¬y | x[′]i[)][:]

_p(¬y | x[′]i[) =][ p][(][¬][y][ |][ x][i][)][p][b][(][x][i]_ _[|][ x]i[′][)]_

+ p(¬y | ¬xi)p(¬xi | x[′]i[)]

= [1][ −] _[r][i]_ (1 − _ei) + [1]_

2 − _ri_ 2 _[e][i]_

= [2][ −] [2][r][i][ +][ e][i][r][i] _._

2(2 − _ri)_


Finally, we write out p(y[′] _| x[′]i[)][ as]_

_p(y[′]_ _| x[′]i[) =][ p][(][y][ |][ x][′]i[)][p][(][y][′][ =][ y][ |][ x][′]i[)]_

+ p(¬y | x[′]i[)][p][(][y][′][ ̸][=][ y][ |][ x][′]i[)]

= [2][ −] _[e][i][r][i]_ _si_

2(2 − _ri)_ [(1][ −] _[s][i][) + 2][ −]2(2[2][r] −[i][ +]r[ e]i)[i][r][i]_

= [2][ −] [2][r][i][s][i][ +][ r][i][e][i][(2][s][i][ −] [1)]

2(2 − _ri)_

2 [)]

= [1][ −] _[r][i][(][s][i][ −]_ _[e][i][s][i][ +][ e][i]_ _._

2 − _ri_

From here, we set p(y[′] _| x[′]i[)][ to][ 1]2_ [to prove the]

forward direction of our proposition. The reverse
direction can be easily verified by substituting the


2pb(y, ¬xi) = 1 − _fi + firipb(y, ¬xi)_

_pb(y, ¬xi) = [1][ −]_ _[f][i]_ _._

2 − _firi_

We now derive the marginal probability pb(¬xi):

_pb(¬xi) = 1 −_ _fi + [1]_

2 _[f][i][r][i][p][b][(][¬][x][i][)]_

2pb(¬xi) = 2 − 2fi + firipb(¬xi)


_pb(¬xi) = [2(1][ −]_ _[f][i][)]_ _._

2 − _firi_


-----

solutions found below back into the above equation.

1 − _ri(si −_ _eisi +_ _[e]2[i]_ [)]

= [1]
2 − _ri_ 2

2 − 2ri(si − _eisi +_ _[e]2 [i]_ [) = 2][ −] _[r][i]_

_ri(1 −_ 2(si − _eisi +_ _[e]2 [i]_ [)) = 0]


_ri(1 −_ 2si + 2eisi − _ei) = 0_

_ri((1 −_ _ei) −_ 2si(1 − _ei)) = 0_

_ri(1 −_ _ei)(1 −_ 2si) = 0

Interestingly, this equation factorizes into three independent solutions, giving three ways to achieve
an unbiased pe(y[′] _| x[′]): ri = 0, ei = 1, and si =_ [1]2 [.]

The implications of these solutions are discussed
in the main text.


-----

