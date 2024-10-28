## ReCLIP: A Strong Zero-Shot Baseline for Referring Expression Comprehension

### Sanjay Subramanian[∗][1] William Merrill[2] Trevor Darrell[1]
 Matt Gardner[3] Sameer Singh[4][,][5] Anna Rohrbach[1]

1UC Berkeley 2New York University 3Microsoft Semantic Machines 4UC Irvine
5Allen Institute for AI (AI2)
### {sanjayss,trevordarrell,anna.rohrbach}@berkeley.edu, wcm9940@nyu.edu, mattgardner@microsoft.com, sameer@uci.edu


### Abstract


Training a referring expression comprehension
(ReC) model for a new visual domain requires
collecting referring expressions, and potentially corresponding bounding boxes, for images in the domain. While large-scale pretrained models are useful for image classification across domains, it remains unclear if
they can be applied in a zero-shot manner to
more complex tasks like ReC. We present ReCLIP, a simple but strong zero-shot baseline
that repurposes CLIP, a state-of-the-art largescale model, for ReC. Motivated by the close
connection between ReC and CLIP’s contrastive pre-training objective, the first component of ReCLIP is a region-scoring method
that isolates object proposals via cropping and
blurring, and passes them to CLIP. However,
through controlled experiments on a synthetic
dataset, we find that CLIP is largely incapable
of performing spatial reasoning off-the-shelf.
Thus, the second component of ReCLIP is a
spatial relation resolver that handles several
types of spatial relations. We reduce the gap
between zero-shot baselines from prior work
and supervised models by as much as 29% on
RefCOCOg, and on RefGTA (video game imagery), ReCLIP’s relative improvement over
supervised ReC models trained on real images
is 8%.

### 1 Introduction


(b) RefGTA (Tanaka et al., 2019)

Figure 1: Predictions from ReCLIP (cyan) and
UNITER-Large (Chen et al., 2020) (red) for the same
referring expression on images from two visual domains. UNITER-Large fails on the GTA (video game)
domain, while ReCLIP selects the correct proposal in
both cases. Close-ups of the two GTA boxes are shown.


(a) RefCOCO+ (Yu et al., 2016)


Visual referring expression comprehension (ReC)—
the task of localizing an object in an image given
a textual referring expression—has applications in
a broad range of visual domains. For example,
ReC is useful for guiding a robot in the real world
(Shridhar et al., 2020) and also for creating natural language interfaces for software applications
with visuals (Wichers et al., 2018). Though the
task is the same across domains, the domain shift
is problematic for supervised referring expression

_∗_ This work was done while Sanjay, Will, and Matt were
affiliated with AI2.


models, as shown in Figure 1: the same simple
referring expression is localized correctly in the
training domain but incorrectly in a new domain.
Collecting task-specific data in each domain
of interest is expensive. Weakly supervised ReC
(Rohrbach et al., 2016) partially addresses this issue, since it does not require the ground-truth box
for each referring expression, but it still assumes
the availability of referring expressions paired with
images and trains on these. Given a large-scale pretrained vision and language model and a method


-----

Figure 2: Overview of ReCLIP. Given object proposals, we isolate the corresponding image regions by cropping
and blurring (only cropping shown here). Using a parser, we extract the noun chunks of the expression. For each
noun chunk, CLIP outputs a distribution over proposals. The relations from the parser and CLIP’s probabilities are
combined by a spatial relation resolver to select the final proposal. In this example, CLIP ranks b3 highest for both
noun chunks, but using the relation resolver we obtain the correct answer b4.


for doing ReC zero-shot—i.e. without any additional training—practitioners could save a great
deal of time and effort. Moreover, as pre-trained
models have become more accurate via scaling (Kaplan et al., 2020), fine-tuning the best models has
become prohibitively expensive–and sometimes infeasible because the model is offered only via API,
e.g. GPT-3 (Brown et al., 2020).
Pre-trained vision and language models like
CLIP (Radford et al., 2021) achieve strong zeroshot performance in image classification across
visual domains (Jia et al., 2021) and in object detection (Gu et al., 2021), but the same success has
not yet been achieved in tasks requiring reasoning over vision and language. For example, Shen
et al. (2021) show that a straightforward zero-shot
approach for VQA using CLIP performs poorly.
Specific to ReC, Yao et al. (2021) introduce a zeroshot approach via Colorful Prompt Tuning (CPT),
which colors object proposals and references the
color in the text prompt to score proposals, but
this has low accuracy. In both of these cases, the
proposed zero-shot method is not aligned closely
enough with the model’s pre-training task of matching naturally occurring images and captions.
In this work, we propose ReCLIP, a simple but
strong new baseline for zero-shot ReC. ReCLIP,
illustrated in Figure 2, has two key components: a
method for scoring object proposals using CLIP
and a method for handling spatial relations between
objects. Our method for scoring region proposals,
Isolated Proposal Scoring (IPS), effectively reduces
ReC to the contrastive pre-training task used by
CLIP and other models. Specifically, we propose
to isolate individual proposals via cropping and
blurring the images and to score these isolated pro

posals with the given expression using CLIP.
To handle relations between objects, we first
consider whether CLIP encodes the spatial information necessary to resolve these relations. We
show through a controlled experiment on CLEVR
images (Johnson et al., 2017) that CLIP and another
pre-trained model ALBEF (Li et al., 2021) are unable to perform its pre-training task on examples
that require spatial reasoning.
Thus, any method that solely relies on these models is unlikely to resolve spatial relations accurately.
Consequently, we propose spatial heuristics for handling spatial relations in which an expression is
decomposed into subqueries, CLIP is used to compute proposal probabilities for each subquery, and
the outputs for all subqueries are combined with
simple rules.
On the standard RefCOCO/g/+ datasets (Mao
et al., 2016; Yu et al., 2016), we find that ReCLIP
outperforms CPT (Yao et al., 2021) by about 20%.
Compared to a stronger GradCAM (Selvaraju et al.,
2017) baseline, ReCLIP obtains better accuracy on
average and has less variance across object types.
Finally, in order to illustrate the practical value of
zero-shot grounding, we also demonstrate that our
zero-shot method surpasses the out-of-domain performance of state-of-the-art supervised ReC models. We evaluate on the RefGTA dataset (Tanaka
et al., 2019), which contains images from a video
game (out of domain for models trained only on
real photos). Using ReCLIP and an object detector
trained outside the target domain, we outperform
UNITER-Large (Chen et al., 2020) (using the same
proposals) and MDETR (Kamath et al., 2021) by
an absolute 4.5% (relative improvement of 8%).
In summary, our contributions include: (1) Re

-----

CLIP, a zero-shot method for referring expression
comprehension, (2) showing that CLIP has low
zero-shot spatial reasoning performance, and (3) a
comparison of our zero-shot ReC performance with
the out-of-domain performance of state-of-the-art
fully supervised ReC systems.[1]

### 2 Background

In this section, we first describe the task at hand
(§2.1) and introduce CLIP, the pre-trained model
we primarily use (§2.2). We then describe two
existing methods for scoring region proposals using
a pre-trained vision and language model: colorful
prompt tuning (§2.3) and GradCAM (§2.4).

**2.1** **Task description**

In referring expression comprehension (ReC), the
model is given an image and a textual referring
expression describing an entity in the image. The
goal of the task is to select the object (bounding
box) that best matches the expression. As in much
of the prior work on REC, we assume access to a
set of object proposals b1, b2, ..., bn, each of which
is a bounding box in the image. Task accuracy is
measured as the percentage of instances for which
the model selects a proposal whose intersectionover-union (IoU) with the ground-truth box is at
least 0.5. In this paper, we focus on the zero-shot
setting in which we apply a pre-trained model to
ReC without using any training data for the task.

**2.2** **Pre-trained model architecture**

The zero-shot approaches that we consider are
general in that the only requirement for the pretrained model is that when given a query consisting of an image and text, it computes a score
for the similarity between the image and text. In
this paper, we primarily use CLIP (Radford et al.,
2021). We focus on CLIP because it was pretrained on 400M image-caption pairs collected
from the web[2] and therefore achieves impressive
zero-shot image classification performance on a
variety of visual domains. CLIP has an imageonly encoder, which is either a ResNet-based architecture (He et al., 2016) or a visual transformer
(Dosovitskiy et al., 2021), and a text-only transformer. We mainly use the RN50x16 and ViTB/32 versions of CLIP. The image encoder takes

[1Our code is available at https://www.github.](https://www.github.com/allenai/reclip)
[com/allenai/reclip.](https://www.github.com/allenai/reclip)
2This dataset is not public.


the raw image and produces an image representation x ∈ R[d], and the text transformer takes the
sequence of text tokens and produces a text representation y ∈ R[d]. In CLIP’s contrastive pretraining task, given a batch of N images and matching captions, each image must be matched with
the corresponding text. The model’s probability
of matching image i with caption j is given by
exp(βxi[T] **yj)/** [�]k[N]=1 [exp(][β][x][i][T][ y][k][)][, where][ β][ is a]
hyperparameter.[3]

We now describe two techniques from prior work
for selecting a proposal using a pre-trained model.

**2.3** **Colorful Prompt Tuning (CPT)**

The first baseline from prior work that we consider
is colorful prompt tuning (CPT), proposed by Yao
et al. (2021) [4]: they shade proposals with different colors and use a masked language prompt in
which the referring expression is followed by “in

[MASK] color”. The color with the highest probability from a pre-trained masked language model
(MLM) (VinVL; (Zhang et al., 2021)) is then chosen. In order to apply this method to models like
CLIP, that provide image-text scores but do not offer an MLM, we create a version of the input image
for each proposal, where the proposal is transparently shaded in red.[5] Our template for the input text
is “[referring expression] is in red color.” Since we
have adapted CPT for non-MLM models, we refer
to this method as CPT-adapted in the experiments.

**2.4** **Gradient-based visualizations**

The second baseline from prior work that we consider is based on gradient-based visualizations,
which are a popular family of techniques for understanding, on a range of computer vision tasks,
which part(s) of an input image are most important
to a model’s prediction. We focus on the most popular technique in this family, GradCAM (Selvaraju
et al., 2017). Our usage of GradCAM follows Li
et al. (2021), in which GradCAM is used to perform weakly supervised referring expression comprehension using the ALBEF model, and Chefer
et al. (2021). In our setting, for a given layer in a
visual transformer, we take the layer’s class-token
(CLS) attention matrix M ∈ R[h,w]. The spatial

3xi and yi are normalized before the dot product.
4CPT is the name given by Yao et al. (2021), but note that
we do not perform few-shot/supervised tuning.
5Specifically, we use the RGB values (240, 0, 30) and
transparency 127/255 that Yao et al. (2021) say works best
with their method. An example is shown in Appendix B.


-----

dimensions h and w are dependent on the model’s
architecture and are generally smaller than the input
dimensions of the image. Then the GradCAM is
computed as G = M
_⊙_ _∂M[∂L]_ [, where][ L][ is the model’s]

output logit (the similarity score for the image-text
pair) and denotes elementwise multiplication.
_⊙_
The procedure for applying GradCAM when the visual encoder is a convolutional network is similar[6];
in place of the attention matrix, we use the activations of the final convolutional layer. Next, we
perform a bicubic interpolation on G so that it has
the same dimensions as the input image. Finally,
we compute for each proposal bi = (x1, y1, x2, y2)
the score _A1[α]_ �xi=2 _x1_ �yj=2 _y1_ _[G][[][i, j][]][, where][ A][ is the]_

area of the image and α is a hyperparameter, and
we choose the proposal with the highest score.

### 3 ReCLIP

ReCLIP consists of two main components: (1) a
region-scoring method that is different from CPT
and GradCAM and (2) a rule-based relation resolver. In this section, we first describe our region
scoring method (§3.1). However, using controlled
experiments on a synthetic dataset, we find that
CLIP has poor zero-shot spatial reasoning performance (§3.2). Therefore, we propose a system that
uses heuristics to resolve spatial relations (§3.3).

**3.1** **Isolated Proposal Scoring (IPS)**

Our proposed method, which we call isolated pro_posal scoring, is based on the observation that_
ReC is similar to the contrastive learning task with
which models like CLIP are pre-trained, except
that rather than selecting one out of several images to match with a given text, we must select
one out of several image regions. Therefore, for
each proposal, we create a new image in which
that proposal is isolated. We consider two methods
of isolation – cropping the image to contain only
the proposal and blurring everything in the image
except for the proposal region. For blurring, we
apply a Gaussian filter with standard deviation σ
to the image RGB values. Appendix A.2 provides
an example of isolation by blurring. The score for
an isolated proposal is obtained by passing it and
the expression through the pre-trained model. To
use cropping and blurring in tandem, we obtain
a score scrop and sblur for each proposal and use

6The convolutional version, following Selvaraju et al.

(2017), applies global average pooling to the gradients, unlike
the transformer version.


Text-pair Text-pair Image-pair Image-pair
Model
Spatial Non-spatial Spatial Non-spatial

CLIP RN50x4 43.39 89.83 48.90 97.36
CLIP RN50x16 51.19 89.83 50.22 96.48
CLIP RN50x64 47.80 94.58 51.54 97.36
CLIP ViT-B/32 48.47 95.25 48.90 96.48
CLIP ViT-B/16 50.51 92.54 50.22 96.92
CLIP ViT-L/14 52.88 96.27 50.66 94.27

Table 1: Accuracy on CLEVR image-text matching task. CLIP
performs well on the non-spatial version of the task but poorly
on the spatial version. Text-pair tasks have 295 instances each;
image-pair tasks have 227 instances each.

_scrop + sblur as the final score. This can be viewed_
as an ensemble of “visual prompts,” analogous to
Radford et al. (2021)’s ensembling of text prompts.

**3.2** **Can we use CLIP to resolve spatial**
**relations?**

A key limitation in Isolated Proposal Scoring is
that relations between objects in different proposals are not taken into account. For example, in
Figure 2, the information about the spatial relationships among the cats is lost when the proposals
are isolated. In order to use CLIP to decide which
object has a specified relation to another object,
the model’s output must encode the spatial relation
in question. Therefore, we design an experiment
to determine whether a pre-trained model, such
as CLIP, can understand spatial relations within
the context of its pre-training task. We generate
synthetic images using the process described for
the CLEVR dataset (Johnson et al., 2017). These
scenes include three shapes–spheres, cubes, and
cylinders–and eight colors–gray, blue, green, cyan,
yellow, purple, brown, red.
In the text-pair version of our tasks, using the
object attribute and position information associated
with each image, we randomly select one of the
pairwise relationships between objects–left, right,
front, or behind–and construct a sentence fragment
based on it. For example: “A blue sphere to the
left of a red cylinder.” We also write a distractor
fragment that replaces the relation with its opposite.
In this case, the distractor would be “A blue sphere
to the right of a red cylinder.” The task, similar to
the contrastive and image-text matching tasks used
to pre-train these models, is to choose the correct
sentence given the image. As a reference point,
we also evaluate on a control (non-spatial) task in
which the correct text is a list of the scene’s objects
and the distractor text is identical except that one
object is swapped with a random object not in the


-----

scene. For example, if the correct text is “A blue
sphere and a red cylinder,” then the distractor text
could be “A blue sphere and a blue cylinder.”
In the image-pair version of our tasks, we have a
single sentence fragment constructed as described
above for the spatial and control (non-spatial) tasks
and two images such that only one matches the text.
Appendix B shows examples of these tasks.
CLIP’s performance on these tasks is shown in
Table 1. Similar results for the pre-trained model
ALBEF (Li et al., 2021) are shown in Appendix D.1
While performance on the control task is quite
good, accuracy on the spatial task is not so different from random chance (50%). This indicates
that the model scores of image-text pairs largely do
not take spatial relations into account.

**3.3** **Spatial Relation Resolver**

Since CLIP lacks sensitivity to spatial relations,
we propose to decompose complex expressions
into simpler primitives. The basic primitive is a
predicate applying to an object, which we use CLIP
to answer. The second primitive is a spatial relation
between objects, for which we use heuristic rules.

**Predicates** A predicate is a textual property that
the referent must satisfy. For example, “the cat”
and “blue airplane” are predicates. We write P (i)
to say that object i satisfies the predicate P . We
model P as a categorical distribution over objects,
and estimate p(i) = Pr[P (i)] with the pre-trained
model using isolated proposal scoring (§ 3.1).

**Relations** We have already discussed the importance of binary spatial relations like “the cat to the
_left of the dog” for the ReC task. We consider_
seven spatial relations–left, right, above, below,
_bigger, smaller, and inside. We write R(i, j) to_
mean that the relation R holds between objects i
and j, and we use heuristics to determine the probability r(i, j) = Pr[R(i, j)]. For example, for left,
we set r(i, j) = 1 if the center point of box i is to
the left of the center point of box j and r(i, j) = 0
otherwise. §C.1 describes all relation semantics.

**Superlative Relations** We also consider superlatives, which refer to an object that has some relation
to all other objects satisfying the same predicate,
e.g. “leftmost dog”. We handle superlatives as a
special case of relations where the empty second argument is filled by copying the predicate specifying
the first argument. Thus, “leftmost dog” effectively
finds the dog that is most likely to the left of other


�
_∝_ _πN_ (i) _rN,N_ _′(i, j)πN_ _′(j)._

_j_

The last line makes the simplifying assumption that
all predicates and relations are independent.[8]

To compute our final output, we ensemble the
distribution πroot for the root node with the output
of plain isolated proposal scoring (with the whole
input expression) by multiplying the proposal probabilities elementwise. This method gives us a prin
7Superlatives of a node are processed after all its relations.
8We write ∝ because πN′ [(][i][)][ is normalized to sum to][ 1][.]


Figure 3: Example extraction of semantic trees from dependency parses. Predicate text in blue. Red arcs show paths
contributing spatial relation left and superlative largest. For
the superlative, we create a parent node with the original node
as the only child, effectively converting it into a relation.

dog(s). Our set of superlative relation types is the
same as our set of relation types, excluding inside.

**Semantic Trees** Having outlined the semantic
formalism underlying our method, we can describe
it procedurally. We first use spaCy (Honnibal and
Johnson, 2015) to build a dependency parse for the
expression. As illustrated in Figure 3, we extract
a semantic tree from the dependency parse, where
each noun chunk becomes a node, and dependency
paths between the heads of noun chunks become
relations between entities based on the keywords
they contain. See §C.2 for extraction details. In
cases where none of our relation/superlative keywords occur in the text, we simply revert to the
plain isolated proposal scoring method using the
full text.
In the tree, each node N contains a predicate PN
and has a set of children; an edge (N, N _[′]) between_
_N and its child N_ _[′]_ corresponds to a relation RN,N _′._
For example, as shown in Figure 3, “a cat to the left
of a dog” would be parsed as a node containing the
predicate “a cat” connected by the relation left to its
child corresponding to “a dog”. We define πN (i)
as the probability that node N refers to object i,
and compute it recursively. For each node N, we
first set πN (i) = pN (i) and then iterate through
each child N _[′]_ and update πN (i) as follows[7]:

_πN[′]_ [(][i][)][ ∝] _[π][N]_ [(][i][)] � Pr �RN,N _′(i, j) ∧_ _PN_ _′(j)�_

_j_


-----

cipled way to combine predicates (PN ) with spatial
relational constraints (RN,N _′) for each node N_ .

### 4 Experiments

**4.1** **Datasets**

We compare ReCLIP to other zero-shot methods
on RefCOCOg (Mao et al., 2016), RefCOCO and
**RefCOCO+ (Yu et al., 2016). These datasets use**
images from MS COCO (Lin et al., 2014). RefCOCO and RefCOCO+ were created in a twoplayer game, and RefCOCO+ is designed to avoid
spatial relations. RefCOCOg includes spatial relations and has longer expressions on average.
For comparing zero-shot methods with the out-ofdomain performance of models trained on COCO,
we use RefGTA (Tanaka et al., 2019), which contains images from the Grand Theft Auto video
game. All referring expressions in RefGTA correspond to people, and the objects (i.e. people)
tend to be much smaller on average than those in
RefCOCO/g/+.

**4.2** **Implementation Details**

We use an ensemble of the CLIP RN50x16 and
ViT-B/32 models (results for individual models
are shown in Appendix G). We ensemble model
outputs by adding together the logits from the
two models elementwise before taking the softmax. GradCAM’s hyperparameter α controls the
effect of the proposal’s area on its score. We select α = 0.5 for all models based on tuning on the
RefCOCOg validation set. We emphasize that the
optimal value of α for a dataset depends on the size
distribution of ground-truth objects. ReCLIP also
has a hyperparameter, namely the standard deviation σ. We try a few values on the RefCOCOg
validation set and choose σ = 100, as we show
in Appendix E.4, isolated proposal scoring has little sensitivity to σ. As discussed by (Perez et al.,
2021), zero-shot experiments often use labeled data
for model selection. Over the course of this work,
we primarily experimented with the RefCOCOg
validation set and to a lesser extent with the RefCOCO+ validation set. For isolated proposal scoring, the main variants explored are documented in
our ablation study (§4.6). Other techniques that we
tried, including for relation-handling, and further
implementation details are given in Appendix E.


**4.3** **Results on RefCOCO/g/+**

Table 2 shows results on RefCOCO, RefCOCO+,
and RefCOCOg. ReCLIP is better than the other
zero-shot methods on RefCOCOg and RefCOCO
and on par with GradCAM on RefCOCO+. However, GradCAM has a much higher variance in its
accuracy between the TestA and TestB splits of RefCOCO+ and RefCOCO. We note that GradCAM’s
hyperparameter α, controlling the effect of proposal size, was tuned on the RefCOCOg validation
set, and RefCOCOg was designed such that boxes
of referents are at least 5% of the image area (Mao
et al., 2016). In the bottom portion of Table 2, we
show that when this 5% threshold, a prior on object
size for this domain, is used to filter proposals for
both GradCAM and ReCLIP, ReCLIP performs on
par with/better than GradCAM on TestA. ReCLIP’s
spatial relation resolver helps on RefCOCOg and
RefCOCO but not on RefCOCO+, which is designed to avoid spatial relations.

**4.4** **Results on RefGTA**

Next, we evaluate on RefGTA to compare our
method’s performance to the out-of-domain accuracy of two state-of-the-art fully supervised ReC
models: UNITER-Large (Chen et al., 2020) and
MDETR (Kamath et al., 2021).
Like ReCLIP, UNITER takes proposals as input.[9] We show results using ground-truth proposals and detections from UniDet (Zhou et al., 2021),
which is trained on the COCO, Objects365 (Shao
et al., 2019), OpenImages (Kuznetsova et al., 2020),
and Mapillary (Neuhold et al., 2017) datasets. Following the suggestion of the UniDet authors, we
use the confidence threshold of 0.5. MDETR does
not take proposals as input.
Table 3 shows our results. For methods that take
proposals (all methods except MDETR), we consider two evaluation settings using UniDet–DT-P,
in which the detected proposals are filtered to have
only proposals whose predicted class label is “person”, and DT, in which all detected proposals are
considered. ReCLIP’s accuracy is more than 15%

9UNITER requires features from the bottom-up
top-down attention model (Anderson et al., 2017).
We use [https://github.com/airsplay/](https://github.com/airsplay/py-bottom-up-attention)
[py-bottom-up-attention to compute the features for](https://github.com/airsplay/py-bottom-up-attention)
RefGTA. We trained UNITER models on RefCOCO+ and
RefCOCOg using features computed from this repository.
On the RefCOCO+ validation set, the resulting model has an
accuracy roughly 0.4% less than that of a model trained and
evaluated using the original features (when using ground-truth
proposals).


-----

RefCOCOg RefCOCO+ RefCOCO
**Model** **Val** **Test** **Val** **TestA** **TestB** **Val** **TestA** **TestB**

Random 18.12 19.10 16.29 13.57 19.60 15.73 13.51 19.20

Supervised SOTA 83.35 81.64 81.13 85.52 72.96 87.51 90.40 82.67

CPT-Blk w/ VinVL (Yao et al., 2021) 32.1 32.3 25.4 25.0 27.0 26.9 27.5 27.4
CPT-Seg w/ VinVL (Yao et al., 2021) 36.7 36.5 31.9 35.2 28.8 32.2 36.1 30.3

**CLIP**
CPT-adapted 22.32 23.65 23.85 21.55 25.92 23.16 21.44 26.95
GradCAM 50.86 49.70 47.83 **56.92** 37.70 42.85 **51.07** 35.21
ReCLIP w/o relations 57.70 57.19 47.43 50.02 43.85 41.97 43.42 39.02
ReCLIP **59.33** **59.01** **47.87** 50.10 **45.10** **45.78** 46.10 **47.07**

**CLIP w/ Object Size Prior**
CPT-adapted 28.98 30.14 26.64 25.13 27.27 26.08 25.38 28.03
GradCAM 52.29 51.28 49.41 59.66 38.62 44.65 53.49 36.19
ReCLIP w/o relations 59.19 59.01 54.66 60.27 46.33 48.53 53.60 40.84
ReCLIP 60.85 61.05 55.07 60.47 47.41 54.04 58.60 49.54

Table 2: Accuracy on the RefCOCOg, RefCOCO+ and RefCOCO datasets. ReCLIP outperforms other zero-shot methods on
RefCOCOg. On RefCOCO+ and RefCOCO, ReCLIP is on par with or better than GradCAM on average and has lower variance
between TestA and TestB, which correspond to different kinds of objects. When taking into account a prior on object size
(filtering out objects smaller than 5% of the image), GradCAM’s advantage on the TestA splits is erased. Best zero-shot results
in each column are in bold, and best zero-shot results using the size prior are underlined. CLIP results use an ensemble of the
RN50x16 and ViT-B/32 CLIP models. CPT-adapted is an adapted version of CPT-Blk. Supervised SOTA refers to MDETR
(Kamath et al., 2021); we use the EfficientNet-B3 version. All methods except MDETR use detected proposals from MAttNet
(Yu et al., 2018). CPT-Seg uses Mask-RCNN segmentation masks from Yu et al. (2018).

|Random 18.12 19.10|16.29 13.57 19.60|15.73 13.51 19.20|
|---|---|---|

|Supervised SOTA 83.35 81.64|81.13 85.52 72.96|87.51 90.40 82.67|
|---|---|---|

|CPT-Blk w/ VinVL (Yao et al., 2021) 32.1 32.3 CPT-Seg w/ VinVL (Yao et al., 2021) 36.7 36.5|25.4 25.0 27.0 31.9 35.2 28.8|26.9 27.5 27.4 32.2 36.1 30.3|
|---|---|---|

|CLIP CPT-adapted 22.32 23.65 GradCAM 50.86 49.70 ReCLIP w/o relations 57.70 57.19 ReCLIP 59.33 59.01|23.85 21.55 25.92 47.83 56.92 37.70 47.43 50.02 43.85 47.87 50.10 45.10|23.16 21.44 26.95 42.85 51.07 35.21 41.97 43.42 39.02 45.78 46.10 47.07|
|---|---|---|

|CLIP w/ Object Size Prior CPT-adapted 28.98 30.14 GradCAM 52.29 51.28 ReCLIP w/o relations 59.19 59.01 ReCLIP 60.85 61.05|26.64 25.13 27.27 49.41 59.66 38.62 54.66 60.27 46.33 55.07 60.47 47.41|26.08 25.38 28.03 44.65 53.49 36.19 48.53 53.60 40.84 54.04 58.60 49.54|
|---|---|---|


higher than the accuracy of UNITER-Large and
roughly 5% more than that of MDETR. ReCLIP
also outperforms GradCAM by about 20%, and the
gap is larger when all UniDet proposals are considered. ReCLIP w/o relations is 1-2% better than
ReCLIP in the settings with ground-truth proposals
and filtered UniDet proposals. One possible reason
for this gap is that the objects of relations in the
expressions could be non-people entities. When
considering all UniDet proposals, the relation resolver in ReCLIP does not hurt accuracy much but
also does not improve accuracy significantly–an additional challenge in this setting is that the number
of proposals is dramatically higher. Appendix F
shows qualitative examples of predictions on RefGTA.

**4.5** **Using another Pre-trained Model**

In order to determine how isolated proposal scoring (IPS) compares to GradCAM and CPT on other
pre-trained models, we present results using ALBEF (Li et al., 2021). ALBEF offers two methods
for scoring image-text pairs–the output used for
its image-text contrastive (ITC) loss and the output used for its image-text matching (ITM) loss.
The architecture providing the ITC output is very
similar to CLIP–has only a shallow interaction between the image and text modalities. The ITM


**Val** **Test**
**Model** GT DT-P DT GT DT-P DT

Random 27.03 21.53 4.86 27.60 21.75 5.13
UNITER-Large
_RefCOCO+_ 49.57 47.52 35.04 50.60 48.30 34.40
_RefCOCOg_ 49.81 48.59 27.58 51.05 49.78 28.31
MDETR
_RefCOCO+_ – – 38.49 – – 39.02
_RefCOCOg_ – – 38.29 – – 39.13
_Pretrained_ – – 54.91 – – 56.60
CLIP GradCAM 51.90 51.03 33.66 51.53 50.73 34.51
ReCLIP 69.84 68.42 60.93 70.79 69.05 **61.38**
_w/o relations_ **71.66** **70.27** **60.98** **72.56** **70.84** 61.31

Table 3: Accuracy on RefGTA dataset. ReCLIP w/o relations
outperforms all other methods. GT denotes use of groundtruth proposals; DT denotes use of detected proposals; DT_P denotes detected proposals filtered to have only people._
Subscripts RefCOCO+/RefCOCOg indicate finetuning dataset;
_Pretrained indicates a model that is not finetuned. MDETR_
does not take proposals as input, so the GT and DT-P columns
are blank. We use the EfficientNet-B3 versions of MDETR.
**Bold indicates best score in a column.**

output is given by an encoder that has deeper interactions between image and text and operates
on top of the ITC encoders’ output. Appendix D
provides more details. The results, shown in Table 4, show that with the ITC output, IPS performs
better than GradCAM, but with the ITM output,
GradCAM performs better. This suggests that IPS
works well across models like CLIP and ALBEF
ITC (i.e. contrastively pre-trained with shallow
modality interactions) but that GradCAM may be


-----

**Model** RefCOCOg RefCOCO+(A) RefCOCO+(B)

**ALBEF ITM (Deep modality interaction)**
CPT-adapted 24.99 26.83 26.43
GradCAM **55.92** **61.75** **42.79**
IPS 55.21 51.82 42.63

**ALBEF ITC (Shallow modality interaction)**
CPT-adapted 21.10 19.00 21.33
GradCAM 47.53 44.60 36.00
IPS **54.07** **45.90** **39.58**

Table 4: Accuracy on RefCOCOg and RefCOCO+ test sets
using ALBEF pre-trained model. IPS does best when using
ALBEF’s ITC architecture, while GradCAM is better for ITM.

(a) ReCLIP is correct, while GradCAM is incorrect

(b) Both ReCLIP and GradCAM are incorrect

Figure 4: RefCOCOg validation examples using
ground-truth proposals. Ground-truth referents are

green, ReCLIP predictions are blue, and GradCAM predictions are red. In 4a, ReCLIP makes the correct prediction based on local context. In 4b, ReCLIP grounds
an incorrect noun chunk from the expression.

better for models with deeper interactions.

**4.6** **Analysis**

**Performance of IPS** Our results show that
among the region scoring methods that we consider,
IPS achieves the highest accuracy for contrastively
pre-trained models like CLIP. Figure 4a gives intuition for this—aside from an object’s attributes,
many referring expressions describe the local context around an object, and IPS focuses on this local
context (as well as object attributes).
Table 5 shows that using both cropping and blurring obtains greater accuracy than either alone.


**Isolation type** RefCOCOg RefCOCO+

Crop 54.43 41.28
Blur 55.96 47.23
max(Crop,Blur) 55.76 44.55
Crop+Blur **57.70** **47.43**

Table 5: Ablation study of isolation types used to score proposals on Val splits of RefCOCOg/RefCOCO+, using detections
from MAttNet (Yu et al., 2018). Crop+Blur is best overall.

**Error Analysis and Limitations** Although ReCLIP outperforms the baselines that we consider,
there is a considerable gap between it and supervised methods. The principal challenge in improving the system is making relation-handling more
flexible. There are several object relation types
that our spatial relation resolver cannot handle; for
instance, those that involve counting: “the second
dog from the right.” Another challenge is in determining which relations require looking at multiple
proposals. For instance, ReCLIP selects a proposal
corresponding to the incorrect noun chunk in Figure 4b because the relation resolver has no rule for
splitting an expression on the relation “with.” Depending on the context, relations like “with” may
or may not require looking at multiple proposals,
so handling them is challenging for a rule-based
system.
In the RefCOCO+ validation set, when using detected proposals, there are 75 instances for which
ReCLIP answers incorrectly but ReCLIP w/o relations answers correctly. We categorize these instances based on their likely sources of error: 4
instances are ambiguous (multiple valid proposals), in 7 instances the parser misses the head noun
chunk, in 14 instances our processing of the parse
leads to omissions of text when doing isolated proposal scoring (e.g. in “girl sitting in back,” the
only noun chunk is “girl,” so this is the only text
used during isolated proposal scoring), 52 cases
in which there is an error in the execution of the
heuristic (e.g. our spatial definition of a relation
does not match the relation in the instance). (There
are 2 instances for which we mark 2 categories.)
The final category (“execution”) includes several
kinds of errors, some examples of which are shown
in Appendix F.

### 5 Related Work

**Referring expression comprehension** Datasets
for ReC span several visual domains, including
photos of everyday scenes (Mao et al., 2016;
Kazemzadeh et al., 2014), video games (Tanaka


-----

et al., 2019), objects in robotic context (Shridhar
et al., 2020; Wang et al., 2021), and webpages
(Wichers et al., 2018).
Spatial heuristics have been used in previous
work (Moratz and Tenbrink, 2006). Our work is
also related to Krishnamurthy and Kollar (2013),
which similarly decomposes the reasoning process
into a parsing step and visual execution steps, but
the visual execution is driven by learned binary
classifiers for each predicate type. In the supervised setting, prior work shows that using an external parser, as we do, leads to lower accuracy
than training a language module jointly with the
remainder of the model (Hu et al., 2017).
There is a long line of work in weakly supervised ReC, where at training time, pairs of referring expressions and images are available but the
ground-truth bounding boxes for each expression
are not (Rohrbach et al., 2016; Liu et al., 2019;
Zhang et al., 2018, 2020; Sun et al., 2021). Our
setting differs from the weakly supervised setting
in that the model is not trained at all on the ReC
task. Sadhu et al. (2019) discuss a zero-shot setting
different from ours in which novel objects are seen
at test time, but the visual domain stays the same.

**Pre-trained vision and language models** Early
pre-trained vision and language models (Tan and
Bansal, 2019; Lu et al., 2019; Chen et al., 2020)
used a cross-modal transformer (Vaswani et al.,
2017) and pre-training tasks like masked language
modeling, image-text matching, and image feature
regression. By contrast, CLIP and similar models
(Radford et al., 2021; Jia et al., 2021) use a separate image and text transformer and a contrastive
pre-training objective. Recent hybrid approaches
augment CLIP’s architecture with a multi-modal
transformer (Li et al., 2021; Zellers et al., 2021).

**Zero-shot application of pre-trained models**
Models pre-trained with the contrastive objective
have exhibited strong zero-shot performance in image classification tasks (Radford et al., 2021; Jia
et al., 2021). Gu et al. (2021) use CLIP can be
to classify objects by computing scores for class
labels with cropped proposals. Our IPS is different
in that it isolates proposals by both cropping and
_blurring. Shen et al. (2021) show that a simple_
zero-shot application of CLIP to visual question
answering performs almost on par with random
chance. Yao et al. (2021) describe a zero-shot
method for ReC based on a pre-trained masked lan

guage model (MLM); we show that their zero-shot
results and a version of their method adapted for
models pre-trained to compute image-text scores
(rather than MLM) are substantially worse than isolated proposal scoring and GradCAM. Concurrent
with our work, Liu et al. (2021) also observe that
CLIP has poor zero-shot accuracy when dealing
with spatial relations.

### 6 Conclusion

We present ReCLIP, a zero-shot method for referring expression comprehension (ReC) that decomposes an expression into subqueries, uses CLIP to
score isolated proposals against these subqueries,
and combines the outputs with spatial heuristics.
ReCLIP outperforms zero-shot ReC approaches
from prior work and also performs well across visual domains: ReCLIP outperforms state-of-the-art
supervised ReC models, trained on natural images,
when evaluated on RefGTA. We also find that CLIP
has low zero-shot spatial reasoning performance,
suggesting the need for pre-training methods that
account more for spatial reasoning.

### 7 Ethical and Broader Impacts

Recent work has shown that pre-trained vision and
language models suffer from biases such as gender bias (Ross et al., 2021; Srinivasan and Bisk,
2021). Agarwal et al. (2021) provide evidence that
CLIP has racial and other biases, which makes
sense since CLIP was trained on data collected
from the web and not necessarily curated carefully.
Therefore, we do not advise deploying our system
directly in the real world immediately. Instead,
practitioners interested in this system should first
perform analysis to measure its biases based on previous work and attempt to mitigate them. We also
note that our work relies heavily on a pre-trained
model whose pre-training required a great deal of
energy, which likely had negative environmental
effects. That being said our zero-shot method does
not require training a new model and in that sense
could be more environmentally friendly than supervised ReC models (depending on the difference in
the cost of inference).

### 8 Acknowledgements

We thank the Berkeley NLP group, Medhini
Narasimhan, and the anonymous reviewers for
helpful comments. We thank Michael Schmitz
for help with AI2 infrastructure. This work was


-----

supported in part by DoD, including DARPA’s
LwLL (FA8750-19-1-0504), and/or SemaFor
(HR00112020054) programs, and Berkeley Artificial Intelligence Research (BAIR) industrial alliance programs. Sameer Singh was supported
in part by the National Science Foundation grant
#IIS-1817183 and in part by the DARPA MCS program under Contract No. N660011924033 with the
United States Office Of Naval Research.

### References

Sandhini Agarwal, Gretchen Krueger, Jack Clark, Alec
Radford, Jong Wook Kim, and Miles Brundage.
2021. Evaluating clip: towards characterization of
broader capabilities and downstream implications.
_arXiv preprint arXiv:2108.02818._

Peter Anderson, Xiaodong He, Chris Buehler, Damien
Teney, Mark Johnson, Stephen Gould, and Lei
Zhang. 2017. Bottom-up and top-down attention for
image captioning and vqa. ArXiv, abs/1707.07998.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-Voss,
Gretchen Krueger, Tom Henighan, Rewon Child,
Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu,
Clemens Winter, Christopher Hesse, Mark Chen,
Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin
Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario
[Amodei. 2020. Language models are few-shot learn-](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)
[ers. In Advances in Neural Information Processing](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)
_Systems 33: Annual Conference on Neural Informa-_
_tion Processing Systems 2020, NeurIPS 2020, De-_
_cember 6-12, 2020, virtual._

Hila Chefer, Shir Gur, and Lior Wolf. 2021. Generic
attention-model explainability for interpreting bimodal and encoder-decoder transformers. In Pro_ceedings of the IEEE/CVF International Conference_
_on Computer Vision, pages 397–406._

Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El
Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and
Jingjing Liu. 2020. Uniter: Universal image-text
representation learning. In ECCV.

Alexey Dosovitskiy, Lucas Beyer, Alexander
Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
Thomas Unterthiner, Mostafa Dehghani, Matthias
Minderer, Georg Heigold, Sylvain Gelly, Jakob
Uszkoreit, and Neil Houlsby. 2021. [An image is](https://openreview.net/forum?id=YicbFdNTTy)
[worth 16x16 words: Transformers for image recog-](https://openreview.net/forum?id=YicbFdNTTy)
[nition at scale.](https://openreview.net/forum?id=YicbFdNTTy) In 9th International Conference
_on Learning Representations, ICLR 2021, Virtual_
_Event, Austria, May 3-7, 2021. OpenReview.net._

Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin
Cui. 2021. [Zero-shot detection via vision and](https://arxiv.org/abs/2104.13921)


[language knowledge distillation.](https://arxiv.org/abs/2104.13921) _ArXiv preprint,_
abs/2104.13921.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian
[Sun. 2016. Deep residual learning for image recog-](https://doi.org/10.1109/CVPR.2016.90)
[nition. In 2016 IEEE Conference on Computer Vi-](https://doi.org/10.1109/CVPR.2016.90)
_sion and Pattern Recognition, CVPR 2016, Las Ve-_
_gas, NV, USA, June 27-30, 2016, pages 770–778._
IEEE Computer Society.

[Matthew Honnibal and Mark Johnson. 2015. An im-](https://doi.org/10.18653/v1/D15-1162)
[proved non-monotonic transition system for depen-](https://doi.org/10.18653/v1/D15-1162)
[dency parsing.](https://doi.org/10.18653/v1/D15-1162) In Proceedings of the 2015 Con_ference on Empirical Methods in Natural Language_
_Processing, pages 1373–1378, Lisbon, Portugal. As-_
sociation for Computational Linguistics.

Ronghang Hu, Marcus Rohrbach, Jacob Andreas,
[Trevor Darrell, and Kate Saenko. 2017. Modeling](https://doi.org/10.1109/CVPR.2017.470)
[relationships in referential expressions with compo-](https://doi.org/10.1109/CVPR.2017.470)
[sitional modular networks. In 2017 IEEE Confer-](https://doi.org/10.1109/CVPR.2017.470)
_ence on Computer Vision and Pattern Recognition,_
_CVPR 2017, Honolulu, HI, USA, July 21-26, 2017,_
pages 4418–4427. IEEE Computer Society.

Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana
Parekh, Hieu Pham, Quoc V. Le, Yun-Hsuan Sung,
[Zhen Li, and Tom Duerig. 2021. Scaling up visual](http://proceedings.mlr.press/v139/jia21b.html)
[and vision-language representation learning with](http://proceedings.mlr.press/v139/jia21b.html)
[noisy text supervision. In Proceedings of the 38th In-](http://proceedings.mlr.press/v139/jia21b.html)
_ternational Conference on Machine Learning, ICML_
_2021, 18-24 July 2021, Virtual Event, volume 139 of_
_Proceedings of Machine Learning Research, pages_
4904–4916. PMLR.

Justin Johnson, Bharath Hariharan, Laurens van der
Maaten, Li Fei-Fei, C. Lawrence Zitnick, and
Ross B. Girshick. 2017. [CLEVR: A diagnostic](https://doi.org/10.1109/CVPR.2017.215)
[dataset for compositional language and elementary](https://doi.org/10.1109/CVPR.2017.215)
[visual reasoning. In 2017 IEEE Conference on Com-](https://doi.org/10.1109/CVPR.2017.215)
_puter Vision and Pattern Recognition, CVPR 2017,_
_Honolulu, HI, USA, July 21-26, 2017, pages 1988–_
1997. IEEE Computer Society.

Aishwarya Kamath, Mannat Singh, Yann LeCun,
Gabriel Synnaeve, Ishan Misra, and Nicolas Carion.
2021. Mdetr - modulated detection for end-to-end
multi-modal understanding. In Proceedings of the
_IEEE/CVF International Conference on Computer_
_Vision (ICCV), pages 1780–1790._

Jared Kaplan, Sam McCandlish, T. J. Henighan, Tom B.
Brown, Benjamin Chess, Rewon Child, Scott Gray,
Alec Radford, Jeff Wu, and Dario Amodei. 2020.
Scaling laws for neural language models. _ArXiv,_
abs/2001.08361.

Sahar Kazemzadeh, Vicente Ordonez, Mark Matten,
and Tamara Berg. 2014. [ReferItGame: Referring](https://doi.org/10.3115/v1/D14-1086)
[to objects in photographs of natural scenes.](https://doi.org/10.3115/v1/D14-1086) In
_Proceedings of the 2014 Conference on Empirical_
_Methods in Natural Language Processing (EMNLP),_
pages 787–798, Doha, Qatar. Association for Computational Linguistics.


-----

Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen,
Yannis Kalantidis, Li-Jia Li, David A. Shamma,
Michael S. Bernstein, and Li Fei-Fei. 2016. Visual genome: Connecting language and vision using crowdsourced dense image annotations. Interna_tional Journal of Computer Vision, 123:32–73._

Jayant Krishnamurthy and Thomas Kollar. 2013.

[Jointly learning to parse and perceive: Connecting](https://doi.org/10.1162/tacl_a_00220)
[natural language to the physical world.](https://doi.org/10.1162/tacl_a_00220) _Transac-_
_tions of the Association for Computational Linguis-_
_tics, 1:193–206._

Alina Kuznetsova, Hassan Rom, Neil Gordon Alldrin,
Jasper R. R. Uijlings, Ivan Krasin, Jordi PontTuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, Tom Duerig, and Vittorio Ferrari. 2020. The open images dataset v4. In_ternational Journal of Computer Vision, 128:1956–_
1981.

Junnan Li, Ramprasaath R. Selvaraju,
Akhilesh Deepak Gotmare, Shafiq Joty, Caiming Xiong, and Steven Hoi. 2021. Align before fuse:
Vision and language representation learning with
momentum distillation. In NeurIPS.

Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James
Hays, Pietro Perona, Deva Ramanan, Piotr Dollár,
and C. Lawrence Zitnick. 2014. Microsoft coco:
Common objects in context. In ECCV.

Nan Liu, Shuang Li, Yilun Du, Josh Tenenbaum, and
[Antonio Torralba. 2021. Learning to compose visual](https://proceedings.neurips.cc/paper/2021/file/c3008b2c6f5370b744850a98a95b73ad-Paper.pdf)
[relations. In Advances in Neural Information Pro-](https://proceedings.neurips.cc/paper/2021/file/c3008b2c6f5370b744850a98a95b73ad-Paper.pdf)
_cessing Systems, volume 34, pages 23166–23178._
Curran Associates, Inc.

Xuejing Liu, Liang Li, Shuhui Wang, Zheng-Jun Zha,
[Dechao Meng, and Qingming Huang. 2019. Adap-](https://doi.org/10.1109/ICCV.2019.00270)
[tive reconstruction network for weakly supervised](https://doi.org/10.1109/ICCV.2019.00270)
[referring expression grounding. In 2019 IEEE/CVF](https://doi.org/10.1109/ICCV.2019.00270)
_International Conference on Computer Vision, ICCV_
_2019, Seoul, Korea (South), October 27 - November_
_2, 2019, pages 2611–2620. IEEE._

Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan
[Lee. 2019. Vilbert: Pretraining task-agnostic visi-](https://proceedings.neurips.cc/paper/2019/hash/c74d97b01eae257e44aa9d5bade97baf-Abstract.html)
[olinguistic representations for vision-and-language](https://proceedings.neurips.cc/paper/2019/hash/c74d97b01eae257e44aa9d5bade97baf-Abstract.html)
[tasks. In Advances in Neural Information Process-](https://proceedings.neurips.cc/paper/2019/hash/c74d97b01eae257e44aa9d5bade97baf-Abstract.html)
_ing Systems 32: Annual Conference on Neural Infor-_
_mation Processing Systems 2019, NeurIPS 2019, De-_
_cember 8-14, 2019, Vancouver, BC, Canada, pages_
13–23.

Junhua Mao, Jonathan Huang, Alexander Toshev, Oana
Camburu, Alan L. Yuille, and Kevin Murphy. 2016.
[Generation and comprehension of unambiguous ob-](https://doi.org/10.1109/CVPR.2016.9)
[ject descriptions. In 2016 IEEE Conference on Com-](https://doi.org/10.1109/CVPR.2016.9)
_puter Vision and Pattern Recognition, CVPR 2016,_
_Las Vegas, NV, USA, June 27-30, 2016, pages 11–20._
IEEE Computer Society.


Reinhard Moratz and Thora Tenbrink. 2006. Spatial
reference in linguistic human-robot interaction: Iterative, empirically supported development of a model
of projective relations. Spatial cognition and compu_tation, 6(1):63–107._

Gerhard Neuhold, Tobias Ollmann, Samuel Rota Bulò,
[and Peter Kontschieder. 2017. The mapillary vistas](https://doi.org/10.1109/ICCV.2017.534)
[dataset for semantic understanding of street scenes.](https://doi.org/10.1109/ICCV.2017.534)
In IEEE International Conference on Computer Vi_sion, ICCV 2017, Venice, Italy, October 22-29, 2017,_
pages 5000–5009. IEEE Computer Society.

Vicente Ordonez, Girish Kulkarni, and Tamara L. Berg.
[2011. Im2text: Describing images using 1 million](https://proceedings.neurips.cc/paper/2011/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html)
[captioned photographs. In Advances in Neural In-](https://proceedings.neurips.cc/paper/2011/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html)
_formation Processing Systems 24: 25th Annual Con-_
_ference on Neural Information Processing Systems_
_2011. Proceedings of a meeting held 12-14 Decem-_
_ber 2011, Granada, Spain, pages 1143–1151._

Ethan Perez, Douwe Kiela, and Kyunghyun Cho.
2021. True few-shot learning with language models.
_ArXiv, abs/2105.11447._

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish
Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
[Gretchen Krueger, and Ilya Sutskever. 2021. Learn-](http://proceedings.mlr.press/v139/radford21a.html)
[ing transferable visual models from natural lan-](http://proceedings.mlr.press/v139/radford21a.html)
[guage supervision. In Proceedings of the 38th In-](http://proceedings.mlr.press/v139/radford21a.html)
_ternational Conference on Machine Learning, ICML_
_2021, 18-24 July 2021, Virtual Event, volume 139 of_
_Proceedings of Machine Learning Research, pages_
8748–8763. PMLR.

Anna Rohrbach, Marcus Rohrbach, Ronghang Hu,
Trevor Darrell, and Bernt Schiele. 2016. Grounding of textual phrases in images by reconstruction.
_ECCV._

Candace Ross, Boris Katz, and Andrei Barbu. 2021.

[Measuring social biases in grounded vision and lan-](https://doi.org/10.18653/v1/2021.naacl-main.78)
[guage embeddings. In Proceedings of the 2021 Con-](https://doi.org/10.18653/v1/2021.naacl-main.78)
_ference of the North American Chapter of the Asso-_
_ciation for Computational Linguistics: Human Lan-_
_guage Technologies, pages 998–1008, Online. Asso-_
ciation for Computational Linguistics.

[Arka Sadhu, Kan Chen, and Ram Nevatia. 2019. Zero-](https://doi.org/10.1109/ICCV.2019.00479)
[shot grounding of objects from natural language](https://doi.org/10.1109/ICCV.2019.00479)
[queries. In 2019 IEEE/CVF International Confer-](https://doi.org/10.1109/ICCV.2019.00479)
_ence on Computer Vision, ICCV 2019, Seoul, Ko-_
_rea (South), October 27 - November 2, 2019, pages_
4693–4702. IEEE.

Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh,
[and Dhruv Batra. 2017. Grad-cam: Visual explana-](https://doi.org/10.1109/ICCV.2017.74)
[tions from deep networks via gradient-based local-](https://doi.org/10.1109/ICCV.2017.74)
[ization. In IEEE International Conference on Com-](https://doi.org/10.1109/ICCV.2017.74)
_puter Vision, ICCV 2017, Venice, Italy, October 22-_
_29, 2017, pages 618–626. IEEE Computer Society._


-----

Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng,
Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun.
2019. Objects365: [A large-scale, high-quality](https://doi.org/10.1109/ICCV.2019.00852)
[dataset for object detection. In 2019 IEEE/CVF In-](https://doi.org/10.1109/ICCV.2019.00852)
_ternational Conference on Computer Vision, ICCV_
_2019, Seoul, Korea (South), October 27 - November_
_2, 2019, pages 8429–8438. IEEE._

Piyush Sharma, Nan Ding, Sebastian Goodman, and
Radu Soricut. 2018. [Conceptual captions:](https://doi.org/10.18653/v1/P18-1238) A
[cleaned, hypernymed, image alt-text dataset for au-](https://doi.org/10.18653/v1/P18-1238)
[tomatic image captioning.](https://doi.org/10.18653/v1/P18-1238) In Proceedings of the
_56th Annual Meeting of the Association for Compu-_
_tational Linguistics (Volume 1: Long Papers), pages_
2556–2565, Melbourne, Australia. Association for
Computational Linguistics.

Sheng Shen, Liunian Harold Li, Hao Tan, Mohit
Bansal, Anna Rohrbach, Kai-Wei Chang, Zhewei
Yao, and Kurt Keutzer. 2021. How much can
clip benefit vision-and-language tasks? _ArXiv,_
abs/2107.06383.

Mohit Shridhar, Dixant Mittal, and David Hsu. 2020.
Ingress: Interactive visual grounding of referring expressions. The International Journal of Robotics Re_search, 39:217 – 232._

Tejas Srinivasan and Yonatan Bisk. 2021. [Worst](https://arxiv.org/abs/2104.08666)
[of both worlds: Biases compound in pre-trained](https://arxiv.org/abs/2104.08666)
[vision-and-language](https://arxiv.org/abs/2104.08666) models. _ArXiv_ _preprint,_
abs/2104.08666.

Mingjie Sun, Jimin Xiao, Eng Gee Lim, Si Liu, and
John Yannis Goulermas. 2021. Discriminative triad
matching and reconstruction for weakly referring expression grounding. IEEE Transactions on Pattern
_Analysis and Machine Intelligence, 43:4189–4195._

[Hao Tan and Mohit Bansal. 2019. LXMERT: Learning](https://doi.org/10.18653/v1/D19-1514)
[cross-modality encoder representations from trans-](https://doi.org/10.18653/v1/D19-1514)
[formers. In Proceedings of the 2019 Conference on](https://doi.org/10.18653/v1/D19-1514)
_Empirical Methods in Natural Language Processing_
_and the 9th International Joint Conference on Natu-_
_ral Language Processing (EMNLP-IJCNLP), pages_
5100–5111, Hong Kong, China. Association for
Computational Linguistics.

Mikihiro Tanaka, Takayuki Itamochi, Kenichi Narioka, Ikuro Sato, Yoshitaka Ushiku, and Tatsuya
[Harada. 2019. Generating easy-to-understand refer-](https://doi.org/10.1109/ICCV.2019.00589)
[ring expressions for target identifications. In 2019](https://doi.org/10.1109/ICCV.2019.00589)
_IEEE/CVF International Conference on Computer_
_Vision, ICCV 2019, Seoul, Korea (South), October_
_27 - November 2, 2019, pages 5793–5802. IEEE._

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
[Kaiser, and Illia Polosukhin. 2017. Attention is all](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
[you need. In Advances in Neural Information Pro-](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
_cessing Systems 30: Annual Conference on Neural_
_Information Processing Systems 2017, December 4-_
_9, 2017, Long Beach, CA, USA, pages 5998–6008._


Ke-Jyun Wang, Yun-Hsuan Liu, Hung-Ting Su, JenWei Wang, Yu-Siang Wang, Winston Hsu, and Wen[Chin Chen. 2021. OCID-ref: A 3D robotic dataset](https://doi.org/10.18653/v1/2021.naacl-main.419)
[with embodied language for clutter scene grounding.](https://doi.org/10.18653/v1/2021.naacl-main.419)
In Proceedings of the 2021 Conference of the North
_American Chapter of the Association for Computa-_
_tional Linguistics: Human Language Technologies,_
pages 5333–5338, Online. Association for Computational Linguistics.

Nevan Wichers, Dilek Z. Hakkani-Tür, and Jindong
Chen. 2018. Resolving referring expressions in images with labeled elements. 2018 IEEE Spoken Lan_guage Technology Workshop (SLT), pages 800–806._

Yuan Yao, Ao Zhang, Zhengyan Zhang, Zhiyuan Liu,
Tat-Seng Chua, and Maosong Sun. 2021. Cpt: Colorful prompt tuning for pre-trained vision-language
models. ArXiv, abs/2109.11797.

Licheng Yu, Zhe Lin, Xiaohui Shen, Jimei Yang, Xin
[Lu, Mohit Bansal, and Tamara L. Berg. 2018. Mat-](https://doi.org/10.1109/CVPR.2018.00142)
[tnet: Modular attention network for referring expres-](https://doi.org/10.1109/CVPR.2018.00142)
[sion comprehension. In 2018 IEEE Conference on](https://doi.org/10.1109/CVPR.2018.00142)
_Computer Vision and Pattern Recognition, CVPR_
_2018, Salt Lake City, UT, USA, June 18-22, 2018,_
pages 1307–1315. IEEE Computer Society.

Licheng Yu, Patrick Poirson, Shan Yang, Alexander C.
Berg, and Tamara L. Berg. 2016. Modeling context
in referring expressions. ECCV.

Rowan Zellers, Ximing Lu, Jack Hessel, Youngjae Yu,
Jae Sung Park, Jize Cao, Ali Farhadi, and Yejin Choi.
2021. Merlot: Multimodal neural script knowledge
models. NeurIPS.

Hanwang Zhang, Yulei Niu, and Shih-Fu Chang. 2018.

[Grounding referring expressions in images by vari-](https://doi.org/10.1109/CVPR.2018.00437)
[ational context. In 2018 IEEE Conference on Com-](https://doi.org/10.1109/CVPR.2018.00437)
_puter Vision and Pattern Recognition, CVPR 2018,_
_Salt Lake City, UT, USA, June 18-22, 2018, pages_
4158–4166. IEEE Computer Society.

Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei
Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. 2021. Vinvl: Revisiting visual representations in vision-language models. 2021 IEEE/CVF
_Conference on Computer Vision and Pattern Recog-_
_nition (CVPR), pages 5575–5584._

Zhu Zhang, Zhou Zhao, Zhijie Lin, Jieming Zhu,
[and Xiuqiang He. 2020. Counterfactual contrastive](https://proceedings.neurips.cc/paper/2020/hash/d27b95cac4c27feb850aaa4070cc4675-Abstract.html)
learning for [weakly-supervised](https://proceedings.neurips.cc/paper/2020/hash/d27b95cac4c27feb850aaa4070cc4675-Abstract.html) vision-language
[grounding. In Advances in Neural Information Pro-](https://proceedings.neurips.cc/paper/2020/hash/d27b95cac4c27feb850aaa4070cc4675-Abstract.html)
_cessing Systems 33: Annual Conference on Neu-_
_ral Information Processing Systems 2020, NeurIPS_
_2020, December 6-12, 2020, virtual._

Xingyi Zhou, Vladlen Koltun, and Philipp Krähenbühl. 2021. Simple multi-dataset detection. ArXiv,
abs/2102.13086.


-----

Figure 5: The visual representation of a proposal using CPT-adapted. The example is taken from the RefCOCOg validation set.

Figure 6: An example of isolating proposals by blurring the remainder of the image using σ = 100

### A Visualization of Region-Scoring Methods

**A.1** **Colorful Prompt Tuning (CPT)**

Figure 5 shows an example of the visual representation of a proposal using CPT-adapted.

**A.2** **Isolated Proposal Scoring (IPS)**

Figure 6 shows the blurred versions of the proposals for an image using σ = 100.

### B Synthetic Spatial Reasoning Experiment

Figure 7 gives an example of the text-pairs version
of the synthetic tasks.
Figure 8 gives an example of the image-pairs
version of the synthetic tasks.


Figure 7: Example image for the synthetic text-pair
tasks. For the spatial task, the text pair corresponding
to this image is “a yellow cube is in front of a blue
cube.” (correct) and “a yellow cube is behind a blue
cube.” (incorrect). For the non-spatial (control) task,
the text pair corresponding to this image is “a blue cube
and a yellow cube” (correct) and “a blue cube and a yellow sphere” (incorrect).

(a) “a blue cube to the left of a yellow cube.”

(b) “a blue cube and a yellow cube”

Figure 8: Examples of the image-pairs version of the
spatial (8a) and non-spatial (8b) tasks. In each case,
the left image is the correct one.

### C Semantic Formalism

**C.1** **Relation Semantics**

We use deterministic heuristics to compute the semantics of the following six relations: left, right,
_above, below, bigger, and smaller. On the other_
hand, we treat inside as a random variable, and use
heuristics to compute the value of its parameter.
For R _left, right, above, below_, we compute
_∈{_ _}_
_R(i, j) by checking whether R holds between the_
center point of box i and box j. For example, if the
center point of i is to the left of the center point of
box j, then left(i, j) = 1.
We compute bigger(i, j) and smaller(i, j) simply by comparing the areas of boxes i and j. For
example, bigger(i, j) checks that the area of box i


-----

is greater than the area of box j.
Finally, for R = inside, we parameterize r(i, j)
as the ratio between the are of the intersection of
boxes i, j compared to the area of box i. Thus,
unlike the other six deterministic rules, inside is
modeled as a random variable.

**C.2** **Relation Extraction**

We identify noun chunks in the dependency parse
as predicates. We then extract relations by looking
for dependency paths between the heads of noun
chunks that contain the following keywords:

  - left: “left”, “west”

  - right: “right”, “east”

  - above: “above”, “north”, “top”, “back”, “behind”

  - below: “below”, “south”, “under”, “front”

  - bigger: “bigger”, “larger”, “closer”

  - smaller: “smaller”, “tinier”, “further”

  - inside: “inside”, “within”, “contained”

We extract superlative relations by looking for dependency paths off the head of a noun chunk containing the following keywords:

  - left: “left”, “west”, “leftmost”, “western”

  - right: “right”, “rightmost”, “east”, “eastern”

  - above: “above”, “north”, “top”

  - below: “below”, “south”, “underneath”,
“front”

  - bigger: “bigger”, “biggest”, “larger”,
“largest”, “closer”, “closest”

  - smaller: “smaller”, “smallest”, “tinier”, “tiniest”, “further”, “furthest”

### D Description of ALBEF

The ALBEF model has an image-only transformer
and a text-only transformer like CLIP but also has
a multi-modal transformer that operates on the outputs of these two transformers. ALBEF is pretrained with three losses: (1) an image-text contrastive (ITC) loss that works just like CLIP’s and
uses the outputs of the image-only and text-only
transformers, (2) an image-text matching (ITM)


Text-pair Text-pair Image-pair Image-pair
Model
Spatial Non-spatial Spatial Non-spatial

ALBEF ITM 49.83 92.20 53.74 90.75
ALBEF ITC 49.83 85.42 51.54 72.25

Table 6: Accuracy on CLEVR image-text matching task. ALBEF performs well on the non-spatial version of the task but
poorly on the spatial version. Text-pair tasks have 295 instances each; image-pair tasks have 227 instances each.

loss–where the task is to decide whether a given
image-text pair match–which uses the outputs of
the multi-modal encoder, and (3) a masked language modeling loss which uses the outputs of the
multi-modal encoder. We explore both the ITC and
ITM scores in our experiments. ALBEF was pretrained on roughly 15M image-caption pairs from
conceptual captions (Sharma et al., 2018), SBU
Captions (Ordonez et al., 2011), COCO (Lin et al.,
2014), and Visual Genome (Krishna et al., 2016).[10]

**D.1** **ALBEF Performance on Synthetic**
**Spatial Reasoning Experiment**

Table 6 shows the zero-shot accuracy of ALBEF
ITM and ITC in the synthetic spatial reasoning
experiment described in §3.2.

### E Implementation Details

**E.1** **Text prompt**

For ALBEF, we pass the input expression directly
to the model, whereas for CLIP, when using GradCAM and ReCLIP (with or without relations), we
use the prefix “a photo of” following the authors’
observations (Radford et al., 2021). For CPT, the
prompt is given in § 2.3.

**E.2** **Position embeddings**

Both CLIP and ALBEF use fixed-size position embeddings, so either the input image must be resized
to fit the dimensions of the embeddings or the size
of the embeddings must be changed. For all models, we resize the image to match the model’s visual input resolution. Resizing of images is done
via bicubic interpolation. Figure 9 shows the how
the performance of the GradCAM method varies
between resizing images and resizing embeddings–
for CLIP RN50x16, there is very little difference,
while for CLIP ViT-B/32 image resizing makes a
larger difference.

10As noted by the ALBEF authors, validation/test images
of RefCOCO+ and RefCOCOg are included in the training set
of COCO.


-----

Figure 9: CLIP RN50x16 and ViT-B/32 Performance
using GradCAM on RefCOCOg validation set comparing resizing of images with resizing of position embeddings, across 10 values of α. These results use groundtruth proposals.

**E.3** **GradCAM Layer**

For CLIP ViT-B/32, we use the last layer of
the visual transformer for GradCAM. For CLIP
RN50x16, we use output of layer 4 for GradCAM.
For ALBEF ITM, we use the third layer of the
multi-modal transformer for GradCAM (following
Li et al. (2021)). For ALBEF ITC, we use the final
layer of the visual transformer for GradCAM.

**E.4** **Hyperparameter sensitivity**

Figure 9 shows the sensitivity of the GradCAM
method to α for the two CLIP models. We choose
_α = 0.5 for all models (including ALBEF), which_
results in the best accuracy for almost models.
For ViT-B/32, α = 0.6 yields slightly higher accuracy by (0.1%) on the RefCOCOg validation
set. Figure 10 shows the sensitivity of the IPS
method to the blur standard deviation σ for the
CLIP RN50x16 model. As shown, the method has
little sensitivity to σ above σ = 20.

**E.5** **Experimentation on validation set**

As discussed by Perez et al. (2021), research on the
zero-shot setting often uses labeled data for model
selection. Aside from variants of IPS documented
in our ablation study (§4.6), we also experimented
on the RefCOCOg validation set (and to a lesser
extent on the RefCOCO+ validation set) with:

1. Drawing a rectangle around the proposal and
using an appropriate text prompt. Performance was somewhat similar to CPT performance.


Figure 10: CLIP RN50x16 Performance using IPS on
RefCOCOg validation set for different values of blur
standard deviation σ. These results use ground-truth
proposals.

2. Ensembling the original text prompt with a
text prompt having only the noun chunk of
the expression containing the head word. This
helped for IPS and is in a sense part of our
rule-based relation-handling.

3. Other techniques for handling superlatives.
For instance, we tried to compute Pr[PN (i) ∧
�
_j≠_ _i[(][¬][P][N]_ [(][j][)][ ∨] [(][P][N] [(][j][)][ ∧] _[R][(][i, j][)))]][. This]_
performed worse than our chosen technique
on the RefCOCOg validation set.

4. Invoking the parser and relation-handling
pipeline on all sentences rather than only those
containing one of the relation/superlative keywords.

We also selected the relation types and keywords
based on these validation sets. Most of these preliminary experiments were performed using the
area threshold mentioned in §4.3.

**E.6** **Description of Computing Infrastructure**

We primarily used a machine with Quadro RTX
8000 GPUs, Google Cloud machines with V100
GPUs, and a machine with TITAN RTX and
GeForce 2080s. These machines used Ubuntu as
the operating system.

**E.7** **Dataset Information**

All datasets that we use are focused on English.
The COCO dataset can be downloaded from
[https://cocodataset.org/#download.](https://cocodataset.org/#download)
The RefCOCO/g/+ datasets can be downloaded from [https://github.com/](https://github.com/lichengunc/refer/tree/master/data)


-----

[lichengunc/refer/tree/master/data.](https://github.com/lichengunc/refer/tree/master/data)
The RefGTA dataset can be downloaded
from [https://github.com/mikittt/](https://github.com/mikittt/easy-to-understand-REG/tree/master/pyutils/refer2)
[easy-to-understand-REG/tree/](https://github.com/mikittt/easy-to-understand-REG/tree/master/pyutils/refer2)
[master/pyutils/refer2. The RefCOCOg](https://github.com/mikittt/easy-to-understand-REG/tree/master/pyutils/refer2)
validation set has 4896 instances, the RefCOCOg
test set has 9602 instances, the RefCOCO+
validation set has 10758 instances, the RefCOCO+
TestA set has 5726 instances, the RefCOCO+
TestB set has 4889 instances, the RefCOCO
validation set has 10834 instances, the RefCOCO
TestA set has 5657 instances, the RefCOCO TestB
set has 5095 instances, the RefGTA validation set
has 17766 instances, and the RefGTA test set has
17646 instances.

### F Qualitative Examples

Figure 12 shows qualitative examples for the RefGTA validation set. Figure 11 shows examples of
the execution errors mentioned in the error analysis
in Section 4.6.

### G Additional Experiment Results

This section presents the full results on the
RefCOCOg/RefCOCO+/RefCOCO datasets, including results without ensembling using CLIP
RN50x16 and ViT-B/32 models and results using
ground-truth proposals. Table 7 shows full results
on the RefCOCOg and RefCOCO+ datasets. Table 8 shows full results on the RefCOCO dataset.


(a) bus behind bus

(b) person behind the fence

(c) chair under dog

(d) smallest train

Figure 11: Examples of execution errors causing ReCLIP to answer incorrectly on instances that it answers
correctly when not using the relation-handling method.
Parts 11a and 11b show cases where the meaning of
“behind” does not match our heuristic, which checks
which proposal’s y-coordinate is smaller. Part 11c

shows an example where “under” means “directly under.” Part 11d shows an example in which due to the
superlative “smallest,” the size of proposals appears to
be weighted more heavily by our approach than scores
CLIP assigns to the proposals based on the text.


-----

RefCOCOg RefCOCO+
**Model** _V alg_ _V ald_ _Testg_ _Testd_ _V alg_ _V ald_ _TestAg_ _TestAd_ _TestBg_ _TestBd_

Random 20.18 18.117 20.34 19.10 16.73 16.29 12.57 13.57 22.13 19.60

UNITER-L (supervised; Chen et al. (2020)) 87.85 74.86 87.73 75.77 84.25 75.90 86.34 81.45 79.75 75.77
MDETR (supervised; Kamath et al. (2021)) – 83.35 – 81.64 – 81.13 – 85.52 – 72.96

Weakly supervised (non-pretrained; Sun et al. (2021)) – – – – 39.18 38.91 40.01 39.91 38.08 37.09

CPT-Blk w/ VinVL (Yao et al., 2021) – 32.1 – 32.3 – 25.4 – 25.0 – 27.0
CPT-Seg w/ VinVL (Yao et al., 2021) – 36.7 – 36.5 – 31.9 – 35.2 – 28.8

**CLIP RN50x16**
CPT-adapted 27.74 25.04 28.81 25.92 24.48 22.09 20.22 19.54 27.80 25.57
GradCAM 54.51 48.35 53.71 47.50 **48.29** **44.53** **52.86** **52.78** 41.13 35.67
ReCLIP w/o relations 62.50 55.88 62.03 54.33 47.12 44.15 46.47 45.97 49.62 41.79
ReCLIP **64.79** **57.66** **64.39** **56.37** **47.92** **44.53** 46.38 45.88 **50.89** **42.87**

**CLIP ViT-B/32**
CPT-adapted 24.16 21.77 24.70 22.78 25.07 23.46 22.28 21.73 28.68 26.32
GradCAM 54.00 49.51 54.01 48.53 48.00 44.64 **52.13** **50.73** 43.85 39.01
ReCLIP w/o relations 62.38 55.35 61.76 54.33 48.53 44.96 50.16 48.24 47.29 41.71
ReCLIP w/o relations **65.48** **56.96** **64.38** **56.15** **49.20** **45.34** 50.23 48.45 **48.58** **42.71**

**CLIP Ensemble**
CPT-adapted 25.96 22.32 25.87 23.65 25.44 23.85 22.00 21.55 28.74 25.92
GradCAM 56.82 50.86 56.15 49.70 51.10 47.83 **57.79** **56.92** 43.24 37.70
ReCLIP w/o relations 65.32 57.70 65.59 57.19 51.54 47.43 51.80 50.02 50.85 43.85
ReCLIP **68.08** **59.33** **67.05** **59.01** **52.12** **47.87** 51.61 50.10 **52.03** **45.10**

Table 7: Accuracy on the RefCOCOg and RefCOCO+ datasets. ReCLIP outperforms other zero-shot methods on RefCOCOg.
On RefCOCO+, ReCLIP is roughly on par with GradCAM but has lower variance between TestA and TestB, which correspond
to different kinds of objects. Subscript g indicates ground-truth proposals are used, and d indicates detected proposals are used.
Best zero-shot results for each model and each column are in bold. See Table 2 for results using object size prior.

RefCOCO
**Model** _V alg_ _V ald_ _TestAg_ _TestAd_ _TestBg_ _TestBd_

Random 16.37 15.73 12.45 13.51 21.32 19.20

UNITER-L (supervised; Chen et al. (2020)) 91.84 81.41 92.65 87.04 91.19 74.17
MDETR (supervised; Kamath et al. (2021)) – 87.51 – 90.40 – 82.67

Weakly supervised (non-pretrained; Sun et al. (2021)) 39.21 38.35 41.14 39.51 37.72 37.01

CPT-Blk w/ VinVL (Yao et al., 2021) – 26.9 – 27.5 – 27.4
CPT-Seg w/ VinVL (Yao et al., 2021) – 32.2 – 36.1 – 30.3

**CLIP RN50x16**
CPT-adapted 23.31 21.48 19.25 18.56 28.36 25.28
GradCAM 44.00 40.49 **47.41** **46.51** 38.17 33.66
ReCLIP w/o relations 40.62 37.61 39.08 38.39 43.55 37.17
ReCLIP **45.94** **41.53** 41.24 40.78 **52.64** **45.55**

**CLIP ViT-B/32**
CPT-adapted 25.12 23.79 23.39 22.87 28.42 26.03
GradCAM 45.41 42.29 **50.13** **49.04** 41.47 36.68
ReCLIP w/o relations 44.37 40.58 45.09 43.98 43.42 37.63
ReCLIP **49.69** **45.77** 48.08 46.99 **52.50** **45.24**

**CLIP Ensemble**
CPT-adapted 24.79 23.16 21.62 21.44 28.89 26.95
GradCAM 46.68 42.85 **51.99** **51.07** 40.10 35.21
ReCLIP w/o relations 45.66 41.97 45.13 43.42 45.40 39.02
ReCLIP **50.51** **45.78** 47.11 46.10 **54.94** **47.07**

Table 8: Accuracy on the RefCOCO dataset. Subscript g indicates ground-truth proposals are used, and d indicates detected
proposals are used. Best zero-shot results for each model and each column are in bold. See Table 2 for results using object size
prior.

|Model V alg V ald Testg Testd|V alg V ald TestAg TestAd TestBg TestBd|
|---|---|
|Random 20.18 18.117 20.34 19.10|16.73 16.29 12.57 13.57 22.13 19.60|
|UNITER-L (supervised; Chen et al. (2020)) 87.85 74.86 87.73 75.77 MDETR (supervised; Kamath et al. (2021)) – 83.35 – 81.64|84.25 75.90 86.34 81.45 79.75 75.77 – 81.13 – 85.52 – 72.96|
|Weakly supervised (non-pretrained; Sun et al. (2021)) – – – –|39.18 38.91 40.01 39.91 38.08 37.09|
|CPT-Blk w/ VinVL (Yao et al., 2021) – 32.1 – 32.3 CPT-Seg w/ VinVL (Yao et al., 2021) – 36.7 – 36.5|– 25.4 – 25.0 – 27.0 – 31.9 – 35.2 – 28.8|
|CLIP RN50x16 CPT-adapted 27.74 25.04 28.81 25.92 GradCAM 54.51 48.35 53.71 47.50 ReCLIP w/o relations 62.50 55.88 62.03 54.33 ReCLIP 64.79 57.66 64.39 56.37|24.48 22.09 20.22 19.54 27.80 25.57 48.29 44.53 52.86 52.78 41.13 35.67 47.12 44.15 46.47 45.97 49.62 41.79 47.92 44.53 46.38 45.88 50.89 42.87|
|CLIP ViT-B/32 CPT-adapted 24.16 21.77 24.70 22.78 GradCAM 54.00 49.51 54.01 48.53 ReCLIP w/o relations 62.38 55.35 61.76 54.33 ReCLIP w/o relations 65.48 56.96 64.38 56.15|25.07 23.46 22.28 21.73 28.68 26.32 48.00 44.64 52.13 50.73 43.85 39.01 48.53 44.96 50.16 48.24 47.29 41.71 49.20 45.34 50.23 48.45 48.58 42.71|
|CLIP Ensemble CPT-adapted 25.96 22.32 25.87 23.65 GradCAM 56.82 50.86 56.15 49.70 ReCLIP w/o relations 65.32 57.70 65.59 57.19 ReCLIP 68.08 59.33 67.05 59.01|25.44 23.85 22.00 21.55 28.74 25.92 51.10 47.83 57.79 56.92 43.24 37.70 51.54 47.43 51.80 50.02 50.85 43.85 52.12 47.87 51.61 50.10 52.03 45.10|


-----

(a) a man in white shorts and white jacket, walking down
(b) a man in white jumpsuit with face mask walking.
a sidewalk.

(c) an african american woman with light colored sweater,
(d) woman in blue shirt in doorway.
brown pants walking down sidewalk near another woman.

(e) a man with yellow helmet behind the fence. (f) a bald black man is walking wearing a tan suit.

(g) a man in all black walking in front of another man. [(h) a man wearing a short-sleeved black top walks by a]
black car.

(j) a man in a blue polo and brown shorts talking on a cell
(i) a woman in a white top.
phone.

Figure 12: Qualitative examples randomly sampled from the RefGTA validation set. Ground-truth referents are


-----

