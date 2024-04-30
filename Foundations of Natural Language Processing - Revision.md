# Ambiguity and Corpora
The main difficulties of NLP are [[#Variability]] and [[#Ambiguity]], two different problems that arise between the relations between the *meaning* of a sentence and the *language* used in a sentence. Ideally, we'd want one-to-one, but this is rarely the case. Instead, we usually get a one-to-many or many-to-one.
## Variability
Variability when one sentence meaning can have multiple language interpretations. For example, the sentence:
 
 > He drew the house
 
could refer to:
- He made a sketch of the house
- *He showed me* his drawing of the house
- He portrayed the house in his paintings
- He drafted the house *in his sketchbook*
## Ambiguity
Ambiguity (arguably the more pressing challenge in NLP) is when the language used in a sentence can have *multiple different **valid** meanings*. There are many different types of ambiguity:
- Homophones: *blew* and *blue*
- Word senses: *bank* (finance or river?)
- Part of speech: *chair* (noun or verb?)
- Syntactic structure: *I saw a girl with a telescope*
- Quantifier scope: *Every child loves some movie*
- Multiple: *I saw her duck*
	- (word senses and syntactic)
- Reference: *John dropped the goblet onto the glass table and it broke.*
- Discourse: *The meeting is cancelled. Nicholas isn't coming to the office today*
- Syntactic Ambiguity: *Put the block in the box on the table in the kitchen*
	- If we have an ambiguous sentence like this with $n$ prepositional phrases, the amount of interpretations is $Cat_{n}$ (*Catalan Numbers*), where $$Cat_{n} = \begin{pmatrix}2n\\n\end{pmatrix} - \begin{pmatrix}2n\\n-1\end{pmatrix} \sim \frac{4^{n}}{n^{3/2}\sqrt{\pi}}$$ 
## Zipf's Law
If we count the amount of words in some [[#Corpora]] to get their *frequency* $f$, and then *rank them in order of most frequent* ($r$), we can plot these values against each other. The result looks something like this:
![[Zipf, unlogged.png]]
This is quite unreadable, so if we use *logarithmic* axes we can see what's going on:
![[Zipf, logged.png]]
This phenomenon is known as ***Zipf's Law***, and is formally defined (in the slides) as $$f \times r \approx k$$, where $k$ is some constant. Most other online sources seem to fix $k=1$ and explicitly write that the *frequency of a word is **inversely proportional** to it's rank in our frequency table*. $$f = \frac{1}{r}$$
The implications of this mean that regardless of how big our corpus is, we will *always* find words we've barely seen before. This proves a problem when we want to find methods of estimating probabilities for *all* words.
## Uncertainty in NLP
Most problems we come across are manifestations of uncertainty
- [[#Ambiguity]] - uncertainty in interpretation
- [[#Variability]] - uncertainty in a specific realisation
- **Robustness** - uncertainty with potential inputs
	- We may have many different types of input (e.g. formal/informal)
- **Context Dependence** - uncertainty with previous information

All of these uncertainties basically force our hand into using *[[#Probabilistic models]]/Machine Learning* 
### Probabilistic models
A probabilistic model has a **inputs** (*set of words+context / utterances*) and **outputs** (*set of Part-of-speech (POS) tags / syntactic analyses*)

| input     | output     |     | input                                                          | output  |
| --------- | ---------- | --- | -------------------------------------------------------------- | ------- |
| $input_1$ | $output_1$ |     | $\textlangle \text{the, } \underline{\text{list}} \textrangle$ |         |
| $input_2$ | $output_2$ |     | $\textlangle \text{We, } \underline{\text{list}} \textrangle$  |         |
| $\dots$   | $\dots$    |     | $\dots$                                                        | $\dots$ |

## Corpora
A *corpus* (plural *corpora*) is a "body of utterances, as words or sentences". Corpora should be naturally occurring to serve as realistic samples of a language. Corpora are also *linguistically [[#Annotation methods|annotated]]*; humans have read this text and marked the structures describing syntax/meaning. 
### Sentiment Analysis
A simple linguistic analysis where we try to tell the sentiment of a given piece of text (good example: a movie review) - is it *positive* or *negative*? This example would be easy to test for accuracy, as we can use the numerical score provided.

A simple sentiment analyser would use a positive and negative "list" (a *sentiment lexicon*) to count the number of positive and negative words. This is a naive approach; words may be ambiguous, may be used in contexts not useful to the overall sentiment
### Tokenisation
We want to split a corpus into separate word/punctuation **tokens** (occurrences), not already separated by spaces. These *tokens* are the individual building blocks for any Language Model to act upon.
# Annotation methods
Annotating is a potentially long and costly job. We may need to consider
- source data (size? Licensing?)
- annotation scheme (complexity? Guidelines)
- annotators (expertise? training?)
- Quality control procedures

We devise a set of *annotation guidelines* to help annotators produce **consistent** data, and to help users interpret the annotations correctly. The *Penn Treebank* has a >300 page long guideline document!
## Inter-annotator agreement
Human annotators are not perfect. Conversely, due to [[#Ambiguity]], there may be multiple possible (equally correct) annotations. *IAA* is the process of getting multiple annotators to "agree" on independently annotated samples. The agreement rate can be thought of as an *upper bound* on the accuracy of a system evaluated on that dataset.

Still, some annotation decisions are far more frequent than others. The **Kappa** coefficient $K$ measures agreement between two people making category judgements, correcting for expected chance agreement.
For example, in a scenario where an item is annotated and 4 coding options are equally likely, then the two annotators will agree 25% of the time. Therefore an agreement of 25% will be assigned $K=0$ and will scale accordingly (e.g. 50% would be $K=0.333$ since 50 is a third of the way from 25 to 100).
%% ## Chance Correction - this never comes up in the slides or J&M? Idk if this is Kappa but under a different name%%
## Gold Standard Evaluation
The gold standard is '*the truth*'; what the original writer *actually* meant when producing the individual text. Gold standards are used for both training and evaluation, but **testing must be done on unseen data**.

When designing a system, we often *tune* it by changing configuration options. If we run several experiments on our "test set", we risk **overfitting** it; this set no longer holds as a reliable *proxy* for new data.
### K-Fold Cross-validation
We often split our dataset into test/train/dev pieces. *Devsets* are sets to work and experiment on; a model *may* overfit these if only used - this is solved by *only training the model on a **training** set*. We then can *test the model* on the ***test set***.

If our model is *too small* to reasonably create sufficiently sized sets, we can use *k-fold cross validation*. This process breaks the data into $k$ pieces and treats one as a held-out set - the remaining are used to train a model. This held out set is used to test these different *folds*. We can then combine all learned information through the use of cross-validation.
# Evaluation methods
The simplest method of measuring a model's *performance* is the **proportion model**: $$\frac{\text{right}}{\text{test set}}\times 100$$
This is okay for some tasks, but not all
## Advanced evaluation methods
We can use the following methods in a system that can produce true/false positives/negatives (a total of 4 possible answers). We use $\text{tp, tn, fp}$ and $\text{fn}$ to represent these  
### Accuracy
The percentage of all observations that were labelled correctly. $$\frac{\text{tp+tn}}{\text{tp+fp+tn+fn}}$$
### Recall
The percentage of items actually present in the input that were correctly identified by the system. $$\frac{\text{tp}}{\text{tp+fn}}$$
### Precision
The percentage of items the system detected *that were actually positive*. $$\frac{\text{tp}}{\text{tp+fp}}$$
### Confusion Matrices
A way of representing all of these different metrics is through use of a *confusion matrix*. ![[A Confusion Matrix.png]]
## F-Score
A combination of [[#Precision]] ($P$) and [[#Recall]] ($R$), the **F-measure** is formally defined as $$F_{\beta} = \frac{(\beta^{2}+1)PR}{\beta^{2}P+R}$$
Values of $\beta > 1$ favour *recall*, and values of $\beta < 1$ favour precision. When $\beta = 1$ they are balanced; this $F_{1}$ metric is the most commonly used and can be more clearly defined as $$F_{1} = \frac{2PR}{P+R}$$
## Significance Testing
If we have a model with 95% accuracy, how can we tell if this is good or bad?

We can use an *upper bound* ([[#Inter-annotator agreement]] rate as discussed earlier) and a lower bound (performance of a "simpler" model) to measure how *significantly* better/worse our model is.

There are two types of significance tests:
-  **Parametric**
	- Used when the underlying distribution is *normal*/*Gaussian*.
- **Non-parametric**
	- Used otherwise
# N-gram LMs
 > *1.5k words into notes and we're finally talking about an actual language model!!*  
 
%%very proud that this metric is actually correct%%

Due to [[#Zipf's Law|the Zipfian curve]] it's very hard to predict possible sentence structures just based on words alone; we need to determine (ideally, what we actually do is *approximate*) the **plausibility** of a sentence. For example, $$
\begin{eqnarray}
P(\text{the cat slept peacefully}) &>& P(\text{slept the peacefully cat}) \\
P(\text{she studies morphosyntax}) &>& P(\text{she studies more faux syntax})
\end{eqnarray}
$$
*N-gram* models are one such language model that approximates this plausibility for us. They can be used for:
- **Spelling correction**
	- Generate possible "correct" spellings for a certain text and pick the best guess
- **Automatic speech recognition**
	- Generate multiple possible text interpretations of speech and pick the best guess
- **Machine translation**
	- Generate multiple possible translations (in the target language) and pick the be- *you get the idea*

N-grams are most well-known for their use in [[#Prediction]]
## Prediction
We want to compute $P(w|h)$, the probability of a word $w$ given a history $h$. For example, if our history is "*its water is so transparent that*" and we want to know the probability of our next word being *the*, we would be finding $$P(\text{the|its water is so transparent that})$$
A method of calculating this probability would be to use *relative frequency counts* $C$: $$
\begin{eqnarray}
P(\text{the|its water is so transparent that}) =\\
\frac{C(\text{its water is so transparent that the})}{C(\text{its water is so transparent that})}
\end{eqnarray}
$$
This is still not sufficient; language is creative and we won't always be able to count entire sentences.

We'll need more clever ways to estimate $P(w|h)$. If we instead change contexts to representing the probability of a sequence of $n$ words $w_{1}, \dots, w_{2}$. If we want to predict the probability of $P(\text{the})$, we can write it as $P(X_{i}=\text{"the"})$. So the probability of a sequence of words can be written both as $$P(X_{1}= w_{1}, X_{2}= w_{2}, \dots, X_{n}= w_{n})$$ and $P(w_{1}, w_{2}, \dots, w_{n})$ or $P(w_{1:n})$.

We can now use the *chain rule* to decompose this probability: $$\begin{eqnarray}
P(w_{1}, \dots, w_{n}) &=& P(w_{1})P(w_{2}|w_{1})P(w_{3}|w_{1:2})\dots P(w_{n}|w_{1:n-1}) \\&=& \displaystyle\prod_{i=1}^n{P(w_{i}|w_{1}, w_{2}, \dots, w_{i-1})}
\end{eqnarray}$$
A **bigram** model ($n=2$), for example, approximates the probability of a word by *only using* the conditional probability of the preceding word; we would approximate the above transparent water example with $$P(\text{the|that})$$
This assumption that we can reasonably estimate the probability of a word based on only the prior is called a **Markov** assumption. N-grams make a Markov assumption that we only need to look $n-1$ words into the past. For n-gram size $N$, $$P(w_{n}|w_{1:n-1})\approx P(w_{n}|w_{n-N+1:n-1})$$
However, given the *bigram assumption for the probability* of an individual word, we can compute the probability of a **complete word sequence** by substituting this into our original equation to get $$P(w_{1:n})\approx \displaystyle\prod^{n}_{k=1}P(w_{k}|w_{k-1})$$
How do we get the individual *bigram probabilities* $P(w_{k}|w_{k-1}$)? We can estimate probabilities with [[#MLE]]
## MLE
We can estimate discrete probabilities with $$P_{RF}=\frac{C(x)}{N}$$ where $C(x)$ is the *count* of $x$ in a large dataset, and $N$ is the total number of items in the dataset.
### Estimating bigram probabilities with MLE
ffffff
## Trigram independence assumption
Under a trigram independence assumption, all 
- $P(\text{mast|I spent three years before the})$
- $P(\text{mast|I went home before the})$
- $P(\text{mast|I saw the sail before the})$
- $P(\text{mast|I revised all week before the})$

are estimated as $P(\text{mast|before the})$. The general rule is $$P_{MLE}(w_{i}|w_{i-2}, w_{i-1})=\frac{C(w_{i-2}, w_{i-1}, w_{i})}{C(w_{i-2}, w_{i-1})}$$
(The amount of times the three words have appeared over the amount of times the previous two words appeared).
# Evaluation of LMs
# Smoothing
# Noisy Channel Model
# Edit Distance
# Expectation Maximisation
# Naive Bayes
# Logistic Regression
# Morphology Parsing
# POS Tagging and HMMs
# Syntax and Parsing
# CYK Algorithm
# Probabilistic Context-Free Grammars
# CYK Pruning
# Parser Evaluation
# Structural Annotation
# Dependency Parsing
# Compositional Semantics
# Discourse Coherence
# Lexical Semantics
# Distributional Semantics
# Neural Embeddings
# Neural Classifiers
# Recurrent Neural Networks
# Neural Language Modelling
# Text Generation and Encoder-Decoder Models
# Attention
# Transformers
# Transfer Learning