---
title: Markov Sequence Model
layout: single
date: 2026-05-10 18:02:14 -0400
# categories: jekyll update
tags: NLP
show_date: true
show_tags: true
permalink: /_posts/markovmodel/
---

A sequence model is a machine learning model that captures patterns in sequential or chronological data and uses them to make predictions.

Examples of sequences include:

- Characters in a word
- Words in a sentence
- Notes in music
- DNA sequence
- Stock prices over time

For example:

```text
h → e → l → l → o
```

The next character depends on previous characters.

A sequence model tries to learn:

> “Given previous elements, what is the next element likely to be?”

---

### Markov Models

A Markov Model is one of the simplest sequence models.

It assumes:

> The future depends only on a limited number of previous states.

This assumption is called the Markov Property.

---

### Understanding the Markov Property

Suppose we are generating text.

In a:

#### 1-Order Markov Model

The next character depends on only 1 previous character.

Example:

```text
P(h | t)
```

> Probability of `h` occurring after `t`

---

#### 2-Order Markov Model

The next character depends on 2 previous characters.

```text
P(l | he)
```

> Probability of `l` occurring after `he`

---

### Learned Probabilities

If we train on many English words, we may learn:

```text
P(h | t) = 0.4
P(e | h) = 0.7
P(l | he) = 0.8
```

- after `t`, `h` appears often
- after `h`, `e` appears often
- after `he`, `l` appears often

---

## My Experiment

I built a simple N-order Markov Model by counting how often character sequences appear in the dataset and constructing probability tables for different context sizes. I then compared how the model’s performance changed as the order of the Markov Model increased.

---

<details markdown="1">
<summary>Code</summary>

```python

import torch

import torch.nn as nn

import matplotlib.pylab as plt

%matplotlib inline




class MarkovModel:
    def __init__(self, n_step, filename):

        self.n_step=n_step

        self.string_to_index = {}

        self.index_to_string = {}

        self.data = open(filename, "r").read().splitlines()

        self.N = torch.tensor([])

        self.N_raw = torch.tensor([])





    def wrangle_data(self):

        chars = sorted(list(set("".join(self.data))))

        self.string_to_index = {s:i+1 for i, s in enumerate(chars)}

        self.string_to_index['.']=0

        self.index_to_string = {i:s for s, i in self.string_to_index.items()}

        n_size = []

        for _ in range(self.n_step):

            n_size.append(len(chars)+1)

            self.N = torch.ones(n_size, dtype=torch.float32)




    def grouped_pairs(self, word):

        return zip(*(word[i:] for i in range(self.n_step)))

    def visualize_N(self):

        plt.figure(figsize=(16,16))

        if self.n_step==2:

            N = self.N

        else:

            N = self.N[0]

            plt.imshow(N, cmap="Blues")



        for i in range(27):

            for j in range(27):

                chstr = self.index_to_string[i]+self.index_to_string[j]

                plt.text(j,i, chstr, ha="center", va="bottom", color='gray')

                plt.text(j,i,int(N[i,j].item()), ha='center', va='top', color='gray')

                plt.axis("off")

    def fill_table(self):

        for w in self.data:

            w = ['.'] + list(w) + ["."]

            for elements in self.grouped_pairs(w):

                # elements = (a,b,c)

                elem =[]

                for ele in elements:

                    elem.append(self.string_to_index[ele])

                    self.N[tuple(elem)]+=1

                    # self.visualize_N()
        # normalize
        self.N_raw = self.N.int()

        self.N = self.N/self.N.sum(axis=-1, keepdim=True)


    def predict_loss(self):

        log_likelihood = 0

        freq = 0

        for w in self.data:

            w = ["."]+list(w)+['.']

            for elements in self.grouped_pairs(w):

                elem = []

                for ele in elements:

                    elem.append(self.string_to_index[ele])

                    prob = self.N[tuple(elem)]

                    logprob = torch.log(prob)

                    freq+=1

                    log_likelihood+=logprob

        negative_log_likelihood = -log_likelihood/freq

        print(f"negative log likelihood = {negative_log_likelihood:.4f}")

        return negative_log_likelihood





    def predict(self, num_outputs=10):

        print("Printing Outputs")

        for _ in range(num_outputs):

            idx = [0 for _ in range(self.n_step-1)]

            out = []

            while True:

                new_idx = torch.multinomial(self.N[tuple(idx)], num_samples=1, replacement=True).item()

                out.append(self.index_to_string[new_idx])

                if new_idx==0:

                    break

                idx.pop(0)

                idx.append(new_idx)

            print("".join(out))

    def __call__(self):

        self.wrangle_data()

        self.fill_table()

        self.predict_loss()

        print("---------------------------------------------")

        self.predict()

        return self.N_raw, self.N
```

</details>

### Bigram Model

```python
bigram_model = MarkovModel(2, "names.txt")
b_N_raw, b_N = bigram_model()
```

### Loss

```text
negative log likelihood = 2.4544
```

### Generated Names

```text
ana.
jeli.
jaut.
marille.
co.
jafoliz.
lerlelynesan.
dre.
denety.
je.
```

---

### Trigram Model

```python
trigram_model = MarkovModel(3, "names.txt")
t_N_raw, t_N = trigram_model()
```

### Loss

```text
negative log likelihood = 2.0927
```

### Generated Names

```text
keah.
frah.
den.
ari.
subee.
qulakierycely.
zane.
rudewhylishuq.
olowaers.
izeth.
```

---

### Four-Gram Model

```python
four_gram = MarkovModel(4, "names.txt")
N = four_gram()
```

### Loss

```text
negative log likelihood = 1.9636
```

### Generated Names

```text
mzza.
pvrtqwgrx.
whor.
wxxew.
ldj.
ngopnbqm.
wjqpgemxmshitley.
atzrhpemberlaqflgfibgnbuqyhwdkzvncnbfbstjiatfmrosey.
jnrv.
g.
```

---

### Understanding Log Likelihood

To evaluate the model, we compute the likelihood of the data.

The likelihood measures:

> How probable is the observed dataset according to the model?

---

### Sequence Probability

Suppose the word is:

```text
emma
```

The probability becomes:

$$
P(e) × P(m|e) × P(m|em) × P(a|mm)
$$

We multiply probabilities of each next character.

---

### Using Log

Multiplying many probabilities creates extremely tiny numbers.

$$
0.1 × 0.01 × 0.05 × 0.001
$$

This quickly approaches zero.

So we use logarithms.

Using log transforms multiplication into addition:

$$
log(ab) = log(a) + log(b)
$$

This is numerically stable and easier to optimize.

---

### Log Likelihood

The total log likelihood becomes:

$$
\log P(Data) = \sum \log P(x_i | context)
$$

A larger log likelihood means:

- model predicts data well
- observed sequences are probable

---

### Negative Log Likelihood (NLL)

Machine learning usually minimizes loss functions.

So instead of maximizing log likelihood, we minimize:

$$
NLL = -(1/N) \sum \log P(x_i | context)
$$

---

### Why Minimizing NLL Works

Suppose:

```text
Correct prediction probability = 0.9
```

Then:

```text
-log(0.9) ≈ 0.105
```

Small loss.

If probability is poor:

```text
Correct prediction probability = 0.01
```

Then:

```text
-log(0.01) ≈ 4.605
```

Huge penalty.

This strongly punishes incorrect predictions.

---

## Observations

As the model order increases:

| Model     | Context Size     | NLL    |
| --------- | ---------------- | ------ |
| Bigram    | 1 previous char  | 2.4544 |
| Trigram   | 2 previous chars | 2.0927 |
| Four-gram | 3 previous chars | 1.9636 |

As the order of the Markov Model increases, the model gets more context for predicting the next character, which improves prediction accuracy on the training data and lowers the negative log likelihood loss(overfitting). However, higher-order models also create many possible character contexts, most of which are rarely or never seen during training. During generation, the model can eventually enter a context that was not observed in the dataset, where the transition probabilities become nearly uniform due to smoothing. At that point, the model begins sampling almost randomly

---
