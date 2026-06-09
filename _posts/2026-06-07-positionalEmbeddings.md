---
title: From Absolute PE to RoPE
layout: single
date: 2026-06-07 18:02:14 -0400
permalink: /_posts/positionalEmbeddings/
categories: jekyll update
tags: NLP CV
show_date: true
show_tags: true
---

In the [Transformer]({{ "/_posts/transformer/" | relative_url }}), I briefly introduced Positional Embeddings and mentioned that they can be learned parameters. While this is true for Absolute Positional Embeddings(ABE), I intentionally skipped over the details at the time. Before diving into Rotary Positional Embeddings (RoPE), it is worth taking a closer look at
Absolute Positional Embeddings(ABE). Understanding the intuition and mathematical properties behind ABE provides a solid foundation for grasping how RoPE extends these ideas to encode relative positional information more effectively.

First lets understand the need for Positional Embedding. We know transformers are position invariant but what does it really mean

Consider the two sentences:

1. **Dog bites man**
2. **Man bites dog**

Both sentences contain the **same words**, but in a **different order**. Humans immediately recognize that they have different meanings because word position matters.

Let's see what happens if we use self-attention **without positional embeddings**.

---

###### Dog Bites Man

###### Query Matrix

$$
Q=
\begin{bmatrix}
1 & 1 & 2 \\
2 & 1 & 1 \\
1 & 2 & 1
\end{bmatrix}
$$

Rows correspond to:

- Row 1 → Dog
- Row 2 → Bites
- Row 3 → Man

###### Key Matrix

$$
K=
\begin{bmatrix}
1 & 2 & 1 \\
0 & 0 & 2 \\
2 & 1 & 0
\end{bmatrix}
$$

Each column corresponds to Dog, Bites and Man

###### Attention Scores

$$
QK^T=
\begin{bmatrix}
5 & 4 & 3 \\
4 & 5 & 4 \\
3 & 3 & 5
\end{bmatrix}
$$

Each entry \(i,j) indicates how strongly token \(i\) attends to token \(j\).

###### Value Matrix

$$
V=
\begin{bmatrix}
1 & 2 & 1 \\
2 & 1 & 1 \\
1 & 1 & 2
\end{bmatrix}
$$

Each row represents Dog, Bites, Man

###### Attention Output

$$
(QK^T)V=
\begin{bmatrix}
16 & 17 & 15 \\
18 & 17 & 17 \\
14 & 14 & 16
\end{bmatrix}
$$

Output representations:

- Dog → \([16,17,15]\)
- Bites → \([18,17,17]\)
- Man → \([14,14,16]\)

---

###### Man Bites Dog

The tokens are now reordered:

- Row 1 → Man
- Row 2 → Bites
- Row 3 → Dog

The Query, Key, and Value matrices are simply permuted versions of the previous ones.

###### Query Matrix

$$
Q=
\begin{bmatrix}
1 & 2 & 1 \\
2 & 1 & 1 \\
1 & 1 & 2
\end{bmatrix}
$$

###### Key Matrix

$$
K=
\begin{bmatrix}
1 & 1 & 2 \\
2 & 0 & 0 \\
0 & 1 & 2
\end{bmatrix}
$$

###### Attention Scores

$$
QK^T=
\begin{bmatrix}
5 & 3 & 3 \\
4 & 5 & 4 \\
3 & 4 & 5
\end{bmatrix}
$$

###### Value Matrix

$$
V=
\begin{bmatrix}
1 & 1 & 2 \\
2 & 1 & 1 \\
1 & 2 & 1
\end{bmatrix}
$$

###### Attention Output

$$
(QK^T)V=
\begin{bmatrix}
14 & 14 & 16 \\
18 & 17 & 17 \\
16 & 17 & 15
\end{bmatrix}
$$

Output representations:

- Man → \([14,14,16]\)
- Bites → \([18,17,17]\)
- Dog → \([16,17,15]\)

---

###### Comparing the Outputs

Output from **Dog Bites Man**:

| Token | Output     |
| ----- | ---------- |
| Dog   | [16,17,15] |
| Bites | [18,17,17] |
| Man   | [14,14,16] |

Output from **Man Bites Dog**:

| Token | Output     |
| ----- | ---------- |
| Man   | [14,14,16] |
| Bites | [18,17,17] |
| Dog   | [16,17,15] |

Notice that the **same output vectors appear in both cases**.

- Dog always receives the representation \([16,17,15]\)
- Bites always receives the representation \([18,17,17]\)
- Man always receives the representation \([14,14,16]\)

regardless of where these words occur in the sentence.

###### The Crucial Observation

**The attention relationships themselves remain unchanged when the tokens are permuted**.

Consider the attention score matrix for **"Dog Bites Man"**:

$$
\begin{bmatrix}
5 & 4 & 3 \\
4 & 5 & 4 \\
3 & 3 & 5
\end{bmatrix}
$$

The third row corresponds to **Man** attending to:

- Dog
- Bites
- Man

giving the attention pattern:

$$
[3,\,3,\,5]
$$

Now consider **"Man Bites Dog"**:

$$
\begin{bmatrix}
5 & 3 & 3 \\
4 & 5 & 4 \\
3 & 4 & 5
\end{bmatrix}
$$

The first row now corresponds to **Man** attending to:

- Man
- Bites
- Dog

giving the attention pattern:

$$
[5,\,3,\,3]
$$

At first glance these rows look different, but notice that they contain the **same relationships**. The only difference is the ordering of the columns because the tokens themselves have been reordered.

So **Man attends to Dog, Bites, and itself in exactly the same way regardless of where Man appears in the sentence**.

The same property holds for Dog and Bites as well.

This reveals the fundamental limitation of attention without positional embedding:

> The attention mechanism is aware of **which tokens exist**, but it is not aware of **where they occur**.

When the input tokens are permuted, the attention matrix is merely permuted in the same way. The relationships between the words remain unchanged. Consequently, the model treats:

- Dog bites man
- Man bites dog

as the same collection of tokens arranged differently, rather than as two sentences with different meanings.

This is precisely why Transformers require **positional embedding**: they inject information about token order so that the attention scores change when the positions of words change.

##### Absolute Positional Embeddings

So how do we add Position Embeddings to the sentences. In the Transformer is All You Need paper, the authors introduced a fixed positional embeddings matrix size same as input embeddings and were summed together

Lets see how these fixed positional embeddings are calculated.
Suppose T,C are the sequence lengths and the channel dimensions

For a position `pos` and channel dimension index `c`:

$$
PE(pos, 2c)
=
\sin\left(\frac{pos}{10000^{\frac{2c}{d\_{model}}}}\right)
$$

$$

PE(pos, 2c+1)
=
\cos\left(\frac{pos}{10000^{\frac{2c}{d\_{model}}}}\right)
$$

where:

- $pos$ = position of the token in the sequence
- $c$ = channel index
- $d\_{model}$ = channel size of the model

The even dimensions use the sine function, while the odd dimensions use the cosine function.

For example:

$$
PE(pos) =
\begin{bmatrix}
\sin\left(\frac{pos}{10000^{0/d*{model}}}\right) \\
\cos\left(\frac{pos}{10000^{0/d*{model}}}\right) \\
\sin\left(\frac{pos}{10000^{2/d*{model}}}\right) \\
\cos\left(\frac{pos}{10000^{2/d*{model}}}\right) \\
\vdots
\end{bmatrix}
$$

This generates a unique position vector for every token position, allowing the Transformer to distinguish between different word orders.

###### Lets dive deeper into Sinusoidal Embeddings

We know that sine and cosine are periodic functions, meaning their values repeat after a fixed interval.

In sinusoidal positional embedding, each channel dimension is assigned a different frequency. The frequency is controlled by the denominator

$$
10000^{\frac{2i}{d\_{\text{model}}}}
$$

As the dimension index \(i\) increases, this denominator becomes larger, causing the angle to change more slowly.

As a result:

- **Lower dimensions** small $i$ have **high-frequency** sine and cosine waves that oscillate rapidly.
- **Higher dimensions** large $i$ have **low-frequency** waves that change much more gradually.

This creates positional signals at multiple scales:

- Early dimensions capture fine-grained differences between nearby positions.
- Later dimensions capture broader, long-range positional relationships.

Consequently, every position is represented by a unique combination of high-frequency and low-frequency sinusoidal patterns.

Lets look at the frequency for a channel along all the sequence

Channel = 2

![FrozenLake board]({{ "assets/images/recentllm/2freq.png" | relative_url }})

---

Channel = 64

![FrozenLake board]({{ "assets/images/recentllm/64freq.png" | relative_url }})

---

Channel = 128

![FrozenLake board]({{ "assets/images/recentllm/128freq.png" | relative_url }})

---

Channel = 256

![FrozenLake board]({{ "assets/images/recentllm/256freq.png" | relative_url }})

---

The graphs above show the positional embedding values for different channel dimensions across the sequence positions.

Notice that the earlier dimensions oscillate much more rapidly. As we move even a small distance along the sequence, the positional embedding values change significantly. This means these dimensions are highly sensitive to small positional differences between nearby tokens.

In contrast, the later dimensions have much lower frequencies and therefore change much more gradually. Moving a few positions along the sequence produces only a small change in their values. These dimensions capture broader positional patterns over longer ranges.

Together, these dimensions provide positional information at multiple scales, allowing the Transformer to distinguish both nearby and distant token positions.

###### The Problem with Absolute Positional Embeddings

Consider these two sentences:

- I told her to do well in exams
- Yesterday, I told her to do well in exams

Although the second sentence contains an additional word at the beginning, the relationship between **I** and **told**, **told** and **her**, and **do** and **well** remains unchanged.

However, with absolute positional embeddings, every word after "Yesterday" is shifted to a new position:

| Word  | Sentence 1 Position | Sentence 2 Position |
| ----- | ------------------- | ------------------- |
| I     | 0                   | 1                   |
| told  | 1                   | 2                   |
| her   | 2                   | 3                   |
| do    | 4                   | 5                   |
| exams | 7                   | 8                   |

As a result, the model sees completely different positional embeddings even though the underlying sentence structure is nearly identical.

Ideally, the model should recognize that:

- **I** is still one token before **told**
- **told** is still one token before **her**
- **well** is still one token before **in**
- **in** is still one token before **exams**

In other words, what matters most is often the **relative distance between tokens**, not their absolute positions within the sequence and with Absolute Positional Embeddings, these relative distance between tokens are not explicitly mentioned.

Furthermore, in the vanilla Transformer, positional embeddings are added directly to token embeddings:

$$
x_i = E(w_i) + P(i)
$$

where:

- $E(w_i)$ is the token embedding
- $P(i)$ is the positional embedding

As a result, the semantic information of the token and the positional information are combined into a single vector before any attention computation takes place.

This means the model no longer receives a "pure" representation of the token. Instead, it receives a mixture of:

- **what the token means**
- **where the token occurs**

Although this approach works well in practice, it forces semantic and positional information to share the same vector space.

This observation motivated later approaches such as **Relative Position Bias**, and **RoPE**, which incorporate positional information directly into the attention mechanism rather than adding it to the token embeddings themselves.

<details markdown="1">
<summary>Code</summary>

```python
class PositionalEmbeddings(nn.Module):

    """
    Transformers do not have understanding of positions.
    So we add positions vector to it.
    (B,S,C) => (B,S,C)
    """

    def __init__(self, seq_len, embdim):
        super().__init__()

        self.dropout = nn.Dropout(0.2)

        # self.pe is a lookup matrix of shape (S, E), self.pe[i] represents ith seq
        self.pe = torch.zeros(seq_len, embdim)

        # positions is just the sequence of the position from 0,seq_len, shape => (SEQ_LEN, 1)
        positions = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        # SHAPE(256)
        emb_skip_dim = torch.arange(0, embdim, step=2, dtype=torch.float32)
        # (seqlen, 1) / (256) => (seqlen, 256)
        z = positions / (10000 ** (emb_skip_dim / embdim))

        self.pe[:, 0::2] = torch.sin(z)
        self.pe[:, 1::2] = torch.cos(z)

        # if you want we can unsqueeze to add B dim in self.pe if not it will broadcast
        # in Vanilla Transformer, Positional embedding are not learnt
        # self.pe = nn.Parameter(self.pe, requires_grad=True)

    def forward(self, x):
        # x shape=> (B,T,C)
        B,T,C = x.shape
        # only adding upto T (Seq len)
        x = x + self.pe[:T, :]
        return x
```

</details>
## Rotary Positional Embeddings (RoPE)

We saw that Absolute Positional Embeddings assign every token a position-specific vector and add it directly to the token embedding.

Although this works, it introduces two problems:

1. Position information is mixed with semantic information before attention is computed.
2. Relative distances between tokens are not explicitly encoded.

What we would really like is a mechanism where the attention score depends on how far apart two tokens are rather than where they are located in the sequence.

For two tokens at positions $m$ and $n$, we would like the attention score to behave like

$$
g(x_m,x_n,m-n)
$$

instead of

$$
g(x_m,x_n,m,n).
$$

In other words, the model should care about the distance between tokens, not their absolute positions.

---

### A Different Idea: Rotate Instead of Add

Absolute positional embeddings inject position by **adding** a vector:

$$
x_i = E(w_i) + P(i).
$$

RoPE takes a completely different approach.

Instead of modifying the token embedding itself, RoPE modifies the **query** and **key** vectors used inside attention.

The idea is simple:

> Assign every position a rotation angle and rotate the query and key vectors by that angle.

Tokens appearing at different positions will therefore have different orientations in space.

The attention score can then depend on the difference between those rotations.

---

### Why Rotation?

Suppose we have a 2D vector

$$
v=
\begin{bmatrix}
x\
y
\end{bmatrix}.
$$

If we rotate it by an angle $$theta$$, we obtain

$$
R(\theta)v
=

\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
x\
y
\end{bmatrix}.
$$

Now imagine two identical vectors.

If one is rotated by $$30^\circ$$ and the other by $$60^\circ$$, their relationship depends only on the angle difference

$$
60^\circ-30^\circ = 30^\circ.
$$

This is exactly the behavior we want for positional information.

If position $m$ rotates a vector by $m\theta$ and position $n$ rotates another vector by $n\theta$, then their interaction should naturally depend on

$$
m\theta-n\theta.
$$

This immediately hints at relative position.

---

### Why Use Complex Numbers?

We could rotate vectors using rotation matrices directly.

However, Transformers perform these operations billions of times.

Using explicit rotation matrices would be expensive.

Complex numbers provide a much more elegant representation.

Recall Euler's formula:

$$
e^{i\phi}
=

\cos\phi+i\sin\phi.
$$

A complex number of unit magnitude represents a point on the unit circle.

Multiplying by

$$
e^{i\phi}
$$

rotates a vector by angle $$\phi$$.

Instead of performing a matrix multiplication for every rotation, we can perform a single complex multiplication.

This makes the mathematics cleaner and the implementation more efficient.

---

### Encoding Position Through Rotation

Consider a query vector $q$ and a key vector $k$.

For a token at position $m$, RoPE rotates the query by

$$
e^{im\theta}.
$$

Similarly, a token at position $n$ rotates the key by

$$
e^{in\theta}.
$$

The rotated vectors become

$$
q_m = q,e^{im\theta}
$$

and

$$
k_n = k,e^{in\theta}.
$$

Notice that the rotation angle grows linearly with position.

Every position therefore corresponds to a unique rotation.

---

### Where Relative Position Appears

Attention computes an inner product between queries and keys.

In the complex domain this becomes

$$
q_m k_n^*
$$

Substituting the rotated vectors gives

$$
(qe^{im\theta})
\left(k e^{i n \theta}\right)^{*}
$$

Using the conjugation rule

$$
(ab)^* = a^* b^*
$$

we obtain

$$
(qe^{im\theta})
(k^*e^{-in\theta})
$$

Rearranging terms,

$$
qk^*
e^{im\theta}
e^{-in\theta}
$$

Combining the exponentials,

$$
qk^*
e^{i(m-n)\theta}
$$

This is the key result.

The positional term depends only on

$$
m-n
$$

The absolute positions $m$ and $n$ have disappeared.

Only their relative distance remains.

This is precisely the behavior we wanted.

###### Extending to High Dimensions

A Transformer does not work with a single complex number.

Instead, RoPE groups every pair of dimensions into a complex value.

For example,

$$
[x_0,x_1,x_2,x_3]
$$

is interpreted as

$$
(x_0+ix_1,;
x_2+ix_3).
$$

Each pair receives its own rotation frequency.

Just like ABE , RoPE uses frequencies of the form:

$$
\frac{1}
{10000^{\frac{2i}{D}}}.
$$

- Lower dimensions rotate more rapidly.

- Higher dimensions rotate more slowly.

As a result, some dimensions capture local positional relationships while others capture long-range relationships.

###### The Main Idea

Absolute positional embeddings add position information to token embeddings.

RoPE injects position directly into attention by rotating queries and keys.

Although each token is rotated using its absolute position, the attention interaction becomes

$$
\operatorname{Real}
\left[
(W_qx_m)
(W_kx_n)^*
e^{i(m-n)\theta}
\right].
$$

The attention score therefore depends naturally on relative distance.

This elegant property is the reason RoPE has become the positional encoding method used in most modern large language models.
