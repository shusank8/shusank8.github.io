---
title: Variational Auto Encoders
layout: single
date: 2026-05-31 18:02:14 -0400
permalink: /_posts/bellmanequation/
categories: jekyll update
tags: CV
show_date: true
show_tags: true
---

We will begin with a brief overview of Autoencoders, discuss their limitations, and then introduce Variational Autoencoders (VAEs) along with the derivation of their loss function.

An Autoencoder is a neural network architecture consisting of two components: an encoder and a decoder. The encoder maps the input data into a lower dimensional representation called the latent space, while the decoder attempts to reconstruct the original data from this latent representation. The network is trained such that the reconstructed output is as close as possible to the input.

Although Autoencoders are effective for dimensionality reduction and data compression, they are not naturally suited for generating new data. This is because the latent space learned by a standard Autoencoder is often irregular, sparse, and unstructured. As a result, randomly sampling points from the latent space may produce unrealistic or meaningless outputs.

To address this limitation, Variational Autoencoders were introduced. VAEs are based on the assumption that the observed data $x$ is generated from an underlying hidden variable $z$, commonly referred to as a latent variable. While we can observe the data $x$, the latent variable $z$ is not directly observable.

From a probabilistic perspective, we assume that the latent variable $z$ is sampled from some prior distribution $p(z)$. Once $z$ is generated, the observed data $x$ is produced according to a conditional distribution $p(x \mid z)$.

Since we only observe $x$ and not the latent variable $z$, the central problem is to determine which latent variables are likely to have generated the observed data. Mathematically, this requires computing the posterior distribution
$p(z \mid x)$

From Bayes' Theorem we know,

$$
p(z \mid x) = \frac{p(x \mid z) p(z)}{p(x)}
$$

where

- $p(z\mid x)$ is posterior,
- $p(x \mid z)$ is likelihood,
- $p(z)$ is prior and
- $p(x)$ is marginal

The difficulty in computing $p(z \mid x)$ arises from the term $p(x)$, known as the evidence or marginal likelihood. Computing $p(x)$ requires integrating over all possible values of the latent variable $z$.

$$
    p(x)=\int p(x \mid z)p(z)dz
$$

For high dimensional latent spaces, this integral is generally intractable and cannot be computed efficiently. Consequently, the exact posterior distribution p(z∣x) is also intractable, making direct inference impossible.

So, we introduce a variational distribution $q_{\phi}(z \mid x)$, parameterized by $\phi$, to approximate the true posterior distribution $p_{\theta}(z \mid x)$, parameterized by $\theta$. Since the true posterior is often intractable, we optimize $\phi$ such that $q_{\phi}(z \mid x)$ closely matches $p_{\theta}(z \mid x)$.

Formally, we seek to minimize the Kullback Leibler (KL) divergence between the two distributions:

$$
D_{\mathrm{KL}}\!\left(q_{\phi}(z \mid x)\,\|\,p_{\theta}(z \mid x)\right)


$$

We Know,

$$
\begin{aligned}
D_{KL}\!\left(q_\phi(z \mid x)\,\|\,p_\theta(z \mid x)\right)
&=
\sum_z q_\phi(z \mid x)
\log
\left(
\frac{q_\phi(z \mid x)}
     {p_\theta(z \mid x)}
\right)\\
&=
\sum_z
q_\phi(z \mid x)
\log
\left(
\frac{q_\phi(z \mid x)p_\theta(x)}
     {p_\theta(x \mid z)p_\theta(z)}
\right) \text{Using Bayes' Rule and $p_\theta(z)$ lacks parameter}\\
&=
\sum_z q_\phi(z \mid x)
\log
\left(
\frac{q_\phi(z \mid x)}
     {p_\theta(x \mid z)p_\theta(z)}
\right)
+
\sum_z q_\phi(z \mid x)\log p_\theta(x)\\
&=
\sum_z q_\phi(z \mid x)
\log
\left(
\frac{q_\phi(z \mid x)}
     {p_\theta(x \mid z)p_\theta(z)}
\right)
+
\log p_\theta(x)
\end{aligned}
$$

$$
\begin{aligned}
\log p_\theta(x)
&=
D_{KL}
-
\sum_z
q_\phi(z \mid x)
\log
\left(
\frac{q_\phi(z \mid x)}
     {p_\theta(x \mid z)p_\theta(z)}
\right)\\
&=
D_{KL}
+
\sum_z
q_\phi(z \mid x)
\log
\left(
\frac{p(x \mid z)p_\theta(z)}
     {q_\phi(z \mid x)}
\right)
\end{aligned}
$$

Since

$$
\log p_\theta(x)
=
D_{KL}\!\left(q_\phi(z \mid x)\,\|\,p_\theta(z \mid x)\right)
+
\text{ELBO},
$$

and

$$
D_{KL}\!\left(q_\phi(z \mid x)\,\|\,p_\theta(z \mid x)\right)\ge 0,
$$

the ELBO is a lower bound on $\log p_\theta(x)$. Therefore, instead of directly maximizing the intractable quantity $\log p_\theta(x)$, we maximize the ELBO. Doing so not only increases the data likelihood but also encourages the approximate posterior $q_\phi(z \mid x)$ to become closer to the true posterior $p_\theta(z\mid x)$, since the KL divergence is minimized when the two distributions are identical.

<details markdown="1">
<summary>$D_{KL} \ge 0$</summary>

$$
\begin {aligned}
D_{KL}\!\left(q\,\|\,p\right) &= \sum_{x} q(x)[ -\log {\frac{p(x)}{q(x)}}]\\
&= \mathbb{E}_{q(x)}[-\log\frac{p(x)}{q(x)}]\\
& \text{ -log is a convex function so applying Jensen Inequality}\\
&\ge -\log(\mathbb{E}_{q(x)}[\frac{p(x)}{q(x)}])\\
& \ge -\log(\sum_{x}q(x)\frac{p(x)}{q(x)})\\
& \ge -\log(1)\\
& \ge 0\\
\end {aligned}
$$

</details>

###### ELBO Loss

$$
\begin{aligned}
&=\sum_z q_\phi(z \mid x)
\log
\left(
\frac{p_\theta(x \mid z)p_\theta(z)}
     {q_\phi(z \mid x)}
\right)\\
&=
\sum_z q_\phi(z \mid x)\log p_\theta(x \mid z)
+
\sum_z q_\phi(z \mid x)
\log
\left(
\frac{p_\theta(z)}
     {q_\phi(z \mid x)}
\right).
\end{aligned}
$$

The first term is the Reconstruction Loss (or expected log-likelihood),

$$
\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)],
$$

which encourages the decoder to accurately reconstruct the observed data $$x$$ from the latent variable $$z$$

The second term is the negative Kullback--Leibler (KL) divergence,

$$
- D_{\mathrm{KL}}\!\left(q_\phi(z \mid x)\,\|\,p_\theta(z)\right),
$$

between the approximate posterior $$q_\phi(z \mid x)$$ and the prior distribution $$p_\theta(z)$$. This term regularizes the latent representation by encouraging $$q_\phi(z \mid x)$$ to remain close to the prior.

While the prior $ p\_\theta(z)$ can, in principle, be any chosen distribution, the standard Variational Autoencoder typically assumes a standard multivariate Gaussian prior:

$$
p_\theta(z) = \mathcal{N}(0, I).
$$

By constraining the latent representations to follow this prior distribution, the model learns a smooth and structured latent space, enabling meaningful interpolation and the generation of new samples.

###### Computing the KL Divergence for Gaussian Latent Variables

Assume

$$
q_\phi(z \mid x)
=
\mathcal N(\mu_q,\sigma_q^2)
$$

and the prior

$$
p_\theta(z)
=
\mathcal N(0,1).
$$

We want

$$
D_{KL}
=
D_{KL}
\!\left(
q_\phi(z \mid x)
\,\|\,p_\theta(z)
\right).
$$

By definition,

$$
D_{KL}
=
\mathbb E_q
\left[
\log \frac{q_\phi(z \mid x)}{p_\theta(z)}
\right]
=
\mathbb E_q[\log q_\phi(z \mid x)]
-
\mathbb E_q[\log p_\theta(z)].
$$

---

###### Compute $ \mathbb E_q[\log q(z \mid x)] $

For a univariate Gaussian,

$$
q_\phi(z \mid x)
=
\frac{1}{\sqrt{2\pi\sigma_q^2}}
\exp
\left(
-\frac{(z-\mu_q)^2}
      {2\sigma_q^2}
\right).
$$

Taking logarithms,

$$
\log q_\phi(z \mid x)
=
-\frac12\log(2\pi)
-\log \sigma_q
-\frac{(z-\mu_q)^2}
       {2\sigma_q^2}.
$$

Taking expectation under \(q\),

$$
\begin{aligned}
\text{As, } \\
&[\mathbb E_q[(z-\mu_q)^2]
=
\sigma_q^2]\\
\text{We get, }\\
\mathbb E_q[\log q_\phi(z \mid x)]
&=
-\frac12\log(2\pi)
-\log \sigma_q
-\frac12\\

\end{aligned}
$$

---

###### Compute $ \mathbb E_q[\log p_\theta(z)] $

For the standard normal prior,

$$
p_\theta(z)
=
\frac{1}{\sqrt{2\pi}}
\exp\!\left(-\frac{z^2}{2}\right),
$$

so

$$
\log p_\theta(z)
=
-\frac12\log(2\pi)
-\frac{z^2}{2}.
$$

Taking expectation,

$$
\mathbb E_q[\log p_\theta(z)]
=
-\frac12\log(2\pi)
-\frac12\mathbb E_q[z^2].
$$

Using

$$
\mathbb E_q[z^2]
=
\mathrm{Var}(z)
+
(\mathbb E[z])^2
=
\sigma_q^2+\mu_q^2,
$$

we get

$$
\mathbb E_q[\log p_\theta(z)]
=
-\frac12\log(2\pi)
-\frac12(\sigma_q^2+\mu_q^2).
$$

---

###### Final KL expression

$$
\begin{aligned}
D_{KL}
&=
\mathbb E_q[\log q(z \mid x)]
-
\mathbb E_q[\log p(z)]\\
&=\log \sigma_q \frac12 + \frac12(\sigma_q^2+\mu_q^2).
\end{aligned}
$$

Therefore,

$$
\boxed{
D_{KL}
\!\left(
\mathcal N(\mu_q,\sigma_q^2)
\,\|\,\mathcal N(0,1)
\right)
=
\frac12
\left(
\mu_q^2
+
\sigma_q^2
-
1
-
\log \sigma_q^2
\right)
}
$$

which is the KL term used in Variational Autoencoders.

---

#### Reparameterization Trick

In a VAE, the encoder predicts the parameters of a Gaussian distribution:

$
q_\phi(z|x) = N(\mu_q, \sigma^2_q)
$

We then want to sample a latent variable:

$
z \sim N(\mu_q, \sigma^2_q)
$

A straightforward implementation is:

```
z = torch.normal(mean=mu, std=std)
```

However, this creates a problem.

Since torch.normal() performs a random sampling operation, gradients cannot flow through the sampling step during backpropagation.

```python
import torch
import torch.nn as nn

x = torch.randn(1)
param_mu = torch.randn(1, 1, requires_grad=True)
param_logvar = torch.randn(1, 1,requires_grad=True)

mu = x @ param_mu
std = torch.exp(0.5*(x @ param_logvar))

z = torch.normal(mean=mu, std=std)

z.backward()

>param_logvar.grad = 0 and param_mu.grad = 0
```

The computation graph is broken at the sampling operation.

So, Instead of sampling directly from $ N(\mu, \sigma^2)$ we sample from a standard normal distribution $ \epsilon \sim N(0,1) $ and construct:

$$
z = \mu + \sigma \epsilon
$$

Now the randomness is contained in $\epsilon$, while $z$ becomes a differentiable function of $\mu$ and $\sigma$

```python

x = torch.randn(1)
param_mu = torch.randn(1, 1, requires_grad=True)
param_logvar = torch.randn(1, 1,requires_grad=True)

mu = x @ param_mu
std = torch.exp(0.5*(x @ param_logvar))
normal = torch.rand_like(std)
z = mu + normal * std

z.backward()

>param_logvar.grad = -0.3634 and param_mu.grad = -0.9655

```


