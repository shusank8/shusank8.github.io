---
title: Deriving the Bellman equation
# date: 2026-04-14
# summary: A systematic derivation of the Bellman expectation equation from the definition of return, using marginalization, conditional probability, and the Markov property.
# tags: [RL]
layout: single
date: 2026-04-14 18:02:14 -0400
permalink: /_posts/bellmanequation/
categories: jekyll update
tags: RL
show_date: true
show_tags: true
---

The Bellman equation is a fundamental recursive relationship in reinforcement learning,
expressing the value of a state in terms of immediate rewards and the value of successor
states. This article presents a detailed and systematic derivation of the Bellman
expectation equation starting from the definition of return and leveraging probabilistic
principles such as marginalization, conditional probability, and the Markov property.

---

## Introduction

In reinforcement learning, an agent interacts with an environment modeled as a Markov
Decision Process (MDP). A central quantity of interest is the state-value function, which
evaluates the expected cumulative reward obtained by following a policy $\pi$ from a given
state.

Formally, the value function is defined as:

$$
V_\pi(s) = \mathbb{E}_\pi\!\left[ G_t \mid S_t = s \right]
$$

where $G_t$ is the return, defined as the discounted sum of future rewards:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}, \quad \gamma \in (0, 1]
$$

---

## Recursive structure of the return

Expanding the definition of the return:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Similarly, the return at the next time step is:

$$
G_{t+1} = R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \cdots
$$

Observing the overlap between these two expressions, we obtain the recursive identity:

$$
G_t = R_{t+1} + \gamma\, G_{t+1}
$$

This decomposition is the cornerstone of the Bellman formulation.

---

## Decomposition of the value function

Substituting the recursive expression of $G_t$ into the value function:

$$
V_\pi(s) = \mathbb{E}_\pi\!\left[ R_{t+1} + \gamma\, G_{t+1} \mid S_t = s \right]
$$

Using linearity of expectation:

$$
V_\pi(s) = \mathbb{E}_\pi[R_{t+1} \mid S_t = s] + \gamma\, \mathbb{E}_\pi[G_{t+1} \mid S_t = s]
$$

We now analyze each term separately.

---

## Expected immediate reward

For discrete random variables, expectation is defined as:

$$
\mathbb{E}[X] = \sum_x x\, P(X = x)
$$

Thus:

$$
\mathbb{E}_\pi[R_{t+1} \mid S_t = s] = \sum_r r\, P(R_{t+1} = r \mid S_t = s)
$$

However, the reward depends on the transition dynamics involving the current state $s$,
action $a$, and next state $s'$. Therefore, we un marginalize over these variables:

$$
P(r \mid s) = \sum_{s', a} P(r, s', a \mid s)
$$

Applying the chain rule of probability:

$$
P(r, s', a \mid s) = P(s', r \mid s, a)\, P(a \mid s)
$$

Recognizing that $P(a \mid s) = \pi(a \mid s)$, we obtain:

$$
P(r \mid s) = \sum_{s', a} P(s', r \mid s, a)\, \pi(a \mid s)
$$

Substituting back:

$$
\mathbb{E}_\pi[R_{t+1} \mid S_t = s] = \sum_{r, s', a} r\, P(s', r \mid s, a)\, \pi(a \mid s)
$$

---

## Expected future return

We now consider:

$$
\mathbb{E}_\pi[G_{t+1} \mid S_t = s] = \sum_g g\, P(G_{t+1} = g \mid S_t = s)
$$

To express this in terms of state transitions, we expand the distribution via
marginalization as g depends on Reward(r) and r is a random variable which in turn depends on actions and state(s')

$$
P(g \mid s) = \sum_{a, s', r} P(g, a, s', r \mid s)
$$

Applying conditional probability:

$$
P(g, a, s', r \mid s) = P(g \mid s', r, a, s)\, P(r, s', a \mid s)
$$

At this stage, we invoke the **Markov property**, which asserts that future returns depend
only on the next state:

$$
P(g \mid s', r, a, s) = P(g \mid s')
$$

Thus:

$$
P(g \mid s) = \sum_{a, s', r} P(g \mid s')\, P(s', r \mid s, a)\, \pi(a \mid s)
$$

Substituting into the expectation:

$$
\mathbb{E}_\pi[G_{t+1} \mid S_t = s] = \sum_{a, s', r} \left(\sum_g g\, P(g \mid s')\right) P(s', r \mid s, a)\, \pi(a \mid s)
$$

Recognizing that:

$$
\sum_g g\, P(g \mid s') = V_\pi(s')
$$

we obtain:

$$
\mathbb{E}_\pi[G_{t+1} \mid S_t = s] = \sum_{a, s', r} V_\pi(s')\, P(s', r \mid s, a)\, \pi(a \mid s)
$$

---

## The Bellman expectation equation

Combining the expressions for the immediate reward and future return:

$$
V_\pi(s) = \sum_{r, s', a} r\, P(s', r \mid s, a)\, \pi(a \mid s) + \gamma \sum_{r, s', a} V_\pi(s')\, P(s', r \mid s, a)\, \pi(a \mid s)
$$

Factoring common terms:

$$
V_\pi(s) = \sum_{r, s', a} \bigl[r + \gamma\, V_\pi(s')\bigr]\, P(s', r \mid s, a)\, \pi(a \mid s)
$$

Reordering yields the standard form:

$$
\boxed{V_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} \bigl[r + \gamma\, V_\pi(s')\bigr]\, P(s', r \mid s, a)}
$$
