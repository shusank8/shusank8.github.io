---
title: Online Monte Carlo and TD learning
# date: 2026-04-26
# summary: Making Monte Carlo Efficient and TD learning
# tags: [RL]

layout: single
date: 2026-04-26 18:02:14 -0400
# categories: jekyll update
tags: RL
permalink: /_posts/tdlearning/
show_date: true
show_tags: true
---

In [Monte Carlo]({{ "/_posts/montecarlo/" | relative_url }}), we played multiple episodes, accumulated rewards through out and averaged it. But there is a real uncertainity about the episodic length in real world. What if the episode is near infinite, we have to wait until the agent finishes MULTIPLE episode to update Q value which is not only slow but also has high space complexity.

First we look into Online Monte Carlo, where rather than going through multiple episodes and updating, we update Q value after each episode.

In Monte Carlo,

$$
\begin{aligned}
Q_k &= \frac{1}{k}\sum_{i=1}^{k} G_i \\
&\quad \text{where } k \text{ is the number of episodes}  \\
Q_{k+1} &= \frac{1}{k+1}\sum_{i=1}^{k+1} G_i \\
&= \frac{1}{k+1}(G_1 + G_2 + \cdots + G_k + G_{k+1}) \\
&= \frac{1}{k+1}(kQ_k + G_{k+1}) \\
&= \frac{k}{k+1}Q_k + \frac{1}{k+1}G_{k+1} \\
&= \left(1 - \frac{1}{k+1}\right)Q_k + \frac{1}{k+1}G_{k+1} \\
&= Q_k - \frac{1}{k+1}Q_k + \frac{1}{k+1}G_{k+1} \\
&= Q_k + \frac{1}{k+1}(G_{k+1} - Q_k)
\end{aligned}
$$

We can see how the value of $Q_{k+1}$ depends on $Q_k$ which eliminates the need of storing rewards from $1 \cdots k-1$.

**_PS_** Please follow this [blog post]({{ "/_posts/montecarlo/" | relative_url }}) for full detailed implementation of generalized policy iteration (GPI) but here we will only be looking at the changes while the iterative process remains the same.

```python
def online_monte_carlo_estimation(policy, env, gamma, num_episodes):

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    N = np.zeros((env.observation_space.n, env.action_space.n))

    for _ in range(num_episodes):

        trajectory = sample_trajectory(policy, env)

        episode_returns = compute_rewards(trajectory, gamma)

        for (state, action), G in episode_returns.items():

            Q[state, action] = Q[state, action] + (1/(N[state, action]+1)) * (G-Q[state, action])

            N[state,action]+=1


    return Q
```

### Temporal Difference Learning

TD learning is a combination of [DP]({{ "/_posts/vanillaPolicyiteration/" | relative_url }}) and [Monte Carlo]({{ "/_posts/montecarlo/" | relative_url }}).
In Monte Carlo, agent has to complete a full episode/s to make an update while in TD learning, agent only needs to wait until next time step to make an update as the update is based on the observed Reward and estimate $Q(s_{t+1})$

$$
Q(s_t) = Q(s_t) + \alpha (R_{t+1} + \gamma (Q_{s_{t+1}}) - Q(s_t))
$$

The crux of TD learning is to make agent update Q value after each step rather than after each episode as see in Online Monte Carlo.

### SARSA

SARSA is on-policy TD learning that follows GPI. On-policy method attempts to improve the policy that is used to make decisions whereas off policy method evaluates a policy different from that used to generate the data.
Over the course of time, if a agent follows on-policy method the policy gradually shifts closer and closer to a deterministic suboptimal policy because of the epsilon greedy.

$$
Q(s_t, a_t) = Q(s_t, a_t) + \alpha (R_{t+1} + \gamma (Q_{s_{t+1}, a_{t+1}}) - Q(s_t, a_t))
$$

```python
def sarsa(env, episodes=2000, alpha=0.1, gamma=0.99, epsilon=0.1):

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for _ in range(episodes):

        state, _ = env.reset()

        action = epsilon_greedy(Q, state, epsilon, env)

        done = False

        while not done:

            next_state, reward, done, _, _ = env.step(action)

            next_action = epsilon_greedy(Q, next_state, epsilon, env)

            Q[state, action] = Q[state, action] + alpha*(reward+gamma*Q[next_state, next_action]-Q[state, action])

            state = next_state

            action = next_action

    return Q
```

### Q Learning

Off Policy TD learning is known as Q Learning which is defined as

$$
Q(s_t, a_t) = Q(s_t, a_t) + \alpha (R_{t+1} + \gamma * max (Q_{s_{t+1}}) - Q(s_t, a_t))
$$

Off Policy uses two policies, one(target) that is learned about and that becomes the optimal policy, and one(behavior) that is more exploratory and is used to generate the data. Off Policy methods are of greater variance and slower to converge as well as they are more powerful as they include on policy methods as the special case where target and behavior policies are the same.

```python
def qlearning(env, episodes=2000, alpha=0.1, gamma=0.99, epsilon=0.1):

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for _ in range(episodes):

        state, _ = env.reset()

        action = epsilon_greedy(Q, state, epsilon, env)

        done = False

        while not done:

            next_state, reward, done, _, _ = env.step(action)

            next_action = epsilon_greedy(Q, next_state, epsilon, env)

            Q[state, action] = Q[state, action] + alpha*(reward+gamma*np.max(Q[next_state])-Q[state, action])

            state = next_state

            action = next_action

    return Q
```

### n-step TD Learning

Monte Carlo needs a full episode to compute the reward making it low bias and high variance while TD(0) is high bias and low variance as the Q values are updated after each time step. As both approaches are two ends of the spectrum, n-step TD Learning falls in between balancing both. n-step TD Learning is a variance of TD learning where an agent updates the Q value after n time step.

```python
def n_td_learning(env, episodes=2000, alpha=0.1, gamma=0.99, epsilon=0.1, n=5):

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    episode_rewards = []

    for _ in range(episodes):

        state, _ = env.reset()

        action = epsilon_greedy(Q, state, epsilon, env)

        done = False

        states = []

        actions = []

        rewards = []

        total_reward = 0

        while not done:

            next_state, reward, done, _, _ = env.step(action)

            next_action = epsilon_greedy(Q, next_state, epsilon, env)

            states.append(state)

            actions.append(action)

            rewards.append(reward)

            total_reward +=reward

            if len(rewards)>=n:

                G = sum([gamma**i * reward[i] for i in range(n)])

                if not done:

                    G+= gamma**n * np.max(Q[next_state])

                Q[states[0], actions[0]] = Q[states[0], actions[0]] + alpha*(G-Q[states[0], actions[0]])

                states.pop(0)

                actions.pop(0)

                rewards.pop(0)

            state = next_state

            action = next_action

            episode_rewards.append(total_reward)

        while rewards:

            G=sum([gamma**i * rewards[i] for i in range(len(rewards)) ])

            Q[states[0], actions[0]] += alpha *(G-Q[states[0], actions[0]])

            states.pop(0)

            actions.pop(0)

            rewards.pop(0)

    return Q, episode_rewards
```
