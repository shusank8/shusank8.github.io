---
title: Bellman Equation using Policy Iteration
# date: 2026-04-15
# summary: A step-by-step implementation of the Bellman Equation using policy iteration on deterministic and stochastic FrozenLake environments.
# tags: [RL]
layout: single
permalink: /_posts/vanillaPolicyiteration/
date: 2026-04-15 18:02:14 -0400
# categories: jekyll update
tags: RL
show_date: true
show_tags: true
---

In Bellman-based methods, we do not start with a known policy. The idea is to begin with a random policy, evaluate how good that policy is, improve it based on the estimated values, and repeat the process until the policy stops changing.

This is a _model-based reinforcement learning method_, which means the environment dynamics are already known to us. In this example, we first work with a _deterministic environment_ so the logic is easier to understand. After that, we extend the same idea to a _stochastic environment_, where the next state is no longer guaranteed.

---

### Bellman Equation

The Bellman expectation equation for a policy is:

$$
v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma\, v_\pi(s')\bigr]
$$

This equation tells us that the value of a state is the expected return we get by following a policy from that state onward.

- $\pi(a \mid s)$ is the probability of taking action $a$ in state $s$
- $p(s', r \mid s, a)$ is the probability of landing in next state $s'$ and receiving reward $r$ given state $s$ and action $a$
- `gamma` is the discount factor
- $v_\pi(s)$ is the value of state `s` under policy $\pi$

In simple terms, Bellman updates say:

1. Look at what action your policy chooses
2. See what next state and reward that action leads to
3. Use the value of that next state to update the current state

### Initialize Environment (Deterministic)

We start by initializing the FrozenLake environment.

![FrozenLake board]({{ "assets/images/frozenlake.png" | relative_url }})

In the FrozenLake 4x4 environment, the agent's goal is to reach the target and collect the reward, but it must navigate cautiously to avoid falling in the lake.

Here, `is_slippery=False` makes the environment deterministic. That means if the agent chooses to move left, it will definitely move left. This is useful for understanding the mechanics of policy iteration before introducing randomness.

```python
env = gym.make("FrozenLake-v1", map_name="4x4", render_mode="rgb_array", is_slippery=False)
```

---

### Understanding the Environment Dynamics

FrozenLake environment dynamics (MDP) are stored in `env.unwrapped.P`.

For each state and action, it tells us:

- the probability of the transition
- the next state
- the reward received
- whether the episode ends

In the deterministic version, each action has only one possible outcome, so the transition list contains a single entry.

A simplified example looks like this:

```python
# {0: {0: [(1.0, 0, 0, False)],
#   1: [(1.0, 4, 0, False)],
#   2: [(1.0, 1, 0, False)],
#   3: [(1.0, 0, 0, False)]}
```

This means:

- from state `0`
- if you take action `1`
- with probability `1`
- you move to state `4`
- get reward `0`
- and the game is not over

---

### Policy Evaluation (Deterministic)

Policy evaluation answers this question:

**If I keep following the current policy, what is the value of each state?**

This function starts with all state values set to zero and repeatedly updates them until they converge.

What the code does:

- Creates a value table initialized to zeros
- Repeatedly loops through every state
- Reads the action chosen by the current policy
- Looks up the next state and reward from the environment dynamics
- Updates the value of the current state using the Bellman update
- Stops when the changes become very small

Because the environment is deterministic, each state-action pair leads to only one next state, so the update is straightforward.

```python
def policy_evaluation(policy, discount_factor, iterations=1000, tol=1e-10):

    Value = np.zeros(env.observation_space.n)

    for _ in range(iterations):

        Value_next = np.copy(Value)

        for s in range(env.observation_space.n):

            action = policy[s]
            prob, new_state, reward, done = env.unwrapped.P[s][action][0]
            Value[s] = reward + discount_factor * Value_next[new_state]

        if np.max(np.abs(Value - Value_next)) < tol:
            break

    return Value
```

---

### Policy Improvement (Deterministic)

Once we know the value of each state, the next step is to improve the policy.

Policy improvement asks:

**For each state, which action gives the highest expected return?**

This function checks all possible actions at every state, computes their one-step lookahead value, and picks the best one.

What happens here:

- For each state, create a list of action values
- For each possible action:
  - look up the next state and reward
  - compute the Q-value using the current value estimates
- Select the action with the highest Q-value
- Store that action in the new policy

This gives us a better policy than the one we started with.

```python
def policy_improvement(values, gamma=0.99):

    new_policy = np.zeros(env.observation_space.n)

    for s in range(env.observation_space.n):

        q_sa = []
        for action in range(env.action_space.n):

            prob, next_state, reward, terminal = env.unwrapped.P[s][action][0]
            Q_value = reward + gamma * values[next_state]
            q_sa.append(Q_value)

        new_policy[s] = np.argmax(q_sa)

    return new_policy
```

---

### Policy Iteration

Now we combine the previous two steps into one full algorithm.

Policy iteration works like this:

1. Start with a random policy
2. Evaluate that policy
3. Improve the policy
4. Repeat until the policy no longer changes

This function does exactly that.

Important details:

- `np.random.randint(...)` creates a random initial policy
- `policy_evaluation(...)` computes the value function under the current policy
- `policy_improvement(...)` produces a new and better policy
- If the new policy is the same as the old one, the algorithm has converged

At convergence, the policy is optimal for the given environment model.

```python
def policy_iteration(env, num_iterations=1000, gamma=0.99):

    policy = np.random.randint(low=0, high=4, size=(16))

    for _ in range(num_iterations):

        value = policy_evaluation(policy, discount_factor=gamma, iterations=num_iterations)
        new_policy = policy_improvement(value, gamma=gamma)

        if np.all(policy == new_policy):
            print("policy has been converged")
            break

        policy = new_policy

    return policy
```

---

### Switching to Stochastic Environment

After understanding the deterministic case, we move to the stochastic version of FrozenLake.

Here, `is_slippery=True` means the chosen action may not always lead to the intended movement. The agent can slip and end up in a different direction. This makes the environment more realistic and forces us to consider probabilities in Bellman updates.

```python
env = gym.make("FrozenLake-v1", map_name="4x4", render_mode="rgb_array", is_slippery=True)
```

---

### How the Stochastic Dynamics Differ

In a stochastic environment, each action can lead to multiple possible next states, each occurring with a certain probability. In the FrozenLake stochastic environment, actions are not deterministic, when the agent chooses a direction, it only moves in the intended direction with a probability of $1/3$ while the remaining probability leads to movement in unintended directions.

A simplified transition snippet looks like this:

```python
# {0: {0: [(0.333..., 0, 0, False),
#          (0.333..., 0, 0, False),
#          (0.333..., 4, 0, False)],
# ...
# }}
```

Now instead of one guaranteed outcome, an action may have several possible results. That means we must sum over all possible transitions when computing state values or action values.

This is the main difference between the deterministic and stochastic implementations.

---

### Policy Evaluation (Stochastic)

The goal stays the same: estimate the value of each state under the current policy.

The difference is in the Bellman update. Since there are multiple possible next states, we compute the **expected value** by summing over all transitions.

What this version does:

- Loops through each state
- Chooses the action dictated by the current policy
- Reads all possible transitions for that action
- Multiplies each outcome by its probability
- Adds them together to get the expected value

This is the proper Bellman expectation update for stochastic environments.

```python
def policy_evaluation(policy, discount_factor, iterations=1000, tol=1e-10):

    Value = np.zeros(env.observation_space.n)

    for _ in range(iterations):

        Value_next = np.copy(Value)

        for s in range(env.observation_space.n):

            action = policy[s]
            env_trans = env.unwrapped.P[s][action]
            V = 0

            for transition in env_trans:
                prob, next_state, reward, done = transition
                V += prob * (reward + discount_factor * Value_next[next_state])

            Value[s] = V

        if np.max(np.abs(Value - Value_next)) < tol:
            break

    return Value
```

---

### Policy Improvement (Stochastic)

Policy improvement also changes slightly in the stochastic case.

Instead of evaluating a single next state for each action, we now compute the expected return over all possible next states.

This function:

- Tries every action in each state
- Looks at all transitions for that action
- Computes the expected Q-value
- Chooses the action with the maximum expected return

So the overall logic is the same as before, but the Q-value computation now includes probability-weighted outcomes.

```python
def policy_improvement(values, gamma=0.99):

    new_policy = np.zeros(env.observation_space.n)

    for s in range(env.observation_space.n):

        q_sa = []

        for action in range(env.action_space.n):

            env_trans = env.unwrapped.P[s][action]
            V = 0

            for transition in env_trans:
                prob, next_state, reward, done = transition
                V += prob * (reward + gamma * values[next_state])

            q_sa.append(V)

        new_policy[s] = np.argmax(q_sa)

    return new_policy
```

---

### Converged Policy From Deterministic Environment

```
[['Down' 'Right' 'Down' 'Left']
 ['Down' 'Left' 'Down' 'Left']
 ['Right' 'Down' 'Down' 'Left']
 ['Left' 'Right' 'Right' 'Left']]
```

It is quite evident the if we start at state 0 and follow the policy above, it takes us to the reward.

### Converged Policy from Stochastic Environment

```
[['Left' 'Up' 'Up' 'Up']
 ['Left' 'Left' 'Left' 'Left']
 ['Up' 'Down' 'Left' 'Left']
 ['Left' 'Right' 'Down' 'Left']]
```

For the policy above, we need to play games to determine how good the policy is because of stochasticity.

```python
num_games = 1000
win = 0
max_step = 200

for _ in range(num_games):

    nextstate, _ = env.reset()

    for _ in range(max_step):

        action = int(converged_policy[nextstate])
        nextstate, reward, done, truncated, _ = env.step(action)

        if done or truncated:
            if reward > 0:
                win += 1
            break

(win/num_games)*100
# Output = 74.8
```

The agent achieved a **74.8% win rate over 1000 games** which is fair enough cause the actions are not deterministic. When the agent chooses a direction, it only moves in the intended direction with a probability of $1/3$ while the remaining probability leads to movement in unintended directions.

### Main Takeaway

This implementation shows how Bellman-based policy iteration works in two settings:

### Deterministic environment

- Each action leads to exactly one next state
- Value updates are simple and direct

### Stochastic environment

- Each action may lead to several next states
- Value updates must sum over all possible outcomes

The key idea in both cases is the same:

- evaluate the current policy
- improve the policy using the value estimates
- repeat until convergence

That is the core of **policy iteration**.

---

### Summary

- We started with a random policy
- We evaluated that policy using Bellman updates
- We improved the policy by choosing better actions
- We repeated the process until the policy converged
- Then we extended the same logic from deterministic to stochastic transitions

This is one of the most important dynamic programming methods in reinforcement learning because it shows how optimal behavior can be computed when the environment model is known.
