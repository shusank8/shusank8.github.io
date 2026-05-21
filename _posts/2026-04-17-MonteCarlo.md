---
title: Monte Carlo Policy Iteration in CliffWalking
# date: 2026-04-17
# summary: Implementation of Monte Carlo policy iteration on the CliffWalking environment, including trajectory sampling, return estimation, policy improvement, and evaluation.
# tags: [RL]

layout: single
date: 2026-04-17 18:02:14 -0400

permalink: /_posts/montecarlo/
# categories: je
tags: RL
show_date: true
show_tags: true
---

> Recommended to go through [Implementation of Bellman Equation]({{ "/_posts/vanillapolicyiteration/" | relative_url }}) first, as some terms here (like stochastic environment) are assumed to be already known.

We move from [Dynamic Programming]({{ "/_posts/vanillapolicyiteration/" | relative_url }}) to Monte Carlo method. Unlike Bellman-based policy iteration, here we are not given the full MDP dynamics we use for updates. Instead, we learn by sampling trajectories from interaction with the environment.

The environment used here is CliffWalking-v1 stochastic environment

![CliffWalking board]({{ "assets/images/CliffWalking.png" | relative_url }})

---

### Understanding the Environment

The environment is a 4 x 12 gridworld.

- The agent starts at the bottom-left corner
- The goal is at the bottom-right corner
- The cells between them on the bottom row are the cliff
- If the agent steps into the cliff, it receives a large negative reward and goes to the starting state

### Rewards and penalties

- Each normal step gives a reward of -1
- Falling into the cliff gives a reward of -100
- The episode ends when the agent reaches the goal

---

### Initialize the Environment

We initialize the CliffWalking stochastic environment.

This environment has four actions:

- `0`: move left
- `1`: move down
- `2`: move right
- `3`: move up

The state is represented as a single integer corresponding to the agent's location in the grid.

For example:

- the start state is `36`, which corresponds to row `3`, column `0`
- the goal state is `47`, which corresponds to row `3`, column `11`

```python
env = gym.make("CliffWalking-v1", render_mode="rgb_array", is_slippery=True)
```

---

### Environment Description

Before training, it helps to understand what the environment returns.

### Observation space

There are `3 x 12 + 1` possible states available to the agent.

The agent cannot remain inside the cliff cells, and the goal ends the episode, so the state space includes:

- the first 3 full rows
- the starting bottom-left cell

Each state is encoded as:

`state = current_row * ncols + current_col`

So if the agent is at row `3` and column `0`, the state is:

`3 * 12 + 0 = 36`

### Starting state

The episode begins in state `36`.

### Reward structure

- Normal move: `-1`
- Cliff: `-100`

### Episode termination

The episode ends when the agent reaches the goal state `47`.

---

### Sampling a Trajectory

Because Monte Carlo methods learn from complete episodes, we need to generate a trajectory by following the policy inside the environment.

What happens here:

- Resets the environment to the start state
- Follows the current policy until the episode ends
- Uses epsilon-greedy exploration
- Stores each experience as a tuple:
  - current state
  - action
  - reward
  - next state
  - done flag

This gives us one full episode that we can later use to compute returns.

### Why epsilon is used

If we always follow the current policy exactly, the agent may never explore better actions.
So with probability `epsilon`, we take a random action. Otherwise, we follow the policy.

```python
def sample_trajectory(policy, env, epsilon=0.1):

    done = False
    trajectory = []
    state, _ = env.reset()

    while not done:

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = policy[state]

        next_state, reward, done, truncated, _ = env.step(action)
        experience = (state, action, reward, next_state, done)
        trajectory.append(experience)
        state = next_state

    return trajectory
```

---

### Computing Returns from a Trajectory

Once we have a full trajectory, we need to compute the return for each state-action pair.

The return is:

$G = r_t + \gamma * r_{t+1} + \gamma^2 * r_{t+2} + ..$

This function walks backward through the trajectory to compute those returns.

What happens here:

- Starts from the end of the episode
- Computes discounted return moving backward
- Stores the return at every time step
- Uses first-visit Monte Carlo
- Only keeps the first occurrence of each state-action pair in the episode

This is important because in first-visit Monte Carlo, we only use the first time a state-action pair appears in an episode.

```python
def compute_rewards(trajectory, gamma):

    G = 0
    returns = {}

    for t in reversed(range(len(trajectory))):
        state, action, reward, next_state, done = trajectory[t]
        G = reward + gamma * G
        returns[(state, action)]=G

    return returns
```

---

### Estimating the Action-Value Function

Now we combine trajectory sampling with return averaging to estimate the Q-function.

This function runs many episodes, collects first-visit returns, and averages them.

What happens here:

- Creates a Q-table with one value per state-action pair
- Creates a dictionary to store all observed returns for each `(state, action)`
- Repeats for many episodes:
  - sample a trajectory
  - compute first-visit returns
  - store the return for each visited state-action pair
- Finally, compute the average return for each pair

That average becomes the Monte Carlo estimate of `Q(s, a)`.

```python
def monte_carlo_estimation(policy, env, gamma, num_episodes):

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    returns = {(s, a): [] for s in range(env.observation_space.n) for a in range(env.action_space.n)}

    for _ in range(num_episodes):
        trajectory = sample_trajectory(policy, env)
        episode_returns = compute_rewards(trajectory, gamma)
        for (state, action), G in episode_returns.items():
            returns[(state, action)].append(G)

    for (state, action), G_list in returns.items():
        if len(G_list) > 0:
            Q[state, action] = np.mean(G_list)

    return Q
```

---

### Policy Improvement

Once we have the Q-values, policy improvement becomes simple.

For every state, we choose the action with the highest estimated Q-value.

This makes the policy greedy with respect to the current Monte Carlo estimate.

```python
def policy_imporvement(Q):
    return np.argmax(Q, axis=-1)
```

---

### Monte Carlo Policy Iteration

Now we put everything together.

This function performs the full Monte Carlo policy iteration loop:

1. Start with a random policy
2. Estimate Q-values from sampled episodes
3. Improve the policy greedily
4. Repeat until the policy changes very little

```python
def mc_policy_iteration(env, gamma=0.99, num_episodes=20000):

    policy = np.random.choice(env.action_space.n, size=(env.observation_space.n))

    while True:
        Q = monte_carlo_estimation(policy, env, gamma, num_episodes)
        new_policy = policy_imporvement(Q)
        diff = sum(abs(policy - new_policy))
        print(diff)

        if diff < 3:
            break

        policy = new_policy

    return policy, Q
```

---

### Running the Algorithm

Now we run the full Monte Carlo policy iteration procedure.

This returns:

- the learned policy
- the estimated Q-table

```python
policy, Q = mc_policy_iteration(env)
```

---

### Evaluating the Learned Policy

After training, lets evaluate the policy.

This evaluation loop plays many games using the learned policy and counts how often the agent succeeds.

What happens here:

- Runs the environment for many episodes
- Always follows the learned policy
- Tracks total reward
- Stops if the episode finishes or gets truncated
- Counts a run as successful if the total return is better than falling into the cliff immediately

The condition `R > -90` is being used as a rough way to identify successful runs.

```python
num_games = 10000
win = 0
max_step = 100

for _ in range(num_games):

    nextstate, _ = env.reset()
    R = 0

    for st in range(max_step):
        action = int(policy[nextstate])
        nextstate, reward, done, truncated, _ = env.step(action)
        R += reward

        if done:
            if R > -90:
                win += 1
            break
```

---

### Final Result

The agent achieved a **72.5% win rate over 10000 games** which is fair enough cause of the stochastic environment. When the agent chooses a direction, it only moves in the intended direction with a probability of $1/3$ while the remaining probability leads to movement in unintended directions.

---

### Main Takeaway

This shows how Monte Carlo control works when we do not directly rely on full model-based Bellman updates.

The key flow is:

- sample trajectories
- compute returns from complete episodes
- estimate Q-values from those returns
- improve the policy greedily
- repeat until the policy stabilizes

Unlike dynamic programming methods, Monte Carlo methods learn from experience rather than directly from transition equations.

---

### Summary

- We used the CliffWalking stochastic environment, where the goal is to reach the destination without falling into the cliff
- We sampled full trajectories using epsilon-greedy exploration
- We computed first-visit Monte Carlo returns
- We estimated `Q(s, a)` by averaging returns
- We improved the policy using greedy action selection
- We evaluated the learned policy over many games
