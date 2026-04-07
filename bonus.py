import gymnasium as gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random
import time
from collections import deque
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.models import Sequential
from cpprb import PrioritizedReplayBuffer


### Hyperparameters
gamma = 0.99
lr = 0.001
batch_size = 32
epsilon_initial = 0.95
target_update_frequency = 50


def build_network():
    model = Sequential([
        Input(shape=(2,)),
        Dense(64, activation='relu', kernel_initializer=HeNormal()),
        Dense(32, activation='relu', kernel_initializer=HeNormal()),
        Dense(3,  activation='linear', kernel_initializer=HeNormal())
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model


### Epsilon-greedy
def epsilon_greedy(states,epsilon,main_dqn):
    prob = np.random.rand()

    if prob < epsilon:
        return np.random.randint(3)
    else:
        states = np.array(states)
        states = np.reshape(states, (1, -1))
        q_values = main_dqn(states, training=False).numpy()
        return np.argmax(q_values[0])


### Training
def train_dqn(replay_buffer, main_dqn, target_dqn, beta, rho = 1):

    if replay_buffer.get_stored_size() < batch_size:
        return

    for j in range(rho):

        batch = replay_buffer.sample(batch_size, beta = beta)
        states      = batch["obs"]
        actions     = batch["act"].astype(int)
        rewards     = batch["rew"].reshape(-1)
        next_states = batch["next_obs"]
        status      = batch["done"].reshape(-1)

        weights = batch["weights"]
        indexes = batch["indexes"]

        tar_q_values = target_dqn(next_states, training=False).numpy()
        max_tar_q_values = np.max(tar_q_values, axis=1).reshape(-1)

        targets = rewards + (1 - status) * gamma * max_tar_q_values

        q_values = main_dqn(states, training = False).numpy()

        td_errors = targets - q_values[np.arange(batch_size), actions]

        for i in range(batch_size):
            q_values[i][actions[i]] = targets[i]

        main_dqn.fit(states,q_values, sample_weight = weights.reshape(-1,), verbose = 0, epochs = 1)

        replay_buffer.update_priorities(indexes, np.abs(td_errors) + 1e-6)


num_episodes = 60


def run_single_seed(seed, truncation_length, rho = 1):

    main_dqn   = build_network()
    target_dqn = build_network()
    target_dqn.set_weights(main_dqn.get_weights())

    env_dict = {
        "obs": {"shape": (2,)},
        "act": {},
        "rew": {},
        "next_obs": {"shape": (2,)},
        "done": {}
    }

    replay_buffer = PrioritizedReplayBuffer(20000, env_dict, alpha = 0.6)

    env     = gym.make("MountainCar-v0", max_episode_steps=truncation_length)
    epsilon = epsilon_initial
    epsilon_decay = (epsilon_initial - 0.05) * 1.25 / num_episodes

    episode_returns = []

    for episode in range(num_episodes):
        observation, info = env.reset(seed=seed)

        timesteps      = 0
        episode_return = 0
        terminated     = False
        truncated      = False

        while not terminated and not truncated:

            action = epsilon_greedy(observation, epsilon, main_dqn)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.add(
                obs=observation,
                act=action,
                rew=reward,
                next_obs=next_observation,
                done=float(done)
            )

            timesteps += 1

            if timesteps % 4 == 0:
                beta = min(1.0, 0.4 + episode * (1.0 - 0.4) / num_episodes)
                train_dqn(replay_buffer, main_dqn, target_dqn, beta, rho = rho)

            if timesteps % target_update_frequency == 0:
                target_dqn.set_weights(main_dqn.get_weights())

            observation     = next_observation
            episode_return += reward

        epsilon = max(epsilon - epsilon_decay, 0.05)
        episode_returns.append(episode_return)

        print(f"  Seed {seed} | Episode {episode+1} | Return {episode_return:.1f}")

    env.close()
    return episode_returns

all_returns = []
seeds = list(range(15))
truncation_length = 2000
rho = 1

for seed in seeds:
    print(f"\n Running Seed {seed} | With PER | Truncation: {truncation_length} | rho : {rho}")
    seed_returns = run_single_seed(seed, truncation_length, rho)
    all_returns.append(seed_returns)

all_returns = np.array(all_returns)
np.save("returns_rho1_PER.npy", all_returns)

"""
Also run the above code again with rho 4 and save the file as returns_rho4_PER.npy
And keep the returns_2000.npy and returns_rho4.npy files also in the same folder.
Only then the below plots will work
"""


# Plots

def compute_ci(data):
    mean = np.mean(data, axis=0)
    ci = stats.sem(data, axis=0) * stats.t.ppf(0.975, df=data.shape[0]-1)
    return mean, mean-ci, mean+ci


files = {
    "ρ=1 (Uniform)": "returns_2000.npy",
    "ρ=1 (PER)": "returns_rho1_PER.npy",
    "ρ=4 (Uniform)": "returns_rho4.npy",
    "ρ=4 (PER)": "returns_rho4_PER.npy",
}


# Learning curves
plt.figure(figsize=(9,5))

for label in files:
    data = np.load(files[label])
    m, lo, hi = compute_ci(data)

    plt.plot(m, label=label)
    plt.fill_between(range(len(m)), lo, hi, alpha=0.15)

plt.title("PER vs Uniform")
plt.xlabel("Episodes")
plt.ylabel("Return")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# Aggregate plot
labels = list(files.keys())
pos = np.arange(len(labels))

means = []
cis = []

for label in labels:
    data = np.load(files[label])
    per_run = np.mean(data, axis=1)

    m = np.mean(per_run)
    ci = 1.96 * np.std(per_run) / np.sqrt(len(per_run))

    means.append(m)
    cis.append(ci)

plt.figure(figsize=(7,5))

plt.plot(pos, means, marker='o')
plt.errorbar(pos, means, yerr=cis, fmt='o', capsize=5)

plt.xticks(pos, labels)
plt.ylabel("Aggregate Performance")
plt.title("PER vs Uniform Comparison")
plt.grid(alpha=0.3)
plt.show()