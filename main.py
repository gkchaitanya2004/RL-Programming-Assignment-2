import gymnasium as gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.models import Sequential


### 1. Hyperparameters
gamma = 0.99
lr = 0.001
batch_size = 32
epsilon_initial = 0.95
target_update_frequency = 50
num_episodes = 60
seeds = range(0, 15)
truncation_length = 2000
rho = 1


### 2. Neural Network (both target and main)
def build_network():
    model = Sequential([
        Input(shape=(2,)),
        Dense(64, activation='relu', kernel_initializer=HeNormal()),
        Dense(32, activation='relu', kernel_initializer=HeNormal()),
        Dense(3,  activation='linear', kernel_initializer=HeNormal())
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

### 3. Epsilon-greedy Implementation
def epsilon_greedy(states,epsilon,main_dqn):
    prob = np.random.rand()

    if prob < epsilon:
        return np.random.randint(3)

    else:
        states = np.array(states)
        states = np.reshape(states, (1, -1))
        q_values = main_dqn(states, training=False).numpy()
        return np.argmax(q_values[0])

### 4. Training Loop
def train_dqn(replay_buffer, main_dqn, target_dqn, rho = 1):

    ## Check if we have enough values to sample
    if len(replay_buffer) < batch_size:
        return

    for j in range(rho):
        ## Sample a random batch from the replay buffer
        batch = random.sample(replay_buffer, batch_size)
        states ,actions, rewards, next_states, status = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        status = np.array(status)

        ## Get the target Q-values from the target DQN network
        tar_q_values = target_dqn(next_states, training=False).numpy()
        max_tar_q_values = np.max(tar_q_values, axis=1)

        ## Bellman equation
        targets = rewards + (1 - status) * gamma * max_tar_q_values

        ## We have state and target values so we can train the main DQN network
        q_values = main_dqn(states, training = False).numpy()

        ## We have q-values but we dont know which actons were so update the values
        for i in range(batch_size):
          q_values[i][actions[i]] = targets[i]

        main_dqn.fit(states,q_values, verbose = 0, epochs = 1)


### 5. Experimentation with different seeds

def run_single_seed(seed, truncation_length, rho = 1):
    main_dqn   = build_network()
    target_dqn = build_network()
    target_dqn.set_weights(main_dqn.get_weights())
    replay_buffer = deque(maxlen=20000)

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

            replay_buffer.append((observation, action, reward, next_observation, float(done)))

            timesteps += 1

            if timesteps % 4 == 0:
                train_dqn(replay_buffer, main_dqn, target_dqn, rho = rho)

            if timesteps % target_update_frequency == 0:
                target_dqn.set_weights(main_dqn.get_weights())

            observation     = next_observation
            episode_return += reward

        epsilon = max(epsilon - epsilon_decay, 0.05)
        episode_returns.append(episode_return)
        print(f"  Seed {seed} | Episode {episode+1} | Return {episode_return:.1f}")

    env.close()
    return episode_returns

### 7. Run over different seeds

all_returns = []
for seed in seeds:
  print(f"\n Running Seed {seed} | Truncation: {truncation_length} | rho : {rho}")
  seed_returns = run_single_seed(seed, truncation_length, rho)
  all_returns.append(seed_returns)

all_returns = np.array(all_returns)

# np.save("returns_2000.npy", all_returns)
"""
Save the npy files since, for the plots, we are using a different python file,
we have to load these npy files in the plots.py file.

Naming Format for the npy files
For part 2 and 3 : returns_{truncation_length}.npy
For part 4a : returns_rho{rho}.npy
For part 4d : 
For batch_size : returns_rho{rho}_bs{batch_size}.npy
For varying target update frequency : returns_rho{rho}_tnr{target_update_frequency}.npy

What all npy files are expected to be present for each 
plot is given in the plots.py file
"""