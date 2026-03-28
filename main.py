import gymnasium as gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.models import Sequential


## DQN Network

### 1. Intialize the replay buffer to store history
replay_buffer = deque(maxlen=10000)

### 2. Build the main DQN network (Responsible for action selection and weight updates)
main_dqn = Sequential([
    Dense(64, activation='relu', kernel_initializer=HeNormal(), input_shape=(2,)),
    Dense(32, activation='relu', kernel_initializer=HeNormal()),
    Dense(3, activation='linear', kernel_initializer=HeNormal())
])

### 3. Build the target DQN network (Respinsible for finding the target Q-values for training)
target_dqn = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),
    Dense(32, activation='relu'),
    Dense(3, activation='linear')
])
target_dqn.set_weights(main_dqn.get_weights()) 

### 4. Epsilon-greedy Implementation
def epsilon_greedy(states,epsilon):
    prob = np.random.rand()

    if prob < epsilon:
        return np.random.randint(3) 
    
    else:
        states = np.array(states)
        states = np.reshape(states, (1, -1))
        q_values = main_dqn(states, training=False).numpy()
        return np.argmax(q_values[0])
    

### Hyperparameters
gamma = 0.99
lr = 0.001
batch_size = 32
epsilon = 0.9
target_update_frequency = 100

### 5. Compile the main DQN network
main_dqn.compile(optimizer=Adam(learning_rate=lr), loss='mse')

### 6. Training Loop
def train_dqn():
    
    ## We sample random batch from the replay buffer so check if we have enough samples to train
    if len(replay_buffer) < batch_size:
        return
    
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

    


env = gym.make("MountainCar-v0", max_episode_steps = 2000) 
observation, info = env.reset(seed=42)
num_episodes = 20
epsilon_decay = (epsilon - 0.1) * 2 / num_episodes
episode_rewards = []

for i in range(num_episodes):
    observation, info = env.reset()
    timesteps = 0
    episode_return = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = epsilon_greedy(observation, epsilon)
        next_observation, reward, terminated, truncated, info = env.step(action)
        status = terminated or truncated

        replay_buffer.append((observation, action, reward, next_observation, status))
        train_dqn()

        timesteps += 1
        if timesteps % target_update_frequency == 0:
            target_dqn.set_weights(main_dqn.get_weights())

        observation = next_observation
        episode_return += reward


    epsilon = max(epsilon - epsilon_decay, 0.1)
    episode_rewards.append(episode_return)
    print(f"Episode {i+1}, return {episode_return}")
    target_dqn.set_weights(main_dqn.get_weights())