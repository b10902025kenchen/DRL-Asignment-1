import numpy as np
import pickle
import random
from simple_custom_taxi_env import SimpleTaxiEnv

# Hyperparameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPISODES = 100  # Training episodes
EPSILON = 1  # Exploration rate
q_table = {}

def load_q_table():
    """Load Q-table if it exists."""
    global q_table
    try:
        with open("q_table.pkl", "rb") as f:
            q_table = pickle.load(f)
        print("Q-table loaded successfully.")
    except FileNotFoundError:
        # q_table = {}
        print("No saved Q-table found, starting fresh.")

def save_q_table():
    """Save the Q-table to a file."""
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print("Q-table saved.")
    print(q_table)

def get_action(state):
    """Choose action using epsilon-greedy policy, only allowing actions 0-3."""
    if state not in q_table:
        q_table[state] = np.zeros(6)
    
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, 3)  # Explore actions 0-3
    return np.argmax(q_table[state])  # Exploit best known action

def update_q_table(state, action, reward, next_state):
    """Update the Q-table using the Q-learning update rule."""
    if next_state not in q_table:
        q_table[next_state] = np.zeros(6)
    q_table[state][action] += ALPHA * (reward + GAMMA * np.max(q_table[next_state]) - q_table[state][action])
    # q_table[state][action] += reward

def train():
    """Train the agent using Q-learning."""
    env = SimpleTaxiEnv()  # Use SimpleTaxiEnv
    load_q_table()

    for episode in range(EPISODES):
        obs, _ = env.reset()
        state = obs
        total_reward = 0

        if state not in q_table:
            q_table[state] = np.zeros(6)

        for _ in range(1000):  # Limit episode length
            action = get_action(state)
            next_obs, reward, done, truncate = env.step(action)
            next_state = next_obs
            print(reward)
            if action >= 4:
                reward_shape = -100000
            elif truncate:
                reward_shape = 10000
            elif reward <= -5:
                reward_shape = -7777
            else:
                reward_shape = 1
            update_q_table(state, action, reward_shape, next_state)

            state = next_state
            total_reward += reward
            if done:
                break

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    save_q_table()

if __name__ == "__main__":
    train()
