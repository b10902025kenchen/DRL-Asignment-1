import numpy as np
import pickle
import random
import gym
from simple_custom_taxi_env import SimpleTaxiEnv

# Hyperparameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPISODES = 100  # Training episodes
EPSILON = 1  # Exploration rate

# Initialize the Q-table
q_table = {}

def load_q_table():
    """Load Q-table if it exists."""
    global q_table
    try:
        with open("q_table.pkl", "rb") as f:
            q_table = pickle.load(f)
        print("Q-table loaded successfully.")
    except FileNotFoundError:
        print("No saved Q-table found, starting fresh.")
    # print(q_table)

def save_q_table():
    """Save the Q-table to a file."""
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print("Q-table saved.")
    # print(q_table)


def get_action(state):
    """Choose action using epsilon-greedy policy, only allowing actions 0-3."""
    taxi_row, taxi_col, r_row, r_col, g_row, g_col, y_row, y_col, b_row, b_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = state

    n = 5

    state=(taxi_row * n + taxi_col) * n * (n-1) + (passenger_look * (n-1)) + destination_look


    # Load Q-table if it's not already loaded
    if not q_table:
        load_q_table()
    # print(q_table)

    # print(state)

    # if state not in q_table:
    #     q_table[state] = np.zeros(6)  # Only 4 possible actions

    # print("00000")
    # return 1

    # if random.uniform(0, 1) > EPSILON:
    #     return random.randint(0, 3)  # Only explore actions 0-3
    return np.argmax(q_table[state])  # Exploit best known action

def update_q_table(state, action, reward, next_state):
    """Update the Q-table using the Q-learning update rule."""
    if next_state not in q_table:
        q_table[next_state] = np.zeros(6)  # Only 4 actions available
    # print(reward)
    # q_table[state][action] += ALPHA * (reward + GAMMA * np.max(q_table[next_state]) - q_table[state][action])
    q_table[state][action] += reward

def train():
    """Train the agent using Q-learning."""
    env = gym.make("Taxi-v3")  # Use the Taxi-v3 environment
    load_q_table()  # Load Q-table if it exists

    for episode in range(EPISODES):
        obs, _ = env.reset()
        # print(obs)
        state = obs
        done = False
        total_reward = 0

        for i in range(1):
            action = random.choice([0, 1, 2, 3, 4, 5])
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = next_obs
            # print(state, next_state)
            # print(action, reward, done, truncated)

            # if truncated:
            #     break

            reward_shape = 1
            if action >= 4:
                reward_shape = -100000
            elif next_state == state:
                # print(q_table[state][action])
                reward_shape = -7777
            

            
            # **Updated Reward Shaping**
            # if reward == -0.1:  # Step penalty (default)
            #     reward = 0  # Neutral step
            # elif reward == -10:  # Wrong pickup/drop-off
            #     reward = 0  
            # elif reward == -5:  # Obstacle penalty (change to stronger penalty)
            #     reward = -1000  
            # elif reward == 20:  # Correct drop-off
            #     reward = 0  

            update_q_table(state, action, reward_shape, next_state)

            state = next_state
            total_reward += reward

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    save_q_table()  # Save Q-table after training

# Uncomment to train the agent
# if __name__ == "__main__":
#     train()
