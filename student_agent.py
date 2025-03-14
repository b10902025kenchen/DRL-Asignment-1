# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

# Hyperparameters
ALPHA = 0.01  # Learning rate
GAMMA = 0.9   # Discount factor
EPISODES = 10000  # Training episodes
EPSILON = 0.1  # Exploration rate

# Policy (weights) initialization
policy_weights = np.zeros((5000, 6))  # Assume state space ~5000

def softmax(x):
    """Compute softmax values for a vector."""
    exp_x = np.exp(x - np.max(x))  # Stability trick
    return exp_x / np.sum(exp_x)

def get_action(obs):
    """Choose action using softmax policy."""
    if isinstance(obs, tuple):  
        obs = obs[0]  # Extract first element if obs is a tuple

    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, 5)  # Exploration

    # Ensure valid state index
    state_index = min(obs, 4999)
    
    # Compute action probabilities using softmax
    action_probs = softmax(policy_weights[state_index])
    
    # Select action based on probability distribution
    return np.random.choice(6, p=action_probs)


def update_policy(state, action, reward, next_state):
    """Update policy using policy gradient learning."""
    state_index = min(state, 4999)
    next_state_index = min(next_state, 4999)

    # Compute advantage (TD error)
    advantage = reward + GAMMA * np.max(policy_weights[next_state_index]) - policy_weights[state_index][action]

    # Policy update using gradient ascent
    policy_weights[state_index][action] += ALPHA * advantage

def save_policy():
    """Save policy weights to a file."""
    with open("policy.pkl", "wb") as f:
        pickle.dump(policy_weights, f)

def load_policy():
    """Load policy weights if they exist."""
    global policy_weights
    try:
        with open("policy.pkl", "rb") as f:
            policy_weights = pickle.load(f)
    except FileNotFoundError:
        print("No saved policy found, starting fresh.")

# Train the agent (only run once)
def train():
    """Train the agent using policy-based learning."""
    env = gym.make("Taxi-v3")  # Use the Taxi-v3 environment

    for episode in range(EPISODES):
        obs, _ = env.reset()
        state = obs
        done = False
        total_reward = 0

        while not done:
            action = get_action(state)
            next_obs, reward, done, _, _ = env.step(action)
            next_state = next_obs

            # Reward shaping
            if reward == -0.1:  # Step penalty
                reward = -1  
            elif reward == -10:  # Wrong pickup/drop-off or fuel out
                reward = -50  
            elif reward == -5:  # Obstacle penalty
                reward = -20  
            elif reward == 20:  # Correct drop-off
                reward = 50  

            update_policy(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        if episode % 500 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    save_policy()

# Load policy when running as an agent
load_policy()
