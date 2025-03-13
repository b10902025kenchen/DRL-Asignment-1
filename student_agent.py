import numpy as np
import pickle
import random
import gym

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((state_size, action_size))
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_size))  # Explore
        return np.argmax(self.q_table[state])  # Exploit
    
    def update_q_table(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + (0 if done else self.gamma * self.q_table[next_state, best_next_action])
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_q_table(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
    
    def load_q_table(self, filename="q_table.pkl"):
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print("No saved Q-table found, starting fresh.")

# Train the agent
def train_agent(env_name="Taxi-v3", episodes=5000):
    env = gym.make(env_name)
    agent = QLearningAgent(state_size=env.observation_space.n, action_size=env.action_space.n)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_table(state, action, reward, next_state, done)
            state = next_state
    
    agent.save_q_table()
    env.close()
    print("Training complete and Q-table saved!")

# Use the trained agent for decision-making
def get_action(obs):
    env = gym.make("Taxi-v3")
    agent = QLearningAgent(state_size=env.observation_space.n, action_size=env.action_space.n)
    agent.load_q_table()
    return agent.choose_action(obs)

if __name__ == "__main__":
    train_agent()
