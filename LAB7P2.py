import numpy as np
import matplotlib.pyplot as plt

class BinaryBandit:
    def __init__(self, p1, p2):
        self.probabilities = [p1, p2]
    
    def pull(self, action):
        """Return reward for given action (0 or 1)"""
        return 1 if np.random.random() < self.probabilities[action] else 0

class EpsilonGreedyAgent:
    def __init__(self, n_actions, epsilon=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.Q = np.zeros(n_actions)  # Action-value estimates
        self.N = np.zeros(n_actions)  # Action counts
    
    def choose_action(self):
        """Choose action using epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploitation: best estimated action
            return np.argmax(self.Q)
    
    def update(self, action, reward):
        """Update action-value estimates"""
        self.N[action] += 1
        # Incremental update: Q = Q + (1/N)(reward - Q)
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

def binary_bandit_simulation(banditA, banditB, num_steps=1000, epsilon=0.1):
    """Run simulation for binary bandit problem"""
    agent = EpsilonGreedyAgent(n_actions=2, epsilon=epsilon)
    rewards = []
    optimal_actions = []
    
    for step in range(num_steps):
        # Choose which bandit to use (alternating or random)
        current_bandit = banditA if step % 2 == 0 else banditB
        
        action = agent.choose_action()
        reward = current_bandit.pull(action)
        agent.update(action, reward)
        
        rewards.append(reward)
        # Determine if action was optimal
        optimal_action = 0 if banditA.probabilities[0] > banditA.probabilities[1] else 1
        optimal_actions.append(1 if action == optimal_action else 0)
    
    return rewards, optimal_actions, agent.Q

# Simulate binary bandits
print("\n=== Binary Bandit Simulation ===")
# Bandit A: action 0 has higher probability
banditA = BinaryBandit(p1=0.8, p2=0.2)
# Bandit B: action 1 has higher probability  
banditB = BinaryBandit(p1=0.3, p2=0.7)

rewards, optimal_actions, final_Q = binary_bandit_simulation(banditA, banditB, 2000)

print(f"Final Q-values: {final_Q}")
print(f"Average reward: {np.mean(rewards):.3f}")
print(f"Optimal action %: {np.mean(optimal_actions)*100:.1f}%")

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(np.cumsum(rewards) / (np.arange(len(rewards)) + 1))
plt.title('Average Reward Over Time')
plt.xlabel('Steps')
plt.ylabel('Average Reward')

plt.subplot(1, 2, 2)
plt.plot(np.cumsum(optimal_actions) / (np.arange(len(optimal_actions)) + 1))
plt.title('Optimal Action Percentage')
plt.xlabel('Steps')
plt.ylabel('% Optimal Actions')
plt.tight_layout()
plt.show()
