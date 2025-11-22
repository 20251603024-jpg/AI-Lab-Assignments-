class ModifiedEpsilonGreedyAgent:
    def __init__(self, n_actions, epsilon=0.1, alpha=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.alpha = alpha  # Step-size parameter for non-stationary problems
        self.Q = np.zeros(n_actions)
        # No need for N counts with constant step-size
    
    def choose_action(self):
        """Choose action using epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q)
    
    def update(self, action, reward):
        """Update with constant step-size for non-stationary problems"""
        self.Q[action] += self.alpha * (reward - self.Q[action])

def compare_agents(n_arms=10, num_steps=10000):
    """Compare standard vs modified epsilon-greedy on non-stationary bandit"""
    # Standard agent (sample average)
    standard_agent = EpsilonGreedyAgent(n_arms, epsilon=0.1)
    # Modified agent (constant step-size)
    modified_agent = ModifiedEpsilonGreedyAgent(n_arms, epsilon=0.1, alpha=0.1)
    
    bandit = NonStationaryBandit(n_arms)
    
    standard_rewards = []
    modified_rewards = []
    standard_optimal = []
    modified_optimal = []
    
    for step in range(num_steps):
        current_optimal = bandit.get_optimal_action()
        
        # Standard agent
        action_std = standard_agent.choose_action()
        reward_std = bandit.pull(action_std)
        standard_agent.update(action_std, reward_std)
        standard_rewards.append(reward_std)
        standard_optimal.append(1 if action_std == current_optimal else 0)
        
        # Modified agent  
        action_mod = modified_agent.choose_action()
        reward_mod = bandit.pull(action_mod)
        modified_agent.update(action_mod, reward_mod)
        modified_rewards.append(reward_mod)
        modified_optimal.append(1 if action_mod == current_optimal else 0)
    
    return (standard_rewards, standard_optimal, 
            modified_rewards, modified_optimal)

print("\n=== Non-Stationary Bandit Comparison ===")
std_rewards, std_optimal, mod_rewards, mod_optimal = compare_agents(10000)

print("Standard Epsilon-Greedy (Sample Average):")
print(f"  Average Reward: {np.mean(std_rewards):.4f}")
print(f"  % Optimal Actions: {np.mean(std_optimal)*100:.2f}%")

print("\nModified Epsilon-Greedy (Constant Step-Size):")
print(f"  Average Reward: {np.mean(mod_rewards):.4f}")
print(f"  % Optimal Actions: {np.mean(mod_optimal)*100:.2f}%")

# Plot comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(np.cumsum(std_rewards) / (np.arange(len(std_rewards)) + 1), 
         label='Standard', alpha=0.7)
plt.plot(np.cumsum(mod_rewards) / (np.arange(len(mod_rewards)) + 1), 
         label='Modified', alpha=0.7)
plt.title('Average Reward Comparison')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(np.cumsum(std_optimal) / (np.arange(len(std_optimal)) + 1), 
         label='Standard', alpha=0.7)
plt.plot(np.cumsum(mod_optimal) / (np.arange(len(mod_optimal)) + 1), 
         label='Modified', alpha=0.7)
plt.title('Optimal Action % Comparison')
plt.xlabel('Steps')
plt.ylabel('% Optimal Actions')
plt.legend()

plt.subplot(1, 3, 3)
# Moving average of rewards
window = 100
std_ma = np.convolve(std_rewards, np.ones(window)/window, mode='valid')
mod_ma = np.convolve(mod_rewards, np.ones(window)/window, mode='valid')
plt.plot(std_ma, label='Standard', alpha=0.7)
plt.plot(mod_ma, label='Modified', alpha=0.7)
plt.title('Moving Average (Window=100)')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.legend()

plt.tight_layout()
plt.show()

# Key insights
print("\n=== Key Insights ===")
print("1. Standard epsilon-greedy (sample average) gives equal weight to all past rewards")
print("2. This causes it to be slow to adapt to changing reward distributions")
print("3. Modified version with constant step-size gives more weight to recent rewards")
print("4. Constant step-size is essential for non-stationary environments")
print("5. Modified agent should show better performance in non-stationary case")