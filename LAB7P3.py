import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random
from collections import defaultdict

class NonStationaryBandit:
    def __init__(self, n_arms=10):
        self.n_arms = n_arms
        self.true_values = np.ones(n_arms) * 0.5  # Start equal
        self.step_size = 0.01
        self.step_count = 0
    
    def step(self):
        """Take random walk step for all arms"""
        self.true_values += np.random.normal(0, self.step_size, self.n_arms)
        self.step_count += 1
    
    def pull(self, action):
        """Return reward for given action"""
        # Add noise to true value
        reward = self.true_values[action] + np.random.normal(0, 0.1)
        # Take step after pull (non-stationary environment)
        self.step()
        return reward
    
    def get_optimal_action(self):
        """Return current optimal action"""
        return np.argmax(self.true_values)
    
    def get_true_values(self):
        """Return current true values of all arms"""
        return self.true_values.copy()

def bandit_nonstat(action):
    """
    Function interface for non-stationary bandit as specified in the problem
    This simulates the bandit_nonstat function that would be provided
    """
    # For the purpose of this simulation, we'll create a persistent bandit
    if not hasattr(bandit_nonstat, 'bandit'):
        bandit_nonstat.bandit = NonStationaryBandit(10)
    
    reward = bandit_nonstat.bandit.pull(action)
    return reward

class EpsilonGreedyAgent:
    """Standard epsilon-greedy agent using sample averages"""
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
        """Update action-value estimates using sample average"""
        self.N[action] += 1
        # Incremental update: Q = Q + (1/N)(reward - Q)
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

class ModifiedEpsilonGreedyAgent:
    """Modified epsilon-greedy agent with constant step-size for non-stationary problems"""
    def __init__(self, n_actions, epsilon=0.1, alpha=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.alpha = alpha  # Constant step-size parameter
        self.Q = np.zeros(n_actions)
    
    def choose_action(self):
        """Choose action using epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q)
    
    def update(self, action, reward):
        """Update with constant step-size for non-stationary problems"""
        self.Q[action] += self.alpha * (reward - self.Q[action])

class OptimisticInitialValuesAgent:
    """Agent with optimistic initial values to encourage exploration"""
    def __init__(self, n_actions, epsilon=0.0, initial_value=5.0, alpha=0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.ones(n_actions) * initial_value  # Optimistic initial values
        self.N = np.zeros(n_actions)
    
    def choose_action(self):
        """Choose action using epsilon-greedy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q)
    
    def update(self, action, reward):
        """Update with constant step-size"""
        self.N[action] += 1
        self.Q[action] += self.alpha * (reward - self.Q[action])

def run_nonstationary_experiment(num_steps=10000, n_arms=10):
    """Compare different agents on non-stationary bandit problem"""
    
    # Initialize agents
    agents = {
        'Standard ε-greedy (ε=0.1)': EpsilonGreedyAgent(n_arms, epsilon=0.1),
        'Modified ε-greedy (α=0.1)': ModifiedEpsilonGreedyAgent(n_arms, epsilon=0.1, alpha=0.1),
        'Modified ε-greedy (α=0.05)': ModifiedEpsilonGreedyAgent(n_arms, epsilon=0.1, alpha=0.05),
        'Optimistic (Q₁=5, α=0.1)': OptimisticInitialValuesAgent(n_arms, initial_value=5.0, alpha=0.1),
    }
    
    # Initialize bandit
    bandit = NonStationaryBandit(n_arms)
    
    # Track results
    results = {name: {'rewards': [], 'optimal': [], 'true_values': []} for name in agents}
    all_true_values = []
    
    for step in range(num_steps):
        current_optimal = bandit.get_optimal_action()
        current_true_values = bandit.get_true_values()
        all_true_values.append(current_true_values.copy())
        
        for name, agent in agents.items():
            action = agent.choose_action()
            reward = bandit.pull(action)
            agent.update(action, reward)
            
            results[name]['rewards'].append(reward)
            results[name]['optimal'].append(1 if action == current_optimal else 0)
            results[name]['true_values'].append(current_true_values.copy())
    
    return results, all_true_values

def analyze_performance(results, window=1000):
    """Analyze and compare agent performance"""
    print("=== Performance Analysis ===")
    print(f"{'Agent':<30} {'Avg Reward':<12} {'% Optimal':<12} {'Last 1000 Avg':<15}")
    print("-" * 70)
    
    performance_data = {}
    
    for name, data in results.items():
        rewards = np.array(data['rewards'])
        optimal = np.array(data['optimal'])
        
        avg_reward = np.mean(rewards)
        pct_optimal = np.mean(optimal) * 100
        last_1000_avg = np.mean(rewards[-1000:]) if len(rewards) >= 1000 else avg_reward
        
        performance_data[name] = {
            'avg_reward': avg_reward,
            'pct_optimal': pct_optimal,
            'last_1000_avg': last_1000_avg
        }
        
        print(f"{name:<30} {avg_reward:<12.4f} {pct_optimal:<12.2f} {last_1000_avg:<15.4f}")
    
    return performance_data

def plot_results(results, all_true_values, num_steps=10000):
    """Create comprehensive visualization of results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Colors for different agents
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    # Plot 1: Average reward over time (moving average)
    ax = axes[0, 0]
    window = 100
    for i, (name, data) in enumerate(results.items()):
        rewards = np.array(data['rewards'])
        # Moving average
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(moving_avg, label=name, color=colors[i], alpha=0.8)
    ax.set_title('Moving Average Reward (Window=100)')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Average Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Percentage of optimal actions
    ax = axes[0, 1]
    for i, (name, data) in enumerate(results.items()):
        optimal = np.array(data['optimal'])
        pct_optimal = np.cumsum(optimal) / (np.arange(len(optimal)) + 1) * 100
        ax.plot(pct_optimal, label=name, color=colors[i], alpha=0.8)
    ax.set_title('Percentage of Optimal Actions')
    ax.set_xlabel('Steps')
    ax.set_ylabel('% Optimal Actions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative reward
    ax = axes[0, 2]
    for i, (name, data) in enumerate(results.items()):
        rewards = np.array(data['rewards'])
        cumulative_reward = np.cumsum(rewards)
        ax.plot(cumulative_reward, label=name, color=colors[i], alpha=0.8)
    ax.set_title('Cumulative Reward')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Cumulative Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: True values evolution (first 3 arms)
    ax = axes[1, 0]
    all_true_values = np.array(all_true_values)
    for arm in range(3):  # Plot first 3 arms
        ax.plot(all_true_values[:1000, arm], label=f'Arm {arm+1}', alpha=0.7)
    ax.set_title('True Values Evolution (First 3 Arms)')
    ax.set_xlabel('Steps')
    ax.set_ylabel('True Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Final performance comparison
    ax = axes[1, 1]
    names = list(results.keys())
    last_1000_rewards = [np.mean(data['rewards'][-1000:]) for data in results.values()]
    bars = ax.bar(names, last_1000_rewards, color=colors[:len(names)], alpha=0.7)
    ax.set_title('Average Reward (Last 1000 Steps)')
    ax.set_ylabel('Average Reward')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 6: Q-value evolution for best modified agent
    ax = axes[1, 2]
    # Get Q-values from the best performing modified agent
    best_mod_agent = None
    for name in results.keys():
        if 'Modified' in name and '(α=0.1)' in name:
            best_mod_agent = name
            break
    
    if best_mod_agent:
        # This would require storing Q-values during simulation
        # For now, we'll plot the true values vs estimated (conceptual)
        ax.plot(all_true_values[:500, 0], label='True Value Arm 1', alpha=0.7)
        ax.plot(all_true_values[:500, 1], label='True Value Arm 2', alpha=0.7)
        ax.set_title('True Values vs Time (Conceptual)')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def statistical_analysis(results):
    """Perform statistical comparison between agents"""
    print("\n=== Statistical Analysis ===")
    
    # Compare the best modified agent with standard agent
    standard_rewards = np.array(results['Standard ε-greedy (ε=0.1)']['rewards'])
    modified_rewards = np.array(results['Modified ε-greedy (α=0.1)']['rewards'])
    
    # T-test for significant difference
    t_stat, p_value = stats.ttest_ind(modified_rewards, standard_rewards)
    
    print(f"Standard vs Modified ε-greedy:")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print("→ Modified agent performs SIGNIFICANTLY better (p < 0.05)")
    else:
        print("→ No significant difference between agents")
    
    # Compare last 1000 steps
    last_1000_std = standard_rewards[-1000:]
    last_1000_mod = modified_rewards[-1000:]
    t_stat_last, p_value_last = stats.ttest_ind(last_1000_mod, last_1000_std)
    
    print(f"\nLast 1000 steps comparison:")
    print(f"P-value: {p_value_last:.6f}")
    if p_value_last < 0.05:
        print("→ Significant difference in recent performance")

def demonstrate_bandit_nonstat_function():
    """Demonstrate the bandit_nonstat function interface"""
    print("\n=== bandit_nonstat Function Demonstration ===")
    
    # Reset the bandit for demonstration
    if hasattr(bandit_nonstat, 'bandit'):
        delattr(bandit_nonstat, 'bandit')
    
    # Test the function interface
    test_rewards = []
    for action in [0, 1, 2, 0, 1, 2]:  # Test sequence of actions
        reward = bandit_nonstat(action)
        test_rewards.append(reward)
        print(f"Action {action} → Reward: {reward:.4f}")
    
    print(f"Average reward in test: {np.mean(test_rewards):.4f}")

def main():
    """Main function to run the complete analysis"""
    print("=== Non-Stationary 10-Armed Bandit Problem ===\n")
    
    # Run the main experiment
    print("Running experiment with 10,000 steps...")
    results, all_true_values = run_nonstationary_experiment(num_steps=10000)
    
    # Analyze performance
    performance_data = analyze_performance(results)
    
    # Statistical analysis
    statistical_analysis(results)
    
    # Plot results
    plot_results(results, all_true_values)
    
    # Demonstrate the function interface
    demonstrate_bandit_nonstat_function()
    
    # Key insights
    print("\n=== Key Insights ===")
    print("1. Standard ε-greedy (sample average) struggles with non-stationary environments")
    print("2. Constant step-size (α) allows agents to track changing reward distributions")
    print("3. Optimal α value depends on the rate of change in the environment")
    print("4. Modified ε-greedy with α=0.1 typically performs best for this setup")
    print("5. The bandit_nonstat function provides the required interface for testing")

if __name__ == "__main__":
    main()