import numpy as np
import matplotlib.pyplot as plt

class GridWorldMDP:
    def __init__(self, reward=-0.04):
        self.rows = 3
        self.cols = 4
        self.actions = ['Up', 'Down', 'Left', 'Right']
        self.action_effects = {
            'Up': [(-1, 0, 0.8), (0, -1, 0.1), (0, 1, 0.1)],  # intended, left, right
            'Down': [(1, 0, 0.8), (0, 1, 0.1), (0, -1, 0.1)],
            'Left': [(0, -1, 0.8), (1, 0, 0.1), (-1, 0, 0.1)],
            'Right': [(0, 1, 0.8), (-1, 0, 0.1), (1, 0, 0.1)]
        }
        
        # Terminal states: (row, col, reward)
        self.terminal_states = {(0, 3): 1.0, (1, 3): -1.0}
        self.walls = [(1, 1)]  # Wall at (1,1)
        
        # Reward for non-terminal states
        self.reward = reward
        
        # Initialize value function
        self.V = np.zeros((self.rows, self.cols))
        for (r, c), reward_val in self.terminal_states.items():
            self.V[r, c] = reward_val
    
    def is_valid_state(self, r, c):
        """Check if state is within bounds and not a wall"""
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return False
        if (r, c) in self.walls:
            return False
        return True
    
    def get_next_state(self, r, c, action):
        """Get possible next states and their probabilities for a given action"""
        next_states = []
        
        for dr, dc, prob in self.action_effects[action]:
            new_r, new_c = r + dr, c + dc
            
            # If next state is invalid, stay in current state
            if not self.is_valid_state(new_r, new_c):
                new_r, new_c = r, c
            
            next_states.append((new_r, new_c, prob))
        
        return next_states
    
    def value_iteration(self, gamma=1.0, theta=1e-6):
        """Perform value iteration to find optimal value function"""
        iterations = 0
        policy = {}
        
        while True:
            delta = 0
            new_V = np.copy(self.V)
            
            for r in range(self.rows):
                for c in range(self.cols):
                    # Skip terminal states
                    if (r, c) in self.terminal_states:
                        continue
                    if (r, c) in self.walls:
                        continue
                    
                    # Calculate value for each action
                    action_values = []
                    for action in self.actions:
                        next_states = self.get_next_state(r, c, action)
                        action_value = 0
                        
                        for nr, nc, prob in next_states:
                            # Immediate reward + discounted future value
                            if (nr, nc) in self.terminal_states:
                                reward_val = self.terminal_states[(nr, nc)]
                            else:
                                reward_val = self.reward
                            
                            action_value += prob * (reward_val + gamma * self.V[nr, nc])
                        
                        action_values.append(action_value)
                    
                    # Update value with maximum action value
                    if action_values:
                        new_V[r, c] = max(action_values)
                        delta = max(delta, abs(new_V[r, c] - self.V[r, c]))
            
            self.V = new_V
            iterations += 1
            
            if delta < theta:
                break
        
        # Extract optimal policy
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.terminal_states or (r, c) in self.walls:
                    policy[(r, c)] = 'Terminal'
                    continue
                
                best_action = None
                best_value = -np.inf
                
                for action in self.actions:
                    next_states = self.get_next_state(r, c, action)
                    action_value = 0
                    
                    for nr, nc, prob in next_states:
                        if (nr, nc) in self.terminal_states:
                            reward_val = self.terminal_states[(nr, nc)]
                        else:
                            reward_val = self.reward
                        
                        action_value += prob * (reward_val + gamma * self.V[nr, nc])
                    
                    if action_value > best_value:
                        best_value = action_value
                        best_action = action
                
                policy[(r, c)] = best_action
        
        return self.V, policy, iterations

    def print_results(self, reward_val):
        """Print value function and policy for given reward"""
        print(f"\n=== Results for r(s) = {reward_val} ===")
        self.reward = reward_val
        self.V = np.zeros((self.rows, self.cols))
        for (r, c), reward_val_term in self.terminal_states.items():
            self.V[r, c] = reward_val_term
        
        V_opt, policy_opt, iterations = self.value_iteration()
        
        print(f"Value Iteration completed in {iterations} iterations")
        print("\nOptimal Value Function:")
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.walls:
                    print("  WALL  ", end="")
                else:
                    print(f"{V_opt[r, c]:7.3f}", end=" ")
            print()
        
        print("\nOptimal Policy:")
        action_symbols = {'Up': '↑', 'Down': '↓', 'Left': '←', 'Right': '→', 'Terminal': 'T'}
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.terminal_states:
                    reward_val = self.terminal_states[(r, c)]
                    print(f" {reward_val:5.1f} ", end="")
                elif (r, c) in self.walls:
                    print("  WALL  ", end="")
                else:
                    print(f"   {action_symbols[policy_opt[(r, c)]]}   ", end="")
            print()

# Solve for different reward values
print("=== Problem 1: 4x3 Grid World MDP ===")
mdp = GridWorldMDP()

rewards = [-2, 0.1, 0.02, 1]
for reward in rewards:
    mdp.print_results(reward)