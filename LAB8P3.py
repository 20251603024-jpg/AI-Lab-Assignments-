class ModifiedGBikeMDP:
    def __init__(self, max_bikes=20, max_move=5, discount=0.9):
        self.max_bikes = max_bikes
        self.max_move = max_move
        self.discount = discount
        
        # Rental parameters
        self.loc1_params = (3, 3)
        self.loc2_params = (4, 2)
        
        # Reward and cost parameters
        self.rental_reward = 10
        self.move_cost = 2
        self.parking_cost = 4
        self.parking_limit = 10
        
        # State space
        self.states = [(i, j) for i in range(max_bikes + 1) for j in range(max_bikes + 1)]
        
        # Action space
        self.actions = list(range(-max_move, max_move + 1))
        
        # Precompute Poisson probabilities
        self.poisson_cache = {}
        self._precompute_poisson()
    
    def _poisson_prob(self, n, lam):
        key = (n, lam)
        if key not in self.poisson_cache:
            self.poisson_cache[key] = poisson.pmf(n, lam)
        return self.poisson_cache[key]
    
    def _precompute_poisson(self):
        max_events = 20
        for lam in [self.loc1_params[0], self.loc1_params[1], 
                   self.loc2_params[0], self.loc2_params[1]]:
            for n in range(max_events + 1):
                self._poisson_prob(n, lam)
    
    def get_transition_probabilities(self, state, action):
        bikes1, bikes2 = state
        
        # Apply free bike movement (employee shuttle)
        if action > 0:  # Moving bikes from loc1 to loc2
            effective_action = max(0, action - 1)  # First bike is free
        else:
            effective_action = action  # No free bike in reverse direction
        
        # Apply action
        bikes1_after_move = bikes1 - action  # Note: using original action for movement
        bikes2_after_move = bikes2 + action
        
        # Ensure bounds
        bikes1_after_move = max(0, min(bikes1_after_move, self.max_bikes))
        bikes2_after_move = max(0, min(bikes2_after_move, self.max_bikes))
        
        # Calculate parking costs
        parking_cost = 0
        if bikes1_after_move > self.parking_limit:
            parking_cost += self.parking_cost
        if bikes2_after_move > self.parking_limit:
            parking_cost += self.parking_cost
        
        transitions = []
        
        for req1 in range(21):
            for ret1 in range(21):
                for req2 in range(21):
                    for ret2 in range(21):
                        p_req1 = self._poisson_prob(req1, self.loc1_params[0])
                        p_ret1 = self._poisson_prob(ret1, self.loc1_params[1])
                        p_req2 = self._poisson_prob(req2, self.loc2_params[0])
                        p_ret2 = self._poisson_prob(ret2, self.loc2_params[1])
                        
                        prob = p_req1 * p_ret1 * p_req2 * p_ret2
                        
                        if prob < 1e-6:
                            continue
                        
                        actual_rent1 = min(req1, bikes1_after_move)
                        actual_rent2 = min(req2, bikes2_after_move)
                        
                        next_bikes1 = bikes1_after_move - actual_rent1 + ret1
                        next_bikes2 = bikes2_after_move - actual_rent2 + ret2
                        
                        next_bikes1 = max(0, min(next_bikes1, self.max_bikes))
                        next_bikes2 = max(0, min(next_bikes2, self.max_bikes))
                        
                        next_state = (next_bikes1, next_bikes2)
                        
                        # Calculate reward with modified costs
                        rental_income = (actual_rent1 + actual_rent2) * self.rental_reward
                        move_cost = abs(effective_action) * self.move_cost
                        total_reward = rental_income - move_cost - parking_cost
                        
                        transitions.append((next_state, total_reward, prob))
        
        return transitions
    
    def policy_iteration(self, theta=1e-6):
        """Perform policy iteration for modified problem"""
        policy = {}
        for state in self.states:
            policy[state] = 0
        
        V = {state: 0 for state in self.states}
        
        policy_stable = False
        iterations = 0
        
        while not policy_stable:
            iterations += 1
            
            # Policy Evaluation
            while True:
                delta = 0
                for state in self.states:
                    bikes1, bikes2 = state
                    if bikes1 == 0 and bikes2 == 0:
                        continue
                    
                    v = V[state]
                    action = policy[state]
                    
                    expected_value = 0
                    transitions = self.get_transition_probabilities(state, action)
                    
                    for next_state, reward, prob in transitions:
                        expected_value += prob * (reward + self.discount * V[next_state])
                    
                    V[state] = expected_value
                    delta = max(delta, abs(v - V[state]))
                
                if delta < theta:
                    break
            
            # Policy Improvement
            policy_stable = True
            for state in self.states:
                bikes1, bikes2 = state
                if bikes1 == 0 and bikes2 == 0:
                    continue
                
                old_action = policy[state]
                
                best_action = None
                best_value = -np.inf
                
                for action in self.actions:
                    if (bikes1 - action < 0 or bikes1 - action > self.max_bikes or
                        bikes2 + action < 0 or bikes2 + action > self.max_bikes):
                        continue
                    
                    action_value = 0
                    transitions = self.get_transition_probabilities(state, action)
                    
                    for next_state, reward, prob in transitions:
                        action_value += prob * (reward + self.discount * V[next_state])
                    
                    if action_value > best_value:
                        best_value = action_value
                        best_action = action
                
                policy[state] = best_action
                
                if best_action != old_action:
                    policy_stable = False
            
            print(f"Modified policy iteration {iterations} completed")
        
        return V, policy, iterations

# Compare both problems
def compare_solutions():
    print("\n=== Problem 3: Modified Gbike with Free Bike and Parking Costs ===")
    
    # Original problem
    original = GBikeMDP()
    V_orig, policy_orig, iter_orig = original.policy_iteration()
    
    # Modified problem
    modified = ModifiedGBikeMDP()
    V_mod, policy_mod, iter_mod = modified.policy_iteration()
    
    print("\n=== COMPARISON RESULTS ===")
    print(f"Original problem iterations: {iter_orig}")
    print(f"Modified problem iterations: {iter_mod}")
    
    print("\nPolicy Comparison for Sample States:")
    print("State\t\tOriginal Action\tModified Action")
    sample_states = [(10, 10), (15, 5), (5, 15), (20, 0), (0, 20)]
    
    for state in sample_states:
        orig_action = policy_orig[state]
        mod_action = policy_mod[state]
        print(f"{state}\t\t{orig_action}\t\t{mod_action}")
    
    print("\nValue Comparison for Sample States:")
    print("State\t\tOriginal Value\tModified Value")
    for state in sample_states:
        orig_val = V_orig[state]
        mod_val = V_mod[state]
        print(f"{state}\t\t{orig_val:.2f}\t\t{mod_val:.2f}")
    
    # Analyze the effect of modifications
    print("\n=== ANALYSIS OF MODIFICATIONS ===")
    print("1. Free bike movement from loc1 to loc2 reduces transportation costs")
    print("2. Parking costs encourage better distribution of bikes")
    print("3. Modified policy shows different movement patterns")
    
    # Find states where policies differ
    differing_states = []
    for state in sample_states:
        if policy_orig[state] != policy_mod[state]:
            differing_states.append(state)
    
    print(f"\nPolicies differ in {len(differing_states)} out of {len(sample_states)} sample states")

# Run comparison
compare_solutions()