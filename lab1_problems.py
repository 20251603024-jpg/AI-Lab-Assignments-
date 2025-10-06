"""
lab1_problems.py

Combined solver for Lab Assignment 1 problems:
- Generic graph search helpers (BFS, DFS)
- Problem A: Missionaries & Cannibals (3M,3C, boat)
- Problem B: Rabbit Leap (EEE_WWW with one blank)
Run: python3 lab1_problems.py
"""
from collections import deque
from itertools import combinations
import copy

# ------------------ Generic helpers ------------------

def bfs_graph(start, successors, goal_test):
    """Generic BFS graph-search returning (states_seq, actions_seq) or None."""
    q = deque([start])
    parent = {start: None}
    action = {start: None}
    while q:
        state = q.popleft()
        if goal_test(state):
            # reconstruct path
            seq = []
            cur = state
            while cur is not None:
                seq.append(cur)
                cur = parent[cur]
            seq.reverse()
            acts = []
            cur = state
            while action[cur] is not None:
                acts.append(action[cur])
                cur = parent[cur]
            acts.reverse()
            return seq, acts
        for act, ns in successors(state):
            if ns not in parent:
                parent[ns] = state
                action[ns] = act
                q.append(ns)
    return None

def dfs_graph(start, successors, goal_test):
    """Generic DFS (stack-based) graph-search returning (states_seq, actions_seq) or None."""
    stack = [start]
    parent = {start: None}
    action = {start: None}
    visited = set()
    while stack:
        state = stack.pop()
        if goal_test(state):
            seq = []
            cur = state
            while cur is not None:
                seq.append(cur)
                cur = parent[cur]
            seq.reverse()
            acts = []
            cur = state
            while action[cur] is not None:
                acts.append(action[cur])
                cur = parent[cur]
            acts.reverse()
            return seq, acts
        if state in visited:
            continue
        visited.add(state)
        # deterministic: push successors in reversed order so the first successor is explored first
        for act, ns in reversed(successors(state)):
            if ns not in visited and ns not in parent:
                parent[ns] = state
                action[ns] = act
                stack.append(ns)
    return None

# ------------------ Problem A: Missionaries & Cannibals ------------------

def mc_successors(state):
    """
    State: tuple (M_left, C_left, boat) where boat: 0 = left, 1 = right
    Total M = 3, C = 3
    Returns list of (action_str, next_state)
    """
    M_left, C_left, boat = state
    M_total = 3
    C_total = 3
    res = []
    # people on current boat side
    if boat == 0:
        # choose 1 or 2 people from left to move to right
        people = ['M'] * M_left + ['C'] * C_left
        indices = list(range(len(people)))
        # generate combinations of size 1 and 2
        for r in (1,2):
            for combo in combinations(indices, r):
                m = sum(1 for i in combo if people[i] == 'M')
                c = sum(1 for i in combo if people[i] == 'C')
                new_M_left = M_left - m
                new_C_left = C_left - c
                new_state = (new_M_left, new_C_left, 1)
                if is_safe_state(new_state, M_total, C_total):
                    act = f"Left->Right: M-{m},C-{c}"
                    res.append((act, new_state))
    else:
        # boat on right: move 1 or 2 from right to left
        M_right = M_total - M_left
        C_right = C_total - C_left
        people = ['M'] * M_right + ['C'] * C_right
        indices = list(range(len(people)))
        for r in (1,2):
            for combo in combinations(indices, r):
                m = sum(1 for i in combo if people[i] == 'M')
                c = sum(1 for i in combo if people[i] == 'C')
                new_M_left = M_left + m
                new_C_left = C_left + c
                new_state = (new_M_left, new_C_left, 0)
                if is_safe_state(new_state, M_total, C_total):
                    act = f"Right->Left: M-{m},C-{c}"
                    res.append((act, new_state))
    return res

def is_safe_state(state, M_total=3, C_total=3):
    """Return True if state obeys constraints (no bank where cannibals > missionaries unless missionaries=0)."""
    M_left, C_left, boat = state
    M_right = M_total - M_left
    C_right = C_total - C_left
    # left bank safe?
    if M_left > 0 and C_left > M_left:
        return False
    # right bank safe?
    if M_right > 0 and C_right > M_right:
        return False
    # also ensure counts in range
    if not (0 <= M_left <= M_total and 0 <= C_left <= C_total and boat in (0,1)):
        return False
    return True

def mc_goal_test(state):
    return state == (0,0,1)  # all moved to right, boat on right

# ------------------ Problem B: Rabbit Leap ------------------

def rabbit_successors(state):
    """
    State: 7-char string like 'EEE_WWW'
    Returns list of (action_str, next_state)
    """
    res = []
    s = list(state)
    n = len(s)
    for i, ch in enumerate(s):
        if ch == 'E':
            # move 1 to right
            if i + 1 < n and s[i+1] == '_':
                ns = s.copy(); ns[i], ns[i+1] = '_', 'E'
                res.append((f"E:{i}->{i+1}", ''.join(ns)))
            # jump 2 to right
            if i + 2 < n and s[i+1] in ('E','W') and s[i+2] == '_':
                ns = s.copy(); ns[i], ns[i+2] = '_', 'E'
                res.append((f"E:{i}->{i+2}", ''.join(ns)))
        elif ch == 'W':
            # move 1 to left
            if i - 1 >= 0 and s[i-1] == '_':
                ns = s.copy(); ns[i], ns[i-1] = '_', 'W'
                res.append((f"W:{i}->{i-1}", ''.join(ns)))
            # jump 2 to left
            if i - 2 >= 0 and s[i-1] in ('E','W') and s[i-2] == '_':
                ns = s.copy(); ns[i], ns[i-2] = '_', 'W'
                res.append((f"W:{i}->{i-2}", ''.join(ns)))
    return res

def rabbit_goal_test(state):
    return state == "WWW_EEE"

# ------------------ Utilities for pretty printing ------------------

def print_solution(states, actions, title="Solution"):
    print(f"--- {title} ---")
    for i, s in enumerate(states):
        act = actions[i-1] if i > 0 else "(start)"
        print(f"{i:2d}: {s}  -- {act}")
    print()

# ------------------ Main: run both problems ------------------

def solve_missionaries_and_cannibals():
    print("Problem A: Missionaries & Cannibals (3M,3C)")
    start = (3,3,0)
    print("Running BFS...")
    bfs_sol = bfs_graph(start, mc_successors, mc_goal_test)
    if bfs_sol:
        states, acts = bfs_sol
        print_solution([format_mc_state(s) for s in states], acts, title="BFS (optimal)")
        print(f"Moves: {len(acts)}")
    else:
        print("No BFS solution found.")
    print("Running DFS...")
    dfs_sol = dfs_graph(start, mc_successors, mc_goal_test)
    if dfs_sol:
        states, acts = dfs_sol
        print_solution([format_mc_state(s) for s in states], acts, title="DFS (found)")
        print(f"Moves: {len(acts)}")
    else:
        print("No DFS solution found.")
    print("="*60)

def format_mc_state(state):
    M_left, C_left, boat = state
    M_total = 3
    C_total = 3
    M_right = M_total - M_left
    C_right = C_total - C_left
    boat_pos = "L" if boat==0 else "R"
    return f"({M_left}M,{C_left}C | {M_right}M,{C_right}C) boat:{boat_pos}"

def solve_rabbit_leap():
    print("Problem B: Rabbit Leap")
    start = "EEE_WWW"
    print("Running BFS...")
    bfs_sol = bfs_graph(start, rabbit_successors, rabbit_goal_test)
    if bfs_sol:
        states, acts = bfs_sol
        print_solution(states, acts, title="BFS (optimal)")
        print(f"Moves: {len(acts)}")
    else:
        print("No BFS solution found.")
    print("Running DFS...")
    dfs_sol = dfs_graph(start, rabbit_successors, rabbit_goal_test)
    if dfs_sol:
        states, acts = dfs_sol
        print_solution(states, acts, title="DFS (found)")
        print(f"Moves: {len(acts)}")
    else:
        print("No DFS solution found.")
    print("="*60)

def main():
    solve_missionaries_and_cannibals()
    solve_rabbit_leap()

if __name__ == "__main__":
    main()
