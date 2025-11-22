import numpy as np
import random
from typing import List, Tuple, Optional
import math
import sys

np.random.seed(0)
random.seed(0)


class HopfieldNetwork:
    def __init__(self, N: int):
        """
        Discrete Hopfield network (binary states -1,+1)
        :param N: number of neurons
        """
        self.N = N
        self.W = np.zeros((N, N), dtype=float)

    def train_hebbian(self, patterns: np.ndarray, normalize: bool = True):
        """
        Hebbian learning rule (outer-product).
        :param patterns: shape (P, N) with values in {-1, +1}
        :param normalize: divide by number of patterns (paper uses 1/P)
        """
        P = patterns.shape[0]
        W = np.zeros((self.N, self.N), dtype=float)
        for p in range(P):
            x = patterns[p].reshape(self.N, 1)
            W += x @ x.T
        if normalize and P > 0:
            W = W / P
        np.fill_diagonal(W, 0.0)
        self.W = W

    def energy(self, s: np.ndarray) -> float:
        """Compute network energy E = -1/2 s^T W s"""
        return -0.5 * s.T @ self.W @ s

    def recall_async(self,
                     input_state: np.ndarray,
                     max_iter: int = 1000,
                     tol: int = 0,
                     random_update: bool = True) -> Tuple[np.ndarray, int, bool]:
        """
        Asynchronous recall: update neurons one by one (random order by default)
        :param input_state: shape (N,) values in {-1, +1}
        :param max_iter: maximum number of single-neuron updates (not epochs)
        :param tol: stop if no flips within tol full epochs
        :param random_update: if True use random ordering per epoch
        :return: final_state, iterations_used, converged_flag
        """
        s = input_state.copy().astype(int)
        N = self.N
        iters = 0
        # We'll measure convergence by an epoch-level check
        no_flip_epochs = 0
        while iters < max_iter and no_flip_epochs <= tol:
            if random_update:
                order = np.random.permutation(N)
            else:
                order = np.arange(N)
            flips = 0
            for i in order:
                iters += 1
                h = self.W[i, :] @ s
                s_new = 1 if h >= 0 else -1  # sign function (tie -> +1)
                if s_new != s[i]:
                    s[i] = s_new
                    flips += 1
                if iters >= max_iter:
                    break
            if flips == 0:
                no_flip_epochs += 1
            else:
                no_flip_epochs = 0
        converged = (no_flip_epochs > 0 or iters < max_iter)
        return s, iters, converged


def generate_random_patterns(P: int, N: int) -> np.ndarray:
    """Generate P random patterns of length N in {-1, +1}"""
    patterns = np.random.choice([-1, 1], size=(P, N))
    return patterns


def add_noise(pattern: np.ndarray, noise_rate: float) -> np.ndarray:
    """Flip sign of a fraction noise_rate of bits in pattern"""
    N = pattern.size
    noisy = pattern.copy()
    k = int(round(noise_rate * N))
    if k == 0:
        return noisy
    idx = np.random.choice(N, size=k, replace=False)
    noisy[idx] = -noisy[idx]
    return noisy


def accuracy_of_recall(original: np.ndarray, recalled: np.ndarray) -> float:
    """Return fraction of bits equal (0..1)"""
    return np.mean(original == recalled)


def associative_memory_experiment(N: int = 100,
                                  stored_P: int = 8,
                                  noise_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
                                  trials_per_level: int = 50) -> dict:
    """
    Run associative memory tests similar to Table in paper.
    Trains network with stored_P random patterns and tests recall at different noise levels.
    Returns dictionary with results.
    """
    results = {}
    # generate patterns and train
    patterns = generate_random_patterns(stored_P, N)
    net = HopfieldNetwork(N)
    net.train_hebbian(patterns, normalize=True)

    # For each noise level, measure fraction of successful recalls (exact match) or average bitwise correctness
    for noise in noise_levels:
        exact_success = 0
        avg_accuracy = 0.0
        for t in range(trials_per_level):
            p_idx = np.random.randint(0, stored_P)
            original = patterns[p_idx]
            noisy = add_noise(original, noise_rate=noise)
            recalled, iters, conv = net.recall_async(noisy, max_iter=1000)
            acc = accuracy_of_recall(original, recalled)
            avg_accuracy += acc
            if np.array_equal(recalled, original):
                exact_success += 1
        avg_accuracy /= trials_per_level
        correction_rate = exact_success / trials_per_level
        results[noise] = {
            "exact_success_rate": correction_rate,
            "avg_bit_accuracy": avg_accuracy
        }
        print(f"[Assoc] noise={int(noise*100)}% -> exact_success={correction_rate:.3f}, avg_bit_acc={avg_accuracy:.3f}")
    return results


def capacity_test(N: int = 100, max_P: int = 20, repeats: int = 5, noise: float = 0.0) -> dict:
    """
    Test practical storage capacity up to max_P patterns.
    For each P, trains and tests recall of all stored patterns (optionally with noise).
    Returns mapping P -> success fraction (all patterns recalled exactly).
    """
    res = {}
    for P in range(1, max_P + 1):
        success_counts = 0
        for r in range(repeats):
            patterns = generate_random_patterns(P, N)
            net = HopfieldNetwork(N)
            net.train_hebbian(patterns, normalize=True)
            # test recall for each pattern
            all_ok = True
            for p in range(P):
                inp = patterns[p]
                if noise > 0:
                    inp = add_noise(inp, noise)
                recalled, iters, conv = net.recall_async(inp, max_iter=2000)
                if not np.array_equal(recalled, patterns[p]):
                    all_ok = False
                    break
            if all_ok:
                success_counts += 1
        res[P] = success_counts / repeats
        print(f"[Capacity] P={P}, fraction_runs_all_patterns_recalled={res[P]:.2f}")
    theoretical = 0.138 * N
    print(f"Theoretical capacity ≈ {theoretical:.2f} patterns (0.138 * N).")
    return res


# ---------------------------
# Eight-rook problem
# ---------------------------
def eight_rook_weights(n: int = 8, same_row_weight: float = -2.0,
                       same_col_weight: float = -2.0, global_inhibition: float = -0.5) -> Tuple[np.ndarray, int]:
    """
    Build weight matrix for n x n board for "n-rooks" constraint:
    - Exactly one rook per row
    - Exactly one rook per column
    We'll encode N = n*n neurons; neuron (r,c) index = r*n + c
    We create a symmetric weight matrix implementing penalties (negative weights for violating constraints).
    Note: This is a penalty-style Hopfield formulation and may require careful scaling of weights and biases.
    """
    N = n * n
    W = np.zeros((N, N), dtype=float)

    def idx(r, c): return r * n + c

    # Penalize two rooks in same row (i != j in same row)
    for r in range(n):
        for c1 in range(n):
            for c2 in range(n):
                if c1 != c2:
                    i, j = idx(r, c1), idx(r, c2)
                    W[i, j] += same_row_weight

    # Penalize two rooks in same column
    for c in range(n):
        for r1 in range(n):
            for r2 in range(n):
                if r1 != r2:
                    i, j = idx(r1, c), idx(r2, c)
                    W[i, j] += same_col_weight

    # global inhibition between all distinct neurons
    for i in range(N):
        for j in range(N):
            if i != j:
                W[i, j] += global_inhibition

    np.fill_diagonal(W, 0.0)
    return W, N


def solve_eight_rooks(n: int = 8, trials: int = 50, max_iter: int = 5000) -> List[np.ndarray]:
    """
    Attempt to find valid n-rooks placements using Hopfield dynamics with many random restarts.
    Returns list of found valid solutions (as boolean board arrays).
    """
    W, N = eight_rook_weights(n)
    found_solutions = []
    for t in range(trials):
        net = HopfieldNetwork(N)
        net.W = W.copy()
        # random initial states biased to few 1s: start with a random placement of n rooks (one per row, but random col)
        state = -np.ones(N, dtype=int)
        for r in range(n):
            c = np.random.randint(0, n)
            state[r * n + c] = 1
        # run asynchronous updates
        s, iters, conv = net.recall_async(state, max_iter=max_iter)
        board = (s.reshape(n, n) == 1)
        # check if valid: exactly one in each row and column
        rows_ok = np.all(board.sum(axis=1) == 1)
        cols_ok = np.all(board.sum(axis=0) == 1)
        if rows_ok and cols_ok:
            found_solutions.append(board.astype(int))
            print(f"[8R] Found valid solution on trial {t}, iters={iters}")
            # optionally stop early if many found
            # break
    if len(found_solutions) == 0:
        print("[8R] No valid solutions found (this is common — Hopfield may get stuck in local minima).")
    return found_solutions


# ---------------------------
# TSP via Hopfield-style network (classic but fragile)
# ---------------------------
def tsp_hopfield_weights(n_cities: int, distance_matrix: np.ndarray,
                         A: float = 500.0, B: float = 500.0, C: float = 200.0, D: float = 500.0) -> Tuple[np.ndarray, int]:
    """
    Build weights for Hopfield TSP (n_cities^2 neurons).
    Classic formulation by Hopfield & Tank (1985), but it's notoriously sensitive and may not produce valid tours easily.
    We follow the generic form:
    - Neuron (i, p) indicates city i at position p in tour.
    - Constraints:
        * Each position p has exactly one city.
        * Each city i appears exactly once.
        * Distance minimization term couples successive positions.
    :param distance_matrix: (n, n) symmetric distances
    :returns: W matrix and N = n^2
    """
    n = n_cities
    N = n * n
    W = np.zeros((N, N), dtype=float)

    def idx(i, p): return i * n + p

    # City visitation constraint (-A)
    for i in range(n):
        for p in range(n):
            for q in range(n):
                if p != q:
                    W[idx(i, p), idx(i, q)] += -A

    # Position assignment constraint (-B)
    for p in range(n):
        for i in range(n):
            for j in range(n):
                if i != j:
                    W[idx(i, p), idx(j, p)] += -B

    # Distance minimization for adjacent positions
    for p in range(n):
        q = (p + 1) % n
        for i in range(n):
            for j in range(n):
                W[idx(i, p), idx(j, q)] += -C * distance_matrix[i, j]

    # Global inhibition (-D)
    for i in range(N):
        for j in range(N):
            if i != j:
                W[i, j] += -D

    np.fill_diagonal(W, 0.0)
    return W, N


def decode_tsp_state(state: np.ndarray, n_cities: int) -> Optional[List[int]]:
    """
    Decode Hopfield state into tour: choose city with +1 at each position p.
    If any position has not exactly one city, return None (invalid).
    """
    n = n_cities
    board = state.reshape((n, n))  # rows: city i, cols: position p
    tour = []
    for p in range(n):
        col = board[:, p]
        ones_idx = np.where(col == 1)[0]
        if ones_idx.size != 1:
            return None
        tour.append(int(ones_idx[0]))
    return tour


def tour_length(tour: List[int], dist: np.ndarray) -> float:
    n = len(tour)
    L = 0.0
    for p in range(n):
        i = tour[p]
        j = tour[(p + 1) % n]
        L += dist[i, j]
    return L


def attempt_tsp(n_cities: int = 10, restarts: int = 20, max_iter: int = 5000) -> List[Tuple[List[int], float]]:
    """
    Try multiple random restarts to find valid tours using Hopfield TSP formulation.
    Returns list of (tour, length) found.
    Warning: this is a best-effort and often finds invalid states.
    """
    # random symmetric distance matrix for demo
    rng = np.random.RandomState(1)
    coords = rng.rand(n_cities, 2)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    # scale distances to similar magnitude used by weights
    dist *= 10.0

    W, N = tsp_hopfield_weights(n_cities, dist, A=500.0, B=500.0, C=200.0, D=500.0)

    found = []
    for r in range(restarts):
        net = HopfieldNetwork(N)
        net.W = W.copy()
        # initialization: try to place one city per position randomly
        state = -np.ones(N, dtype=int)
        # create a random permutation
        perm = np.random.permutation(n_cities)
        for p in range(n_cities):
            city = perm[p]
            state[city * n_cities + p] = 1
        s_final, iters, conv = net.recall_async(state, max_iter=max_iter)
        tour = decode_tsp_state(s_final, n_cities)
        if tour is not None:
            L = tour_length(tour, dist)
            found.append((tour, L))
            print(f"[TSP] Found valid tour on restart {r} length={L:.3f}")
    if len(found) == 0:
        print("[TSP] No valid tours found (common — Hopfield/Tank often requires careful parameter tuning).")
    return found


# ---------------------------
# Main: run experiments mapping to paper's reported results
# ---------------------------
def main():
    N = 100  # 10x10 grid used in paper
    stored_P = 8  # as used in paper
    print("=== Associative Memory Experiment ===")
    assoc_results = associative_memory_experiment(N=N, stored_P=stored_P,
                                                 noise_levels=[0.1, 0.2, 0.3, 0.4, 0.5],
                                                 trials_per_level=50)

    print("\n=== Capacity Test ===")
    capacity_results = capacity_test(N=N, max_P=20, repeats=5, noise=0.0)

    print("\n=== Eight-Rook Attempts ===")
    eight_rook_solutions = solve_eight_rooks(n=8, trials=50, max_iter=3000)
    print(f"Eight-rook solutions found: {len(eight_rook_solutions)} (each is an 8x8 0/1 board)")

    print("\n=== TSP Attempts (10 cities) ===")
    tsp_found = attempt_tsp(n_cities=10, restarts=20, max_iter=3000)
    print(f"TSP tours found: {len(tsp_found)}")

    # Summarize associative memory results into table printed as text
    print("\n=== Summary: Error Correction Table ===")
    print("ErrorRate | ExactSuccessRate | AvgBitAccuracy")
    for noise, res in assoc_results.items():
        print(f"{int(noise*100):>8}% | {res['exact_success_rate']:.3f}           | {res['avg_bit_accuracy']:.3f}")

    print("\n=== Theoretical capacity check ===")
    print(f"Theoretical P_max ~ 0.138 * N = {0.138 * N:.2f}")
    # find largest P with high practical recall in capacity_results (>=0.6 of runs succeed)
    good_Ps = [P for P, frac in capacity_results.items() if frac >= 0.6]
    if good_Ps:
        print(f"Practically reliable up to P={max(good_Ps)} (>=0.6 runs succeeded).")
    else:
        print("No P had >=0.6 reliability in this run.")


if __name__ == "__main__":
    main()
