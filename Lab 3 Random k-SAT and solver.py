#!/usr/bin/env python3
"""
Lab 3 — Random k-SAT Generation and Solving
Parts B & C: Hill Climbing, Beam Search, and Variable-Neighborhood Descent (VND)
Author: Generated for student submission
"""

import random, time, math, itertools
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------
#  PART B — Random k-SAT Instance Generator
# ---------------------------------------------------------------------
def gen_k_sat(k: int, n: int, m: int, seed: int = None) -> Dict:
    """Generate random k-SAT instance with n variables and m clauses."""
    rng = random.Random(seed)
    clauses = []
    for _ in range(m):
        vars_ = rng.sample(range(n), k)
        clause = []
        for v in vars_:
            sign = rng.choice([True, False])
            clause.append((v, sign))  # (variable, positive?)
        clauses.append(tuple(clause))
    return {"k": k, "n": n, "m": m, "clauses": tuple(clauses)}

def eval_clause(clause, assign):
    """Evaluate a clause given a Boolean assignment."""
    for v, pos in clause:
        val = assign[v]
        if (val and pos) or ((not val) and (not pos)):
            return True
    return False

def count_unsat(instance, assign):
    """Count number of unsatisfied clauses."""
    return sum(0 if eval_clause(c, assign) else 1 for c in instance["clauses"])

def random_assignment(n, rng):
    """Return random Boolean assignment list."""
    return [rng.choice([False, True]) for _ in range(n)]

def flip(assign, idx):
    """Return a copy with bit idx flipped."""
    a = assign[:]
    a[idx] = not a[idx]
    return a

# ---------------------------------------------------------------------
#  PART C — Solvers
# ---------------------------------------------------------------------
def make_break_counts(instance, assign):
    """Compute make/break counts used in heuristic H2."""
    n = instance["n"]
    make, breakc = [0]*n, [0]*n
    sat = [eval_clause(c, assign) for c in instance["clauses"]]
    for ci, clause in enumerate(instance["clauses"]):
        if sat[ci]:
            sat_lits, sat_idx = 0, None
            for (v, pos) in clause:
                val = assign[v]
                if (val and pos) or ((not val) and (not pos)):
                    sat_lits += 1; sat_idx = v
            if sat_lits == 1:
                breakc[sat_idx] += 1
        else:
            for (v, pos) in clause:
                val = assign[v]
                if (val and (not pos)) or ((not val) and pos):
                    make[v] += 1
    return make, breakc

def heuristic1(instance, assign):
    """H1 = number of satisfied clauses."""
    return instance["m"] - count_unsat(instance, assign)

def heuristic2(instance, assign):
    """H2 = make - break score."""
    make, breakc = make_break_counts(instance, assign)
    return sum(m - b for m, b in zip(make, breakc))

# -------------------- Hill-Climbing ---------------------------------
def hill_climb(instance, max_steps=1500, restarts=6, rng=None, heuristic=1):
    rng = rng or random.Random()
    best_global, best_cost_global = None, instance["m"]
    steps_used = 0
    for _ in range(restarts):
        assign = random_assignment(instance["n"], rng)
        cur_unsat = count_unsat(instance, assign)
        for _ in range(max_steps):
            steps_used += 1
            if cur_unsat == 0:
                return True, assign, steps_used, 0
            if heuristic == 1:
                cand = [(count_unsat(instance, flip(assign, i)), i) for i in range(instance["n"])]
                best_cost, idx = min(cand)
            else:
                make, breakc = make_break_counts(instance, assign)
                deltas = [(breakc[i]-make[i], i) for i in range(instance["n"])]
                deltas.sort()
                best_cost = cur_unsat + deltas[0][0]
                idx = deltas[0][1]
            if best_cost < cur_unsat:
                assign = flip(assign, idx); cur_unsat = best_cost
            else:
                idx = rng.randrange(instance["n"])
                assign = flip(assign, idx)
                cur_unsat = count_unsat(instance, assign)
        if cur_unsat < best_cost_global:
            best_cost_global, best_global = cur_unsat, assign[:]
    return False, best_global, steps_used, best_cost_global

# -------------------- Beam Search -----------------------------------
def beam_search(instance, width=3, max_iters=400, rng=None, heuristic=1):
    rng = rng or random.Random()
    n = instance["n"]
    beam = [random_assignment(n, rng) for _ in range(width)]
    iters = 0
    while iters < max_iters:
        iters += 1
        for a in beam:
            if count_unsat(instance, a) == 0:
                return True, a, iters, 0
        neighbors = []
        for a in beam:
            if heuristic == 1:
                base = instance["m"] - count_unsat(instance, a)
                scored = []
                for i in range(n):
                    a2 = flip(a, i)
                    scored.append((-(instance["m"] - count_unsat(instance, a2) - base), a2))
                scored.sort()
                neighbors.extend([s[1] for s in scored[:min(10, n)]])
            else:
                make, breakc = make_break_counts(instance, a)
                order = sorted(range(n), key=lambda i: -(make[i]-breakc[i]))
                for i in order[:min(10, n)]:
                    neighbors.append(flip(a, i))
        if heuristic == 1:
            beam = sorted(neighbors, key=lambda x: count_unsat(instance, x))[:width]
        else:
            beam = sorted(neighbors, key=lambda x: -heuristic2(instance, x))[:width]
    best = min(beam, key=lambda x: count_unsat(instance, x))
    return False, best, iters, count_unsat(instance, best)

# -------------------- VND -------------------------------------------
def vnd(instance, max_iters=1200, rng=None):
    rng = rng or random.Random()
    a = random_assignment(instance["n"], rng)
    iters = 0
    def improve_N1(a):
        base = count_unsat(instance, a); best = (base, None)
        for i in range(instance["n"]):
            c = count_unsat(instance, flip(a, i))
            if c < best[0]: best = (c, flip(a, i))
        return best[1]
    def improve_N2(a):
        unsat = [ci for ci, c in enumerate(instance["clauses"]) if not eval_clause(c, a)]
        if not unsat: return None
        ci = rng.choice(unsat)
        clause = instance["clauses"][ci]
        base = count_unsat(instance, a)
        best, best_delta = None, 1e9
        for (v, pos) in clause:
            a2 = flip(a, v)
            c = count_unsat(instance, a2)
            d = c - base
            if d < best_delta:
                best_delta, best = d, a2
        return best if best_delta < 0 else None
    def improve_N3(a):
        base = count_unsat(instance, a); best, best_c = None, base
        n = instance["n"]
        for _ in range(min(20, n*(n-1)//2)):
            i, j = rng.randrange(n), rng.randrange(n)
            if i == j: continue
            a2 = flip(flip(a, i), j)
            c = count_unsat(instance, a2)
            if c < best_c: best, best_c = a2, c
        return best if best_c < base else None
    neighborhoods = [improve_N1, improve_N2, improve_N3]
    k = 0
    while iters < max_iters:
        iters += 1
        if count_unsat(instance, a) == 0:
            return True, a, iters, 0
        new_a = neighborhoods[k](a)
        if new_a is not None:
            a, k = new_a, 0
        else:
            k += 1
            if k >= len(neighborhoods):
                a = random_assignment(instance["n"], rng); k = 0
    return False, a, iters, count_unsat(instance, a)

# ---------------------------------------------------------------------
#  Experiment / Main
# ---------------------------------------------------------------------
def run_suite(seed=21):
    rng = random.Random(seed)
    settings = [(20, int(3.5*20)), (20, int(4.2*20))]
    algos = [
        ("HillClimb-H1", lambda inst: hill_climb(inst, heuristic=1, rng=rng)),
        ("HillClimb-H2", lambda inst: hill_climb(inst, heuristic=2, rng=rng)),
        ("Beam(w=3)-H1", lambda inst: beam_search(inst, width=3, heuristic=1, rng=rng)),
        ("Beam(w=4)-H2", lambda inst: beam_search(inst, width=4, heuristic=2, rng=rng)),
        ("VND", lambda inst: vnd(inst, rng=rng)),
    ]
    results = []
    for (n, m) in settings:
        inst = gen_k_sat(3, n, m, seed=rng.randrange(10**6))
        for name, fn in algos:
            t0 = time.time()
            ok, assign, iters, cost = fn(inst)
            elapsed = int((time.time()-t0)*1000)
            results.append({
                "n": n, "m": m, "algo": name, "solved": ok,
                "iters": iters, "best_unsat": cost, "time_ms": elapsed
            })
    return results

if __name__ == "__main__":
    print("\n=== Lab 3: Random k-SAT (Parts B & C) ===")
    results = run_suite()
    print(f"{'n,m':<8}  {'Algorithm':<16}  {'Solved':<6}  {'Iters':<6}  {'Unsat':<6}  {'Time(ms)':<8}")
    print("-"*64)
    for r in results:
        print(f"({r['n']},{r['m']})  {r['algo']:<16}  {str(r['solved']):<6}  {r['iters']:<6}  {r['best_unsat']:<6}  {r['time_ms']:<8}")
