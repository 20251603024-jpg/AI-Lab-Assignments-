#!/usr/bin/env python3
\"\"\"Lab 2 — Sentence Alignment & Plagiarism Detection using A* Search
Runnable script: produces alignment traces and summary for example test cases,
and can also be used with two text files as command-line arguments:
    python3 lab2_plagiarism_astar.py [doc1.txt] [doc2.txt]

Author: Generated for student submission
\"\"\"

import re, heapq, textwrap, argparse, sys, json

# ------------------ Text processing & distance ------------------

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r\"[^a-z0-9\\s\\.\\!\\?]\", \" \", s)  # keep sentence punctuation
    s = re.sub(r\"\\s+\", \" \", s).strip()
    return s

def split_sentences(doc: str):
    # naive split on sentence-ending punctuation followed by whitespace
    doc = re.sub(r\"([a-zA-Z0-9])\\s*\\n\", r\"\\1. \", doc)
    parts = re.split(r\"(?<=[\\.\\!\\?])\\s+\", doc.strip())
    return [p.strip() for p in parts if p.strip()]

def tokens(s: str):
    return [t for t in re.split(r\"\\W+\", s.lower()) if t]

def levenshtein(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    prev = list(range(m+1))
    for i in range(1, n+1):
        cur = [i] + [0]*m
        ai = a[i-1]
        for j in range(1, m+1):
            cost = 0 if ai == b[j-1] else 1
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
        prev = cur
    return prev[m]

def norm_edit_distance(s1: str, s2: str) -> float:
    t1, t2 = tokens(s1), tokens(s2)
    if not t1 and not t2:
        return 0.0
    d = levenshtein(t1, t2)
    denom = max(len(t1), len(t2), 1)
    return d / denom

def sentence_similarity(s1: str, s2: str) -> float:
    return 1.0 - norm_edit_distance(s1, s2)

# ------------------ A* alignment ------------------

def astar_align(sents1, sents2, gap_penalty=0.7):
    n, m = len(sents1), len(sents2)
    start = (0,0)
    goal = (n, m)
    def h(i,j):
        return abs((n - i) - (m - j)) * gap_penalty
    open_heap = []
    heapq.heappush(open_heap, (h(0,0), 0.0, start, None, None))
    best_g = {start: 0.0}
    parents = {}
    actions = {}
    while open_heap:
        f, g, (i,j), parent, action = heapq.heappop(open_heap)
        if (i,j) in parents:
            continue
        parents[(i,j)] = parent
        actions[(i,j)] = action
        if (i,j) == goal:
            break
        # MATCH
        if i < n and j < m:
            s_cost = norm_edit_distance(sents1[i], sents2[j])
            ng = g + s_cost
            ns = (i+1, j+1)
            if ng < best_g.get(ns, float("inf")):
                best_g[ns] = ng
                heapq.heappush(open_heap, (ng + h(*ns), ng, ns, (i,j), ("MATCH", i, j, s_cost)))
        # GAP in doc2 (skip sents1[i])
        if i < n:
            ng = g + gap_penalty
            ns = (i+1, j)
            if ng < best_g.get(ns, float("inf")):
                best_g[ns] = ng
                heapq.heappush(open_heap, (ng + h(*ns), ng, ns, (i,j), ("GAP1", i, None, gap_penalty)))
        # GAP in doc1 (skip sents2[j])
        if j < m:
            ng = g + gap_penalty
            ns = (i, j+1)
            if ng < best_g.get(ns, float("inf")):
                best_g[ns] = ng
                heapq.heappush(open_heap, (ng + h(*ns), ng, ns, (i,j), ("GAP2", None, j, gap_penalty)))
    # reconstruct path
    path = []
    s = goal
    total_cost = best_g.get(goal, float("inf"))
    while s is not None:
        act = actions.get(s)
        path.append((s, act, best_g.get(s, None)))
        s = parents.get(s)
    path.reverse()
    # build alignment steps
    align_steps = []
    cum = 0.0
    step_idx = 0
    for k in range(1, len(path)):
        (i,j), act, g = path[k]
        if act is None:
            continue
        typ, i_sent, j_sent, cost = act
        cum += cost
        step_idx += 1
        rec = {
            "step": step_idx,
            "i": i,
            "j": j,
            "type": typ,
            "s1": (sents1[i_sent] if i_sent is not None else ""),
            "s2": (sents2[j_sent] if j_sent is not None else ""),
            "cost": round(cost, 4),
            "cum_cost": round(cum, 4)
        }
        if typ == "MATCH":
            rec["similarity"] = round(sentence_similarity(rec["s1"], rec["s2"]), 4)
        else:
            rec["similarity"] = None
        align_steps.append(rec)
    return align_steps, total_cost

# ------------------ Utilities & reporting ------------------

def summarize_alignment(steps, sim_threshold=0.8):
    aligned_pairs = [s for s in steps if s["type"] == "MATCH"]
    n_pairs = len(aligned_pairs)
    plag_pairs = [s for s in aligned_pairs if s["similarity"] is not None and s["similarity"] >= sim_threshold]
    avg_sim = sum([s["similarity"] for s in aligned_pairs]) / n_pairs if n_pairs else 0.0
    plag_pct = (len(plag_pairs) / max(n_pairs, 1)) * 100.0
    return {
        "n_total_steps": len(steps),
        "n_pairs": n_pairs,
        "avg_similarity": round(avg_sim, 4),
        "plag_pairs": len(plag_pairs),
        "plag_percent": round(plag_pct, 2)
    }

def steps_to_text(steps, max_rows=None):
    rows = steps if max_rows is None else steps[:max_rows]
    header = f\"{'Step':>4} | {'Type':<6} | {'Cost':>6} | {'Cum':>6} | {'Similarity':>10} | Sentence 1  ||  Sentence 2\"
    lines = [header, '-'*len(header)]
    for rec in rows:
        sim = \"\" if rec['similarity'] is None else f\"{rec['similarity']:.3f}\"
        line = f\"{rec['step']:>4} | {rec['type']:<6} | {rec['cost']:>6.3f} | {rec['cum_cost']:>6.3f} | {sim:>10} | {rec['s1']} || {rec['s2']}\"
        lines.append(textwrap.fill(line, width=140, subsequent_indent=' '*12))
    if max_rows is not None and len(steps) > max_rows:
        lines.append(f\"... ({len(steps)-max_rows} more rows)\")
    return \"\\n\".join(lines)

# ------------------ Example test cases & main ------------------

EXAMPLES = [
    (\"Identical Documents\", \"A* search is a best-first search algorithm that finds the least-cost path. It uses a priority queue ordered by f(n) = g(n) + h(n). When the heuristic is admissible, A* is optimal and complete. A consistent heuristic also guarantees that f-values are non-decreasing. In practice, A* is widely used in pathfinding and planning.\", 
     \"A* search is a best-first search algorithm that finds the least-cost path. It uses a priority queue ordered by f(n) = g(n) + h(n). When the heuristic is admissible, A* is optimal and complete. A consistent heuristic also guarantees that f-values are non-decreasing. In practice, A* is widely used in pathfinding and planning.\"),
    (\"Slightly Modified\", \"A* search is a best-first search algorithm that finds the least-cost path. It uses a priority queue ordered by f(n) = g(n) + h(n). When the heuristic is admissible, A* is optimal and complete.\", 
     \"A star (A*) search is a best-first method that discovers the lowest-cost path. It employs a priority queue ordered by f(n) = g(n) + h(n). If the heuristic never overestimates, A* remains optimal and complete.\"),
    (\"Completely Different\", \"A* search is a best-first search algorithm that finds the least-cost path. It uses a priority queue ordered by f(n) = g(n) + h(n).\", 
     \"Cooking pasta requires boiling water and adding salt. Stir the sauce separately with tomatoes and garlic. Exercise daily for good health.\"),
    (\"Partial Overlap\", \"A* search is commonly used in path planning for robots. The heuristic guides the search toward the goal efficiently. Graph search can be memory intensive in big problems.\", 
     \"A* search is commonly used in path planning for robots. In practice, A* is widely used in pathfinding and planning. Basketball is a popular sport around the world.\")
]

def run_examples():
    for title, d1, d2 in EXAMPLES:
        print('\\n' + '='*80)
        print(f\"Example: {title}\")
        s1 = split_sentences(normalize_text(d1))
        s2 = split_sentences(normalize_text(d2))
        steps, total_cost = astar_align(s1, s2, gap_penalty=0.7)
        stats = summarize_alignment(steps)
        print('\\nDocument 1 sentences:')
        for i, s in enumerate(s1, 1): print(f\"  {i}. {s}\")
        print('\\nDocument 2 sentences:')
        for i, s in enumerate(s2, 1): print(f\"  {i}. {s}\")
        print('\\nAlignment trace:')
        print(steps_to_text(steps, max_rows=200))
        print('\\nSummary:')
        print(json.dumps(stats, indent=2))

def align_files(path1, path2):
    with open(path1, 'r', encoding='utf-8') as f: d1 = f.read()
    with open(path2, 'r', encoding='utf-8') as f: d2 = f.read()
    s1 = split_sentences(normalize_text(d1))
    s2 = split_sentences(normalize_text(d2))
    steps, total_cost = astar_align(s1, s2, gap_penalty=0.7)
    stats = summarize_alignment(steps)
    print('\\n' + '='*80)
    print(f\"Alignment: {path1}  <-->  {path2}\")
    print(f\"Sentences: doc1={len(s1)}, doc2={len(s2)}\")
    print('\\nAlignment trace:')
    print(steps_to_text(steps, max_rows=500))
    print('\\nSummary:')
    print(json.dumps(stats, indent=2))

def main(argv):
    parser = argparse.ArgumentParser(description='A* Sentence Alignment (Lab 2)')
    parser.add_argument('file1', nargs='?', help='First document (optional)')
    parser.add_argument('file2', nargs='?', help='Second document (optional)')
    args = parser.parse_args(argv)
    if args.file1 and args.file2:
        align_files(args.file1, args.file2)
    else:
        run_examples()

if __name__ == '__main__':
    main(sys.argv[1:])
