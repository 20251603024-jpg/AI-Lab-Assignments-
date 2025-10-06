#!/usr/bin/env python3
\"\"\"Lab 4 — Jigsaw Puzzle Reconstruction using Simulated Annealing
Produces: puzzle_original.png, puzzle_scrambled.png, puzzle_recovered.png, sa_energy.png
Usage:
    python3 lab4_jigsaw_sa.py            # run demo with synthetic image
    python3 lab4_jigsaw_sa.py --image path/to/image.png --grid 4 --iters 12000
\"\"\"

import argparse, random, math, time
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------- Synthetic image -----------------------
def make_synthetic_image(size=256, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(3):
        x = np.linspace(0, 255, size, dtype=np.uint8)
        grad = np.tile(x, (size,1))
        if i == 1: grad = grad.T
        img[:,:,i] = grad
    for _ in range(40):
        r = int(rng.integers(6, 24))
        cx = int(rng.integers(r, size-r))
        cy = int(rng.integers(r, size-r))
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        yy, xx = np.ogrid[:size, :size]
        mask = (xx-cx)**2 + (yy-cy)**2 <= r*r
        img[mask] = color
    for k in range(-20, 21):
        rr = np.clip(np.arange(size)+k, 0, size-1)
        img[np.arange(size), rr, :] = 255 - img[np.arange(size), rr, :]
    return img

# ----------------------- Tiling utilities -----------------------
def tile_image(img, grid=4):
    h, w, _ = img.shape
    assert h % grid == 0 and w % grid == 0, \"Image dimensions must be divisible by grid.\"
    th, tw = h//grid, w//grid
    tiles = []
    for r in range(grid):
        for c in range(grid):
            tiles.append(img[r*th:(r+1)*th, c*tw:(c+1)*tw].copy())
    return tiles, (th, tw)

def compose_image(tiles, grid, tile_size):
    th, tw = tile_size
    H, W = th*grid, tw*grid
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, grid)
        out[r*th:(r+1)*th, c*tw:(c+1)*tw] = tile
    return out

# ----------------------- Edge profiles & pairwise costs -----------------------
def edge_profiles(tiles):
    L,R,T,B = [],[],[],[]
    for t in tiles:
        L.append(t[:,0,:].astype(np.int16))
        R.append(t[:,-1,:].astype(np.int16))
        T.append(t[0,:,:].astype(np.int16))
        B.append(t[-1,:,:].astype(np.int16))
    return np.array(L), np.array(R), np.array(T), np.array(B)

def pair_costs(L, R, T, B):
    n = L.shape[0]
    right_cost = np.zeros((n,n), dtype=np.float32)
    down_cost = np.zeros((n,n), dtype=np.float32)
    for a in range(n):
        # difference between right edge of a and left edge of b
        right_cost[a,:] = np.mean(np.abs(R[a][None,:,:] - L), axis=(1,2))
        # difference between bottom edge of a and top edge of b
        down_cost[a,:]  = np.mean(np.abs(B[a][None,:,:] - T), axis=(1,2))
    return right_cost, down_cost

# ----------------------- Energy and delta -----------------------
class JigsawEnergy:
    def __init__(self, RC, DC, grid):
        self.RC = RC; self.DC = DC; self.grid = grid; self.N = grid*grid
    def energy_full(self, placement):
        E = 0.0; g = self.grid
        for r in range(g):
            for c in range(g):
                idx = r*g + c
                t = placement[idx]
                if c < g-1:
                    E += self.RC[t, placement[idx+1]]
                if r < g-1:
                    E += self.DC[t, placement[idx+g]]
        return float(E)
    def delta_swap(self, placement, p, q):
        if p == q: return 0.0
        g = self.grid
        def contrib(pos, tile_at):
            r, c = divmod(pos, g); E = 0.0
            # right neighbor
            if c < g-1:
                nb = placement[pos+1] if pos+1 not in (p,q) else (placement[q] if pos+1==p else placement[p])
                E += self.RC[tile_at, nb]
            # left neighbor
            if c > 0:
                nbp = pos-1
                nb = placement[nbp] if nbp not in (p,q) else (placement[q] if nbp==p else placement[p])
                E += self.RC[nb, tile_at]
            # down neighbor
            if r < g-1:
                nb = placement[pos+g] if pos+g not in (p,q) else (placement[q] if pos+g==p else placement[p])
                E += self.DC[tile_at, nb]
            # up neighbor
            if r > 0:
                nbp = pos-g
                nb = placement[nbp] if nbp not in (p,q) else (placement[q] if nbp==p else placement[p])
                E += self.DC[nb, tile_at]
            return E
        t_p, t_q = placement[p], placement[q]
        before = contrib(p, t_p) + contrib(q, t_q)
        after  = contrib(p, t_q) + contrib(q, t_p)
        return after - before

# ----------------------- Simulated Annealing -----------------------
def simulated_annealing(energy_model, placement_init, T0=60.0, alpha=0.997, iters=12000, rng=None):
    rng = rng or random.Random(1)
    placement = placement_init.copy()
    E = energy_model.energy_full(placement)
    best_E = E; best = placement.copy()
    history = [E]; accepts = 0
    T = T0; N = len(placement)
    start_time = time.time()
    for k in range(1, iters+1):
        p = rng.randrange(N); q = rng.randrange(N)
        while q == p: q = rng.randrange(N)
        dE = energy_model.delta_swap(placement, p, q)
        if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-12)):
            placement[p], placement[q] = placement[q], placement[p]
            E += dE; accepts += 1
            if E < best_E:
                best_E = E; best = placement.copy()
        history.append(E); T *= alpha
    elapsed_ms = int((time.time() - start_time) * 1000)
    return best, best_E, history, accepts, elapsed_ms

# ----------------------- CLI and demo -----------------------
def parse_args():
    p = argparse.ArgumentParser(description='Lab 4: Jigsaw via Simulated Annealing')
    p.add_argument('--image', type=str, default=None, help='Path to input image (optional). If absent, a synthetic image is used.')
    p.add_argument('--grid', type=int, default=4, help='Tile grid size (g for g x g).')
    p.add_argument('--iters', type=int, default=12000, help='SA iterations.')
    p.add_argument('--seed', type=int, default=3, help='Random seed.')
    return p.parse_args()

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    # load or generate image
    if args.image:
        img = np.array(Image.open(args.image).convert('RGB'))
    else:
        img = make_synthetic_image(size=256, seed=args.seed)
    grid = args.grid
    tiles, tile_size = tile_image(img, grid=grid)
    N = grid*grid
    # scramble tiles (random permutation)
    perm = np.arange(N)
    rng.shuffle(perm)
    scrambled_tiles = [tiles[i] for i in perm]
    scrambled_img = compose_image(scrambled_tiles, grid, tile_size)
    # compute pairwise edge costs on original tiles (so that correct adjacency is low cost)
    L,R,T,B = edge_profiles(tiles)
    RC, DC = pair_costs(L, R, T, B)
    energy = JigsawEnergy(RC, DC, grid)
    # initial placement: perm (tile indices at each board position)
    placement0 = perm.copy().tolist()
    best_place, best_E, history, accepts, elapsed_ms = simulated_annealing(energy, placement0, T0=60.0, alpha=0.997, iters=args.iters, rng=random.Random(args.seed))
    recovered_tiles = [tiles[i] for i in best_place]
    recovered_img = compose_image(recovered_tiles, grid, tile_size)
    out = Path('.')
    Image.fromarray(img).save(out / 'puzzle_original.png')
    Image.fromarray(scrambled_img).save(out / 'puzzle_scrambled.png')
    Image.fromarray(recovered_img).save(out / 'puzzle_recovered.png')
    # energy plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4)); plt.plot(history); plt.xlabel('Iteration'); plt.ylabel('Energy'); plt.tight_layout()
    plt.savefig(out / 'sa_energy.png'); plt.close()
    # summary
    print('Grid:', grid, 'Tiles:', N)
    print('Initial energy:', history[0])
    print('Final best energy:', best_E)
    print('Accepts:', accepts, 'Iterations:', args.iters)
    print('Runtime (ms):', elapsed_ms)
    print('Saved: puzzle_original.png, puzzle_scrambled.png, puzzle_recovered.png, sa_energy.png')

if __name__ == '__main__':
    main()
