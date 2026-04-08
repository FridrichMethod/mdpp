"""Benchmark: sequential vs threaded vs multiprocess trajectory loading.

Measures wall time and peak memory (RSS) for each strategy.

Run directly:  python tests/core/bench_load_trajectories.py
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import mdtraj as md

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results")
TOPOLOGY = "step5_production_complex_fit.pdb"
TRAJECTORY = "step5_production_complex_fit.xtc"
SYSTEM = "BirA-bioAMP"
STRIDE = 10
N_FRAMES = 1000


def _find_replica_paths() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for i in range(1, 7):
        replica_dir = RESULTS_DIR / f"replica{i}"
        matches = sorted(replica_dir.glob(f"{SYSTEM}_*/"))
        if matches:
            sim_dir = matches[0]
            pairs.append((sim_dir / TRAJECTORY, sim_dir / TOPOLOGY))
    return pairs


def _load_one(args: tuple[str, str, int, int]) -> int:
    xtc, top, stride, n_frames = args
    topology = md.load_topology(top)
    with md.open(xtc) as fh:
        traj = fh.read_as_traj(topology, n_frames=n_frames, stride=stride)
    return traj.n_frames


def _get_rss_mb() -> float:
    """Get current RSS in MB (Linux only)."""
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def bench_sequential(pairs: list[tuple[Path, Path]]) -> tuple[float, float]:
    rss_before = _get_rss_mb()
    t0 = time.perf_counter()
    for xtc, top in pairs:
        _load_one((str(xtc), str(top), STRIDE, N_FRAMES))
    elapsed = time.perf_counter() - t0
    rss_after = _get_rss_mb()
    return elapsed, rss_after - rss_before


def bench_threads(pairs: list[tuple[Path, Path]], workers: int) -> tuple[float, float]:
    args = [(str(xtc), str(top), STRIDE, N_FRAMES) for xtc, top in pairs]
    rss_before = _get_rss_mb()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_load_one, args))
    elapsed = time.perf_counter() - t0
    rss_after = _get_rss_mb()
    return elapsed, rss_after - rss_before


def bench_pool(pairs: list[tuple[Path, Path]], workers: int) -> tuple[float, float]:
    args = [(str(xtc), str(top), STRIDE, N_FRAMES) for xtc, top in pairs]
    rss_before = _get_rss_mb()
    t0 = time.perf_counter()
    with Pool(processes=workers) as pool:
        pool.map(_load_one, args)
    elapsed = time.perf_counter() - t0
    rss_after = _get_rss_mb()
    return elapsed, rss_after - rss_before


def main() -> None:
    pairs = _find_replica_paths()
    if not pairs:
        print("No replica directories found under results/. Skipping benchmark.")
        return

    print(f"Loading {len(pairs)} replicas of {SYSTEM}, stride={STRIDE}, n_frames={N_FRAMES}")
    print(f"{'Method':<35} {'Time':>7} {'Speedup':>8} {'RSS delta':>10}")
    print("-" * 65)

    t_seq, mem_seq = bench_sequential(pairs)
    print(f"{'Sequential':<35} {t_seq:>6.2f}s {'1.00x':>8} {mem_seq:>+8.1f} MB")

    for w in [2, 4, 6]:
        t, mem = bench_threads(pairs, w)
        label = f"ThreadPoolExecutor (workers={w})"
        print(f"{label:<35} {t:>6.2f}s {t_seq / t:>7.2f}x {mem:>+8.1f} MB")

    for w in [2, 4, 6]:
        t, mem = bench_pool(pairs, w)
        label = f"mp.Pool (workers={w})"
        print(f"{label:<35} {t:>6.2f}s {t_seq / t:>7.2f}x {mem:>+8.1f} MB")


if __name__ == "__main__":
    main()
