"""Benchmark: sequential vs threaded vs multiprocess trajectory loading.

Run directly:  python tests/core/bench_load_trajectories.py
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import mdtraj as md

# ---------------------------------------------------------------------------
# Configuration -- point at real trajectories for a meaningful benchmark.
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
    chunks: list[md.Trajectory] = []
    loaded = 0
    chunk_size = min(n_frames, 1000)
    for chunk in md.iterload(xtc, top=top, stride=stride, chunk=chunk_size):
        need = n_frames - loaded
        if chunk.n_frames >= need:
            chunks.append(chunk[:need])
            loaded += need
            break
        chunks.append(chunk)
        loaded += chunk.n_frames
    traj = md.join(chunks)
    return traj.n_frames


def bench_sequential(pairs: list[tuple[Path, Path]]) -> float:
    t0 = time.perf_counter()
    for xtc, top in pairs:
        _load_one((str(xtc), str(top), STRIDE, N_FRAMES))
    return time.perf_counter() - t0


def bench_threads(pairs: list[tuple[Path, Path]], workers: int) -> float:
    args = [(str(xtc), str(top), STRIDE, N_FRAMES) for xtc, top in pairs]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_load_one, args))
    return time.perf_counter() - t0


def bench_pool(pairs: list[tuple[Path, Path]], workers: int) -> float:
    args = [(str(xtc), str(top), STRIDE, N_FRAMES) for xtc, top in pairs]
    t0 = time.perf_counter()
    with Pool(processes=workers) as pool:
        pool.map(_load_one, args)
    return time.perf_counter() - t0


def bench_process_pool_executor(pairs: list[tuple[Path, Path]], workers: int) -> float:
    args = [(str(xtc), str(top), STRIDE, N_FRAMES) for xtc, top in pairs]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_load_one, args))
    return time.perf_counter() - t0


def main() -> None:
    pairs = _find_replica_paths()
    if not pairs:
        print("No replica directories found under results/. Skipping benchmark.")
        return

    print(f"Loading {len(pairs)} replicas of {SYSTEM}, stride={STRIDE}, n_frames={N_FRAMES}\n")

    t_seq = bench_sequential(pairs)
    print(f"  Sequential:                    {t_seq:.2f}s")

    for w in [2, 4, 6]:
        t_thr = bench_threads(pairs, w)
        print(f"  Threads (workers={w}):           {t_thr:.2f}s  ({t_seq / t_thr:.2f}x)")

    for w in [2, 4, 6]:
        t_pool = bench_pool(pairs, w)
        print(f"  mp.Pool (workers={w}):            {t_pool:.2f}s  ({t_seq / t_pool:.2f}x)")

    for w in [2, 4, 6]:
        t_ppe = bench_process_pool_executor(pairs, w)
        print(f"  ProcessPoolExecutor (workers={w}): {t_ppe:.2f}s  ({t_seq / t_ppe:.2f}x)")


if __name__ == "__main__":
    main()
