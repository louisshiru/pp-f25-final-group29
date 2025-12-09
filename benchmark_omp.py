#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt


def run_once(binary: str, dataset: str, threads: int) -> float:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    start = time.perf_counter()
    subprocess.run([binary, dataset] if dataset else [binary], env=env, check=True)
    end = time.perf_counter()
    return end - start


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark tsp_ga2opt_omp across thread counts.")
    parser.add_argument(
        "--binary",
        default="./tsp_ga2opt_omp",
        help="Path to tsp_ga2opt_omp executable (default: ./tsp_ga2opt_omp)",
    )
    parser.add_argument(
        "--dataset",
        default="qa194.tsp",
        help="Dataset to pass to the solver; leave empty to use the executable's default dataset.",
    )
    parser.add_argument("--min-threads", type=int, default=1, help="Minimum thread count to test (default: 1)")
    parser.add_argument("--max-threads", type=int, default=8, help="Maximum thread count to test (default: 8)")
    parser.add_argument(
        "--output",
        default="omp_timing.png",
        help="Filename for the generated line chart (default: omp_timing.png)",
    )
    args = parser.parse_args()

    if args.min_threads < 1 or args.max_threads < args.min_threads:
        print("Invalid thread range", file=sys.stderr)
        return 1

    # threads_range = list(range(args.min_threads, args.max_threads + 1))
    threads_range = [1,2,4,6,8,12]
    timings = []

    for t in threads_range:
        print(f"Running with {t} thread(s)...")
        try:
            duration = run_once(args.binary, args.dataset, t)
        except subprocess.CalledProcessError as exc:
            print(f"Run failed for {t} thread(s): {exc}", file=sys.stderr)
            return 1
        timings.append(duration)
        print(f"  Time: {duration:.3f} s")

    for i in range(len(threads_range)):
        print(f"Running with {threads_range[i]} thread(s)...")
        print(f"  Time: {timings[i]:.3f} s")
    

    plt.figure()
    plt.plot(threads_range, timings, marker="o")
    plt.title("tsp_ga2opt_omp runtime vs. threads")
    plt.xlabel("OMP_NUM_THREADS")
    plt.ylabel("Time (s)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Chart saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
