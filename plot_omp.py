#!/usr/bin/env python3
import matplotlib.pyplot as plt


def main() -> int:
    output = "omp_timing_server.png"
    threads_range = [1,2,4,6,8,12]
    timings = [86.756,46.992,25.235,18.734,15.927,12.671]

    plt.figure()
    plt.plot(threads_range, timings, marker="o")
    plt.title("tsp_ga2opt_omp runtime vs. threads")
    plt.xlabel("OMP_NUM_THREADS")
    plt.ylabel("Time (s)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output)
    print(f"Chart saved to {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
