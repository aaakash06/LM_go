import argparse
import subprocess
import sys


def run(command):
    print()
    print("=" * 70)
    print("RUNNING:")
    print(" ".join(command))
    print("=" * 70)

    subprocess.run(
        command,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--full",
        action="store_true",
    )

    args = parser.parse_args()

    if args.full:
        steps = 300
    else:
        steps = args.steps

    # ------------------------------------------------------------
    # 1. Correctness tests
    # ------------------------------------------------------------

    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "-v",
        ]
    )

    # ------------------------------------------------------------
    # 2. Attention benchmark
    # ------------------------------------------------------------

    run(
        [
            sys.executable,
            "-m",
            "experiments.benchmark_attention",
        ]
    )

    # ------------------------------------------------------------
    # 3. Training experiments
    # ------------------------------------------------------------

    run(
        [
            sys.executable,
            "run_experiments.py",
            "--steps",
            str(steps),
            "--fresh",
        ]
    )

    # ------------------------------------------------------------
    # 4. Generate plots
    # ------------------------------------------------------------

    run(
        [
            sys.executable,
            "-m",
            "experiments.plot_results",
            "--training",
            "results/training_results.csv",
            "--benchmark",
            "results/benchmark_results.csv",
        ]
    )

    print()
    print("=" * 70)
    print("RESEARCH PIPELINE COMPLETE")
    print("=" * 70)
    print()
    print(
        "Results:"
    )
    print(
        "  results/training_results.csv"
    )
    print(
        "  results/benchmark_results.csv"
    )
    print()
    print(
        "Figures:"
    )
    print(
        "  results/figures/"
    )
    print()


if __name__ == "__main__":
    main()