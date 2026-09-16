import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run controlled attention experiments."
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )

    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[128, 256, 512],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--results",
        default="results/training_results.csv",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete old training results first.",
    )

    args = parser.parse_args()

    if args.fresh and os.path.exists(
        args.results
    ):
        os.remove(args.results)

    os.makedirs(
        "results",
        exist_ok=True,
    )

    total = (
        len(args.seeds)
        * len(args.seq_lens)
        * 3
    )

    current = 0

    for seed in args.seeds:
        for seq_len in args.seq_lens:
            for attention in [
                "reference",
                "sdpa",
                "chunked",
            ]:
                current += 1

                print()
                print(
                    "#" * 70
                )

                print(
                    f"Experiment "
                    f"{current}/{total}"
                )

                print(
                    f"attention={attention}"
                )

                print(
                    f"sequence_length={seq_len}"
                )

                print(
                    f"seed={seed}"
                )

                print(
                    "#" * 70
                )

                command = [
                    sys.executable,
                    "-m",
                    "experiments.train_experiment",
                    "--attention",
                    attention,
                    "--seq-len",
                    str(seq_len),
                    "--seed",
                    str(seed),
                    "--steps",
                    str(args.steps),
                    "--batch-size",
                    str(args.batch_size),
                    "--results",
                    args.results,
                ]

                subprocess.run(
                    command,
                    check=True,
                )

    print()
    print(
        "=" * 70
    )

    print(
        "ALL TRAINING EXPERIMENTS COMPLETE"
    )

    print(
        f"Results: {args.results}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()