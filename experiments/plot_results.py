import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_results(path):
    df = pd.read_csv(path)

    os.makedirs(
        "results/figures",
        exist_ok=True,
    )

    # Validation loss.
    plt.figure()

    for attention_type in df[
        "attention_type"
    ].unique():
        subset = df[
            df["attention_type"]
            == attention_type
        ]

        grouped = (
            subset.groupby("seq_len")[
                "val_loss"
            ]
            .mean()
            .reset_index()
        )

        plt.plot(
            grouped["seq_len"],
            grouped["val_loss"],
            marker="o",
            label=attention_type,
        )

    plt.xlabel("Sequence length")
    plt.ylabel("Mean validation loss")
    plt.title("Validation Loss vs Sequence Length")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/validation_loss.png",
        dpi=160,
    )

    plt.close()

    # Perplexity.
    plt.figure()

    for attention_type in df[
        "attention_type"
    ].unique():
        subset = df[
            df["attention_type"]
            == attention_type
        ]

        grouped = (
            subset.groupby("seq_len")[
                "val_perplexity"
            ]
            .mean()
            .reset_index()
        )

        plt.plot(
            grouped["seq_len"],
            grouped["val_perplexity"],
            marker="o",
            label=attention_type,
        )

    plt.xlabel("Sequence length")
    plt.ylabel("Mean validation perplexity")
    plt.title(
        "Validation Perplexity vs Sequence Length"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/perplexity.png",
        dpi=160,
    )

    plt.close()

    # Throughput.
    plt.figure()

    for attention_type in df[
        "attention_type"
    ].unique():
        subset = df[
            df["attention_type"]
            == attention_type
        ]

        grouped = (
            subset.groupby("seq_len")[
                "tokens_per_second"
            ]
            .mean()
            .reset_index()
        )

        plt.plot(
            grouped["seq_len"],
            grouped["tokens_per_second"],
            marker="o",
            label=attention_type,
        )

    plt.xlabel("Sequence length")
    plt.ylabel("Tokens / second")
    plt.title(
        "Training Throughput vs Sequence Length"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/throughput.png",
        dpi=160,
    )

    plt.close()


def plot_benchmark_results(path):
    df = pd.read_csv(path)

    os.makedirs(
        "results/figures",
        exist_ok=True,
    )

    plt.figure()

    for attention_type in df[
        "attention_type"
    ].unique():
        subset = df[
            df["attention_type"]
            == attention_type
        ]

        plt.plot(
            subset["seq_len"],
            subset["milliseconds"],
            marker="o",
            label=attention_type,
        )

    plt.xlabel("Sequence length")
    plt.ylabel("Milliseconds / iteration")
    plt.title("Attention Runtime Scaling")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/attention_runtime.png",
        dpi=160,
    )

    plt.close()

    plt.figure()

    for attention_type in df[
        "attention_type"
    ].unique():
        subset = df[
            df["attention_type"]
            == attention_type
        ]

        plt.plot(
            subset["seq_len"],
            subset["tokens_per_second"],
            marker="o",
            label=attention_type,
        )

    plt.xlabel("Sequence length")
    plt.ylabel("Tokens / second")
    plt.title("Attention Throughput Scaling")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/attention_throughput.png",
        dpi=160,
    )

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training",
        default=None,
    )

    parser.add_argument(
        "--benchmark",
        default=None,
    )

    args = parser.parse_args()

    if args.training:
        plot_training_results(
            args.training
        )

    if args.benchmark:
        plot_benchmark_results(
            args.benchmark
        )

    print(
        "Plots written to results/figures/"
    )