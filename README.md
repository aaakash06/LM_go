# Efficient Attention in Small Language Models

A controlled experimental study of different causal self-attention implementations inside a decoder-only Transformer language model.

This project implements and compares:

* **Reference attention** — explicit scaled dot-product attention with a causal mask
* **PyTorch SDPA** — PyTorch's fused `scaled_dot_product_attention`
* **Chunked attention** — a memory-aware implementation that computes attention over query chunks

The goal is not to assume that one implementation is universally better, but to **measure the trade-offs between correctness, training behavior, runtime, throughput, and memory usage** under controlled conditions.

---

## Research Question

Modern Transformer implementations often replace straightforward attention code with optimized kernels or memory-aware algorithms.

This project asks:

> **How do different causal-attention implementations affect computational efficiency and language-modeling behavior when the underlying Transformer architecture is held constant?**

More specifically:

1. Does fused PyTorch SDPA provide measurable throughput improvements over explicit attention?
2. How does attention runtime change as sequence length increases?
3. How does a chunked attention implementation compare with full attention?
4. Do different attention implementations produce meaningfully different language-modeling results under the same training setup?
5. What computational trade-offs become visible on Apple Silicon using the MPS backend?

---

## Experimental Design

To make the comparison meaningful, the experiments keep the following factors fixed:

* Transformer architecture
* Number of parameters
* Vocabulary
* Dataset
* Optimizer
* Learning-rate configuration
* Batch size
* Number of training steps
* Random seeds
* Model initialization
* Evaluation procedure
* Hardware/backend

The primary experimental variables are:

* **Attention implementation**

  * Reference
  * SDPA
  * Chunked
* **Sequence length**

  * 128
  * 256
  * 512
* **Random seed**

  * 1
  * 2
  * 3

This produces:

**3 attention implementations × 3 sequence lengths × 3 seeds = 27 training runs**

A separate attention benchmark evaluates sequence lengths up to 1024 tokens.

---

## Results

### Attention Benchmark

The attention implementations were benchmarked on Apple Silicon using PyTorch's MPS backend.

| Attention | Seq. Length | Tokens/sec |
| --------- | ----------: | ---------: |
| Reference |         128 |      ~715K |
| SDPA      |         128 |     ~2.64M |
| Chunked   |         128 |     ~1.71M |
| Reference |         256 |     ~1.60M |
| SDPA      |         256 |     ~2.64M |
| Chunked   |         256 |     ~1.70M |
| Reference |         512 |     ~1.15M |
| SDPA      |         512 |     ~1.93M |
| Chunked   |         512 |     ~1.14M |
| Reference |        1024 |      ~707K |
| SDPA      |        1024 |     ~1.21M |
| Chunked   |        1024 |      ~584K |

The benchmark shows a substantial throughput advantage for SDPA in these measured configurations, particularly at shorter and medium sequence lengths.

The chunked implementation is primarily included as a research implementation and comparison point. The benchmark does **not** establish that chunking reduces MPS memory usage in this particular setup; the recorded memory measurements should therefore be interpreted cautiously.

### Training Experiments

Training was evaluated across:

* 3 attention implementations
* 3 sequence lengths
* 3 random seeds
* 300 training steps in the full experiment configuration

The resulting CSV contains:

* training loss
* validation loss
* validation perplexity
* parameter count
* total training time
* tokens/second
* device/backend
* sequence length
* attention implementation
* random seed

The experiments are designed to examine whether computational differences between attention implementations translate into observable differences during language-model training.

Raw results are stored in:

```text
results/
├── training_results.csv
└── benchmark_results.csv
```

Generated figures are stored in:

```text
results/figures/
├── validation_loss.png
├── perplexity.png
├── throughput.png
├── attention_runtime.png
└── attention_throughput.png
```

---

# Model Architecture

The language model is a small decoder-only Transformer implemented directly in PyTorch.

The default configuration is:

```text
Embedding dimension: 256
Transformer layers: 4
Attention heads: 4
Feed-forward dimension: 1024
Maximum sequence length: configurable
Normalization: RMSNorm
Activation: SwiGLU
Position representation: learned positional embeddings
Dropout: configurable
Language-model head: tied with token embeddings
```

The model uses a **pre-normalization Transformer architecture**.

Conceptually:

```text
Input Tokens
     │
     ▼
Token Embeddings
     │
     +── Positional Embeddings
     │
     ▼
┌─────────────────────────┐
│ Transformer Block       │
│                         │
│ RMSNorm                 │
│    │                    │
│    ▼                    │
│ Causal Self-Attention   │
│    │                    │
│    +──── Residual        │
│                         │
│ RMSNorm                 │
│    │                    │
│    ▼                    │
│ SwiGLU Feed-Forward     │
│    │                    │
│    +──── Residual        │
└─────────────────────────┘
     │
     ▼
   Repeat
     │
     ▼
RMSNorm
     │
     ▼
Tied LM Head
     │
     ▼
Logits
```

---

# Attention Implementations

## 1. Reference Attention

The reference implementation explicitly constructs the attention computation.

For queries, keys, and values:

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

Attention scores are:

$$
S = \frac{QK^T}{\sqrt{d_k}}
$$

A causal mask prevents each token from attending to future tokens:

$$
S_{ij} = -\infty \quad \text{if } j > i
$$

The attention probabilities are:

$$
A = \text{softmax}(S)
$$

and the output is:

$$
O = AV
$$

This implementation is intentionally straightforward and serves as the primary correctness reference.

---

## 2. PyTorch SDPA

The second implementation uses:

```python
torch.nn.functional.scaled_dot_product_attention
```

with causal attention enabled.

Conceptually:

```python
F.scaled_dot_product_attention(
    q,
    k,
    v,
    is_causal=True
)
```

This delegates the attention computation to PyTorch's optimized implementation.

The surrounding model architecture and projection layers remain the same as the reference implementation, allowing the experiment to focus on the attention computation itself.

---

## 3. Chunked Attention

The third implementation divides the query sequence into smaller chunks.

Instead of materializing the full:

```text
[B, H, T, T]
```

attention-score tensor at once, it processes query positions in blocks.

For example, with:

```text
sequence length = 512
chunk size = 128
```

the query sequence is processed as:

```text
queries 0–127
queries 128–255
queries 256–383
queries 384–511
```

Each chunk computes attention against the full key/value sequence.

The approach reduces the size of the temporary score tensor for each individual computation, while potentially introducing additional computation or kernel-launch overhead.

This makes it useful as a simple implementation for studying the trade-off between memory behavior and computational efficiency.

---

# Correctness Testing

Because optimized attention implementations can introduce subtle numerical or masking errors, the repository includes automated tests.

The test suite checks:

### Reference vs. SDPA

The outputs of the explicit and SDPA implementations are compared under equivalent inputs.

### Reference vs. Chunked

The chunked implementation is compared against the reference implementation.

### Non-divisible sequence lengths

Chunked attention is tested with sequence lengths that are not exact multiples of the chunk size.

### Gradient equivalence

The implementations are tested not only on forward outputs but also on gradients.

### Causal masking

The tests verify that future tokens cannot influence earlier positions.

### Model behavior

Additional tests verify:

* output tensor shapes
* language-model loss calculation
* parameter-count consistency
* autoregressive generation

### Optimizer

The custom AdamW implementation is tested on a simple optimization problem to verify that parameters move toward the expected solution.

Run all tests with:

```bash
python -m pytest tests -v
```

---

# Repository Structure

```text
language-model-research/
│
├── src/
│   ├── __init__.py
│   ├── attention.py
│   ├── flash_attention.py
│   ├── chunked_attention.py
│   ├── model.py
│   ├── optimizer.py
│   ├── tokenizer.py
│   ├── data.py
│   ├── training.py
│   └── evaluation.py
│
├── experiments/
│   ├── __init__.py
│   ├── train_experiment.py
│   ├── benchmark_attention.py
│   └── plot_results.py
│
├── tests/
│   ├── __init__.py
│   ├── test_attention.py
│   ├── test_chunked_attention.py
│   ├── test_model.py
│   └── test_optimizer.py
│
├── results/
│   ├── training_results.csv
│   ├── benchmark_results.csv
│   └── figures/
│
├── checkpoints/
│
├── train.py
├── run_experiments.py
├── run_all.py
├── requirements.txt
└── README.md
```

---

# Data Pipeline

The project intentionally uses a lightweight character-level tokenizer so that the experiments can focus on Transformer implementation rather than tokenizer engineering.

The tokenizer:

1. Builds a vocabulary from the training corpus.
2. Maps characters to integer token IDs.
3. Converts token IDs back into text.

For autoregressive language modeling, each sequence is split into input and target tokens:

```text
Input:
t0 t1 t2 t3 t4

Target:
t1 t2 t3 t4 t5
```

The model therefore learns to predict the next token at every position.

The data pipeline performs a 90/10 train/validation split.

---

# Training Objective

The model is trained using next-token prediction with cross-entropy loss.

Given an input sequence:

```text
x1, x2, ..., xT
```

the model predicts:

```text
x2, x3, ..., xT+1
```

The training objective is:

$$
\mathcal{L}
=
-\frac{1}{T}
\sum_{t=1}^{T}
\log p(x_{t+1}|x_{\leq t})
$$

Validation perplexity is calculated as:

$$
PPL = e^{\mathcal{L}}
$$

Lower perplexity corresponds to lower average negative log-likelihood on the evaluation data.

---

# Optimizer

The repository includes a custom implementation of AdamW rather than relying exclusively on PyTorch's built-in optimizer.

AdamW maintains first- and second-moment estimates:

$$
m_t = \beta_1m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2
$$

with bias correction followed by the parameter update.

Weight decay is applied separately from the gradient update, following the AdamW formulation.

This provides another from-scratch systems component while keeping the training pipeline self-contained.

---

# Hardware

The experiments were run on:

```text
Apple Silicon
PyTorch MPS backend
```

The training and benchmarking code automatically selects the available accelerator:

```text
MPS → CUDA → CPU
```

The selected device is printed during execution and stored in the experiment results.

Because performance characteristics are hardware- and backend-dependent, the benchmark numbers in this repository should be interpreted specifically as measurements from the tested Apple Silicon environment.

They should not be treated as universal rankings of the attention implementations across GPUs, CPUs, or other accelerators.

---

# Installation

Clone the repository and enter the project directory:

```bash
git clone <your-repository-url>
cd language-model-research
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The current requirements are:

```text
torch
numpy
pandas
matplotlib
pytest
```

---

# Quick Start

## 1. Run the tests

```bash
python -m pytest tests -v
```

This verifies the attention implementations, model, and optimizer.

---

## 2. Run a simple training job

```bash
python train.py
```

This runs a small decoder-only Transformer training example using SDPA attention.

At the end, the script evaluates the model and performs autoregressive generation.

---

## 3. Benchmark attention implementations

```bash
python -m experiments.benchmark_attention
```

This benchmarks:

```text
reference
sdpa
chunked
```

over:

```text
128
256
512
1024
```

token sequences.

The results are written to:

```text
results/benchmark_results.csv
```

---

# Running the Research Experiments

The full controlled experiment consists of:

```text
3 attention implementations
× 3 sequence lengths
× 3 random seeds
= 27 experiments
```

For a faster test run:

```bash
python run_experiments.py --steps 50
```

For a larger experiment:

```bash
python run_experiments.py --steps 200
```

For the full configuration:

```bash
python run_experiments.py --steps 300 --fresh
```

The `--fresh` option removes previously generated experiment results before starting a new experiment sweep.

Results are saved to:

```text
results/training_results.csv
```

---

# Generating Figures

After running the experiments:

```bash
python -m experiments.plot_results \
    --training results/training_results.csv \
    --benchmark results/benchmark_results.csv
```

This generates:

```text
results/figures/
├── validation_loss.png
├── perplexity.png
├── throughput.png
├── attention_runtime.png
└── attention_throughput.png
```

These figures make it easier to inspect:

* validation-loss differences
* perplexity differences
* training throughput
* attention runtime scaling
* attention throughput across sequence lengths

---

# Run Everything

For a quick end-to-end smoke test:

```bash
python run_all.py --steps 50
```

For the larger experiment configuration:

```bash
python run_all.py --full
```

The pipeline performs:

```text
1. Unit tests
      ↓
2. Attention benchmarks
      ↓
3. Training experiments
      ↓
4. Result aggregation
      ↓
5. Plot generation
```

---

# Metrics

The project records several complementary metrics.

## Validation Loss

Measures the average cross-entropy loss on held-out data.

Lower values indicate that the model assigns higher probability to the observed validation tokens.

---

## Perplexity

Perplexity is:

$$
PPL = e^{loss}
$$

It provides another way of interpreting language-model likelihood.

---

## Training Throughput

Measured in:

```text
tokens / second
```

This captures the amount of training data processed per unit time.

---

## Attention Runtime

The benchmark records forward-pass runtime for the attention implementations.

This helps isolate the computational cost of the attention mechanism from the rest of the Transformer.

---

## Memory

The benchmark also records available PyTorch/MPS memory measurements.

Memory measurements are backend-dependent and should be interpreted as instrumentation from the tested environment rather than as a universal measurement of peak memory complexity.

---

# Why Multiple Random Seeds?

A single training run can be affected by random initialization and stochastic optimization.

Therefore, each configuration is evaluated using three seeds:

```text
seed 1
seed 2
seed 3
```

This makes it possible to inspect whether observed differences are consistent across independent runs rather than relying on a single trajectory.

For a larger research study, the same methodology could be extended to additional seeds.

---

# Why Multiple Sequence Lengths?

Self-attention has a quadratic relationship with sequence length in its conventional formulation.

The attention-score tensor has the shape:

```text
[B, H, T, T]
```

where `T` is sequence length.

Therefore, increasing sequence length can substantially increase the computational and memory cost of attention.

Testing:

```text
128 → 256 → 512 → 1024
```

allows the benchmark to examine how each implementation behaves as the sequence becomes longer.

---

# Experimental Controls

A meaningful systems comparison requires controlling factors other than the implementation being studied.

The experiments therefore keep the core model configuration constant:

```text
d_model = 256
n_layers = 4
n_heads = 4
d_ff = 1024
```

The attention implementations use the same:

* QKV projection structure
* output projection
* number of heads
* head dimension
* causal masking semantics
* Transformer block structure

This means the primary architectural difference is the implementation of the attention computation itself.

---

# Reproducibility

The training code explicitly sets random seeds before constructing each experiment.

Each result row records:

```text
attention_type
seq_len
seed
steps
batch_size
parameters
train_loss
val_loss
val_perplexity
total_seconds
tokens_per_second
device
```

This makes individual experiment configurations recoverable from the results file.

For reproducible comparisons, use the same:

* Python environment
* PyTorch version
* hardware
* experiment configuration
* random seeds

Hardware-level differences can still affect runtime measurements.

---

# Interpreting the Results

The central purpose of this project is **measurement rather than assumption**.

For example, a faster attention implementation does not automatically imply:

* lower validation loss
* lower perplexity
* lower memory usage
* better scaling on every device
* better overall model quality

Likewise, a memory-aware algorithm may introduce additional computational overhead.

The experiments therefore treat:

```text
modeling behavior
+
runtime
+
throughput
+
memory
```

as separate dimensions.

This makes the project useful as a small-scale systems/ML research study rather than simply a benchmark claiming that one implementation is universally superior.

---

# Limitations

This project is intentionally small and has several limitations.

### Small model

The Transformer is designed to run locally on consumer hardware. Results may differ substantially for larger models.

### Small dataset

The included corpus is intentionally lightweight and is not representative of large-scale language-model training.

### Character-level tokenization

Character-level modeling has very different sequence statistics from modern subword tokenizers such as BPE or SentencePiece.

### Short training runs

The experiments are designed for controlled local experimentation rather than convergence-scale language-model training.

### Hardware specificity

The benchmark was performed on Apple Silicon using MPS. Results may differ on NVIDIA GPUs, AMD GPUs, CPUs, or other accelerators.

### MPS memory measurements

The recorded memory values are backend instrumentation and should not be interpreted as a direct measurement of the theoretical peak memory complexity of each algorithm.

### Limited seeds

Three seeds provide some robustness against initialization variance, but a larger study could use more seeds.

### Limited sequence lengths

The training study evaluates up to 512 tokens, while the standalone benchmark extends to 1024. Larger context lengths would provide a broader view of scaling behavior.

---

# Potential Extensions

Several extensions would make the study substantially more comprehensive.

## RoPE vs. Learned Positional Embeddings

Replace learned positional embeddings with Rotary Position Embeddings (RoPE) and investigate whether positional representation interacts with attention implementation.

---

## Larger Context Windows

Extend the benchmark to:

```text
2048
4096
8192
```

where hardware permits.

This would make scaling behavior more pronounced.

---

## Larger Models

Repeat the experiments with progressively larger configurations:

```text
256d / 4 layers
512d / 8 layers
768d / 12 layers
```

while keeping the experimental methodology consistent.

---

## More Random Seeds

Increase from:

```text
3 seeds
```

to:

```text
5–10 seeds
```

for more robust statistical analysis.

---

## GPU Comparison

Run the same benchmark on:

* NVIDIA CUDA
* Apple MPS
* CPU

to investigate backend-specific behavior.

---

## Longer Training

Run each configuration for substantially more tokens and compare:

* convergence rate
* final validation loss
* final perplexity
* training efficiency

rather than only short-run behavior.

---

## Memory Profiling

Use more specialized memory profiling tools to distinguish:

* allocated memory
* reserved memory
* peak memory
* temporary attention tensors

from simple current-memory measurements.

---

## Statistical Analysis

Aggregate the three seeds for each configuration and report:

```text
mean
standard deviation
confidence intervals
```

for validation loss, perplexity, and throughput.

This would provide a stronger quantitative analysis of experimental variance.

---

# Research Workflow

The repository is structured around a repeatable experimental workflow:

```text
                 ┌───────────────────┐
                 │ Attention         │
                 │ Implementations    │
                 └─────────┬─────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          Reference       SDPA       Chunked
              │            │            │
              └────────────┼────────────┘
                           ▼
                 Same Transformer
                 Same Parameters
                           │
                           ▼
                  Controlled Training
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          Seq 128       Seq 256      Seq 512
              │            │            │
              └────────────┼────────────┘
                           ▼
                    Multiple Seeds
                           │
                           ▼
                 ┌───────────────────┐
                 │ CSV Experiment    │
                 │ Results           │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Analysis + Plots  │
                 └───────────────────┘
```

The important design principle is that **the experimental pipeline is automated rather than relying on manually running and recording individual experiments**.

---

# From-Scratch Components

This project intentionally implements several core components rather than treating the Transformer as a black box.

Implemented directly in the repository:

* causal self-attention
* fused-QKV attention projections
* chunked attention
* Transformer blocks
* RMSNorm
* SwiGLU feed-forward networks
* decoder-only language model
* character tokenizer
* autoregressive dataset
* AdamW optimizer
* cosine-style learning-rate utilities
* training loop
* evaluation
* perplexity calculation
* autoregressive generation
* attention benchmarks
* experiment orchestration
* correctness tests
* result visualization

PyTorch is still used for tensor operations, automatic differentiation, neural-network modules, and the MPS backend.

---

# Design Philosophy

The project follows three principles.

### 1. Correctness First

The explicit reference implementation provides a simple baseline against which optimized implementations can be tested.

### 2. Controlled Comparisons

Model architecture, parameters, data, and training conditions are kept constant whenever possible.

### 3. Measure Rather Than Assume

Performance claims are based on collected measurements rather than assumptions about which implementation should be faster or more memory-efficient.

---

# Example Result Interpretation

A useful analysis does not simply say:

> "SDPA is better."

Instead, the data can be interpreted dimension-by-dimension:

```text
                 Reference     SDPA       Chunked
-----------------------------------------------------
Runtime              ...        ...          ...
Throughput           ...        ...          ...
Validation Loss      ...        ...          ...
Perplexity           ...        ...          ...
Memory               ...        ...          ...
```

This allows computational and modeling behavior to be discussed separately.

For example, the current benchmark demonstrates that SDPA achieved higher measured attention throughput than the reference implementation on the tested MPS configurations. The chunked implementation did not show a throughput advantage in these measurements, illustrating why empirical benchmarking is important.

---

# Project Goals

The project is intended to demonstrate practical understanding of:

* Transformer internals
* causal self-attention
* PyTorch tensor programming
* optimized attention APIs
* memory-aware computation
* language-model training
* optimizer implementation
* experimental design
* reproducibility
* performance benchmarking
* statistical variation across seeds
* ML systems engineering

The broader goal is to bridge the gap between:

```text
"How does a Transformer work?"
```

and:

```text
"How does the implementation of a Transformer affect
the actual computational behavior of a training system?"
```

---

# Future Research Direction

A natural next stage would be to turn this project into a larger attention-systems study.

A possible progression is:

```text
Phase 1
Correctness
    ↓
Phase 2
Small-model benchmarking
    ↓
Phase 3
Multi-seed training experiments
    ↓
Phase 4
Long-context evaluation
    ↓
Phase 5
Larger models
    ↓
Phase 6
Cross-hardware comparison
    ↓
Phase 7
Statistical analysis
```

This would transform the current implementation-focused project into a more extensive empirical study of efficient Transformer inference and training.

---

# License

Add the license of your choice before publishing the repository.

For example:

```text
MIT License
```

If this repository is intended for public use, include the complete license text in a `LICENSE` file.

---

# Acknowledgments

This project is inspired by the implementation-first approach to modern language-model training and systems research, including coursework and materials surrounding Stanford's CS336: Language Modeling from Scratch.

The implementation in this repository is an independent educational/research project.

---

# Summary

This repository implements a small decoder-only Transformer from scratch and uses it as a controlled environment for studying causal-attention implementations.

The experimental pipeline compares:

```text
Reference Attention
        vs.
PyTorch SDPA
        vs.
Chunked Attention
```

across multiple:

```text
sequence lengths
random seeds
performance metrics
```

The project combines low-level Transformer implementation with reproducible experimentation, automated benchmarking, correctness testing, and quantitative analysis.

The emphasis is not on declaring a universally superior attention implementation, but on understanding **how implementation choices affect the measured behavior of a language-model training system under controlled conditions**.
