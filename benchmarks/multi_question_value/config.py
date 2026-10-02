"""Shared config for the multi-question value benchmark.

Plain-classifier hyperparameters deliberately mirror the VSS recipe
(vss-prototype-clinc-slot-ho-qmask.yaml) so the comparison isolates the
architecture: same lr, weight decay, warmup, epochs, batch size, seeds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

QUESTION_COUNTS = [1, 2, 4, 8, 16, 32, 50]
SEEDS = [1, 2, 3]
SYNTH_QUESTION_POOL = 32  # distinct question types per synthetic state


@dataclass
class PlainConfig:
    """Hyperparameters mirroring the VSS training recipe."""
    vocab_size: int = 16384
    hash_buckets: int = 8192
    hidden_size: int = 256
    layers: int = 6
    heads: int = 8
    intermediate_size: int = 1024
    dropout: float = 0.1
    max_sequence_length: int = 4096
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 150
    epochs: int = 8
    batch_size: int = 32
    clip_grad_norm: float = 1.0
    score_bins: int = 64
    seeds: list[int] = field(default_factory=lambda: list(SEEDS))


@dataclass
class BenchConfig:
    # synthetic task
    n_synthetic_states: int = 800
    synth_question_pool: int = SYNTH_QUESTION_POOL
    # real datasets use their existing converted JSONL under data/
    # evaluation
    question_counts: list[int] = field(default_factory=lambda: list(QUESTION_COUNTS))
    n_eval_states: int = 300
    eval_seed: int = 42
    # latency: largest practical on CPU — one iteration is a FULL request
    # (Q sequential forwards for mode A), so 5 warmup + 20 measured already
    # costs minutes per (Q, mode); percentiles are reported over the 20
    warmup_iters: int = 5
    measured_iters: int = 20
    # interference / permutation analyses
    n_interference_states: int = 100
    # environment bookkeeping (filled at runtime)
    torch_version: str = ""
    torch_threads: int = 0
