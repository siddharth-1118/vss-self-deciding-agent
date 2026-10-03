"""VSS configuration objects.

All architectural values are configurable; configs/vss-*.yaml ship presets.
Category tags used across docs:
  - [public]  publicly documented concepts (typed decisions, calibration, ...)
  - [std]     standard ML techniques (RoPE, RMSNorm, SwiGLU, ...)
  - [vss]     VSS-original engineering decisions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class ModelConfig:
    """Architecture hyperparameters."""

    hidden_size: int = 256
    layers: int = 6
    heads: int = 8
    kv_heads: int = 8  # GQA [std]: kv_heads < heads shares K/V projections
    intermediate_size: int = 1024
    max_sequence_length: int = 4096
    dropout: float = 0.0
    vocab_size: int = 24576
    hash_buckets: int = 8192  # [vss] OOV words hashed into extra buckets
    score_bins: int = 64  # ordinal bins for Score heads [vss]
    option_slots: int = 1024  # shared choice-option slots [vss]
    header_only_choice: bool = False  # serialize choice blocks without option text [vss]
    tokenizer_type: str = "word"  # "word" (hash fallback) or "bpe"
    question_masked: bool = False  # block attention between question blocks [vss-qmask]
    # [vss] slot-CE regime: score option slots directly without the per-pair
    # option-refinement MLP. Makes train/eval scoring paths identical so
    # full-schema ranking consistency is trained explicitly.
    use_refine_choice: bool = True
    rope_theta: float = 10000.0
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.hidden_size % self.heads != 0:
            raise ValueError(f"hidden_size={self.hidden_size} not divisible by heads={self.heads}")
        if self.heads % self.kv_heads != 0:
            raise ValueError(f"heads={self.heads} not divisible by kv_heads={self.kv_heads}")
        if self.pad_token_id != 0:
            raise ValueError("VSS reserves token id 0 as PAD; pad_token_id must be 0")


@dataclass
class InferenceConfig:
    abstain_threshold: float = 0.55
    enable_abstention: bool = True
    # [vss] how per-answer confidence is produced:
    #   max_prob        - argmax class probability
    #   calibrated_head - auxiliary correctness head P(correct | inputs)
    #   blend           - product of the two, rescaled to [0, 1]
    confidence_mode: Literal["max_prob", "calibrated_head", "blend"] = "blend"
    deterministic: bool = True
    batch_size: int = 32


@dataclass
class TrainingConfig:
    seed: int = 13
    batch_size: int = 32
    grad_accum: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    # Hard cap on the warmup as a FRACTION of the run. `warmup_steps` alone
    # silently dominates short runs: on an 800-example / 8-epoch schedule
    # (200 optimizer steps) a 150-step warmup spent 75% of training ramping
    # up and left 50 steps of decay, which is a large part of why the
    # synthetic VSS runs looked permanently underfit. At 10% the synthetic
    # schedule is repaired (150 -> 20 steps) while every real-data schedule
    # (2.3k-2.7k steps, 150-step warmup = 5.6-6.6%) is left untouched, so this
    # change cannot be confused with a hyperparameter change on real data.
    warmup_frac: float | None = 0.10
    # GLOBAL optimizer-step budget for the whole run (not per epoch). The
    # cosine schedule is computed over this budget, so a step cap is the
    # right knob for step-matching two systems with different cost per step
    # (VSS consumes a whole multi-question state per step; the plain baseline
    # consumes one (state, question) row).
    max_steps: int | None = None
    epochs: int = 3
    clip_grad_norm: float = 1.0
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    scheduler: Literal["cosine", "linear"] = "cosine"
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {"choice": 1.0, "noul": 1.0, "score": 1.0}
    )
    score_ordinal_weight: float = 0.25
    log_every: int = 50
    eval_every: int = 300
    # optimizer steps between mid-epoch checkpoints. These carry the pinned
    # schedule shape and the accumulated epoch history, so a kill+resume is
    # equivalent to an uninterrupted run (docs/convergence_audit.md finding 6).
    ckpt_every: int = 100
    checkpoint_dir: str = "runs/prototype"
    num_workers: int = 0
    # --- convergence control (docs/convergence_audit.md) ---
    # Stop after `patience` epochs without eval loss improving by more than
    # `early_stop_min_delta`. None disables early stopping (train to `epochs`).
    early_stop_patience: int | None = None
    early_stop_min_delta: float = 1e-3
    # Stop when eval loss has not improved for this many epochs, even if the
    # LR schedule has not finished (catches premature schedule termination).
    min_epochs: int = 1


@dataclass
class VSSConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": vars(self.model),
            "inference": vars(self.inference),
            "training": vars(self.training),
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VSSConfig":
        known = {"model", "inference", "training"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config sections: {sorted(unknown)}")
        return cls(
            model=ModelConfig(**d.get("model", {})),
            inference=InferenceConfig(**d.get("inference", {})),
            training=TrainingConfig(**d.get("training", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "VSSConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})


# Preset registry: VSS.from_pretrained("vss-small") resolves through this.
PRESETS: dict[str, str] = {
    "vss-prototype": "vss-prototype.yaml",
    "vss-small": "vss-small.yaml",
    "vss-base": "vss-base.yaml",
    "vss-large": "vss-large.yaml",
}


def preset_config_path(name: str) -> Path | None:
    """Locate a shipped preset YAML by model name."""
    fname = PRESETS.get(name)
    if fname is None:
        return None
    here = Path(__file__).resolve()
    for root in list(here.parents)[:5]:
        cand = root / "configs" / fname
        if cand.exists():
            return cand
    return None
