"""Shared test fixtures: a tiny model config and small synthetic datasets."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from vss.data.schema import TrainingExample  # noqa: E402
from vss.data.synthetic import SyntheticGenerator  # noqa: E402
from vss.model.config import ModelConfig  # noqa: E402
from vss.model.vss_model import VSSModel  # noqa: E402


@pytest.fixture(scope="session")
def tiny_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=32,
        layers=2,
        heads=4,
        kv_heads=2,
        intermediate_size=64,
        max_sequence_length=512,
        dropout=0.0,
        vocab_size=1024,
        hash_buckets=256,
        score_bins=16,
        option_slots=128,
    )


@pytest.fixture(scope="session")
def tiny_model(tiny_config: ModelConfig) -> VSSModel:
    torch.manual_seed(0)
    return VSSModel(tiny_config)


@pytest.fixture(scope="session")
def synth_train() -> list[TrainingExample]:
    gen = SyntheticGenerator(seed=7)
    return gen.generate({"basic": 40, "noul": 30, "score": 30, "multi": 30, "dynamic": 20, "hard": 20})


@pytest.fixture(scope="session")
def synth_eval() -> list[TrainingExample]:
    gen = SyntheticGenerator(seed=8)
    return gen.generate({"basic": 12, "noul": 8, "score": 8, "multi": 8, "dynamic": 6, "hard": 6})
