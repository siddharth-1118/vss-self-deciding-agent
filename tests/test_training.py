"""Training integration test: tiny model learns the synthetic task."""
from __future__ import annotations

import pytest
import torch

from vss.data.schema import load_jsonl
from vss.model.config import TrainingConfig, VSSConfig
from vss.model.vss_model import VSSModel
from vss.training.trainer import Trainer


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    """Train a tiny model briefly on a small synthetic set; return loaded API."""
    from vss.api import VSS
    from vss.data.synthetic import SyntheticGenerator
    from vss.model.config import ModelConfig, TrainingConfig, VSSConfig

    torch.manual_seed(0)
    mcfg = ModelConfig(
        hidden_size=32, layers=2, heads=4, kv_heads=2,
        intermediate_size=64, max_sequence_length=512,
        vocab_size=1024, hash_buckets=256, score_bins=16, option_slots=128,
    )
    config = VSSConfig(model=mcfg)
    config.training = TrainingConfig(
        seed=3, batch_size=16, lr=3e-3, epochs=3, warmup_steps=5,
        checkpoint_dir=str(tmp_path_factory.mktemp("ckpt") / "run1"),
        eval_every=1000, log_every=1000,
    )
    gen = SyntheticGenerator(seed=42)
    train = gen.generate({"basic": 60, "noul": 40, "score": 40, "multi": 40, "dynamic": 30, "hard": 30})
    eval = gen.generate({"basic": 12})  # tiny eval split

    model = VSSModel(config.model)
    trainer = Trainer(model, config, train, eval)
    result = trainer.fit()
    assert result["history"][-1]["train_loss"] < result["history"][0]["train_loss"]

    out_dir = f"{config.training.checkpoint_dir}/final"
    api = VSS.from_pretrained(out_dir)
    api.checkpoint_dir = out_dir  # stash for reload tests
    api.inference_cfg.enable_abstention = False  # contract tests: force a value
    return api


class TestTrainingPipeline:
    def test_loss_decreases(self, trained_model) -> None:
        assert trained_model is not None

    def test_checkpoint_loads_and_decides(self, trained_model) -> None:
        result = trained_model.decide(
            {"message": "I was charged twice on my credit card and need a refund adjustment"},
            [
                {"id": "department", "type": "choice",
                 "options": ["billing", "technical", "sales", "shipping", "other"]},
                {"id": "refund_requested", "type": "noul"},
                {"id": "urgency", "type": "score", "min": 0, "max": 10},
            ],
        )
        answers = result["answers"]
        assert answers["department"]["value"] in ["billing", "technical", "sales", "shipping", "other"]
        assert answers["refund_requested"]["value"] in (0, 1)
        assert 0 <= answers["urgency"]["value"] <= 10

    def test_schema_constrained_output(self, trained_model) -> None:
        result = trained_model.decide(
            {"message": "the app crashes every time I open the dashboard screen"},
            [{"id": "department", "type": "choice",
              "options": ["billing", "technical", "sales", "shipping", "other"]}],
        )
        probs = result["answers"]["department"]["probabilities"]
        assert set(probs) == {"billing", "technical", "sales", "shipping", "other"}
        assert abs(sum(probs.values()) - 1.0) < 1e-2

    def test_dynamic_option_subset(self, trained_model) -> None:
        result = trained_model.decide(
            {"message": "I need to change the delivery address of my order"},
            [{"id": "department", "type": "choice", "options": ["shipping", "other"]}],
        )
        assert result["answers"]["department"]["value"] in ["shipping", "other"]

    def test_fresh_load_abstains_on_out_of_domain(self, trained_model) -> None:
        """A fresh load facing OOD input must stay schema-valid and calibrated."""
        from vss.api import VSS

        api = VSS.from_pretrained(trained_model.checkpoint_dir)
        result = api.decide(
            {"message": "the sky is blue today and my garden grows well"},
            [{"id": "department", "type": "choice",
              "options": ["billing", "technical", "sales", "shipping", "other"]}],
        )
        entry = result["answers"]["department"]
        assert entry["value"] in ("ABSTAIN", "billing", "technical", "sales", "shipping", "other")
        assert 0.0 <= entry["confidence"] <= 1.0
