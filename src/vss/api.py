"""Public VSS API.

Example:
    from vss import VSS

    model = VSS.from_pretrained("runs/prototype/final")
    result = model.decide(
        state={"message": "My payment was charged twice."},
        questions=[
            {"id": "department", "type": "choice",
             "options": ["billing", "technical", "sales", "other"]},
            {"id": "refund", "type": "noul"},
        ],
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .inference.batching import run_batch
from .inference.engine import decide_once
from .inference.schema import validate_output, validate_request
from .model.config import InferenceConfig, VSSConfig
from .model.vss_model import VSSModel


class VSS:
    """User-facing decision model handle."""

    def __init__(self, model: VSSModel, inference_cfg: InferenceConfig) -> None:
        self.model = model
        self.inference_cfg = inference_cfg
        self.device = next(model.parameters()).device.type

    # ------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, path: str, device: str | None = None) -> "VSS":
        """Load a trained checkpoint directory (config.yaml + model.safetensors)."""
        p = Path(path)
        if not (p / "config.yaml").exists():
            raise FileNotFoundError(f"{p} is not a VSS checkpoint (missing config.yaml)")
        cfg: VSSConfig = VSSConfig.load(p / "config.yaml")
        model = VSSModel.load_pretrained(str(p), device=device or "cpu")
        if device is None and torch.cuda.is_available():
            model = model.to("cuda")
        return cls(model, cfg.inference)

    # ------------------------------------------------------------ decisions
    def decide(
        self,
        state: dict[str, Any] | list[Any],
        questions: list[dict[str, Any]],
        *,
        abstain_threshold: float | None = None,
        enable_abstention: bool | None = None,
    ) -> dict[str, Any]:
        """One forward pass: state + typed questions -> typed answers."""
        req = validate_request({"state": state, "questions": questions})
        qs = [q.model_dump(exclude_none=True) for q in req.questions]
        thr = self.inference_cfg.abstain_threshold if abstain_threshold is None else abstain_threshold
        abst = self.inference_cfg.enable_abstention if enable_abstention is None else enable_abstention
        result = decide_once(
            self.model,
            req.state,
            qs,
            abstain_threshold=thr,
            enable_abstention=abst,
            confidence_mode=self.inference_cfg.confidence_mode,
            device=self.device,
        )
        validate_output(result["answers"], req)
        return result

    def decide_batch(
        self,
        states: list[dict[str, Any] | list[Any]],
        questions_list: list[list[dict[str, Any]]],
        *,
        batch_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Batched inference: chunks of one padded forward each."""
        bs = batch_size or self.inference_cfg.batch_size
        out: list[dict[str, Any]] = []
        for i in range(0, len(states), bs):
            chunk_s = states[i : i + bs]
            chunk_q = questions_list[i : i + bs]
            reqs = [validate_request({"state": s, "questions": q}) for s, q in zip(chunk_s, chunk_q)]
            qs = [[q.model_dump(exclude_none=True) for q in r.questions] for r in reqs]
            per = run_batch(
                self.model, chunk_s, qs,
                abstain_threshold=self.inference_cfg.abstain_threshold,
                enable_abstention=self.inference_cfg.enable_abstention,
                confidence_mode=self.inference_cfg.confidence_mode,
                device=self.device,
            )
            for r, ans in zip(reqs, per):
                validate_output(ans["answers"], r)
            out.extend(per)
        return out
