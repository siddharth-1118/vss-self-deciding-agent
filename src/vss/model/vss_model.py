"""Top-level VSS model: encoder + question adapter + decision heads.

One forward pass evaluates every question against the shared state
representation. See docs/architecture.md for the full diagram and
provenance tags ([public] / [std] / [vss]).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..model.config import ModelConfig, preset_config_path, VSSConfig
from ..model.questions import QuestionSpec
from .encoder import VSSEncoder
from .transformer import TransformerEncoder


class VSSModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer: object | None = None  # attached after fit/load
        self.encoder = TransformerEncoder(
            vocab_size=config.vocab_size,
            dim=config.hidden_size,
            layers=config.layers,
            heads=config.heads,
            kv_heads=config.kv_heads,
            ffn_dim=config.intermediate_size,
            max_seq_len=config.max_sequence_length,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
            pad_token_id=config.pad_token_id,
        )
        from .encoder import VSSEncoder as _E  # local import to avoid cycle

        self.vss_encoder = _E(self.encoder, None)  # tokenizer injected later
        from .questions import DecisionHeads

        self.heads = DecisionHeads(
            dim=config.hidden_size,
            num_option_slots=config.option_slots,
            score_bins=config.score_bins,
            use_refine=getattr(config, "use_refine_choice", True),
        )

    # ------------------------------------------------------------ inference
    def forward(
        self,
        states: list[dict],
        questions_list: list[list[dict]],
        device: torch.device | str = "cpu",
    ) -> dict:
        specs = [[QuestionSpec.from_dict(q) for q in qs] for qs in questions_list]
        if not getattr(self.config, "header_only_choice", False):
            serial_questions = questions_list
        else:
            # mark choice blocks as header-only for serialization [vss]
            serial_questions = [
                [{**q, "header_only_choice": True} for q in qs] for qs in questions_list
            ]
        enc = self.vss_encoder.encode_batch(states, serial_questions, device)
        H = self.encoder(enc["token_ids"])
        qvecs = []
        for b, spans in enumerate(enc["spans"]):
            for (s, e) in spans:
                s = max(0, min(s, H.shape[1] - 1))
                e = max(s + 1, min(e, H.shape[1]))
                qvecs.append(H[b, s:e].mean(dim=0))
        if not qvecs:
            qvecs = [H.new_zeros(H.shape[-1])]
        q = torch.stack(qvecs, 0)
        heads_out = self.heads(q, [s for qs in specs for s in qs])
        rows = heads_out["rows"]
        # group rows back per example, aligned with questions_list [vss]
        per_example_rows: list[list[dict]] = []
        idx = 0
        for qs in specs:
            per_example_rows.append(rows[idx : idx + len(qs)])
            idx += len(qs)
        return {
            "hidden": H,
            "question_vectors": q,
            "token_ids": enc["token_ids"],
            "spans": enc["spans"],
            "rows": rows,
            "per_example_rows": per_example_rows,
        }

    # ------------------------------------------------------------ utilities
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------ save/load
    def save_pretrained(self, path: str) -> None:
        """Save safetensors weights + tokenizer + config into a directory."""
        from pathlib import Path

        from safetensors.torch import save_file

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        flat = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(flat, str(out / "model.safetensors"))
        from .config import VSSConfig

        VSSConfig(model=self.config).save(out / "config.yaml")
        if self.tokenizer is not None:
            self.tokenizer.save(out / "vocab.json")

    @classmethod
    def load_pretrained(cls, path: str, device: str = "cpu") -> "VSSModel":
        from pathlib import Path

        from safetensors.torch import load_file

        p = Path(path)
        config = VSSConfig.load(p / "config.yaml").model
        model = cls(config)
        state = load_file(str(p / "model.safetensors"))
        model.load_state_dict(state, strict=True)
        vocab = p / "vocab.json"
        if vocab.exists():
            from .tokenizer import VSSTokenizer

            model.tokenizer = VSSTokenizer.load(vocab)
            model.vss_encoder.tokenizer = model.tokenizer
        return model.to(device)

    @classmethod
    def from_preset(cls, name: str, device: str = "cpu") -> "VSSModel":
        """Build an (untrained) model from a preset config name."""
        cfg_path = preset_config_path(name)
        if cfg_path is None:
            raise KeyError(f"unknown preset: {name}")
        config = VSSConfig.load(cfg_path).model
        return cls(config).to(device)
