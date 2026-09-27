"""Export a VSS checkpoint.

- safetensors export happens automatically in save_pretrained; this script
  additionally exports ONNX when the `onnx` extra is installed.

Usage:
    python scripts/export.py --model runs/prototype/final --format onnx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from vss.model.vss_model import VSSModel  # noqa: E402


def export_onnx(model: VSSModel, out_dir: Path, max_len: int) -> Path:
    import torch

    try:
        import onnx  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "onnx not installed. Run: pip install 'vss[onnx]' or pip install onnx onnxruntime"
        ) from e

    model.eval()
    vocab_total = model.config.vocab_size + model.config.hash_buckets
    dummy = torch.randint(1, vocab_total, (1, min(64, max_len)))

    class Wrapper(torch.nn.Module):
        def __init__(self, m: VSSModel) -> None:
            super().__init__()
            self.m = m

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            return self.m.encoder(token_ids)

    wrapper = Wrapper(model).eval()
    out_path = out_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["token_ids"],
        output_names=["hidden"],
        dynamic_axes={"token_ids": {0: "batch", 1: "seq"}, "hidden": {0: "batch", 1: "seq"}},
        opset_version=17,
    )
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument("--format", choices=["onnx", "safetensors"], default="safetensors")
    args = ap.parse_args()

    model = VSSModel.load_pretrained(args.model)
    out_dir = Path(args.model)
    if args.format == "safetensors":
        model.save_pretrained(str(out_dir))
        print(json.dumps({"exported": str(out_dir / "model.safetensors"),
                          "parameters": model.num_parameters()}))
        return 0
    path = export_onnx(model, out_dir, model.config.max_sequence_length)
    print(json.dumps({"exported": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
