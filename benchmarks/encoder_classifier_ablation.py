"""Encoder-classifier ablation (docs/validation_report.md U1, experiment 1).

Question: does VSS's typed-question serialization + slot head add value or
cost on real data? We train the SAME bidirectional encoder (same size, seed,
optimizer, warmup, epochs, batches) end-to-end as a plain state classifier
(CLS-style mean-pool over the <STATE> region + linear head over 151 intents)
and compare against the committed VSS slot-ho run.

Matched budget: same batch size, epochs, AdamW lr/wd, warmup+cosine schedule,
gradient clipping, seed. The classifier sees ONLY the serialized state block
(no question block), which is the minimal-faithful "plain head" comparison.

Usage:
    python benchmarks/encoder_classifier_ablation.py --train data/clinc150/train.jsonl \
        --valid data/clinc150/validation_1000.jsonl --test data/clinc150/test.jsonl \
        --labels data/raw/clinc_label_names.json --epochs 8 --seed 13
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class ClassifierModel(nn.Module):
    """Same transformer encoder, plain classification head.

    Inputs are the *state-only* serialization lines, tokenized with the same
    word+hash tokenizer fitted on state-only text. The encoder is wrapped in
    an nn.Module container so its weights actually join the optimizer — the
    VSSModel wraps it in `VSSEncoder`, a deliberate non-Module front-end.
    """

    def __init__(self, inner_encoder, hidden: int, n_classes: int) -> None:
        super().__init__()
        self.enc = _EncoderBox(inner_encoder)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        H = self.enc(token_ids)              # [B, T, hidden]
        mask = (token_ids != 0).unsqueeze(-1).float()
        pooled = (H * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(pooled)             # [B, n_classes]


class _EncoderBox(nn.Module):
    """nn.Module wrapper around TransformerEncoder so parameters register."""

    def __init__(self, inner) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.inner(token_ids)



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/clinc150/train.jsonl")
    ap.add_argument("--valid", default="data/clinc150/validation_1000.jsonl")
    ap.add_argument("--test", default="data/clinc150/test.jsonl")
    ap.add_argument("--labels", default="data/raw/clinc_label_names.json")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3.0e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=150)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ckpt-encoder", default="runs/clinc150-slot-ho/best.pt",
                    help="VSS checkpoint supplying the encoder init + tokenizer")
    ap.add_argument("--ckpt-dir", default="runs/ablation-cls",
                    help="progress-checkpoint dir (supports --resume after timeouts)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--scratch", action="store_true",
                    help="random encoder init instead of VSS checkpoint weights")
    ap.add_argument("--out", default="benchmarks/ablation/encoder_classifier.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    import random

    random.seed(args.seed)

    from vss.model.config import ModelConfig
    from vss.model.serialize import serialize_state
    from vss.model.tokenizer import VSSTokenizer
    from vss.model.transformer import TransformerEncoder
    from vss.model.vss_model import VSSModel

    # ---------- load VSS checkpoint for encoder weights + tokenizer ----------
    ck = torch.load(args.ckpt_encoder, map_location="cpu", weights_only=False)
    raw = ck["config"]
    raw = raw.get("model", raw) if isinstance(raw, dict) else raw
    cfg = ModelConfig(**raw)
    vss = VSSModel(cfg)
    vss.load_state_dict(ck["model"])
    tok_path = Path(args.ckpt_encoder).parent / "final" / "vocab.json"
    tok = VSSTokenizer.load(str(tok_path))

    labels = sorted(json.loads(Path(args.labels).read_text()))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n_classes = len(labels)

    def load_split(path: str):
        out = []
        for line in open(path, encoding="utf-8"):
            ex = json.loads(line)
            text = serialize_state(ex["state"])
            ids = tok.encode(text)
            out.append((ids, label_to_idx[ex["questions"][0]["answer"]]))
        return out

    print("loading splits...", flush=True)
    train = load_split(args.train)
    valid = load_split(args.valid)
    test = load_split(args.test)
    print(f"train={len(train)} valid={len(valid)} test={len(test)}", flush=True)

    # ---------- model: same encoder architecture ----------
    encoder = vss.vss_encoder  # reuse the trained encoder module directly
    if args.scratch:
        torch.manual_seed(args.seed)  # re-randomize encoder weights
        for name, p in encoder.encoder.named_parameters():
            if "norm" in name and p.dim() == 1:
                torch.nn.init.ones_(p)  # RMSNorm gains must start at 1
            elif p.dim() > 1:
                torch.nn.init.trunc_normal_(p, std=0.02)
            else:
                torch.nn.init.zeros_(p)  # biases
        if args.resume and (Path(args.ckpt_dir) / "last.pt").exists():
            print("NOTE: --scratch + --resume: checkpoint weights will override the "
                  "random init below (expected)", flush=True)
    model = ClassifierModel(encoder.encoder, cfg.hidden_size, n_classes)
    if args.scratch:
        _probe = sum(1 for n, p in model.named_parameters()
                     if "norm" in n and float(p.std()) < 1e-4)
        print(f"init check: {_probe} norm params with ~zero std (expect 0)", flush=True)
        _emb_std = float(model.enc.inner.token_emb.weight.std())
        print(f"init check: token_emb std {_emb_std:.4f} (expect ~0.02)", flush=True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train} (expect ~11.2M if encoder joined)", flush=True)
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup, sched_kind = args.warmup_steps, "cosine"

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def batches(split, bs, shuffle_seed=None):
        idx = list(range(len(split)))
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(idx)
        for s in range(0, len(idx), bs):
            chunk = [split[i] for i in idx[s : s + bs]]
            T = max(len(ids) for ids, _ in chunk)
            x = torch.zeros(len(chunk), T, dtype=torch.long)
            for j, (ids, _) in enumerate(chunk):
                x[j, : len(ids)] = torch.tensor(ids)
            y = torch.tensor([y for _, y in chunk])
            yield x, y

    def evaluate(split) -> float:
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in batches(split, args.batch_size):
                pred = model(x).argmax(-1)
                correct += int((pred == y).sum())
        return correct / len(split)

    # ---------- train ----------
    model.train()
    step = 0
    best_val, best_epoch = 0.0, -1
    t0 = time.time()
    ckpt_dir = Path(args.ckpt_dir)
    start_epoch = 0
    if args.resume and (ckpt_dir / "last.pt").exists():
        st = torch.load(ckpt_dir / "last.pt", weights_only=False, map_location="cpu")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step = st["step"]
        start_epoch = st["epoch"] + 1
        best_val, best_epoch = st["best_val"], st["best_epoch"]
        print(f"resumed at epoch {start_epoch} step {step}", flush=True)

    def save_progress(epoch: int) -> None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step,
                    "epoch": epoch, "best_val": best_val, "best_epoch": best_epoch},
                   ckpt_dir / "last.pt")

    for epoch in range(start_epoch, args.epochs):
        ep_loss, nb = 0.0, 0
        for x, y in batches(train, args.batch_size, shuffle_seed=args.seed + epoch):
            loss = torch.nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            ep_loss += float(loss)
            nb += 1
            step += 1
            if step % args.save_every == 0:
                save_progress(epoch)
        val_acc = evaluate(valid)
        best_val = max(best_val, val_acc)
        best_epoch = epoch if val_acc >= best_val else best_epoch
        save_progress(epoch)
        print(f"epoch {epoch}: train {ep_loss / max(1, nb):.4f} val_acc {val_acc:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    test_acc = evaluate(test)
    print(f"TEST accuracy {test_acc:.4f} (best val {best_val:.4f} @ epoch {best_epoch})", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "ablation": ("same encoder architecture, plain linear head, state-only input; "
                     + ("random init" if args.scratch else "warm start from VSS slot-ho encoder")),
        "train_args": {k: getattr(args, k) for k in (
            "train", "epochs", "batch_size", "lr", "weight_decay", "warmup_steps", "seed")},
        "n_classes": n_classes,
        "total_steps": total_steps,
        "best_val_accuracy": round(best_val, 4),
        "test_accuracy_state_only_classifier": round(test_acc, 4),
        "vss_reference_full_schema_accuracy": 0.7398,
        "vss_reference_artifact": "benchmarks/ood/clinc150_slot_ho_best.json",
        "wall_seconds": round(time.time() - t0, 1),
        "note": "identical optimizer/schedule/epochs/seed as the VSS run; the only "
                "differences are the input (state-only serialization, no question "
                "block) and the head (linear over 151 intents instead of slot-CE).",
    }, indent=2))
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
