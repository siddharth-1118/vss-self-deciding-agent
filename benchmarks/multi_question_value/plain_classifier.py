"""System A/B: plain classifier called once per question.

A deliberately simple and strong conventional classifier:
  - SAME word+hash tokenizer as VSS, fitted on the same training texts
  - SAME transformer encoder architecture (hidden 256, 6 layers, 8 heads,
    SwiGLU 1024, RoPE) trained FROM SCRATCH on the same split
  - input per row = the same canonical serialization VSS sees:
    serialize_example(state, [question])
  - typed heads: choice -> linear over the label inventory (masked CE over
    the question's declared options + a trained ABSTAIN class, mirroring
    VSS's abstain logit), noul -> sigmoid, score -> 64 ordinal bins with
    expected-value decoding (identical to VSS's ScoreHead).
  - one forward pass per (state, question) pair: mode A runs the pairs
    sequentially, mode B batches them.

Fairness notes: identical train/validation/test splits, identical epochs,
optimizer, lr schedule, batch size, seeds and gradient clipping as the VSS
recipe (config.PlainConfig mirrors the qmask YAML). The encoder is wrapped
in an nn.Module so its parameters actually join the optimizer (VSSEncoder
is deliberately not a Module — see benchmarks/encoder_classifier_ablation.py).
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import PlainConfig
from dataset import MultiQuestionExample

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vss.model.serialize import serialize_example  # noqa: E402
from vss.model.tokenizer import VSSTokenizer  # noqa: E402
from vss.model.transformer import TransformerEncoder  # noqa: E402
from vss.training.losses import score_losses  # noqa: E402

ABSTAIN = "ABSTAIN"


class PlainEncoderBox(nn.Module):
    """nn.Module wrapper so the TransformerEncoder's params register."""

    def __init__(self, inner: TransformerEncoder) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.inner(token_ids)


class PlainClassifier(nn.Module):
    """(state, question) -> one typed decision. No cross-question pathway."""

    def __init__(self, cfg: PlainConfig, n_labels: int, with_abstain: bool) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_labels = n_labels
        self.with_abstain = with_abstain
        self.encoder = PlainEncoderBox(
            TransformerEncoder(
                vocab_size=cfg.vocab_size,
                dim=cfg.hidden_size,
                layers=cfg.layers,
                heads=cfg.heads,
                kv_heads=cfg.heads,
                ffn_dim=cfg.intermediate_size,
                max_seq_len=cfg.max_sequence_length,
                dropout=cfg.dropout,
                rope_theta=10000.0,
                pad_token_id=0,
                question_masked=False,
            )
        )
        self.choice_head = nn.Linear(cfg.hidden_size, n_labels)
        self.noul_head = nn.Linear(cfg.hidden_size, 1)
        self.score_head = nn.Linear(cfg.hidden_size, cfg.score_bins)
        self.tokenizer: VSSTokenizer | None = None

    # ------------------------------------------------------------- features
    def forward(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        H = self.encoder(token_ids)              # [B, T, D]
        mask = (token_ids != 0).unsqueeze(-1).float()
        pooled = (H * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        out = {
            "choice": self.choice_head(pooled),  # [B, n_labels]
            "noul": torch.sigmoid(self.noul_head(pooled)).squeeze(-1),  # [B]
            "score": self.score_head(pooled),    # [B, bins]
        }
        return out

    # -------------------------------------------------------------- labeling
    def label_index(self, label: str) -> int:
        """Union-space index; ABSTAIN sits at the end when present."""
        return self._label_to_idx[label]

    def attach_labels(self, labels: list[str]) -> None:
        self._label_to_idx = {l: i for i, l in enumerate(labels)}

    # ------------------------------------------------------------ row encode
    def encode_rows(
        self, pairs: list[tuple[dict, dict]], device: str
    ) -> tuple[torch.Tensor, list[dict]]:
        """Tokenize (state, question) pairs into a padded batch."""
        assert self.tokenizer is not None, "fit tokenizer first"
        seqs = []
        for state, q in pairs:
            req = dict(q.as_request())
            text = serialize_example(state, [req])
            seqs.append(self.tokenizer.encode(text))
        T = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), T, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return x, pairs


def build_label_inventory(examples: list[MultiQuestionExample],
                          with_abstain: bool) -> list[str]:
    """Sorted union of all declared choice options (+ ABSTAIN if trained)."""
    labels: set[str] = set()
    has_abstain = False
    for ex in examples:
        for q in ex.questions:
            if q.type == "choice":
                if q.answer == ABSTAIN:
                    has_abstain = True
                    continue
                labels.update(q.options or ())
    labels.add(ABSTAIN) if (with_abstain and has_abstain) else None
    return sorted(labels)


def init_scratch_(model: PlainClassifier, seed: int) -> None:
    """Same scratch recipe as the ablation benchmark (norm gains = 1)."""
    torch.manual_seed(seed)
    for name, p in model.named_parameters():
        if "norm" in name and p.dim() == 1:
            torch.nn.init.ones_(p)
        elif p.dim() > 1:
            torch.nn.init.trunc_normal_(p, std=0.02)
        else:
            torch.nn.init.zeros_(p)


# ------------------------------------------------------------------ training

def _row_target(q, label_to_idx: dict[str, int], bins: int) -> dict:
    if q.type == "choice":
        if q.answer == ABSTAIN:
            return {"choice": label_to_idx[ABSTAIN], "abstain": True}
        return {"choice": label_to_idx[q.answer]}
    if q.type == "noul":
        return {"noul": float(int(q.answer))}
    return {"score": float(q.answer), "min": q.min or 0.0, "max": q.max or 10.0}


def _batch_loss(
    out: dict[str, torch.Tensor],
    pairs: list[tuple[dict, dict]],
    label_to_idx: dict[str, int],
    bins: int,
    device: str,
) -> torch.Tensor:
    """Masked multi-task loss: CE over declared options (+ABSTAIN), BCE,
    Huber + ordinal (identical to VSS's score_losses)."""
    losses: list[torch.Tensor] = []
    for i, (state, q) in enumerate(pairs):
        if q.type == "choice":
            opts = list(q.options or ())
            allowed = [label_to_idx[o] for o in opts]
            if q.answer == ABSTAIN:
                allowed.append(label_to_idx[ABSTAIN])
                tgt = len(opts)          # local index of ABSTAIN in the mask
            else:
                tgt = opts.index(q.answer)  # local index within declared options
            logits = out["choice"][i, allowed].unsqueeze(0)
            losses.append(F.cross_entropy(
                logits, torch.tensor([tgt], device=device)))
        elif q.type == "noul":
            p = out["noul"][i].reshape(-1).clamp(1e-6, 1 - 1e-6)
            y = torch.tensor([float(int(q.answer))], device=device)
            losses.append(F.binary_cross_entropy(p, y))
        else:
            lo, hi = q.min or 0.0, q.max or 10.0
            probs = F.softmax(out["score"][i]).unsqueeze(0)
            centers = torch.linspace(lo, hi, bins, device=device)
            s = score_losses(probs, centers,
                             torch.tensor([float(q.answer)], device=device), lo, hi)
            losses.append(s["huber"] + 0.25 * s["ordinal"])
    return torch.stack(losses).mean()


def train_plain(
    model: PlainClassifier,
    train_examples: list[MultiQuestionExample],
    valid_examples: list[MultiQuestionExample],
    cfg: PlainConfig,
    seed: int,
    ckpt_dir: str,
    save_every: int = 100,
    log_every: int = 100,
) -> dict:
    """Train from scratch with the VSS recipe; resumable progress checkpoints.

    The 600s shell cap is handled by re-invoking with --resume: last.pt keeps
    {model, opt, sched, step, epoch, best_val}; a mid-epoch checkpoint replays
    the remainder of its epoch (skip_batches), mirroring the ablation script.
    """
    device = "cpu"
    torch.manual_seed(seed)
    random.seed(seed)
    label_to_idx = model._label_to_idx
    bins = cfg.score_bins

    # fit tokenizer on training texts (idempotent; keeps loaded vocab)
    if model.tokenizer is None:
        tok = VSSTokenizer(vocab_size=cfg.vocab_size, hash_buckets=cfg.hash_buckets)
        texts = [serialize_example(ex.state, [q.as_request()])
                 for ex in train_examples for q in ex.questions]
        tok.fit(texts[:20000])
        model.tokenizer = tok

    train_rows = [(ex.state, q) for ex in train_examples for q in ex.questions]
    valid_rows = [(ex.state, q) for ex in valid_examples for q in ex.questions]
    steps_per_epoch = math.ceil(len(train_rows) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    warmup = cfg.warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    ckpt = Path(ckpt_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    start_epoch, global_step, best_val = 0, 0, float("inf")
    resume_batch = 0
    if (ckpt / "last.pt").exists():
        st = torch.load(ckpt / "last.pt", weights_only=False, map_location="cpu")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        global_step = st["step"]
        best_val = st["best_val"]
        if st.get("partial"):
            start_epoch, resume_batch = st["epoch"], st["batch_index"]
        else:
            start_epoch = st["epoch"] + 1
        print(f"resumed at epoch {start_epoch} batch {resume_batch} "
              f"step {global_step}", flush=True)

    def batches(rows: list, bs: int, shuffle_seed: int | None = None):
        idx = list(range(len(rows)))
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(idx)
        for s in range(0, len(idx), bs):
            yield [rows[i] for i in idx[s : s + bs]]

    def eval_val() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for chunk in batches(valid_rows, cfg.batch_size):
                x, _ = model.encode_rows(chunk, device)
                out = model(x)
                loss = _batch_loss(out, chunk, label_to_idx, bins, device)
                tot += float(loss) * len(chunk)
                n += len(chunk)
        model.train()
        return tot / max(1, n)

    history: list[dict] = []
    model.train()
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        ep_batches = list(batches(train_rows, cfg.batch_size, seed + epoch))
        ep_loss, nb = 0.0, 0
        for bi, chunk in enumerate(ep_batches):
            if bi < resume_batch:
                continue
            x, _ = model.encode_rows(chunk, device)
            out = model(x)
            loss = _batch_loss(out, chunk, label_to_idx, bins, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            opt.step()
            sched.step()
            global_step += 1
            ep_loss += float(loss)
            nb += 1
            if nb % save_every == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": global_step,
                            "epoch": epoch, "best_val": best_val,
                            "partial": True, "batch_index": bi + 1},
                           ckpt / "last.pt")
            if nb % log_every == 0:
                print(f"  epoch {epoch} batch {nb}/{len(ep_batches)} "
                      f"loss {float(loss):.4f}", flush=True)
        resume_batch = 0
        val = eval_val()
        history.append({"epoch": epoch, "train_loss": ep_loss / max(1, nb),
                        "val_loss": val, "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch}: train {ep_loss / max(1, nb):.4f} val {val:.4f} "
              f"({history[-1]['seconds']}s)", flush=True)
        is_best = val < best_val
        best_val = min(best_val, val)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": global_step,
                    "epoch": epoch, "best_val": best_val,
                    "partial": False, "batch_index": 0},
                   ckpt / "last.pt")
        if is_best:
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": val}, ckpt / "best.pt")

    torch.save({"model": model.state_dict(), "epoch": cfg.epochs - 1,
                "val_loss": best_val, "label_to_idx": label_to_idx},
               ckpt / "final.pt")
    if model.tokenizer is not None:
        model.tokenizer.save(ckpt / "vocab.json")
    return {"history": history, "best_val": best_val, "steps": global_step}


def load_plain(ckpt_dir: str, cfg: PlainConfig | None = None) -> PlainClassifier:
    """Rebuild a trained plain classifier from final.pt + vocab.json."""
    ckpt = Path(ckpt_dir)
    st = torch.load(ckpt / "final.pt", weights_only=False, map_location="cpu")
    label_to_idx: dict[str, int] = st["label_to_idx"]
    cfg = cfg or PlainConfig()
    model = PlainClassifier(cfg, n_labels=len(label_to_idx),
                            with_abstain=ABSTAIN in label_to_idx)
    model.attach_labels(sorted(label_to_idx, key=label_to_idx.get))
    model.load_state_dict(st["model"])
    model.tokenizer = VSSTokenizer.load(ckpt / "vocab.json")
    model.eval()
    return model


# ---------------------------------------------------------------- inference

@torch.no_grad()
def predict_rows(
    model: PlainClassifier,
    pairs: list[tuple[dict, dict]],
    batch_size: int = 32,
    device: str = "cpu",
) -> list[dict]:
    """Per-row prediction records (schema mirrors metrics.py expectations).

    For choice: distribution over the row's DECLARED options (+ ABSTAIN when
    the inventory has one), renormalized — mirroring VSS's engine output.
    For score: expected value over the row's [min, max] centers.
    """
    model.eval()
    label_to_idx = model._label_to_idx
    records: list[dict] = []
    for s in range(0, len(pairs), batch_size):
        chunk = pairs[s : s + batch_size]
        x, _ = model.encode_rows(chunk, device)
        out = model(x)
        for i, (state, q) in enumerate(chunk):
            rec: dict = {"qid": q.id, "type": q.type, "gold": q.answer}
            if q.type == "choice":
                opts = list(q.options or ())
                allowed = [label_to_idx[o] for o in opts]
                if label_to_idx.get(ABSTAIN) is not None:
                    allowed.append(label_to_idx[ABSTAIN])
                    opts_ab = opts + [ABSTAIN]
                else:
                    opts_ab = opts
                probs = torch.softmax(out["choice"][i, allowed], dim=-1)
                dist = {o: float(p) for o, p in zip(opts_ab, probs)}
                pred = max(dist, key=dist.get)
                rec.update({
                    "pred": pred,
                    "probs": dist,
                    "max_prob": dist[pred],
                    "conf": dist[pred],
                    "correct": pred == q.answer,
                })
            elif q.type == "noul":
                p = float(out["noul"][i])
                pred = 1 if p >= 0.5 else 0
                rec.update({
                    "pred": pred,
                    "p_true": p,
                    "max_prob": max(p, 1 - p),
                    "conf": max(p, 1 - p),
                    "correct": pred == int(q.answer),
                })
            else:
                lo, hi = q.min or 0.0, q.max or 10.0
                probs = torch.softmax(out["score"][i], dim=-1)
                centers = torch.linspace(lo, hi, model.cfg.score_bins)
                value = float((probs * centers).sum())
                gold_bin = int(round((float(q.answer) - lo) / (hi - lo)
                                     * (model.cfg.score_bins - 1)))
                pred_bin = int(torch.argmax(probs))
                pred_bin_val = float(centers[pred_bin])
                tol = 0.10 * (hi - lo)
                rec.update({
                    "pred": round(value, 4),
                    "value": value,
                    "probs": [float(p) for p in probs],
                    "gold_bin": gold_bin,
                    "pred_bin_value": pred_bin_val,
                    "max_prob": float(probs[gold_bin]),
                    "conf": float(probs[pred_bin]),
                    "correct": abs(value - float(q.answer)) <= tol,
                })
            records.append(rec)
    return records
