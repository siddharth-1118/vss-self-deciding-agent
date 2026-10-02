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
from vss.training.trainer import effective_warmup, lr_lambda  # noqa: E402

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
            req = q.as_request()
            if getattr(q, "header_only_choice", False):
                # same canonical token stream VSS consumes (option text is
                # stripped by the VSS model's own forward — see vss_model.py)
                req = {**req, "header_only_choice": True}
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
            if getattr(q, "header_only_choice", False):
                # Full-inventory CE over the entire head (the U1 ablation's
                # proven-strong plain-head recipe), matching the eval-time
                # softmax over the row's declared options.  Do NOT restrict
                # the support to the train row's declared subset: with
                # header-only text the head cannot know the subset, and train
                # rows (15 opts) vs eval rows (151 opts) would be different
                # tasks — that mismatch is what collapsed the v1 baseline.
                logits = out["choice"][i].unsqueeze(0)
                tgt = label_to_idx[q.answer]
                losses.append(F.cross_entropy(
                    logits, torch.tensor([tgt], device=device)))
                continue
            opts = list(q.options or ())
            allowed = [label_to_idx[o] for o in opts]
            if q.answer == ABSTAIN:
                allowed.append(label_to_idx[ABSTAIN])
                tgt = len(opts)      # local index of ABSTAIN in the mask
            else:
                tgt = opts.index(q.answer)  # local index within options
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
        texts = []
        for ex in train_examples:
            for q in ex.questions:
                req = q.as_request()
                if getattr(q, "header_only_choice", False):
                    req = {**req, "header_only_choice": True}
                texts.append(serialize_example(ex.state, [req]))
        tok.fit(texts[:20000])
        model.tokenizer = tok

    train_rows = [(ex.state, q) for ex in train_examples for q in ex.questions]
    valid_rows = [(ex.state, q) for ex in valid_examples for q in ex.questions]
    steps_per_epoch = math.ceil(len(train_rows) / cfg.batch_size)
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    if getattr(cfg, "max_steps", None):
        # global step budget: the cosine schedule is computed over it, so the
        # two systems can be step-matched (same lr_lambda as VSS)
        total_steps = min(total_steps, int(cfg.max_steps))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    # Same schedule helpers as the VSS trainer (shared import below): the
    # warmup is capped at a fraction of the run and step 0 gets a non-zero
    # LR. The baseline must not be handicapped by a schedule defect, and must
    # not be handed an advantage VSS does not get -- one code path, both arms.
    warmup = effective_warmup(cfg.warmup_steps, cfg.warmup_frac, total_steps)

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, warmup, total_steps, "cosine")
    )

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

    def eval_val() -> tuple[float, dict]:
        """Row-weighted validation loss + per-task metrics.

        Choice accuracy uses the SAME rule as predict_rows (softmax over the
        row's declared options + ABSTAIN) so the selection signal matches the
        metric the benchmark reports.
        """
        model.eval()
        tot, n = 0.0, 0
        n_choice = n_correct = 0
        noul_n = noul_ok = 0
        score_err: list[float] = []
        conf: list[float] = []
        correct: list[float] = []
        has_abstain = label_to_idx.get(ABSTAIN) is not None
        with torch.no_grad():
            for chunk in batches(valid_rows, cfg.batch_size):
                x, _ = model.encode_rows(chunk, device)
                out = model(x)
                loss = _batch_loss(out, chunk, label_to_idx, bins, device)
                tot += float(loss) * len(chunk)
                n += len(chunk)
                for i, (_state, q) in enumerate(chunk):
                    if q.type == "choice":
                        opts = list(q.options or ())
                        allowed = [label_to_idx[o] for o in opts]
                        names = list(opts)
                        if has_abstain:
                            allowed.append(label_to_idx[ABSTAIN])
                            names.append(ABSTAIN)
                        probs = torch.softmax(out["choice"][i, allowed], dim=-1)
                        pred = names[int(torch.argmax(probs))]
                        ok = float(pred == q.answer)
                        n_choice += 1
                        n_correct += int(ok)
                        correct.append(ok)
                        conf.append(float(probs[int(torch.argmax(probs))]))
                    elif q.type == "noul":
                        p = float(out["noul"][i])
                        noul_n += 1
                        noul_ok += int((1 if p >= 0.5 else 0) == int(q.answer))
                    else:
                        lo, hi = q.min or 0.0, q.max or 10.0
                        probs = torch.softmax(out["score"][i], dim=-1)
                        centers = torch.linspace(lo, hi, bins)
                        score_err.append(abs(float((probs * centers).sum())
                                             - float(q.answer)))
        model.train()
        m = {
            "choice_accuracy": n_correct / max(1, n_choice),
            "choice_n": float(n_choice),
            "noul_accuracy": noul_ok / max(1, noul_n),
            "noul_n": float(noul_n),
            "score_mae": sum(score_err) / max(1, len(score_err)),
            "score_n": float(len(score_err)),
        }
        if correct:
            m["row_accuracy"] = sum(correct) / len(correct)
        return tot / max(1, n), m

    # history/early-stop state live in the checkpoint so a resumed run keeps
    # the same curve and the same patience counter
    _st = (torch.load(ckpt / "last.pt", weights_only=False, map_location="cpu")
           if (ckpt / "last.pt").exists() else None)
    history: list[dict] = list(_st.get("history", [])) if _st else []
    epochs_without_improvement = int(_st.get("epochs_without_improvement", 0)) if _st else 0
    pinned_total = int(_st.get("schedule_total_steps", 0)) if _st else 0
    if pinned_total:
        # keep the LR curve identical to the uninterrupted run
        total_steps, warmup = pinned_total, int(_st.get("schedule_warmup", warmup))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: lr_lambda(s, warmup, total_steps, "cosine")
        )
        sched.load_state_dict(_st["sched"])
    stopped_early = False
    model.train()

    def _save(path: Path, payload: dict) -> None:
        """Atomic progress checkpoint: a killed process can truncate a plain
        torch.save, so write to tmp and replace (last.pt stays loadable)."""
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        ep_batches = list(batches(train_rows, cfg.batch_size, seed + epoch))
        ep_loss, nb = 0.0, 0
        grad_norms: list[float] = []
        upd_norms: list[float] = []
        for bi, chunk in enumerate(ep_batches):
            if bi < resume_batch:
                continue
            x, _ = model.encode_rows(chunk, device)
            out = model(x)
            loss = _batch_loss(out, chunk, label_to_idx, bins, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            grad_norms.append(float(gn))
            before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
            opt.step()
            with torch.no_grad():
                upd, base = 0.0, 0.0
                for p0, p1 in zip(before, (p for p in model.parameters()
                                           if p.requires_grad)):
                    d = float((p1.detach() - p0).norm())
                    upd += d * d
                    base += float(p0.norm()) ** 2
                upd_norms.append((upd ** 0.5) / max(1e-12, base ** 0.5))
            sched.step()
            global_step += 1
            ep_loss += float(loss)
            if getattr(cfg, "max_steps", None) and global_step >= int(cfg.max_steps):
                break  # global step budget exhausted
            nb += 1
            if nb % save_every == 0:
                _save(ckpt / "last.pt",
                      {"model": model.state_dict(), "opt": opt.state_dict(),
                       "sched": sched.state_dict(), "step": global_step,
                       "epoch": epoch, "best_val": best_val,
                       "partial": True, "batch_index": bi + 1,
                       "history": history,
                       "epochs_without_improvement": epochs_without_improvement,
                       "schedule_total_steps": total_steps,
                       "schedule_warmup": warmup})
            if nb % log_every == 0:
                print(f"  epoch {epoch} batch {nb}/{len(ep_batches)} "
                      f"loss {float(loss):.4f}", flush=True)
        resume_batch = 0
        val, vm = eval_val()
        rec = {
            "epoch": epoch,
            "train_loss": ep_loss / max(1, nb),
            "eval_loss": val,
            "val_loss": val,
            "lr": float(opt.param_groups[0]["lr"]),
            "seconds": round(time.time() - t0, 1),
            "grad_norm_mean": sum(grad_norms) / max(1, len(grad_norms)),
            "grad_norm_max": max(grad_norms) if grad_norms else 0.0,
            "param_update_rel_mean": sum(upd_norms) / max(1, len(upd_norms)),
            "param_update_rel_max": max(upd_norms) if upd_norms else 0.0,
            "n_updates": float(len(upd_norms)),
            **vm,
        }
        history.append(rec)
        print(f"epoch {epoch}: train {ep_loss / max(1, nb):.4f} val {val:.4f} "
              f"choice_acc={vm.get('choice_accuracy')} lr={rec['lr']:.2e} "
              f"gnorm={rec['grad_norm_mean']:.3f} ({rec['seconds']}s)", flush=True)
        is_best = val < best_val - getattr(cfg, "early_stop_min_delta", 0.0)
        if is_best:
            best_val = val
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        _save(ckpt / "last.pt",
              {"model": model.state_dict(), "opt": opt.state_dict(),
               "sched": sched.state_dict(), "step": global_step,
               "epoch": epoch, "best_val": best_val,
               "partial": False, "batch_index": 0, "history": history,
               "epochs_without_improvement": epochs_without_improvement,
               "schedule_total_steps": total_steps, "schedule_warmup": warmup})
        if is_best:
            _save(ckpt / "best.pt",
                  {"model": model.state_dict(), "epoch": epoch,
                   "val_loss": val})
        patience = getattr(cfg, "early_stop_patience", None)
        if (patience is not None
                and epoch + 1 >= getattr(cfg, "min_epochs", 1)
                and epochs_without_improvement >= patience):
            print(f"early stop at epoch {epoch}: no val improvement >"
                  f" {getattr(cfg, 'early_stop_min_delta', 0.0)} for"
                  f" {epochs_without_improvement} epochs", flush=True)
            stopped_early = True
            break

    torch.save({"model": model.state_dict(), "epoch": cfg.epochs - 1,
                "val_loss": best_val, "label_to_idx": label_to_idx},
               ckpt / "final.pt")
    if model.tokenizer is not None:
        model.tokenizer.save(ckpt / "vocab.json")
    return {"history": history, "best_val": best_val, "steps": global_step,
            "schedule_total_steps": total_steps, "schedule_warmup": warmup,
            "stopped_early": stopped_early, "epochs_run": len(history)}


def load_plain(ckpt_dir: str, cfg: PlainConfig | None = None) -> PlainClassifier:
    """Rebuild a trained plain classifier.

    Checkpoint selection mirrors the VSS protocol: best validation-loss
    checkpoint (best.pt) when present, else the final epoch. Label inventory
    comes from final.pt; vocab.json must exist in the dir.
    """
    ckpt = Path(ckpt_dir)
    st_final = torch.load(ckpt / "final.pt", weights_only=False, map_location="cpu")
    label_to_idx: dict[str, int] = st_final["label_to_idx"]
    cfg = cfg or PlainConfig()
    model = PlainClassifier(cfg, n_labels=len(label_to_idx),
                            with_abstain=ABSTAIN in label_to_idx)
    model.attach_labels(sorted(label_to_idx, key=label_to_idx.get))
    best_path = ckpt / "best.pt"
    if best_path.exists():
        st = torch.load(best_path, weights_only=False, map_location="cpu")
        model.load_state_dict(st["model"])
    else:
        model.load_state_dict(st_final["model"])
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
