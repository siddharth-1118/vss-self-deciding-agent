"""Trainer: multi-task training over validated TrainingExamples.

Features:
  - one forward pass per batch with all question types mixed [public]
  - weighted multi-task loss (see losses.py)
  - cosine/linear warmup schedule, gradient clipping
  - checkpoint save/resume with tokenizer + optimizer state
  - curriculum via ordered stage lists (see curriculum.py)
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from ..data.schema import TrainingExample
from ..model.config import VSSConfig
from ..model.vss_model import VSSModel
from .losses import combined_loss


def build_targets(example: TrainingExample) -> list[dict[str, Any]]:
    """Per-question targets aligned with the example's question order."""
    out = []
    for q in example.questions:
        if q.type == "choice":
            if q.answer == "ABSTAIN":
                out.append({"answer_index": len(q.options or []), "abstain": True})
            else:
                out.append({"answer_index": q.options.index(q.answer) if q.options else 0})
        elif q.type == "noul":
            out.append({"answer": int(q.answer)})
        else:
            out.append({"answer": float(q.answer), "min": q.min or 0.0, "max": q.max or 10.0})
    return out


def lr_lambda(step: int, warmup: int, total: int, scheduler: str) -> float:
    """LR multiplier at `step`.

    Step 0 gets `1/warmup` rather than 0: LambdaLR applies this before the
    first optimizer step, so returning 0 made the first update a no-op.
    """
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    if scheduler == "cosine":
        return 0.5 * (1 + math.cos(math.pi * progress))
    return 1.0 - progress


def effective_warmup(warmup_steps: int, warmup_frac: float | None, total: int) -> int:
    """Clamp warmup to a fraction of the run.

    `warmup_steps` is a cap, not a target: a fixed 150 steps is 6% of a
    2.6k-step run and 75% of a 200-step run. See docs/convergence_audit.md
    finding 2.
    """
    w = int(warmup_steps)
    if warmup_frac is not None:
        w = min(w, max(1, int(warmup_frac * total)))
    return max(1, min(w, max(1, total)))


class Trainer:
    def __init__(
        self,
        model: VSSModel,
        config: VSSConfig,
        train_examples: list[TrainingExample],
        eval_examples: list[TrainingExample] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train = train_examples
        self.eval = eval_examples or []
        self.tcfg = config.training
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.opt = torch.optim.AdamW(
            model.parameters(), lr=self.tcfg.lr, weight_decay=self.tcfg.weight_decay
        )
        self._build_vocab()

    # ---------------------------------------------------------------- vocab
    def _build_vocab(self) -> None:
        """Fit tokenizer on training states (idempotent; keeps loaded vocab)."""
        from ..model.serialize import serialize_example
        from ..model.tokenizer import VSSTokenizer

        if self.model.tokenizer is None:
            tok: VSSTokenizer
            if getattr(self.config.model, "tokenizer_type", "word") == "bpe":
                from ..model.tokenizer_bpe import BPETokenizer

                tok = BPETokenizer(vocab_size=self.config.model.vocab_size)
            else:
                tok = VSSTokenizer(
                    vocab_size=self.config.model.vocab_size,
                    hash_buckets=self.config.model.hash_buckets,
                )
            texts = []
            for ex in self.train[:20000]:
                texts.append(
                    serialize_example(
                        ex.state, [q.as_request() for q in ex.questions]
                    )
                )
            tok.fit(texts)
            self.model.tokenizer = tok
            self.model.vss_encoder.tokenizer = tok
        else:
            tok = self.model.tokenizer
            assert isinstance(tok, VSSTokenizer)
    # ---------------------------------------------------------------- train
    def train_epoch(
        self,
        examples: list[TrainingExample],
        epoch: int,
        global_step: int,
        sched: torch.optim.lr_scheduler.LambdaLR | None = None,
        step_hook=None,
        skip_batches: int = 0,
    ) -> tuple[int, list[float], dict[str, float]]:
        rng = random.Random(self.tcfg.seed + epoch)
        order = list(range(len(examples)))
        rng.shuffle(order)
        bs = self.tcfg.batch_size
        if self.tcfg.max_steps:  # GLOBAL step cap (was per-epoch; see config)
            order = order[: max(1, self.tcfg.max_steps) * bs]
        losses: list[float] = []
        grad_norms: list[float] = []
        update_norms: list[float] = []
        self.model.train()
        batches = [order[s : s + bs] for s in range(0, len(order), bs)]
        for bi, batch_idx in enumerate(batches):
            if bi < skip_batches:
                continue  # already trained in a previous (partial) run [vss]
            batch = [examples[i] for i in batch_idx]
            states = [ex.state for ex in batch]
            qs = [[q.as_request() for q in ex.questions] for ex in batch]
            targets = [build_targets(ex) for ex in batch]

            out = self.model(states, qs, device=self.device)
            flat_rows, flat_targets = [], []
            for row_group, tgt_group in zip(out["per_example_rows"], targets):
                flat_rows.extend(row_group)
                flat_targets.extend(tgt_group)
            loss, parts = combined_loss(
                flat_rows, flat_targets,
                self.tcfg.loss_weights, self.tcfg.score_ordinal_weight,
            )
            (loss / self.tcfg.grad_accum).backward()
            if (global_step + 1) % self.tcfg.grad_accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.clip_grad_norm
                )
                grad_norms.append(float(gn))
                # parameter-update statistics: how far the weights actually
                # moved this step, relative to their own scale. A zero here
                # with a non-zero grad norm means updates are being killed
                # (clipping at ~0, zero LR, or a detached graph).
                before = [p.detach().clone() for p in self.model.parameters()
                          if p.requires_grad]
                self.opt.step()
                with torch.no_grad():
                    upd, base = 0.0, 0.0
                    for p0, p1 in zip(before, (p for p in self.model.parameters()
                                               if p.requires_grad)):
                        d = float((p1.detach() - p0).norm())
                        upd += d * d
                        base += float(p0.norm()) ** 2
                    update_norms.append((upd ** 0.5) / max(1e-12, base ** 0.5))
                self.opt.zero_grad(set_to_none=True)
                if sched is not None:
                    sched.step()
            losses.append(parts["total"])
            global_step += 1
            if self.tcfg.max_steps and global_step >= self.tcfg.max_steps:
                break  # global step budget exhausted (docs/convergence_report.md §5)
            if step_hook is not None:
                step_hook(global_step, bi + 1)
            if global_step % self.tcfg.log_every == 0:
                print(f"  epoch {epoch} step {global_step} loss {parts['total']:.4f}", flush=True)
        stats = {
            "grad_norm_mean": sum(grad_norms) / max(1, len(grad_norms)),
            "grad_norm_max": max(grad_norms) if grad_norms else 0.0,
            "param_update_rel_mean": sum(update_norms) / max(1, len(update_norms)),
            "param_update_rel_max": max(update_norms) if update_norms else 0.0,
            "n_updates": float(len(update_norms)),
        }
        return global_step, losses, stats

    # ------------------------------------------------------------- schedule
    def fit(self, resume_from: str | None = None) -> dict[str, Any]:
        tcfg = self.tcfg
        torch.manual_seed(tcfg.seed)
        random.seed(tcfg.seed)

        total_steps = (len(self.train) * tcfg.epochs) // max(1, tcfg.batch_size)
        if tcfg.max_steps:
            total_steps = min(total_steps, tcfg.max_steps)
        total_steps = max(1, total_steps)
        warmup = effective_warmup(tcfg.warmup_steps, tcfg.warmup_frac, total_steps)

        start_epoch, global_step, best_loss = 0, 0, float("inf")
        ckpt_dir = Path(tcfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, Any]] = []
        epochs_without_improvement = 0
        stopped_early = False

        resume_batch = 0
        if resume_from:
            state = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(state["model"])
            self.opt.load_state_dict(state["optimizer"])
            partial = state.get("partial", False)
            # a mid-epoch checkpoint replays the remainder of its (partial)
            # epoch; a post-epoch checkpoint continues with the next [vss]
            start_epoch = state["epoch"] if partial else state["epoch"] + 1
            resume_batch = state.get("batch_index", 0) if partial else 0
            global_step = state["global_step"]
            best_loss = state.get("best_loss", float("inf"))
            history = list(state.get("history", []))
            epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
            # PIN the schedule shape. Recomputing total_steps from the current
            # `epochs` would silently reshape the LR curve on every resume, so
            # "resume with more epochs" is not comparable to an uninterrupted
            # run. The pinned values win.
            pinned_total = state.get("schedule_total_steps")
            if pinned_total:
                total_steps = int(pinned_total)
                warmup = int(state.get("schedule_warmup", warmup))
            print(f"resumed from {resume_from}: starting epoch {start_epoch}"
                  + (f" at batch {resume_batch}" if partial else "")
                  + f"; schedule pinned total={total_steps} warmup={warmup}",
                  flush=True)

        sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt,
            lambda s: lr_lambda(s, warmup, total_steps, tcfg.scheduler),
        )
        self._pinned_total_steps = total_steps
        self._pinned_warmup = warmup
        # fast-forward the schedule to the resumed step
        for _ in range(global_step):
            sched.step()

        ckpt_every = max(1, int(getattr(tcfg, "ckpt_every", 100)))
        for epoch in range(start_epoch, tcfg.epochs):
            # The step budget is GLOBAL, so on any dataset large enough that
            # `max_steps` is smaller than len(train)*epochs the budget runs out
            # partway through an epoch. Without this guard the loop keeps
            # spinning: each later epoch runs a single batch at lr==0 and then
            # evaluates, which burns wall-clock AND increments
            # `epochs_without_improvement`, so early stopping fires on epochs
            # that could not have learned anything. See
            # docs/convergence_audit.md finding 5.
            if tcfg.max_steps and global_step >= tcfg.max_steps:
                print(f"step budget exhausted ({global_step}/{tcfg.max_steps})"
                      f" at epoch {epoch}; stopping", flush=True)
                break
            t0 = time.time()
            steps_at_epoch_start = global_step

            def hook(gs: int, batch_index: int, _epoch: int = epoch, _s0: int = steps_at_epoch_start) -> None:
                if (gs - _s0) % ckpt_every == 0:
                    # `history` MUST be passed through: _checkpoint defaults it to
                    # [], so omitting it made every mid-epoch checkpoint wipe the
                    # per-epoch records, and a resumed run silently restarted its
                    # history from the current epoch. Observed on banking77, where
                    # the run JSON began at epoch 1 and epoch 0 survived only in
                    # the log. See docs/convergence_audit.md finding 6.
                    self._checkpoint(ckpt_dir / "last.pt", _epoch, gs, best_loss,
                                     partial=True, batch_index=batch_index,
                                     history=history,
                                     epochs_without_improvement=epochs_without_improvement)

            global_step, losses, stats = self.train_epoch(
                self.train, epoch, global_step, sched,
                step_hook=hook, skip_batches=resume_batch,
            )
            resume_batch = 0  # only the first resumed epoch replays partially
            mean_loss = sum(losses) / max(1, len(losses))
            metrics = self.evaluate_detailed() if self.eval else {}
            eval_loss = metrics.get("loss")
            rec = {
                "epoch": epoch,
                "train_loss": mean_loss,
                "eval_loss": eval_loss,
                "lr": float(self.opt.param_groups[0]["lr"]),
                "seconds": round(time.time() - t0, 1),
                **stats,
                **{k: v for k, v in metrics.items() if k != "loss"},
            }
            history.append(rec)
            print(f"epoch {epoch}: train {mean_loss:.4f}"
                  + (f" eval {eval_loss:.4f}" if eval_loss is not None else "")
                  + f" choice_acc={metrics.get('choice_accuracy')}"
                  + f" lr={rec['lr']:.2e}"
                  + f" gnorm={stats['grad_norm_mean']:.3f}"
                  + f" upd={stats['param_update_rel_mean']:.2e}", flush=True)
            is_best = eval_loss is not None and eval_loss < best_loss - tcfg.early_stop_min_delta
            if is_best:
                best_loss = eval_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            self._checkpoint(ckpt_dir / "last.pt", epoch, global_step, best_loss,
                             history=history, epochs_without_improvement=epochs_without_improvement)
            if is_best:
                self._checkpoint(ckpt_dir / "best.pt", epoch, global_step, best_loss,
                                 history=history, epochs_without_improvement=epochs_without_improvement)
            self.model.save_pretrained(str(ckpt_dir / "final"))
            if (tcfg.early_stop_patience is not None
                    and epoch + 1 >= tcfg.min_epochs
                    and epochs_without_improvement >= tcfg.early_stop_patience):
                print(f"early stop at epoch {epoch}: no eval-loss improvement >"
                      f" {tcfg.early_stop_min_delta} for {epochs_without_improvement} epochs",
                      flush=True)
                stopped_early = True
                break

        return {
            "history": history,
            "best_loss": best_loss,
            "steps": global_step,
            "schedule_total_steps": total_steps,
            "schedule_warmup": warmup,
            "stopped_early": stopped_early,
            "epochs_run": len(history),
        }

    # ----------------------------------------------------------------- eval
    @torch.no_grad()
    def evaluate(self) -> float:
        """Mean validation loss over the eval split (per-row, not per-batch)."""
        return self.evaluate_detailed()["loss"]

    @torch.no_grad()
    def evaluate_detailed(self) -> dict[str, float]:
        """Validation loss plus per-task metrics, accumulated per ROW.

        `evaluate()` used to average per-batch means, which silently weights a
        batch of 1 noul rows the same as a batch of 32 choice rows. Metrics
        here are row-weighted, so they are comparable across epochs and across
        batch sizes.
        """
        self.model.eval()
        loss_sum = 0.0
        n_rows = 0
        per_type_loss: dict[str, list[float]] = {}
        n_choice = n_choice_correct = 0
        n_abstain = 0
        correct: list[float] = []
        conf: list[float] = []
        noul_correct = noul_n = 0
        score_err: list[float] = []
        bs = self.tcfg.batch_size
        for start in range(0, len(self.eval), bs):
            batch = self.eval[start : start + bs]
            states = [ex.state for ex in batch]
            qs = [[q.as_request() for q in ex.questions] for ex in batch]
            out = self.model(states, qs, device=self.device)
            for row_group, ex in zip(out["per_example_rows"], batch):
                flat_targets = build_targets(ex)
                per_row = _per_row_loss(row_group, flat_targets, self.tcfg)
                for row, tgt, rl in zip(row_group, flat_targets, per_row):
                    n_rows += 1
                    loss_sum += rl
                    per_type_loss.setdefault(row["type"], []).append(rl)
                    if row["type"] == "choice":
                        logits = torch.cat(
                            [row["logits"], row["abstain_logit"].reshape(1)]
                        )
                        pred = int(torch.argmax(logits))
                        gold = (logits.shape[-1] - 1 if tgt.get("abstain")
                                else int(tgt["answer_index"]))
                        n_choice += 1
                        n_choice_correct += int(pred == gold)
                        n_abstain += int(tgt.get("abstain", False))
                        probs = torch.softmax(logits.float(), dim=-1)
                        correct.append(float(pred == gold))
                        conf.append(float(probs[pred]))
                    elif row["type"] == "noul":
                        noul_n += 1
                        noul_correct += int(
                            (1 if float(row["prob"]) >= 0.5 else 0) == int(tgt["answer"])
                        )
                    else:
                        score_err.append(abs(float(row["value"]) - float(tgt["answer"])))
        m: dict[str, float] = {"loss": loss_sum / max(1, n_rows), "n_rows": float(n_rows)}
        for t, vals in per_type_loss.items():
            m[f"{t}_loss"] = sum(vals) / max(1, len(vals))
        m["choice_accuracy"] = n_choice_correct / max(1, n_choice)
        m["choice_n"] = float(n_choice)
        m["abstain_rows"] = float(n_abstain)
        m["noul_accuracy"] = noul_correct / max(1, noul_n)
        m["noul_n"] = float(noul_n)
        m["score_mae"] = sum(score_err) / max(1, len(score_err))
        m["score_n"] = float(len(score_err))
        if correct:
            m["row_accuracy"] = sum(correct) / len(correct)
            m["ece"] = _ece(conf, correct)
        else:
            m["row_accuracy"] = float("nan")
            m["ece"] = float("nan")
        return m

    # ----------------------------------------------------------- checkpoint
    def _checkpoint(
        self,
        path: Path,
        epoch: int,
        global_step: int,
        best_loss: float,
        partial: bool = False,
        batch_index: int = 0,
        history: list[dict[str, Any]] | None = None,
        epochs_without_improvement: int = 0,
    ) -> None:
        # atomic write: a killed process (600s shell cap) can truncate a plain
        # torch.save mid-write; write tmp then replace so last.pt stays loadable
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
            "partial": partial,
            "batch_index": batch_index,
            "seed": self.tcfg.seed,
            "config": self.config.to_dict(),
            # pinned schedule shape + early-stopping state, so a resumed run
            # follows the identical LR curve (see fit())
            "schedule_total_steps": getattr(self, "_pinned_total_steps", None),
            "schedule_warmup": getattr(self, "_pinned_warmup", None),
            "history": history or [],
            "epochs_without_improvement": epochs_without_improvement,
        }
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(path)


def _per_row_loss(
    rows: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    tcfg,
) -> list[float]:
    """Weighted per-row contribution to the total loss (row-weighted mean)."""
    from .losses import choice_loss, noul_loss, score_losses

    out: list[float] = []
    for row, tgt in zip(rows, targets):
        w = 0.0
        if row["type"] == "choice":
            logits = torch.cat([row["logits"], row["abstain_logit"].reshape(1)]).unsqueeze(0)
            idx = (logits.shape[-1] - 1) if tgt.get("abstain") else int(tgt["answer_index"])
            w += tcfg.loss_weights.get("choice", 1.0) * float(choice_loss(logits, torch.tensor([idx])))
        elif row["type"] == "noul":
            y = torch.tensor(float(tgt["answer"]))
            w += tcfg.loss_weights.get("noul", 1.0) * float(
                noul_loss(row["prob"].reshape(-1), y.reshape(-1))
            )
        else:
            y = torch.tensor(float(tgt["answer"]))
            s = score_losses(row["probs"], row["centers"], y, tgt["min"], tgt["max"])
            w += tcfg.loss_weights.get("score", 1.0) * float(s["huber"])
            w += tcfg.score_ordinal_weight * float(s["ordinal"])
        w += tcfg.loss_weights.get("calibration", 1.0) * float(
            torch.nn.functional.binary_cross_entropy(
                row["calibration"].reshape(1).clamp(1e-6, 1 - 1e-6),
                torch.tensor([_correct(row, tgt)]),
            )
        )
        out.append(w)
    return out


def _correct(row: dict[str, Any], tgt: dict[str, Any]) -> float:
    if row["type"] == "choice":
        logits = torch.cat([row["logits"].detach(), row["abstain_logit"].detach().reshape(1)])
        pred = int(torch.argmax(logits))
        gold = logits.shape[-1] - 1 if tgt.get("abstain") else int(tgt["answer_index"])
        return float(pred == gold)
    if row["type"] == "noul":
        return float((1 if float(row["prob"]) >= 0.5 else 0) == int(tgt["answer"]))
    tol = 0.1 * (tgt["max"] - tgt["min"])
    return float(abs(float(row["value"]) - float(tgt["answer"])) <= tol)


def _ece(conf: list[float], correct: list[float], bins: int = 10) -> float:
    if not conf:
        return float("nan")
    tot = 0.0
    n = len(conf)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo <= c < hi or (b == bins - 1 and c == hi)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        avg = sum(conf[i] for i in idx) / len(idx)
        tot += (len(idx) / n) * abs(acc - avg)
    return tot
