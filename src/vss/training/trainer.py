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
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    if scheduler == "cosine":
        return 0.5 * (1 + math.cos(math.pi * progress))
    return 1.0 - progress


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
    ) -> tuple[int, list[float]]:
        rng = random.Random(self.tcfg.seed + epoch)
        order = list(range(len(examples)))
        rng.shuffle(order)
        bs = self.tcfg.batch_size
        if self.tcfg.max_steps:  # cap steps per epoch
            order = order[: self.tcfg.max_steps * bs]
        losses: list[float] = []
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.clip_grad_norm)
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                if sched is not None:
                    sched.step()
            losses.append(parts["total"])
            global_step += 1
            if step_hook is not None:
                step_hook(global_step, bi + 1)
            if global_step % self.tcfg.log_every == 0:
                print(f"  epoch {epoch} step {global_step} loss {parts['total']:.4f}", flush=True)
        return global_step, losses

    # ------------------------------------------------------------- schedule
    def fit(self, resume_from: str | None = None) -> dict[str, Any]:
        tcfg = self.tcfg
        torch.manual_seed(tcfg.seed)
        random.seed(tcfg.seed)

        total_steps = (len(self.train) * tcfg.epochs) // max(1, tcfg.batch_size)
        if tcfg.max_steps:
            total_steps = min(total_steps, tcfg.max_steps)

        start_epoch, global_step, best_loss = 0, 0, float("inf")
        ckpt_dir = Path(tcfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

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
            print(f"resumed from {resume_from}: starting epoch {start_epoch}"
                  + (f" at batch {resume_batch}" if partial else ""), flush=True)

        sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt,
            lambda s: lr_lambda(s, tcfg.warmup_steps, total_steps, tcfg.scheduler),
        )
        # fast-forward the schedule to the resumed step
        for _ in range(global_step):
            sched.step()

        history: list[dict[str, Any]] = []
        ckpt_every = 100  # steps between mid-epoch checkpoints [vss]
        for epoch in range(start_epoch, tcfg.epochs):
            t0 = time.time()
            steps_at_epoch_start = global_step

            def hook(gs: int, batch_index: int, _epoch: int = epoch, _s0: int = steps_at_epoch_start) -> None:
                if (gs - _s0) % ckpt_every == 0:
                    self._checkpoint(ckpt_dir / "last.pt", _epoch, gs, best_loss, partial=True, batch_index=batch_index)

            global_step, losses = self.train_epoch(
                self.train, epoch, global_step, sched,
                step_hook=hook, skip_batches=resume_batch,
            )
            resume_batch = 0  # only the first resumed epoch replays partially
            mean_loss = sum(losses) / max(1, len(losses))
            eval_loss = self.evaluate() if self.eval else None
            history.append(
                {"epoch": epoch, "train_loss": mean_loss, "eval_loss": eval_loss,
                 "seconds": round(time.time() - t0, 1)}
            )
            print(f"epoch {epoch}: train {mean_loss:.4f}"
                  + (f" eval {eval_loss:.4f}" if eval_loss is not None else ""), flush=True)
            is_best = eval_loss is not None and eval_loss < best_loss
            if is_best:
                best_loss = eval_loss
            self._checkpoint(ckpt_dir / "last.pt", epoch, global_step, best_loss)
            if is_best:
                self._checkpoint(ckpt_dir / "best.pt", epoch, global_step, best_loss)
            self.model.save_pretrained(str(ckpt_dir / "final"))

        return {"history": history, "best_loss": best_loss, "steps": global_step}

    # ----------------------------------------------------------------- eval
    @torch.no_grad()
    def evaluate(self) -> float:
        """Mean validation loss over the eval split."""
        self.model.eval()
        losses: list[float] = []
        bs = self.tcfg.batch_size
        for start in range(0, len(self.eval), bs):
            batch = self.eval[start : start + bs]
            states = [ex.state for ex in batch]
            qs = [[q.as_request() for q in ex.questions] for ex in batch]
            out = self.model(states, qs, device=self.device)
            for row_group, ex in zip(out["per_example_rows"], batch):
                flat_rows = row_group
                flat_targets = build_targets(ex)
                loss, parts = combined_loss(
                    flat_rows, flat_targets, self.tcfg.loss_weights, self.tcfg.score_ordinal_weight
                )
                losses.append(parts["total"])
        return sum(losses) / max(1, len(losses))

    # ----------------------------------------------------------- checkpoint
    def _checkpoint(
        self,
        path: Path,
        epoch: int,
        global_step: int,
        best_loss: float,
        partial: bool = False,
        batch_index: int = 0,
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
        }
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
