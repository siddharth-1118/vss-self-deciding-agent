"""Audit probe 2: parameter-update coverage and split leakage."""
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "multi_question_value"))

import torch

torch.set_num_threads(6)

import dataset  # noqa: E402
from vss.data.schema import TrainingExample, dump_jsonl, load_jsonl  # noqa: E402
from vss.model.config import VSSConfig  # noqa: E402
from vss.model.vss_model import VSSModel  # noqa: E402
from vss.training.trainer import Trainer, build_targets  # noqa: E402
from vss.training.losses import combined_loss  # noqa: E402

cfg = VSSConfig.load(str(REPO / "configs" / "vss-prototype-clinc-slot-ho-qmask.yaml"))

# ---------------------------------------------------------------- leakage
print("=== PROBE 5: split leakage (synthetic) ===")
splits = {s: dataset.load_synthetic(s) for s in ("train", "validation", "test")}
print({k: len(v) for k, v in splits.items()})


def state_key(ex):
    return repr(sorted(ex.state.items()))


def qkey(ex):
    return tuple(sorted((q.id, q.answer) for q in ex.questions))


tr_s = {state_key(e) for e in splits["train"]}
va_s = {state_key(e) for e in splits["validation"]}
te_s = {state_key(e) for e in splits["test"]}
print(f"  train&val states={len(tr_s & va_s)} train&test={len(tr_s & te_s)} "
      f"val&test={len(va_s & te_s)}")
tr_q = {qkey(e) for e in splits["train"]}
te_q = {qkey(e) for e in splits["test"]}
print(f"  train&test (state,question-set) keys={len(tr_q & te_q)}")

for ds in ("clinc150", "banking77"):
    d = REPO / "data" / ds
    if not d.exists():
        print(f"  {ds}: no split dir")
        continue
    rows = {}
    for sp in ("train", "validation", "test"):
        p = d / f"{sp}.jsonl"
        if p.exists():
            rows[sp] = load_jsonl(str(p))
    tr = {str(e.state.get("text", e.state)) for e in rows.get("train", [])}
    te = {str(e.state.get("text", e.state)) for e in rows.get("test", [])}
    print(f"  {ds}: sizes={ {k: len(v) for k, v in rows.items()} } "
          f"train&test utterances={len(tr & te)}")

# ------------------------------------------------- parameter update coverage
print("\n=== PROBE 6: parameter update coverage (synthetic batch) ===")
tr_ex = [TrainingExample.model_validate(e.to_training_dict()) for e in splits["train"][:32]]
model = VSSModel(cfg.model)
tr = Trainer(model, cfg, tr_ex, tr_ex[:8])
states = [ex.state for ex in tr_ex]
qs = [[q.as_request() for q in ex.questions] for ex in tr_ex]
out = model(states, qs)
flat_rows, flat_tgts = [], []
for g, t in zip(out["per_example_rows"], tr_ex):
    flat_rows.extend(g)
    flat_tgts.extend(build_targets(t))
loss, parts = combined_loss(flat_rows, flat_tgts, cfg.training.loss_weights,
                            cfg.training.score_ordinal_weight)
loss.backward()
print(f"  loss={float(loss):.4f} parts={ {k: round(v, 4) for k, v in parts.items()} }")
none_grad, zero_grad = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.grad is None:
        none_grad.append(n)
    elif float(p.grad.abs().max()) == 0.0:
        zero_grad.append(n)
tot = sum(1 for _ in model.named_parameters())
print(f"  params={tot} no-grad={len(none_grad)} zero-grad={len(zero_grad)}")
if none_grad:
    print(f"  NO GRAD: {none_grad[:12]}")
if zero_grad:
    print(f"  ZERO GRAD: {zero_grad[:12]}")
n_none = sum(p.numel() for n, p in model.named_parameters() if n in set(none_grad))
print(f"  no-grad param count: {n_none:,}")

# ------------------------------------------------- LR schedule shape
print("\n=== PROBE 7: LR schedule shape ===")
from vss.training.trainer import lr_lambda  # noqa: E402

for name, n, warm, ep, bs in [
    ("vss synthetic (as run)", 800, 150, 8, 32),
    ("vss synthetic (30 ep)", 800, 150, 30, 32),
    ("plain synthetic (as run)", 6400, 150, 8, 32),
    ("vss clinc150", 10625, 150, 8, 32),
    ("vss banking77", 9079, 150, 8, 32),
]:
    total = (n * ep) // bs
    lrs = [3e-4 * lr_lambda(s, warm, total, "cosine") for s in range(total)]
    frac_warm = warm / total
    # mean lr over the second half of training
    tail = sum(lrs[total // 2:]) / max(1, len(lrs[total // 2:]))
    print(f"  {name:<28} total_steps={total:>5} warmup_frac={frac_warm:5.1%} "
          f"lr[0]={lrs[0]:.2e} lr[-1]={lrs[-1]:.2e} mean_lr_2nd_half={tail:.2e}")
