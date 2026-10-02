"""Ad-hoc audit probes for the convergence audit (not part of the test suite)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "multi_question_value"))

import torch

torch.set_num_threads(6)
torch.manual_seed(0)

import dataset  # noqa: E402
from vss.model.config import VSSConfig  # noqa: E402
from vss.model.vss_model import VSSModel  # noqa: E402

cfg = VSSConfig.load(str(REPO / "configs" / "vss-prototype-clinc-slot-ho-qmask.yaml"))
cfg.model.dropout = 0.0  # determinism for the probe
model = VSSModel(cfg.model)

states_all = dataset.load_synthetic("test")[:64]
qs_all = [[q.as_request() for q in ex.questions][:4] for ex in states_all]

from vss.model.serialize import serialize_example  # noqa: E402
from vss.model.tokenizer import VSSTokenizer  # noqa: E402

tok = VSSTokenizer(vocab_size=cfg.model.vocab_size, hash_buckets=cfg.model.hash_buckets)
tok.fit([serialize_example(ex.state, [q.as_request() for q in ex.questions])
         for ex in dataset.load_synthetic("train")[:2000]])
model.tokenizer = tok
model.vss_encoder.tokenizer = tok

# --- probe 1: is the state-block token length heterogeneous across examples?
lens = []
for ex, qs in zip(states_all, qs_all):
    text, spans = model.vss_encoder._serialize_with_spans(ex.state, qs)
    lens.append(spans[0][0])  # token index where question 1 starts == state length
uniq = sorted(set(lens))
print(f"PROBE1 state-block token lengths: min={min(lens)} max={max(lens)} "
      f"n_distinct={len(uniq)} first5={uniq[:5]}")

# --- probe 2: do per-question outputs depend on batch composition?
def run(batch_idx):
    st = [states_all[i].state for i in batch_idx]
    qs = [qs_all[i] for i in batch_idx]
    with torch.no_grad():
        out = model(st, qs)
    rows = []
    for b, grp in enumerate(out["per_example_rows"]):
        for r in grp:
            if r["type"] == "choice":
                rows.append((batch_idx[b], r))
    return rows

sel = list(range(8))
solo = run(sel)
batched = run(sel)

# now the same 8 examples but each presented inside a batch whose FIRST
# example has a different (much longer) state block
order = sorted(range(len(states_all)), key=lambda i: -lens[i])
grouped = run(order[:1] + [i for i in sel[1:]])

print("PROBE2 max |logit diff| same-batch vs reordered-batch:", end=" ")
worst = 0.0
for (bi, a) in solo:
    for (bj, b) in grouped:
        if bi == bj:
            va = torch.cat([a["logits"], a["abstain_logit"].reshape(1)])
            vb = torch.cat([b["logits"], b["abstain_logit"].reshape(1)])
            worst = max(worst, float((va - vb).abs().max()))
print(f"{worst:.3e}")

# --- probe 3: the stable-position values actually assigned
enc = model.vss_encoder.encode_batch(
    [states_all[i].state for i in sel], [qs_all[i] for i in sel]
)
pos = enc["positions"]
print(f"PROBE3 positions shape={tuple(pos.shape)}")
for b, i in enumerate(sel):
    spans = enc["spans"][b]
    sl = spans[0][0]
    q1 = (spans[0][1] - spans[0][0])
    own_state_len = lens[i]
    print(f"  ex{i:>2} true_state_len={own_state_len:>4} pos_used_state_len={int(pos[b][:sl].max().item() + 1):>4} "
          f"q1_start_pos={int(pos[b, spans[0][0]].item()):>4} (expected {own_state_len})")

# --- probe 4: real datasets — are state lengths heterogeneous there?
from vss.data.schema import load_jsonl  # noqa: E402

for ds in ("clinc150", "banking77"):
    exs = load_jsonl(str(REPO / "data" / ds / "train.jsonl"))[:64]
    ls = []
    for ex in exs:
        text, spans = model.vss_encoder._serialize_with_spans(
            ex.state, [q.as_request() for q in ex.questions[:1]]
        )
        ls.append(spans[0][0])
    print(f"PROBE4 {ds}: state token len min={min(ls)} max={max(ls)} "
          f"n_distinct={len(set(ls))}")
