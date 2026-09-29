"""System C: VSS runner (existing qmask checkpoints, unmodified).

Loads trained VSS runs (best.pt + final/config.yaml + final/vocab.json),
runs multi-question inference, and extracts per-question records aligned to
gold. Uses the model's raw rows (logits/probs/calibration) so distribution
metrics (Brier/NLL) are computed on the same distributions the heads produce,
while decision semantics (abstention threshold, blend confidence, trained
abstain class) go through the existing inference engine — VSS is evaluated
exactly as shipped, never modified to win the benchmark.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from vss.inference.engine import build_answer_rows, group_answers_by_example  # noqa: E402
from vss.inference.batching import run_batch  # noqa: E402
from vss.model.config import ModelConfig  # noqa: E402
from vss.model.questions import QuestionSpec  # noqa: E402
from vss.model.tokenizer import VSSTokenizer  # noqa: E402
from vss.model.vss_model import VSSModel  # noqa: E402

ABSTAIN = "ABSTAIN"


def load_vss(run_dir: str, device: str = "cpu") -> VSSModel:
    """Load a trained VSS run: best.pt weights + final/ tokenizer.

    Checkpoint format: torch.save({"model": ..., "config": {...}}, best.pt)
    where config may nest under "model" (see scripts/eval_ood.py).
    """
    run = Path(run_dir)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    raw = ck["config"]
    raw = raw.get("model", raw) if isinstance(raw, dict) else raw
    cfg = ModelConfig(**raw)
    model = VSSModel(cfg)
    model.load_state_dict(ck["model"])
    tok = VSSTokenizer.load(str(run / "final" / "vocab.json"))
    model.tokenizer = tok
    model.vss_encoder.tokenizer = tok
    return model.to(device).eval()


def default_inference_kwargs() -> dict:
    """The shipped inference configuration (qmask YAML)."""
    return {
        "abstain_threshold": 0.55,
        "enable_abstention": True,
        "confidence_mode": "blend",
        "device": "cpu",
    }


@torch.no_grad()
def predict_examples(
    model: VSSModel,
    examples,  # list[MultiQuestionExample]
    batch_size: int = 8,
    device: str = "cpu",
    **infer_kwargs,
) -> list[list[dict]]:
    """Per-request lists of per-question records (metrics.py schema).

    One padded forward per batch of requests (the shipped batched path); the
    VSS system answers all Q questions of a request in that single pass.
    """
    model.eval()
    kw = default_inference_kwargs() | infer_kwargs
    kw["device"] = device
    out_records: list[list[dict]] = []
    for s in range(0, len(examples), batch_size):
        chunk = examples[s : s + batch_size]
        states = [ex.state for ex in chunk]
        questions_list = [[q.as_request() for q in ex.questions] for ex in chunk]
        answers = run_batch(model, states, questions_list, **kw)
        # raw rows for distribution metrics (aligned with questions order)
        with torch.no_grad():
            out = model(states, questions_list, device=device)
        rows_by_request = out["per_example_rows"]
        for ex, ans, rows in zip(chunk, answers, rows_by_request):
            recs: list[dict] = []
            for q, row in zip(ex.questions, rows):
                a = ans["answers"][q.id]
                rec: dict = {"qid": q.id, "type": q.type, "gold": q.answer}
                if q.type == "choice":
                    if a["value"] == ABSTAIN:
                        p_full = torch.softmax(
                            torch.cat([row["logits"].float(),
                                       row["abstain_logit"].float().view(1)]), -1)
                        rec.update({
                            "pred": ABSTAIN,
                            "probs": {o: float(p_full[i]) for i, o in
                                      enumerate(q.options or ())},
                            "max_prob": float(p_full.max()),
                            "conf": float(a.get("confidence", 0.0)),
                            "correct": q.answer == ABSTAIN,
                            "abstained": True,
                            "abstain_prob": float(p_full[-1]),
                        })
                    else:
                        probs = torch.softmax(row["logits"].float(), -1)
                        dist = {o: float(p) for o, p in zip(q.options or (), probs)}
                        rec.update({
                            "pred": a["value"],
                            "probs": dist,
                            "max_prob": float(probs.max()),
                            "conf": float(a.get("confidence", 0.0)),
                            "correct": a["value"] == q.answer,
                            "abstained": False,
                        })
                elif q.type == "noul":
                    p_true = float(row["prob"])
                    rec.update({
                        "pred": a["value"],
                        "p_true": p_true,
                        "max_prob": max(p_true, 1 - p_true),
                        "conf": float(a.get("confidence", 0.0)),
                        "correct": a["value"] == int(q.answer),
                        "abstained": a["value"] == ABSTAIN,
                    })
                else:
                    value = float(row["value"])
                    lo, hi = q.min or 0.0, q.max or 10.0
                    probs = row["probs"].float().reshape(-1)
                    centers = torch.linspace(lo, hi, probs.shape[-1])
                    tol = 0.10 * (hi - lo)
                    gold_bin = int(round((float(q.answer) - lo) / (hi - lo)
                                         * (probs.shape[-1] - 1)))
                    pred_bin = int(torch.argmax(probs))
                    rec.update({
                        "pred": round(value, 4),
                        "value": value,
                        "probs": [float(p) for p in probs],
                        "gold_bin": gold_bin,
                        "pred_bin_value": float(centers[pred_bin]),
                        "max_prob": float(probs[gold_bin]),
                        "conf": float(a.get("confidence", 0.0)),
                        "correct": abs(value - float(q.answer)) <= tol,
                        "abstained": a["value"] == ABSTAIN,
                    })
                recs.append(rec)
            out_records.append(recs)
    return out_records


def latency_vss(
    model: VSSModel,
    examples,  # list[MultiQuestionExample] with >= q questions each
    q: int,
    warmup_iters: int,
    measured_iters: int,
    device: str = "cpu",
) -> dict:
    """Single-request latency for mode C: one request carrying Q questions.

    Each iteration = full request (serialize + tokenize + one padded forward
    + answer extraction), mirroring what the API/CLI path does.
    """
    model.eval()
    ex = examples[0]
    state = ex.state
    questions = [q.as_request() for q in ex.questions[:q]]
    kw = default_inference_kwargs() | {"device": device}

    def once() -> None:
        run_batch(model, [state], [questions], **kw)

    for _ in range(warmup_iters):
        once()
    times: list[float] = []
    for _ in range(measured_iters):
        t0 = time.perf_counter()
        once()
        times.append(time.perf_counter() - t0)
    return _lat_stats(times)


def _lat_stats(times_s: list[float]) -> dict:
    ts = sorted(times_s)
    n = len(ts)
    mean = sum(ts) / n
    return {
        "mean_ms": mean * 1000.0,
        "p50_ms": ts[n // 2] * 1000.0,
        "p95_ms": ts[int(n * 0.95)] * 1000.0,
        "p99_ms": ts[min(n - 1, int(n * 0.99))] * 1000.0,
    }
