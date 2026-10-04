"""VSS command-line interface.

Commands:
  vss train      --config configs/vss-prototype.yaml [--resume runs/.../last.pt]
  vss decide     --model runs/prototype/final --state '{"message": "..."}' --questions questions.json
  vss serve      --model runs/prototype/final
  vss evaluate   --model runs/prototype/final --data data/eval.jsonl
  vss validate   --data data/train.jsonl
  vss generate   --out data/generated/train.jsonl [--counts basic=400,...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_train(args: argparse.Namespace) -> int:
    from .data.schema import TrainingExample, load_jsonl
    from .model.config import VSSConfig
    from .model.vss_model import VSSModel
    from .training.trainer import Trainer

    cfg = VSSConfig.load(args.config)
    train = load_jsonl(args.train)
    eval_examples = load_jsonl(args.eval) if args.eval else []
    if not train:
        print("no training data", file=sys.stderr)
        return 1
    model = VSSModel(cfg.model)
    trainer = Trainer(model, cfg, train, eval_examples)
    result = trainer.fit(resume_from=args.resume)
    print(json.dumps({"history": result["history"], "best_loss": result["best_loss"]}, indent=2))
    print(f"checkpoint: {cfg.training.checkpoint_dir}/final")
    return 0


def _load_json_arg(value: str, what: str) -> object:
    """Accept EITHER inline JSON or a path to a JSON file.

    `--state` took inline JSON while `--questions` took only a path, with no
    help text saying so. Passing inline JSON to `--questions` produced a raw
    `OSError: [Errno 22] Invalid argument` traceback instead of an explanation.
    Both now accept both forms, and a bad value produces one clear line on
    stderr with a non-zero exit rather than a stack trace.
    """
    text = value.strip()
    if text[:1] in "{[":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise _CliError(f"--{what}: invalid inline JSON ({exc})") from None
    path = Path(text)
    if not path.exists():
        raise _CliError(
            f"--{what}: {text!r} is neither inline JSON (starting with {{ or [) "
            f"nor an existing file. Write the JSON to a file, or pass it inline."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _CliError(f"--{what}: {path} is not valid JSON ({exc})") from None


class _CliError(Exception):
    """User-facing CLI error: printed as one line, no traceback."""


def _cmd_decide(args: argparse.Namespace) -> int:
    from .api import VSS

    state_arg = args.state if args.state is not None else args.state_file
    if state_arg is None:
        raise _CliError("--state (inline JSON) or --state-file (path) is required")
    state = _load_json_arg(state_arg, "state")
    questions = _load_json_arg(args.questions, "questions")
    if not isinstance(questions, list):
        raise _CliError("--questions must be a JSON array of question objects")
    model = VSS.from_pretrained(args.model)
    result = model.decide(state, questions)
    print(json.dumps(result, indent=2))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .inference.server import main as serve_main

    sys.argv = ["vss-serve", "--model", args.model, "--host", args.host, "--port", str(args.port)]
    serve_main()
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .api import VSS
    from .data.schema import load_jsonl

    model = VSS.from_pretrained(args.model)
    examples = load_jsonl(args.data)
    preds: list[str] = []
    golds: list[str] = []
    confs: list[float] = []
    correct: list[int] = []
    for ex in examples:
        result = model.decide(
            ex.state, [q.as_request() for q in ex.questions]
        )
        for q in ex.questions:
            a = result["answers"][q.id]
            if q.type == "choice":
                preds.append(str(a["value"]))
                golds.append(str(q.answer))
                is_abstain = a["value"] == "ABSTAIN"
                confs.append(float(a["confidence"]))
                correct.append(0 if is_abstain else int(a["value"] == q.answer))
    from .eval.metrics import accuracy, f1_scores, roc_pr
    from .model.calibration import brier_score, expected_calibration_error, nll as nll_fn

    report = {
        "n": len(examples),
        "accuracy": accuracy(preds, golds),
        "f1": f1_scores(preds, golds),
        "calibration_choice_only": {
            "ece": expected_calibration_error(confs, correct),
            "brier": brier_score(confs, correct),
            "nll": nll_fn(confs, correct),
            **roc_pr(confs, correct),
        },
    }
    print(json.dumps(report, indent=2))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from .data.schema import validate_jsonl_file

    valid, errors = validate_jsonl_file(args.data)
    print(f"valid examples: {valid}")
    if errors:
        print(f"errors: {len(errors)}")
        for e in errors[:20]:
            print(" -", e)
        return 1
    print("all examples valid")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from .data.schema import dump_jsonl
    from .data.synthetic import SyntheticGenerator

    gen = SyntheticGenerator(seed=args.seed)
    counts: dict[str, int] = {}
    if args.counts:
        for part in args.counts.split(","):
            k, v = part.split("=")
            counts[k.strip()] = int(v)
    examples = gen.generate(counts or None)
    dump_jsonl(examples, args.out)
    print(f"wrote {len(examples)} examples to {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="vss", description="VSS decision model CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--train", required=True)
    t.add_argument("--eval")
    t.add_argument("--resume")
    t.set_defaults(fn=_cmd_train)

    d = sub.add_parser("decide")
    d.add_argument("--model", required=True)
    d.add_argument("--state", help='state as inline JSON, e.g. \'{"message": "..."}\'')
    d.add_argument("--state-file", help="path to a JSON file holding the state")
    d.add_argument("--questions",
                   help='questions as inline JSON or a path to a JSON file, '
                        'e.g. \'[{"id":"dept","type":"choice","options":["a","b"]}]\'')
    d.set_defaults(fn=_cmd_decide)

    s = sub.add_parser("serve")
    s.add_argument("--model", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=_cmd_serve)

    e = sub.add_parser("evaluate")
    e.add_argument("--model", required=True)
    e.add_argument("--data", required=True)
    e.set_defaults(fn=_cmd_evaluate)

    v = sub.add_parser("validate")
    v.add_argument("--data", required=True)
    v.set_defaults(fn=_cmd_validate)

    g = sub.add_parser("generate")
    g.add_argument("--out", required=True)
    g.add_argument("--counts")
    g.add_argument("--seed", type=int, default=13)
    g.set_defaults(fn=_cmd_generate)

    args = ap.parse_args()
    try:
        return args.fn(args)
    except _CliError as exc:
        # A user mistake is not a crash. One line on stderr, non-zero exit.
        print(f"vss {getattr(args, 'cmd', '')}: error: {exc}".replace("  ", " "),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
