"""REST API [public: POST /v1/decide contract].

Run:  vss-serve --model runs/prototype/final --port 8000
"""
from __future__ import annotations

import argparse
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..api import VSS
from ..data.schema import DecisionRequest

app = FastAPI(title="VSS decision API", version="0.1.0")
_model: VSS | None = None


class DecideBody(BaseModel):
    state: dict[str, Any] | list[Any]
    questions: list[dict[str, Any]]


def load_model(path: str) -> VSS:
    global _model
    _model = VSS.from_pretrained(path)
    return _model


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_loaded": str(_model is not None)}


@app.post("/v1/decide")
def decide(body: DecideBody) -> dict[str, Any]:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        req = DecisionRequest.model_validate(body.model_dump())
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    result = _model.decide(req.state, [q.model_dump(exclude_none=True) for q in req.questions])
    return {"answers": result["answers"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="VSS REST server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    import uvicorn

    load_model(args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
