"""High-level VSS encoder: tokenizes serialized input, embeds questions,
produces the shared decision representation.

Flow [vss]:

    state + questions --canonical-serialize--> tokens
    tokens -> TransformerEncoder -> contextual states H
    question <QUESTION> block positions are extracted; each question's
    contextual vector becomes its query representation q_i.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .serialize import serialize_example
from .tokenizer import VSSTokenizer
from .transformer import TransformerEncoder


def _question_spans(token_ids: list[list[int]]) -> list[list[int]]:
    """Locate tokens of each `<QUESTION>` block in the serialized sequence.

    Works on the *decoded string level* is expensive; instead we re-serialize
    incrementally in `VSSEncoder.encode_inputs` and track spans there. This
    helper exists for tests/debug on pre-tokenized sequences.
    """
    return token_ids  # pragma: no cover - replaced by encode_inputs span tracking


class VSSEncoder:
    """Serialization/tokenization front-end over a TransformerEncoder.

    Deliberately NOT an nn.Module: it owns no parameters, so registering it
    in VSSModel would duplicate every encoder weight in checkpoints [vss].
    """

    def __init__(self, encoder: TransformerEncoder, tokenizer: VSSTokenizer | None) -> None:
        self.encoder = encoder
        self.tokenizer = tokenizer

    # ------------------------------------------------------------- encoding
    def encode_batch(
        self,
        states: list[dict],
        questions_list: list[list[dict]],
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Tokenize + pad a batch, return token ids and per-question spans.

        Returns dict with:
          token_ids  LongTensor [B, T]
          spans      list (len B) of list (len Q_i) of (start, end) token idx
        """
        self.tokenizer.require_built()
        seqs: list[list[int]] = []
        spans: list[list[tuple[int, int]]] = []
        max_len = 0
        for state, questions in zip(states, questions_list):
            text, span = self._serialize_with_spans(state, questions)
            ids = self.tokenizer.encode(text)
            ids = ids[: self.encoder.max_seq_len]
            spans.append(span)
            seqs.append(ids)
            max_len = max(max_len, len(ids))

        pad = self.encoder.pad_token_id
        batch_ids = [ids + [pad] * (max_len - len(ids)) for ids in seqs]
        token_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
        return {"token_ids": token_ids, "spans": spans}

    def _serialize_with_spans(
        self, state: dict, questions: list[dict]
    ) -> tuple[str, list[tuple[int, int]]]:
        """Serialize and record character spans of each <QUESTION> block."""
        from .serialize import serialize_question, serialize_state

        parts = [serialize_state(state)]
        char_spans: list[tuple[int, int]] = []
        cursor = len(parts[0])
        for q in questions:
            qtext = serialize_question(q)
            parts.append(qtext)
            cursor += 1  # newline separator
            char_spans.append((cursor, cursor + len(qtext)))
            cursor += len(qtext)
        text = "\n".join(parts)

        # char span -> approximate token span via re-encoding of the slice;
        # token boundaries align because blocks are whitespace-separated.
        tok_spans: list[tuple[int, int]] = []
        offset = 0
        for cs, ce in char_spans:
            # count tokens strictly before cs and before ce in full text
            prefix = text[:cs]
            mid = text[cs:ce]
            n_before = len(self.tokenizer.encode(prefix))
            n_block = len(self.tokenizer.encode(mid))
            tok_spans.append((n_before, n_before + n_block))
        _ = offset
        return text, tok_spans

    # --------------------------------------------------------------- forward
    def forward(
        self,
        states: list[dict],
        questions_list: list[list[dict]],
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Run the encoder; gather one contextual vector per question.

        Question representation = mean-pool over its <QUESTION> block tokens.
        """
        enc = self.encode_batch(states, questions_list, device)
        token_ids = enc["token_ids"]
        H = self.encoder(token_ids)  # [B, T, D]
        qvecs: list[torch.Tensor] = []
        for b, spans in enumerate(enc["spans"]):
            for (s, e) in spans:
                s = max(0, min(s, H.shape[1] - 1))
                e = max(s + 1, min(e, H.shape[1]))
                qvecs.append(H[b, s:e].mean(dim=0))
        if not qvecs:
            D = H.shape[-1]
            qvecs = [H.new_zeros(D)]
        q = torch.stack(qvecs, dim=0)  # [sum_i Q_i, D]
        return {"hidden": H, "question_vectors": q, "token_ids": token_ids, "spans": enc["spans"]}
