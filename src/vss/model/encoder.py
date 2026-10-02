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

        # per-question-block attention mask [vss-qmask]: each question block
        # attends to the state and to itself only; the state attends to itself
        # only. Every question vector is then a pure function of (state, that
        # question), which provably removes the measured cross-question
        # interference (benchmarks/interference). Built only when the encoder
        # is configured for it; None keeps the legacy bidirectional behavior.
        qmask = None
        if getattr(self.encoder, "question_masked", False) and any(spans):
            T = token_ids.shape[1]
            neg = torch.finfo(torch.float32).min
            # PER-EXAMPLE masks [B,1,T,T]. A single shared [T,T] matrix is
            # wrong: the loop below writes disjoint row sets per example, so
            # the last example's layout wins and every other example is
            # masked against another example's span geometry. That silently
            # removed block isolation for batched requests whose span layouts
            # differ, and made answers depend on batch composition and on
            # question order (reordering shifts spans). One mask per example
            # is also what the isolation guarantee is defined against.
            m = torch.zeros(len(spans), T, T, dtype=torch.float32, device=device)
            for b, spans_b in enumerate(spans):
                qs = [(s, e) for (s, e) in spans_b if e > s]
                if not qs:
                    continue
                # widen spans for boundary-token drift, but never into the
                # previous/next question block (spans may touch: e_i == s_{i+1})
                wide: list[tuple[int, int]] = []
                for i, (s, e) in enumerate(qs):
                    prev_e = qs[i - 1][1] if i > 0 else 0
                    next_s = qs[i + 1][0] if i + 1 < len(qs) else T
                    ws = max(prev_e, s - 1)
                    we = min(next_s, e + 1)
                    if we > ws:
                        wide.append((ws, we))
                is_q = torch.zeros(T, dtype=torch.bool, device=device)
                for s, e in wide:
                    is_q[s:e] = True
                # state rows: mask ALL question columns
                state_rows = ~is_q
                m[b, state_rows, :] = torch.where(is_q, neg, 0.0)
                # question rows: mask other question columns
                for s, e in wide:
                    other_q = is_q.clone()
                    other_q[s:e] = False
                    m[b, s:e, :] = torch.where(other_q, neg, 0.0)
            qmask = m.unsqueeze(1)  # [B,1,T,T]
        # stable per-token positions [vss-qmask]: each serialization block
        # occupies positions independent of how many blocks FOLLOW it, so a
        # question's RoPE phase never changes when other questions are
        # appended. Required together with the attention mask for isolation.
        positions = None
        if getattr(self.encoder, "question_masked", False):
            seq_len = token_ids.shape[1]
            # PER-EXAMPLE state length. Taking example 0's state length for the
            # whole batch is wrong whenever states differ in length: the
            # question blocks of a shorter example get RoPE positions that
            # overlap its own state block (the state assignment below runs
            # last and overwrites them), and its question-to-state relative
            # distances are shifted by (L_0 - L_b). That reintroduces exactly
            # the batch-composition dependence the per-example attention mask
            # was written to remove. Inert on the synthetic set (all states
            # serialize to the same 32 tokens) but live on both real datasets
            # (CLINC150 7-24 tokens, Banking77 10-30). See
            # docs/convergence_audit.md finding 3.
            pos = torch.zeros(token_ids.shape[0], seq_len, dtype=torch.long, device=device)
            for b, spans_b in enumerate(spans):
                own_state_len = min(
                    spans_b[0][0] if spans_b else seq_len, seq_len
                )
                for i, (s, e) in enumerate(spans_b):
                    if e <= s:
                        continue
                    pos[b, s:e] = own_state_len + torch.arange(e - s, device=device)
                n_state = min(own_state_len, seq_len)
                pos[b, :n_state] = torch.arange(n_state, device=device)
            positions = pos
        return {"token_ids": token_ids, "spans": spans, "question_mask": qmask,
                "positions": positions}

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
        H = self.encoder(token_ids, attn_bias=enc.get("question_mask"),
                         positions=enc.get("positions"))  # [B, T, D]
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
