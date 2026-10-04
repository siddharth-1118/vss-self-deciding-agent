# MODEL CARD — VSS-Prototype

> **The full, evidence-bearing model card is [`docs/model_card.md`](docs/model_card.md).**
> This page is a short quick-reference. Where the two differ, the one in
> `docs/` is authoritative. Release status, the real-data benchmark and the full
> limitations list live there and in
> [`docs/release_readiness.md`](docs/release_readiness.md).

**Release status: RESEARCH PREVIEW (v0.1.0).** Not a competitive model: on
three seeds each, a tuned plain classifier beats VSS on both real datasets
(Banking77 0.878 vs 0.830; CLINC150 0.918 vs 0.705). See the linked card.

## Model details

- **Name**: VSS-Prototype (v0.1)
- **Type**: System-One decision model — state + typed questions -> parallel
  probabilistic decisions in one transformer forward pass
- **Architecture**: bidirectional transformer encoder (RoPE, RMSNorm, SwiGLU,
  GQA), per-question adapters, Choice/Noul/Score heads, auxiliary calibration
  head. **11,164,483 parameters** for the benchmark configuration used by the
  convergence sweep; **13,655,364** for `configs/vss-prototype.yaml`, which
  drives the verified end-to-end reference run.
- **Context**: 4096 tokens (canonical serialized state + question blocks)
- **Languages**: any (word-vocab + hash OOV); synthetic training data is English

## Intended use

Routing, triage, classification, scoring and binary decisions over support
tickets, agent/tool decisions, and similar structured decision workloads
where outputs are consumed by machines, not humans reading prose. The model
is designed to be retrained on domain data via the documented JSONL format.

## Out-of-scope use

- Open-ended text generation (the model does not generate text)
- High-stakes autonomous decisions without human review
- Use as a factual QA system or chatbot

## Training data

Synthetic, programmatically labeled support-domain examples across six
curriculum stages (basic/noul/score/multi/dynamic/hard) with negation traps,
irrelevant context, typos, and in-domain `ABSTAIN` targets. No LLM was used to
label data. No proprietary or scraped data is included. Real-data benchmarks
(Banking77, CLINC150) are used for evaluation and fine-tuning only, never
mixed into the synthetic training set.

## Measured performance (CPU, held-out synthetic split, n=480)

| metric | value |
|---|---|
| choice accuracy (incl. correct abstentions) | 1.000 |
| choice abstain rate | 0.100 (exactly the ABSTAIN-target share) |
| ECE / Brier / NLL | 0.0044 / 0.0008 / 0.0049 |
| score mean relative error | 0.042 |
| p50 latency, 1 / 5 / 10 / 50 questions | 12.1 / 22.2 / 34.8 / 174.7 ms |

Reproducible via `benchmarks/README.md`. **These numbers are in-distribution
synthetic results only** — the eval split comes from the same generator
family as training data. They demonstrate a working, calibrated pipeline;
they do not establish real-world robustness.

## Limitations and known issues

- **Not competitive with a plain classifier on real data.** Three seeds each on
  Banking77 and CLINC150: plain leads both. Synthetic accuracy does not predict
  real-data accuracy — the synthetic tie above coexists with both real losses.
- **Unstable across seeds.** On Banking77 VSS spans 0.765–0.895 (sd 0.065)
  across seeds where plain spans 0.870–0.885 (sd 0.008). Do not quote a single
  VSS number as expected performance.
- **`ABSTAIN` is not an out-of-distribution detector.** Measured, not assumed:
  the abstain logit separates OOS at AUROC 0.664 (chance = 0.5). Emitted
  *confidence* does better (AUROC 0.807 on CLINC150 test) but the operating
  point that catches most OOS rejects 33% of legitimate in-scope traffic and
  still misses ~20% of OOS. Treat it as selective prediction with published
  costs, not as a novelty alarm. Details and the trade-off curve:
  `docs/benchmark_report.md` §8.
- Training is on synthetic English support-domain data; the eval split shares
  the generator family, so the numbers above are in-distribution. The robustness
  suite has since been run on real public datasets — see `docs/`.
- AUROC/AUPRC of confidence are undefined on a split with zero errors; the
  report emits `null` rather than a fabricated value.
- Calibration metrics reflect the synthetic difficulty distribution, not
  any deployment distribution. On real data the answered-subset ECE is 0.122.

## Ethical considerations

Decision models can encode biases from training data. Abstention exists so
the model can decline rather than guess; deployments should monitor
abstention rates across subgroups and keep humans in the loop for
consequential decisions.
