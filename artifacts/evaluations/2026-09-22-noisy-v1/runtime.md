# Runtime benchmark

Measured sequentially on Intel Core i7-8700 CPU, PyTorch 2.7.1+cu118, four computation threads, one interop thread, greedy decoding up to 100 tokens. Both models received identical source texts from the saved historical evaluation. CUDA was unavailable.

| Measurement | Previous quick-TF-decay model | New noisy-v1 model |
|---|---:|---:|
| Load checkpoint | 1.51 s | 0.72 s |
| First translation after load | 0.12 s | 0.16 s |
| Mean single-request latency | 132.52 ms | 227.30 ms |
| Median single-request latency | 109.14 ms | 169.36 ms |
| 95th percentile single-request latency | 302.25 ms | 494.27 ms |
| Batch throughput (batch size 16) | 36.34 sentences/s | 18.76 sentences/s |

The new model has 1.72× the mean single-request latency and 51.6% of the previous model’s batch throughput on these inputs.

Single-request measurements cover 60 inputs after five warmup requests, using `Translator.translate`. Batch throughput uses the median of three passes over 128 inputs after one warmup batch. Timing includes tokenization, generation and detokenization, excluding quality metrics and checkpoint loading. Different generated lengths affect end-to-end timing. No concurrent evaluation jobs were running; other host activity was not controlled. Load measurements are single observations with potentially warm filesystem caches; the old file includes optimizer state while the new file is inference-only, so load times are not an architecture comparison.

The earlier quality-evaluation jobs ran concurrently, making their saved durations unsuitable for a controlled speed comparison.

Reproduce from the repository root:

```bash
.venv-3-12/bin/python artifacts/evaluations/2026-09-22-noisy-v1/benchmark_runtime.py
```

Raw timings and actual translations are in `runtime.json`.
