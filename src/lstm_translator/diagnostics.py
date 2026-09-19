"""Opt-in, synchronized training diagnostics; never retain activation tensors."""

from contextlib import contextmanager
import hashlib
import json
import sys

import torch


class DebugLogger:
    def __init__(self, path=None):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, record):
        line = json.dumps(record, sort_keys=True)
        print(f"[debug] {line}", file=sys.stderr, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as output:
                output.write(line + "\n")


def tokenizer_info(tokenizer):
    state = tokenizer.to_dict()
    return dict(sha256=hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
                vocabulary_size=len(tokenizer.tokens), merges=len(tokenizer.rules),
                format_version=state['format_version'],
                text_processing_version=state['text_processing_version'])


class BatchDiagnostics:
    def __init__(self, logger, device, step):
        self.logger = logger
        self.device = torch.device(device)
        self.step = step
        self.stage = "batch"
        self.decoder_step = 0

    def emit(self, event, *, synchronize=True, **fields):
        record = dict(event=event, step=self.step, phase=self.stage,
                      decoder_step=self.decoder_step, **fields)
        if self.device.type == "cuda":
            if synchronize:
                torch.cuda.synchronize(self.device)
            record.update(
                allocated_gib=torch.cuda.memory_allocated(self.device) / 2**30,
                reserved_gib=torch.cuda.memory_reserved(self.device) / 2**30,
                peak_allocated_gib=torch.cuda.max_memory_allocated(self.device) / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(self.device) / 2**30,
            )
        self.logger(record)

    def phase(self, stage):
        self.stage = stage
        self.emit("phase_start")

    @contextmanager
    def watch(self, model, source, target, lengths):
        handles = []
        try:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            def stats(values, width):
                values = values.float()
                return dict(min=int(values.min()), median=float(values.median()),
                            p95=float(torch.quantile(values, .95)), max=int(values.max()),
                            tokens=int(values.sum()),
                            padding_fraction=1 - float(values.sum()) / (len(values) * width))
            target_lengths = target.ne(0).sum(dim=1)
            batch, source_width = source.shape
            target_steps = target.shape[1] - 1
            element_size = next(model.parameters()).element_size()
            self.emit("batch", source_shape=list(source.shape), target_shape=list(target.shape),
                      source_lengths=stats(lengths, source_width),
                      target_lengths=stats(target_lengths, target.shape[1]),
                      source_bpe_over_127=int((lengths - 1 > 127).sum()),
                      target_bpe_over_127=int((target_lengths - 2 > 127).sum()),
                      logits_gib=batch * target_steps * model.target_vocabulary_size * element_size / 2**30,
                      attention_one_tensor_per_step_gib=(
                          batch * source_width * model.config.hidden_size * 2 * target_steps * element_size / 2**30
                          if model.config.attention else 0))

            def encoder_start(module, args):
                self.phase("encoder")

            def encoder_end(module, args, output):
                self.phase("encoder_complete")

            def decoder_start(module, args):
                self.decoder_step += 1
                self.stage = "decoder"

            def decoder_end(module, args, output):
                if self.decoder_step == 1 or self.decoder_step % 16 == 0 or self.decoder_step == target_steps:
                    self.emit("decoder_progress")
                if self.decoder_step == target_steps:
                    self.stage = "stack_logits"

            handles.append(model.encoder.register_forward_pre_hook(encoder_start))
            handles.append(model.encoder.register_forward_hook(encoder_end))
            handles.append(model.decoder.register_forward_pre_hook(decoder_start))
            handles.append(model.decoder.register_forward_hook(decoder_end))
            yield
        except torch.OutOfMemoryError:
            # Do not synchronize a failing CUDA operation or allocate GPU tensors.
            self.emit("oom", synchronize=False)
            raise
        finally:
            for handle in handles:
                handle.remove()
