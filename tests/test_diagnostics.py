"""CPU coverage for diagnostics, including simulated OOM cleanup."""
import unittest

import torch

from lstm_translator import Seq2Seq, Seq2SeqConfig
from lstm_translator.diagnostics import BatchDiagnostics
from lstm_translator.training import _train_batch


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model = Seq2Seq(8, 8, Seq2SeqConfig(hidden_size=4, num_layers=1, embedding_dim=3))
        self.source = torch.tensor([[4, 2, 0], [5, 4, 2]])
        self.target = torch.tensor([[1, 4, 2, 0], [1, 5, 4, 2]])
        self.lengths = torch.tensor([2, 3])

    def test_logging_preserves_training_and_reports_lengths(self):
        import copy
        plain = copy.deepcopy(self.model)
        records = []
        outputs = []
        for model, diagnostic in [(plain, None), (self.model, BatchDiagnostics(records.append, 'cpu', 1))]:
            outputs.append(_train_batch(model, self.source, self.target, self.lengths,
                torch.nn.CrossEntropyLoss(ignore_index=0), torch.optim.AdamW(model.parameters()),
                'cpu', 1.0, 1.0, diagnostics=diagnostic))
        self.assertEqual(outputs[0], outputs[1])
        for left, right in zip(plain.parameters(), self.model.parameters()):
            self.assertTrue(torch.equal(left, right))
        batch = records[0]
        self.assertEqual(batch['source_lengths']['tokens'], 5)
        self.assertEqual(batch['target_lengths']['tokens'], 7)
        self.assertEqual(batch['source_shape'], [2, 3])
        self.assertEqual(records[-1]['phase'], 'complete')
        self.assertFalse(self.model.decoder._forward_hooks)

    def test_oom_reports_stage_and_removes_hooks(self):
        records = []
        diagnostic = BatchDiagnostics(records.append, 'cpu', 7)
        with self.assertRaises(torch.OutOfMemoryError):
            with diagnostic.watch(self.model, self.source, self.target, self.lengths):
                diagnostic.phase('backward')
                raise torch.OutOfMemoryError('simulated')
        self.assertEqual(records[-1]['event'], 'oom')
        self.assertEqual(records[-1]['phase'], 'backward')
        self.assertFalse(self.model.encoder._forward_pre_hooks)
        self.assertFalse(self.model.decoder._forward_hooks)


if __name__ == '__main__':
    unittest.main()
