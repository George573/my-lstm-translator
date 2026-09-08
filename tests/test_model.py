import torch

from lstm_translator.model import Seq2Seq, Seq2SeqConfig, Vocabulary


def test_vocabulary_is_deterministic():
    assert Vocabulary({"z", "a"}).token_to_index == Vocabulary({"a", "z"}).token_to_index


def test_model_output_shape():
    config = Seq2SeqConfig(hidden_size=4, num_layers=1, embedding_dim=3)
    model = Seq2Seq(8, 9, config)
    source = torch.tensor([[4, 5, 2], [6, 2, 0]])
    target = torch.tensor([[1, 5, 6, 2], [1, 7, 2, 0]])

    output = model(source, target, teacher_forcing_ratio=1.0)

    assert output.shape == (2, 3, 9)
