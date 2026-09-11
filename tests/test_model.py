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


def test_encoder_state_is_unchanged_by_extra_padding():
    torch.manual_seed(1)
    config = Seq2SeqConfig(hidden_size=4, num_layers=1, embedding_dim=3)
    model = Seq2Seq(8, 9, config).eval()
    short = torch.tensor([[4, 5, 2]])
    padded = torch.tensor([[4, 5, 2, 0, 0]])
    target = torch.tensor([[1, 5, 2]])
    lengths = torch.tensor([3])

    with torch.no_grad():
        short_output = model(short, target, lengths, teacher_forcing_ratio=1.0)
        padded_output = model(padded, target, lengths, teacher_forcing_ratio=1.0)

    assert torch.allclose(short_output, padded_output, atol=1e-6)


def test_greedy_generation_has_bounded_length():
    model = Seq2Seq(8, 9, Seq2SeqConfig(hidden_size=4, num_layers=1, embedding_dim=3))
    generated = model.generate(torch.tensor([[4, 5, 2]]), max_length=5)
    assert generated.shape[0] == 1
    assert generated.shape[1] <= 5
