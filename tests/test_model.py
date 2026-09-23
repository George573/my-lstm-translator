import copy

import pytest
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


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_generation_rejects_invalid_temperature(temperature):
    model = Seq2Seq(8, 9, Seq2SeqConfig(hidden_size=4, num_layers=1, embedding_dim=3))
    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        model.generate(torch.tensor([[4, 2]]), do_sample=True, temperature=temperature)


def test_sampling_uses_temperature_and_preserves_eos(monkeypatch):
    model = Seq2Seq(8, 9, Seq2SeqConfig(hidden_size=4, num_layers=1, embedding_dim=3))
    inputs, distributions = [], []
    logits = torch.arange(9, dtype=torch.float).expand(2, 1, 9)

    def decode(tokens, hidden, *args):
        inputs.append(tokens.clone())
        return logits, hidden

    def sample(probabilities, num_samples):
        distributions.append(probabilities.clone())
        assert num_samples == 1
        # First item finishes immediately; its later predictions must be ignored.
        return torch.tensor([[2], [4]]) if len(distributions) == 1 else torch.tensor([[5], [2]])

    monkeypatch.setattr(model.decoder, "forward", decode)
    monkeypatch.setattr(torch, "multinomial", sample)
    generated = model.generate(torch.tensor([[4, 2], [5, 2]]), max_length=5,
                               do_sample=True, temperature=0.5)

    assert generated.tolist() == [[2, 2], [4, 2]]
    assert [tokens.tolist() for tokens in inputs] == [[[1], [1]], [[2], [4]]]
    for probabilities in distributions:
        torch.testing.assert_close(probabilities, torch.softmax(logits[:, 0] / 0.5, dim=-1))


def test_sampling_can_select_non_greedy_tokens_and_is_seeded():
    model = Seq2Seq(8, 9, Seq2SeqConfig(hidden_size=4, num_layers=1, embedding_dim=3)).eval()
    with torch.no_grad():
        model.decoder.output.weight.zero_()
        model.decoder.output.bias.fill_(-float("inf"))
        model.decoder.output.bias[4:6] = 0
    source = torch.tensor([[4, 2]]).expand(64, -1)
    assert model.generate(source, max_length=1).unique().tolist() == [4]
    torch.manual_seed(42)
    sampled = model.generate(source, max_length=3, do_sample=True)
    torch.manual_seed(42)
    repeated = model.generate(source, max_length=3, do_sample=True)
    assert sampled.shape == (64, 3)
    assert sampled.unique().tolist() == [4, 5]
    assert torch.equal(sampled, repeated)


def reference_forward(model, source, target, lengths, ratio):
    """Original per-step projection and slice-write path for regression checks."""
    encoded, hidden = model.encoder(source, lengths)
    hidden = model._bridge_hidden(hidden)
    mask = source.ne(0)
    outputs = encoded.new_zeros(source.size(0), target.size(1) - 1,
                                model.target_vocabulary_size)
    decoder_input = target[:, :1]
    for step in range(target.size(1) - 1):
        predictions, hidden = model.decoder(decoder_input, hidden, encoded, mask)
        outputs[:, step] = predictions[:, 0]
        predicted = predictions.argmax(-1)
        if ratio == 0:
            decoder_input = predicted
        elif ratio == 1:
            decoder_input = target[:, step + 1:step + 2]
        else:
            teacher_mask = torch.rand(source.size(0), 1, device=source.device) < ratio
            decoder_input = torch.where(teacher_mask, target[:, step + 1:step + 2], predicted)
    return outputs


@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
def test_optimized_decoder_matches_original_outputs_and_gradients(attention, ratio):
    torch.manual_seed(7)
    model = Seq2Seq(8, 9, Seq2SeqConfig(hidden_size=4, num_layers=2,
                                      embedding_dim=3, dropout=0.2, attention=attention))
    reference = copy.deepcopy(model)
    source = torch.tensor([[4, 5, 2], [6, 2, 0]])
    target = torch.tensor([[1, 5, 6, 2], [1, 7, 2, 0]])
    lengths = torch.tensor([3, 2])
    torch.manual_seed(19)
    expected = reference_forward(reference, source, target, lengths, ratio)
    torch.manual_seed(19)
    actual = model(source, target, lengths, teacher_forcing_ratio=ratio)
    torch.testing.assert_close(actual, expected)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    criterion(actual.reshape(-1, 9), target[:, 1:].reshape(-1)).backward()
    criterion(expected.reshape(-1, 9), target[:, 1:].reshape(-1)).backward()
    for (name, parameter), (_, original) in zip(model.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(parameter.grad, original.grad, msg=lambda msg: f"{name}: {msg}")


def test_encoder_projection_runs_once_per_sequence():
    model = Seq2Seq(8, 9, Seq2SeqConfig(hidden_size=4, num_layers=1, embedding_dim=3))
    calls = []
    handle = model.decoder.attention.encoder_projection.register_forward_hook(
        lambda *args: calls.append(1))
    try:
        source = torch.tensor([[4, 5, 2]])
        model(source, torch.tensor([[1, 5, 6, 2]]))
        assert len(calls) == 1
        calls.clear()
        model.generate(source, max_length=5)
        assert len(calls) == 1
    finally:
        handle.remove()


@pytest.mark.parametrize("attention", [False, True])
def test_full_teacher_forcing_batches_embedding_and_output_projection(attention):
    model = Seq2Seq(8, 9, Seq2SeqConfig(hidden_size=4, num_layers=1,
                                      embedding_dim=3, attention=attention))
    calls = {"embedding": [], "output": []}
    handles = [
        module.register_forward_hook(
            lambda module, args, output, name=name: calls[name].append(args[0].shape))
        for name, module in (("embedding", model.decoder.embedding),
                             ("output", model.decoder.output))
    ]
    try:
        model(torch.tensor([[4, 5, 2]]), torch.tensor([[1, 5, 6, 2]]),
              teacher_forcing_ratio=1.0)
        assert calls["embedding"] == [torch.Size([1, 3])]
        assert len(calls["output"]) == 1
        assert calls["output"][0][:2] == (1, 3)
    finally:
        for handle in handles:
            handle.remove()
