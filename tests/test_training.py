from lstm_translator import TrainingConfig


def test_training_progress_is_opt_in():
    assert TrainingConfig().show_progress is False


def test_training_progress_can_be_enabled():
    assert TrainingConfig(show_progress=True).show_progress is True
