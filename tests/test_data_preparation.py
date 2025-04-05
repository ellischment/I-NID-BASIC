# tests/test_data_preparation.py
import pytest
import pandas as pd
from src.data_preparation import create_long_tailed_distribution, split_data_known_novel


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'text': ['check balance'] * 15 + ['transfer money'] * 10 +
                ['open account'] * 8 + ['loan request'] * 5,
        'intent': [0] * 15 + [1] * 10 + [2] * 8 + [3] * 5
    })


def test_long_tail_creation(sample_data):
    # Test gamma=1 (balanced)
    balanced = create_long_tailed_distribution(sample_data, gamma=1, num_classes=4)
    assert len(balanced) == len(sample_data)

    # Test gamma=3 creates long tail
    long_tail = create_long_tailed_distribution(sample_data, gamma=3, num_classes=4)
    counts = long_tail['intent'].value_counts().to_dict()
    assert counts[0] > counts[3]  # Head class should have more samples


def test_data_splitting(sample_data):
    labeled, unlabeled, test = split_data_known_novel(sample_data)

    # Basic checks
    assert len(labeled) > 0
    assert len(unlabeled) > 0
    assert len(test) > 0

    # Verify no overlap between labeled and test
    assert set(labeled.index).isdisjoint(set(test.index))

    # Verify test is balanced
    test_counts = test['intent'].value_counts()
    assert test_counts.max() - test_counts.min() <= 2  # Nearly balanced