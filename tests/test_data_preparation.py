import pytest
import pandas as pd
import numpy as np
import os

from src.data_preparation import (  # Updated import path
    create_long_tailed_distribution,
    split_data_known_novel,
    prepare_dataset,
    PROCESSED_DIR
)


@pytest.fixture
def sample_data():
    """Generate synthetic dataset for testing"""
    intents = [f"intent_{i}" for i in range(10)]
    texts = [f"sample text {i}" for i in range(100)]
    return pd.DataFrame({
        'text': texts * 10,
        'intent': sorted(intents * 100)
    })

def test_create_long_tailed_distribution(sample_data):
    """Test long-tail distribution creation"""
    gamma = 3
    num_classes = 10
    result = create_long_tailed_distribution(sample_data, gamma, num_classes)

    # Verify distribution follows long-tail pattern
    counts = result['intent'].value_counts().sort_index()
    for i in range(1, num_classes):
        assert counts[i - 1] >= counts[i], f"Distribution not decreasing for gamma={gamma}"


def test_split_data_known_novel(sample_data):
    """Test known/novel split with 75%/25% ratio"""
    labeled, unlabeled, test = split_data_known_novel(sample_data)

    all_intents = sample_data['intent'].unique()
    known_intents = labeled['intent'].unique()

    # Verify 75%/25% split
    assert len(known_intents) / len(all_intents) == pytest.approx(0.75, 0.1)

    # Verify 10% labeled data
    original_known = sample_data[sample_data['intent'].isin(known_intents)]
    assert len(labeled) / len(original_known) == pytest.approx(0.1, 0.05)

    # Verify test set is balanced
    test_counts = test['intent'].value_counts()
    assert test_counts.std() / test_counts.mean() < 0.2  # Low relative std


def test_prepare_dataset_with_realistic_data(monkeypatch, tmp_path):
    test_data = pd.DataFrame({
        'text': ["text1", "text2", "text3", "text4"],
        'intent': ["intent1", "intent1", "intent2", "intent3"]
    })

    def mock_load(dataset_name, config):
        return {'train': test_data.to_dict('records')}

    monkeypatch.setattr("src.data_preparation.load_dataset", mock_load)
    monkeypatch.setattr("src.data_preparation.PROCESSED_DIR", str(tmp_path))

    saved_files = {}
    original_to_csv = pd.DataFrame.to_csv

    def mock_to_csv(self, path, *args, **kwargs):
        saved_files[os.path.basename(path)] = self
        original_to_csv(self, path, *args, **kwargs)

    monkeypatch.setattr("pandas.DataFrame.to_csv", mock_to_csv)

    prepare_dataset("test_ds", None, num_classes=3, gamma_values=[3])

    output_dir = tmp_path / "test_ds_gamma3"
    assert output_dir.exists()

    expected_files = ["train_labeled.csv", "train_unlabeled.csv", "test.csv"]
    for file in expected_files:
        assert file in saved_files
        assert not saved_files[file].empty
def test_empty_data_handling():
    """Test handling of empty input"""
    empty_df = pd.DataFrame(columns=['text', 'intent'])
    labeled, unlabeled, test = split_data_known_novel(empty_df)
    assert labeled.empty and unlabeled.empty and test.empty


def test_small_dataset_handling():
    """Test minimum labeled samples adjustment"""
    tiny_data = pd.DataFrame({
        'text': ['a', 'b', 'c'],
        'intent': ['x', 'x', 'y']
    })
    labeled, _, _ = split_data_known_novel(tiny_data)
    assert len(labeled) >= 1  # Should have at least 1 sample