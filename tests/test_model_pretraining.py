# tests/test_pretraining.py
import pytest
import torch
from transformers import BertTokenizer
from src.model_pretraining import IntentDataset, PretrainConfig
import pandas as pd

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'text': ['check balance', 'transfer money', 'open account'],
        'intent': [0, 1, 2]
    })


def test_intent_dataset(sample_data):
    config = PretrainConfig(batch_size=2, max_seq_length=32)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = IntentDataset(sample_data, tokenizer, config)

    # Test basic functionality
    assert len(dataset) == 3
    item = dataset[0]
    assert 'input_ids' in item
    assert 'attention_mask' in item
    assert 'labels' in item

    # Test batching
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))
    assert batch['input_ids'].shape == (2, config.max_seq_length)