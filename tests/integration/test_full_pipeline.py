# tests/integration/test_full_pipeline.py
import pytest
import pandas as pd
import torch
from pathlib import Path
from src.data_preparation import prepare_dataset
from src.model_pretraining import pretrain_model, PretrainConfig
from src.pseudo_labeling import PseudoLabelGenerator
from src.representation_training import RepresentationLearner
from src.evaluation import evaluate_model
from torch.utils.data import DataLoader
import numpy as np


@pytest.mark.integration
def test_full_pipeline(tmp_path):
    # ========== 1. Data Preparation ==========
    test_data = pd.DataFrame({
        'text': ['check balance'] * 5 + ['transfer money'] * 4 +
                ['open account'] * 3 + ['loan request'] * 2,
        'intent': [0] * 5 + [1] * 4 + [2] * 3 + [3] * 2
    })

    # Save test data structure
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    test_data.to_csv(data_dir / "test.csv", index=False)

    # ========== 2. Model Pretraining ==========
    config = PretrainConfig(
        batch_size=8,
        num_epochs=1,
        max_seq_length=32,
        pretrained_model_path=str(tmp_path / "pretrained")
    )

    model, tokenizer = pretrain_model(test_data, test_data, config)

    # Verify model output
    test_input = tokenizer("test input", return_tensors="pt")
    with torch.no_grad():
        output = model(**test_input)
    assert output.logits.shape[1] == 4  # 4 classes in test data

    # ========== 3. Pseudo-Labeling ==========
    plg = PseudoLabelGenerator(model, tokenizer)
    texts = test_data['text'].tolist()
    embeddings = plg.get_embeddings(texts)
    pseudo_labels = plg.generate_labels(test_data['text'].tolist(), n_clusters=4)

    assert embeddings.shape == (len(texts), 768)  # Проверяем размерность BERT-эмбеддингов
    assert not np.isnan(embeddings).any()
    assert len(pseudo_labels) == len(test_data)
    assert len(set(pseudo_labels)) <= 4

    # ========== 4. Representation Learning ==========
    test_data['pseudo_label'] = pseudo_labels
    rep_learner = RepresentationLearner(model, tokenizer, config)

    # Test training step
    test_dataset = rep_learner._prepare_dataset(test_data[:2], test_data[2:])
    test_loader = DataLoader(test_dataset, batch_size=4)
    batch = next(iter(test_loader))
    losses = rep_learner.train_step(batch, torch.optim.AdamW(model.parameters(), lr=1e-5))

    assert 'total_loss' in losses
    assert losses['total_loss'] > 0

    # ========== 5. Evaluation ==========
    metrics = rep_learner.evaluate(test_loader)
    required_metrics = {'accuracy', 'nmi', 'ari'}
    assert all(m in metrics for m in required_metrics)

    # Test standalone evaluation
    eval_metrics = evaluate_model(model=model, test_data=test_data)
    assert 0 <= eval_metrics['accuracy'] <= 1