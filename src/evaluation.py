from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score, adjusted_rand_score
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
import pandas as pd
import logging
import numpy as np
import os
from src.config import Config

config = Config()

def evaluate_model(model=None, test_data=None):
    if model is None:
        model = BertModel.from_pretrained(config.finetuned_model_path)
    if test_data is None:
        test_data = pd.read_csv(os.path.join(config.data_processed_path, "test.csv"))

    logging.info("Starting model evaluation")

    test_data = pd.read_csv(os.path.join(config.data_processed_path, "test_data.csv"))

    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name)
    model = BertModel.from_pretrained(config.finetuned_model_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    test_encodings = tokenizer(list(test_data['text']), truncation=True, padding=True, max_length=config.max_seq_length)
    test_labels = torch.tensor(test_data['label_index'].tolist())

    test_dataset = TensorDataset(
        torch.tensor(test_encodings['input_ids']),
        torch.tensor(test_encodings['attention_mask']),
        test_labels
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            predicted_labels = np.argmax(logits, axis=1)
            predictions.extend(predicted_labels)

    true_labels = test_labels.cpu().numpy()

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    nmi = normalized_mutual_info_score(true_labels, predictions)
    ari = adjusted_rand_score(true_labels, predictions)

    logging.info(f"Evaluation completed: Accuracy={accuracy:.4f}, F1={f1:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}")
    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}")

if __name__ == "__main__":
    evaluate_model()