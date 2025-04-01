import numpy as np
import ot
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
import pandas as pd
import logging
import os

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/imbanid.log', level=logging.INFO)


def generate_pseudo_labels():
    logging.info("Starting pseudo-label generation")

    # Load unlabeled data
    unlabeled_data = pd.read_csv("data/processed/unlabeled_data.csv")

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("models/pretrained/bert_pretrained")

    # Device setup (GPU if available, otherwise CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Tokenize unlabeled data
    unlabeled_encodings = tokenizer(list(unlabeled_data['text']), truncation=True, padding=True, max_length=128)
    unlabeled_dataset = TensorDataset(
        torch.tensor(unlabeled_encodings['input_ids']),
        torch.tensor(unlabeled_encodings['attention_mask'])
    )
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=False)

    # Get predictions from the model
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in unlabeled_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Extract CLS token embeddings
            predictions.extend(logits)

    P = np.array(predictions)  # Convert predictions to a numpy array
    logging.info(f"Obtained {P.shape[0]} predictions")

    # Relaxed Optimal Transport (ROT) for pseudo-label generation
    a = np.ones(P.shape[0]) / P.shape[0]  # Uniform distribution over samples
    b = np.ones(P.shape[1]) / P.shape[1]  # Uniform distribution over classes
    Q = ot.sinkhorn(a, b, -np.log(P + 1e-10), reg=0.1)  # Solve ROT problem with entropy regularization

    # Generate pseudo-labels by taking the argmax of the transport matrix Q
    pseudo_labels = np.argmax(Q, axis=1)

    max_probs = np.max(Q, axis=1)
    confidence_threshold = 0.5  # Порог уверенности
    pseudo_labels[max_probs < confidence_threshold] = -1  # Шумные примеры

    unlabeled_data['pseudo_labels'] = pseudo_labels
    unlabeled_data.to_csv("data/processed/unlabeled_data_with_pseudo_labels.csv", index=False)
    logging.info("Pseudo-labels generated and saved")


if __name__ == "__main__":
    generate_pseudo_labels()