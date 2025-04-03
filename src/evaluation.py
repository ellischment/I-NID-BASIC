from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score, adjusted_rand_score
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
import pandas as pd
import logging
import numpy as np
import os

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/imbanid.log', level=logging.INFO)

def evaluate_model(model=None, test_data=None):
    if model is None:
        model = BertModel.from_pretrained("models/finetuned/bert_finetuned")
    if test_data is None:
        test_data = pd.read_csv("data/processed/test.csv")

    logging.info("Starting model evaluation")

    # Load test data
    test_data = pd.read_csv("data/processed/test_data.csv")

    # Initialize tokenizer and fine-tuned model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("models/finetuned/bert_finetuned")

    # Device setup (GPU if available, otherwise CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Tokenize test data
    test_encodings = tokenizer(list(test_data['text']), truncation=True, padding=True, max_length=128)
    test_labels = torch.tensor(test_data['label_index'].tolist())

    # Create TensorDataset and DataLoader for test data
    test_dataset = TensorDataset(
        torch.tensor(test_encodings['input_ids']),
        torch.tensor(test_encodings['attention_mask']),
        test_labels
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Get predictions from the model
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Extract CLS token embeddings
            predicted_labels = np.argmax(logits, axis=1)  # Get predicted labels
            predictions.extend(predicted_labels)

    # Convert test labels to numpy array for evaluation
    true_labels = test_labels.cpu().numpy()

    # Compute evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    nmi = normalized_mutual_info_score(true_labels, predictions)
    ari = adjusted_rand_score(true_labels, predictions)

    # Log evaluation results
    logging.info(f"Evaluation completed: Accuracy={accuracy:.4f}, F1={f1:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}")
    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}")


if __name__ == "__main__":
    evaluate_model()