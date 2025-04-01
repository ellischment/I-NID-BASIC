from transformers import BertTokenizer, BertModel, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import logging
import os

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/imbanid.log', level=logging.INFO)


def train_model():
    logging.info("Starting training on pseudo-labeled data")

    # Load labeled, unlabeled (with pseudo-labels), and test data
    labeled_data = pd.read_csv("data/processed/labeled_data.csv")
    unlabeled_data = pd.read_csv("data/processed/unlabeled_data_with_pseudo_labels.csv")
    test_data = pd.read_csv("data/processed/test_data.csv")

    # Combine labeled and pseudo-labeled data for training
    all_training_data = pd.concat([labeled_data, unlabeled_data])

    # Initialize tokenizer and pre-trained model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("models/pretrained/bert_pretrained")

    # Tokenize the training data
    train_encodings = tokenizer(list(all_training_data['text']), truncation=True, padding=True, max_length=128)
    train_labels = torch.tensor(all_training_data['pseudo_labels'].tolist())

    # Create TensorDataset and DataLoader for training
    train_dataset = TensorDataset(
        torch.tensor(train_encodings['input_ids']),
        torch.tensor(train_encodings['attention_mask']),
        train_labels
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize optimizer and device (GPU if available, otherwise CPU)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Training loop
    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)

            # Compute loss using the CLS token embeddings
            loss = torch.nn.CrossEntropyLoss()(outputs.last_hidden_state[:, 0, :], labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Log average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

    # Save the fine-tuned model
    os.makedirs("models/finetuned", exist_ok=True)
    model.save_pretrained("models/finetuned/bert_finetuned")
    logging.info("Training completed, model saved")


if __name__ == "__main__":
    train_model()