from transformers import BertTokenizer, BertModel, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
import pandas as pd
import os

# Создаем директорию для логов, если она не существует
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/imbanid.log', level=logging.INFO)


def pretrain_model(labeled_data, test_data):
    """
    Pretrain the model using labeled and test data.
    :param labeled_data: DataFrame with columns 'text' and 'label_index'.
    :param test_data: DataFrame with columns 'text' and 'label_index'.
    """
    logging.info("Starting model pretraining")
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize the data
    train_encodings = tokenizer(list(labeled_data['text']), truncation=True, padding=True, max_length=128,
                                return_tensors='pt')
    train_labels = torch.tensor(labeled_data['label_index'].tolist())

    # Отладочный вывод
    print("Train encodings (input_ids):", train_encodings['input_ids'])
    print("Train labels:", train_labels)

    val_encodings = tokenizer(list(test_data['text']), truncation=True, padding=True, max_length=128,
                              return_tensors='pt')
    val_labels = torch.tensor(test_data['label_index'].tolist())

    # Проверяем, что размеры совпадают
    assert len(train_encodings['input_ids']) == len(train_labels), "Size mismatch between input_ids and labels"
    assert len(train_encodings['attention_mask']) == len(
        train_labels), "Size mismatch between attention_mask and labels"

    # Create datasets
    train_dataset = TensorDataset(
        train_encodings['input_ids'],
        train_encodings['attention_mask'],
        train_labels
    )
    val_dataset = TensorDataset(
        val_encodings['input_ids'],
        val_encodings['attention_mask'],
        val_labels
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize optimizer and device
    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Training loop
    for epoch in range(3):  # You can increase the number of epochs
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            # Use the first token's hidden state for classification
            loss = torch.nn.CrossEntropyLoss()(outputs.last_hidden_state[:, 0, :], labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

    # Save the model
    os.makedirs("models/pretrained", exist_ok=True)
    model.save_pretrained("models/pretrained/bert_pretrained")
    logging.info("Pretraining completed, model saved")


if __name__ == "__main__":
    pretrain_model()