from transformers import BertTokenizer, BertModel, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
import pandas as pd
import os
from typing import Optional, Tuple

os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/imbanid.log', level=logging.INFO)


def pretrain_model(
        labeled_data: pd.DataFrame,
        test_data: pd.DataFrame,
        tokenizer: Optional[BertTokenizer] = None,
        model: Optional[BertModel] = None,
        save_path: str = "models/pretrained/bert_pretrained"
) -> Tuple[BertModel, BertTokenizer]:
    """
    Pretrain the model using labeled and test data.

    Args:
        labeled_data: DataFrame with columns 'text' and 'label_index'
        test_data: DataFrame with columns 'text' and 'label_index'
        tokenizer: Optional pre-initialized tokenizer
        model: Optional pre-initialized model
        save_path: Path to save the pretrained model

    Returns:
        Tuple of (model, tokenizer)
    """
    logging.info("Starting model pretraining")

    # Initialize tokenizer and model if not provided
    tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained('bert-base-uncased')
    model = model if model is not None else BertModel.from_pretrained('bert-base-uncased')

    # Tokenize data
    train_encodings = tokenizer(
        list(labeled_data['text']),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    train_labels = torch.tensor(labeled_data['label_index'].tolist())

    val_encodings = tokenizer(
        list(test_data['text']),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    val_labels = torch.tensor(test_data['label_index'].tolist())

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

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Setup training
    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(3):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)

            # Use CLS token for classification
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            loss = criterion(cls_embeddings, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logging.info(f"Pretraining completed, model saved to {save_path}")

    return model, tokenizer


if __name__ == "__main__":
    # Example usage
    data = {
        'text': ['sample text 1', 'sample text 2'],
        'label_index': [0, 1]
    }
    df = pd.DataFrame(data)
    pretrain_model(df, df)
