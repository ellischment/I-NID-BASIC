import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from config import config

def train_step(model, batch, device, scaler=None):
    inputs, masks, labels = batch
    inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

    with autocast(enabled=scaler is not None):
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss / config.gradient_accumulation  # Используем config

    if scaler:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return loss.item()


def setup_training(model):
    scaler = GradScaler() if config.mixed_precision else None  # Используем config
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)  # Используем config

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model, optimizer, scaler
