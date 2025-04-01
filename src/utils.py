import logging
import torch
from transformers import BertTokenizer

def setup_logging(log_file="logs/imbanid.log"):
    """Настройка логирования."""
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

def load_tokenizer(model_name="bert-base-uncased"):
    """Загрузка токенизатора."""
    return BertTokenizer.from_pretrained(model_name)

def move_to_device(model, device):
    """Перемещение модели на устройство (CPU/GPU)."""
    return model.to(device)

def save_model(model, path):
    """Сохранение модели."""
    model.save_pretrained(path)
    logging.info(f"Модель сохранена в {path}")

def load_model(path):
    """Загрузка модели."""
    model = BertModel.from_pretrained(path)
    logging.info(f"Модель загружена из {path}")
    return model