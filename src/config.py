from dataclasses import dataclass
import os
import torch
from typing import List
import ipdb  # Interactive debugger
from IPython.core.debugger import set_trace

# Добавьте этот декоратор для отладки функций
def debug_break(func):
    def wrapper(*args, **kwargs):
        set_trace()  # Автоматическая остановка при вызове
        return func(*args, **kwargs)
    return wrapper

@dataclass
class Config:
    """Configuration class for ImbaNID model"""
    batch_size: int = 1024 if torch.cuda.get_device_capability()[0] >= 8 else 512
    mixed_precision: bool = True
    gradient_accumulation: int = 2

    # System settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    colab: bool = os.environ.get('COLAB_GPU') is not None

    # Data parameters
    known_ratio: float = 0.75
    labeled_ratio: float = 0.1
    gamma_values: List[int] = (3, 5, 10)

    # Model architecture
    model_name: str = "bert-base-uncased"
    tokenizer_name: str = "bert-base-uncased"
    max_seq_length: int = 128

    # Training hyperparameters
    batch_size: int = 512 if not colab else 64  # Smaller for Colab
    num_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # ImbaNID specific parameters
    temperature: float = 0.07
    rho: float = 0.7
    tau_g: float = 0.9
    lambda1: float = 0.05
    lambda2: float = 2  # 7 for balanced datasets
    omega: float = 0.5  # Weight for contrastive loss

    # Path configurations (auto-detected)
    @property
    def pretrained_model_path(self) -> str:
        return os.path.join("models", "pretrained", "bert_pretrained")

    @property
    def finetuned_model_path(self) -> str:
        return os.path.join("models", "finetuned", "bert_finetuned")

    @property
    def data_raw_path(self) -> str:
        return os.path.join("data", "raw")

    @property
    def data_processed_path(self) -> str:
        return os.path.join("data", "processed")

    def __post_init__(self):
        """Post-initialization validation"""
        assert 0 < self.known_ratio <= 1, "known_ratio must be in (0, 1]"
        assert 0 < self.labeled_ratio <= 1, "labeled_ratio must be in (0, 1]"
        assert all(g > 1 for g in self.gamma_values), "gamma_values must be > 1"
        assert self.batch_size > 0, "batch_size must be positive"

        # Auto-adjust for Colab environment
        if self.colab:
            self.batch_size = min(self.batch_size, 64)
            self.num_epochs = max(self.num_epochs, 1)

        self.debug_phase = None  # 'data|pretrain|rot|contrastive|eval'
        self.breakpoints = {
            'data': True,  # Остановка после загрузки данных
            'pretrain': True,  # Остановка после претрейна
            'rot': True,  # Остановка после ROT
            'contrastive': True,  # Остановка при контрастивном обучении
            'eval': True  # Остановка после оценки
        }


# Global configuration instance
config = Config()