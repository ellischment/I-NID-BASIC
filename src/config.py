from dataclasses import dataclass, field
import os
import torch
from typing import List, Dict, Optional, Tuple
import ipdb
from IPython.core.debugger import set_trace
import logging
import pandas as pd

def debug_break(func):
    def wrapper(*args, **kwargs):
        set_trace()
        return func(*args, **kwargs)
    return wrapper

@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    colab: bool = os.environ.get('COLAB_GPU') is not None
    seed: int = 42
    num_workers: int = 4 if not os.environ.get('COLAB_GPU') else 2

    dataset_name: str = "clinc_oos"
    dataset_variant: str = "plus"
    num_classes: int = None
    known_ratio: float = 0.75
    labeled_ratio: float = 0.1
    gamma_values: List[int] = field(default_factory=lambda: [3, 5, 10])
    text_column: str = "text"
    label_column: str = "intent"

    model_name: str = "bert-base-uncased"
    tokenizer_name: str = "bert-base-uncased"
    max_seq_length: int = 128
    hidden_size: int = 768
    projection_dim: int = 256

    batch_size: int = 512 if not os.environ.get('COLAB_GPU') else 64
    num_epochs: int = 10
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1
    patience: int = 3

    temperature: float = 0.07
    rho: float = 0.7
    tau_g: float = 0.9
    lambda1: float = 0.05
    lambda2: float = 2.0
    omega: float = 0.5
    mlm_probability: float = 0.15
    moving_avg_param: float = 0.9

    use_amp: bool = True
    gradient_accumulation: int = 2
    eval_steps: int = 100

    debug_phase: Optional[str] = None
    breakpoints: Dict[str, bool] = field(default_factory=lambda: {
        'data': True,
        'pretrain': True,
        'rot': True,
        'contrastive': True,
        'eval': True
    })

    @property
    def pretrained_model_path(self) -> str:
        return os.path.join("models", "pretrained", f"{self.model_name}_pretrained")

    @property
    def finetuned_model_path(self) -> str:
        return os.path.join("models", "finetuned", f"{self.model_name}_finetuned")

    @property
    def data_raw_path(self) -> str:
        return os.path.join("data", "raw")

    @property
    def data_processed_path(self) -> str:
        return os.path.join("data", "processed", f"{self.dataset_name}_gamma{self.gamma_values[0]}")

    @property
    def log_dir(self) -> str:
        return os.path.join("logs", f"{self.dataset_name}_{self.model_name}")

    def __post_init__(self):
        if self.num_classes is None:
            try:
                train_df = pd.read_csv(os.path.join(self.data_processed_path, "train_labeled.csv"))
                self.num_classes = len(train_df['intent'].unique())
            except:
                self.num_classes = 150

        assert 0 < self.known_ratio <= 1, "known_ratio must be in (0, 1]"
        assert 0 < self.labeled_ratio <= 1, "labeled_ratio must be in (0, 1]"
        assert all(g > 1 for g in self.gamma_values), "gamma_values must be > 1"

        assert self.model_name in ["bert-base-uncased", "bert-large-uncased"], "Unsupported model"

        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"

        if self.colab:
            self.batch_size = min(self.batch_size, 64)
            self.num_workers = 2
            self.gradient_accumulation = max(self.gradient_accumulation, 2)

        os.makedirs(self.pretrained_model_path, exist_ok=True)
        os.makedirs(self.finetuned_model_path, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(self.log_dir, "training.log"),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    @property
    def label_encoder_path(self):
        return os.path.join(self.data_processed_path, "label_encoder.pkl")

    def update(self, params: Dict):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid config parameter: {key}")

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save(self, path: str):
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str):
        import yaml
        with open(path, 'r') as f:
            params = yaml.safe_load(f)
        return cls(**params)

config = Config()