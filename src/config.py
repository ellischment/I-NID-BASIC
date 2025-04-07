from dataclasses import dataclass, field
import os
import torch
from typing import List, Dict, Optional, Tuple, Any
import logging
import pandas as pd
from pathlib import Path


@dataclass
class Config:
    # Device and environment settings
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    colab: bool = field(default_factory=lambda: os.environ.get('COLAB_GPU') is not None)
    seed: int = 42
    num_workers: int = field(default_factory=lambda: 2 if os.environ.get('COLAB_GPU') else 4)

    # Dataset configuration
    dataset_name: str = "clinc_oos"
    dataset_variant: str = "plus"
    num_classes: int = None
    known_ratio: float = 0.75
    labeled_ratio: float = 0.1
    gamma_values: List[int] = field(default_factory=lambda: [3, 5, 10])
    text_column: str = "text"
    label_column: str = "intent"

    # Model architecture
    model_name: str = "bert-base-uncased"
    tokenizer_name: str = "bert-base-uncased"
    max_seq_length: int = 128
    hidden_size: int = 768
    projection_dim: int = 256

    # Training hyperparameters
    batch_size: int = field(default_factory=lambda: 64 if os.environ.get('COLAB_GPU') else 512)
    num_epochs: int = 10
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1
    patience: int = 3

    # Loss function parameters
    temperature: float = 0.07
    rho: float = 0.7
    tau_g: float = 0.9
    lambda1: float = 0.05
    lambda2: float = 2.0
    omega: float = 0.5
    mlm_probability: float = 0.15
    moving_avg_param: float = 0.9

    # Training setup
    use_amp: bool = True
    gradient_accumulation: int = field(default_factory=lambda: 2 if os.environ.get('COLAB_GPU') else 1)
    eval_steps: int = 100

    # Debugging
    debug_phase: Optional[str] = None
    breakpoints: Dict[str, bool] = field(default_factory=lambda: {
        'data': False,
        'pretrain': False,
        'rot': False,
        'contrastive': False,
        'eval': False
    })

    def __post_init__(self):
        # Initialize paths and directories
        self._init_paths()
        self._init_logging()

        # Validate configuration
        self._validate_config()

        # Set Colab-specific defaults
        if self.colab:
            self._set_colab_defaults()

        # Initialize num_classes if not provided
        if self.num_classes is None:
            self._init_num_classes()

    def _init_paths(self):
        """Initialize all required directories."""
        paths = [
            self.pretrained_model_path,
            self.finetuned_model_path,
            self.data_raw_path,
            self.data_processed_path,
            self.log_dir
        ]

        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)

    def _init_logging(self):
        """Initialize logging configuration."""
        logging.basicConfig(
            filename=os.path.join(self.log_dir, "training.log"),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info(f"Initialized config with device: {self.device}")

    def _validate_config(self):
        """Validate configuration parameters."""
        assert 0 < self.known_ratio <= 1, "known_ratio must be in (0, 1]"
        assert 0 < self.labeled_ratio <= 1, "labeled_ratio must be in (0, 1]"
        assert all(g > 1 for g in self.gamma_values), "gamma_values must be > 1"
        assert self.model_name in ["bert-base-uncased", "bert-large-uncased"], "Unsupported model"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"

    def _set_colab_defaults(self):
        """Set Colab-specific defaults."""
        self.batch_size = min(self.batch_size, 64)
        self.num_workers = 2
        self.gradient_accumulation = max(self.gradient_accumulation, 2)

    def _init_num_classes(self):
        """Initialize num_classes from data if not provided."""
        try:
            train_path = os.path.join(self.data_processed_path, "train_labeled.csv")
            if os.path.exists(train_path):
                train_df = pd.read_csv(train_path)
                self.num_classes = len(train_df[self.label_column].unique())
                logging.info(f"Auto-detected num_classes: {self.num_classes}")
            else:
                self.num_classes = 150
                logging.warning("Using default num_classes=150 as train_labeled.csv not found")
        except Exception as e:
            self.num_classes = 150
            logging.error(f"Error detecting num_classes: {str(e)}, using default 150")

    @property
    def pretrained_model_path(self) -> str:
        """Path to save/load pretrained models."""
        return str(Path("models") / "pretrained" / self.model_name)

    @property
    def finetuned_model_path(self) -> str:
        """Path to save/load fine-tuned models."""
        return str(Path("models") / "finetuned" / self.model_name)

    @property
    def data_raw_path(self) -> str:
        """Path to raw data files."""
        return str(Path("data") / "raw")

    @property
    def data_processed_path(self) -> str:
        """Path to processed data files."""
        return str(Path("data") / "processed" / f"{self.dataset_name}_gamma{min(self.gamma_values)}")

    @property
    def log_dir(self) -> str:
        """Directory for log files."""
        return str(Path("logs") / f"{self.dataset_name}_{self.model_name}")

    @property
    def label_encoder_path(self) -> str:
        """Path to label encoder file."""
        return str(Path(self.data_processed_path) / "label_encoder.pkl")

    def update(self, params: Dict[str, Any]) -> None:
        """Update configuration parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid config parameter: {key}")
        self.__post_init__()  # Re-validate after update

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        import yaml
        with open(path, 'r') as f:
            params = yaml.safe_load(f)
        return cls(**params)


config = Config()