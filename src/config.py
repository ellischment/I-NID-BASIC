# Пути к данным
DATA_PATHS = {
    "clinc": "data/raw/clinc_oos",
    "banking": "data/raw/banking77",
    "stackoverflow": "data/raw/stackoverflow",
    "processed": "data/processed",
    "models": "models"
}

# Гиперпараметры для создания длиннохвостого датасета
DATASET_CONFIG = {
    "gamma": [3, 5, 10],  # Коэффициенты дисбаланса
    "known_intent_ratio": 0.75,  # Доля известных интентов
    "labeled_ratio": 0.1  # Доля размеченных данных
}

# Гиперпараметры для модели
MODEL_CONFIG = {
    "pretrained_model": "bert-base-uncased",
    "batch_size": 32,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Гиперпараметры для генерации псевдометок
PSEUDO_LABELING_CONFIG = {
    "reg": 0.1,  # Регуляризация для Sinkhorn
    "confidence_threshold": 0.9  # Порог уверенности для псевдометок
}

# Гиперпараметры для оценки
EVALUATION_CONFIG = {
    "metrics": ["accuracy", "f1", "nmi", "ari"]
}