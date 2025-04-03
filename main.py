import os
import sys
import logging
from datetime import datetime
import pandas as pd
import torch
from torch.utils.data import DataLoader  # Добавлен импорт DataLoader
from src.config import Config
from src.data_preparation import prepare_dataset
from src.model_pretraining import pretrain_model
from src.pseudo_labeling import PseudoLabelGenerator
from src.representation_learning import RepresentationLearner
from transformers import BertModel, BertTokenizer

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/imbanid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    # Initialize configuration
    cfg = Config()

    # Enable TF32 for A100 GPU
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 1. Data Preparation
    logging.info("=== Preparing Datasets ===")
    prepare_dataset("clinc_oos", "plus", 150, cfg.gamma_values)

    # Load processed data
    data_dir = f"data/processed/clinc_oos_gamma{cfg.gamma_values[0]}"
    labeled_df = pd.read_csv(f"{data_dir}/train_labeled.csv")
    unlabeled_df = pd.read_csv(f"{data_dir}/train_unlabeled.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")

    # 2. Model Pretraining
    logging.info("=== Pretraining Model ===")
    model, tokenizer = pretrain_model(labeled_df, test_df, cfg)

    # 3. Pseudo-Label Generation with ROT
    logging.info("=== Generating Pseudo-Labels ===")
    plg = PseudoLabelGenerator(model=model, tokenizer=tokenizer,
                               lambda1=cfg.lambda1, lambda2=cfg.lambda2)

    # Estimate number of classes
    n_classes = len(labeled_df['label'].unique()) + 10  # Simple heuristic

    # Generate pseudo-labels
    embeddings = plg.get_embeddings(unlabeled_df['text'].tolist())
    unlabeled_df['pseudo_label'] = plg.generate_pseudo_labels(embeddings, n_classes)
    unlabeled_df.to_csv(f"{data_dir}/train_unlabeled_pseudo.csv", index=False)

    # 4. Representation Learning
    logging.info("=== Representation Learning ===")
    representation_learner = RepresentationLearner(model, tokenizer, cfg)

    # Train with both labeled and pseudo-labeled data
    representation_learner.train(
        labeled_data=labeled_df,
        unlabeled_data=unlabeled_df,
        num_epochs=cfg.num_epochs
    )

    # Save fine-tuned model
    model.save_pretrained(cfg.finetuned_model_path)
    tokenizer.save_pretrained(cfg.finetuned_model_path)

    # 5. Evaluation
    logging.info("=== Evaluating Model ===")
    test_dataset = representation_learner._prepare_dataset(pd.DataFrame(), test_df)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)
    metrics = representation_learner.evaluate(test_loader)

    logging.info(f"Final Metrics - ACC: {metrics['accuracy']:.4f}, "
                 f"NMI: {metrics['nmi']:.4f}, ARI: {metrics['ari']:.4f}")
    print(f"\n=== Final Results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"NMI: {metrics['nmi']:.4f}")
    print(f"ARI: {metrics['ari']:.4f}")


if __name__ == "__main__":
    print("=== Environment Check ===")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    if 'torch' in sys.modules:
        print(f"PyTorch version: {torch.__version__}")
        print(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.get_device_name(0)}")

    main()