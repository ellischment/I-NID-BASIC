import os
import sys
from pathlib import Path

# Автоматическая настройка путей
BASE_DIR = Path(__file__).parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import argparse
import logging
from datetime import datetime
from src.config import Config
from src.data_preparation import prepare_dataset
from src.model_pretraining import pretrain_model
from src.pseudo_labeling import PseudoLabelGenerator
from src.representation_learning import RepresentationLearner
from src.evaluation import evaluate_model
import pandas as pd


# Initialize configuration
cfg = Config()

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/imbanid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Debug mode (1 epoch, 1% data)")
    parser.add_argument("--skip_train", action="store_true", help="Skip training phase")
    args = parser.parse_args()

    try:
        logging.info("=== Starting ImbaNID ===")

        # 1. Data preparation
        if args.debug:
            cfg.num_epochs = 1  # Override for debug mode
            gamma_to_use = [cfg.gamma_values[0]]
            sample_frac = 0.01
        else:
            gamma_to_use = cfg.gamma_values
            sample_frac = 1.0

        prepare_dataset("clinc_oos", "plus", 150, gamma_to_use)
        logging.info("Data preparation complete")

        # Load prepared data
        data_dir = f"data/processed/clinc_oos_gamma{gamma_to_use[0]}"
        labeled_df = pd.read_csv(f"{data_dir}/train_labeled.csv")
        unlabeled_df = pd.read_csv(f"{data_dir}/train_unlabeled.csv")
        test_df = pd.read_csv(f"{data_dir}/test.csv")

        if args.debug:
            labeled_df = labeled_df.sample(frac=sample_frac, random_state=42)
            unlabeled_df = unlabeled_df.sample(frac=sample_frac, random_state=42)
            test_df = test_df.sample(frac=sample_frac, random_state=42)

        # 2. Model pretraining
        if not args.skip_train:
            model, tokenizer = pretrain_model(
                labeled_df,
                test_df,
                epochs=cfg.num_epochs
            )
            logging.info("Pretraining complete")

            # 3. Pseudo-label generation
            plg = PseudoLabelGenerator(model=model, tokenizer=tokenizer)
            pseudo_df = plg.process(
                unlabeled_df,
                output_path=f"{data_dir}/pseudo_labeled.csv" if not args.debug else None
            )
            logging.info(f"Generated {len(pseudo_df)} pseudo-labels")

            # 4. Representation learning
            rl = RepresentationLearner(cfg)
            rl.train(labeled_df, pseudo_df, debug=args.debug)
            logging.info("Representation learning complete")

        # 5. Evaluation
        evaluate_model(model if not args.skip_train else None, test_df)
        logging.info("Evaluation complete")

    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()