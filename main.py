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
        prepare_dataset("clinc_oos", "plus", 150, [cfg.gamma_values[0]])  # Use first gamma for debug
        logging.info("Data preparation complete")

        # Load prepared data
        labeled_df = pd.read_csv("data/processed/clinc_oos_gamma3/train_labeled.csv")
        unlabeled_df = pd.read_csv("data/processed/clinc_oos_gamma3/train_unlabeled.csv")
        test_df = pd.read_csv("data/processed/clinc_oos_gamma3/test.csv")

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
            pseudo_df = plg.process(unlabeled_df)
            logging.info(f"Generated {len(pseudo_df)} pseudo-labels")

            # 4. Representation learning
            rl = RepresentationLearner(cfg)
            rl.train(labeled_df, pseudo_df)
            logging.info("Representation learning complete")

        # 5. Evaluation
        evaluate_model(model if not args.skip_train else None, test_df)
        logging.info("Evaluation complete")

    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()