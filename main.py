import os
import sys
import ipdb  # Interactive debugger
from pathlib import Path
import argparse
import logging
from datetime import datetime
import pandas as pd
from src.config import Config
from src.data_preparation import prepare_dataset
from src.model_pretraining import pretrain_model
from src.pseudo_labeling import PseudoLabelGenerator
from src.representation_learning import RepresentationLearner
from src.evaluation import evaluate_model

# Automatic path configuration
BASE_DIR = Path(__file__).parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Initialize configuration with hyperparameters
cfg = Config()

# Setup logging to file
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/imbanid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def debug_breakpoint(name, local_vars):
    """Universal debugging breakpoint with variable inspection"""
    print(f"\n=== DEBUG BREAKPOINT: {name} ===")
    print("Available variables:")
    for var_name, var_val in local_vars.items():
        print(f"- {var_name} ({type(var_val)})")
    ipdb.set_trace()


def main():
    # Configure command-line arguments
    parser = argparse.ArgumentParser(description="ImbaNID Training Pipeline")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode (reduces data and epochs)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training phase (evaluation only)")
    parser.add_argument("--breakpoints", type=str, default="all",
                        help="Comma-separated breakpoints (data,pretrain,rot,contrastive,eval)")
    args = parser.parse_args()

    try:
        logging.info("=== Starting ImbaNID Training ===")

        # Parse which breakpoints to activate
        breakpoints = args.breakpoints.split(",") if args.breakpoints != "all" else [
            "data", "pretrain", "rot", "contrastive", "eval"
        ]

        # ============= 1. DATA PREPARATION =============
        if args.debug:
            cfg.num_epochs = 1  # Reduced epochs for debugging
            gamma_to_use = [cfg.gamma_values[0]]  # Use first gamma value
            sample_frac = 0.01  # Use 1% of data
        else:
            gamma_to_use = cfg.gamma_values  # Use all gamma values
            sample_frac = 1.0  # Use full dataset

        # Prepare dataset with specified imbalance ratio
        prepare_dataset("clinc_oos", "plus", 150, gamma_to_use)
        logging.info("Data preparation complete")

        # Load processed datasets
        data_dir = f"data/processed/clinc_oos_gamma{gamma_to_use[0]}"
        labeled_df = pd.read_csv(f"{data_dir}/train_labeled.csv")
        unlabeled_df = pd.read_csv(f"{data_dir}/train_unlabeled.csv")
        test_df = pd.read_csv(f"{data_dir}/test.csv")

        if args.debug:  # Subsample data if in debug mode
            labeled_df = labeled_df.sample(frac=sample_frac, random_state=42)
            unlabeled_df = unlabeled_df.sample(frac=sample_frac, random_state=42)
            test_df = test_df.sample(frac=sample_frac, random_state=42)

        if "data" in breakpoints:
            debug_breakpoint("DATA_LOADED", locals())

        # ============= 2. MODEL PRETRAINING =============
        if not args.skip_train:
            # Train initial model on labeled data
            model, tokenizer = pretrain_model(
                labeled_df,
                test_df,
                epochs=cfg.num_epochs
            )
            logging.info("Pretraining complete")

            if "pretrain" in breakpoints:
                debug_breakpoint("MODEL_PRETRAINED", {
                    "model": model,
                    "tokenizer": tokenizer,
                    "labeled_df": labeled_df
                })

            # ========= 3. PSEUDO-LABEL GENERATION =========
            plg = PseudoLabelGenerator(model=model, tokenizer=tokenizer)
            pseudo_df = plg.process(
                unlabeled_df,
                output_path=f"{data_dir}/pseudo_labeled.csv" if not args.debug else None
            )
            logging.info(f"Generated {len(pseudo_df)} pseudo-labels")

            if "rot" in breakpoints:
                debug_breakpoint("PSEUDO_LABELS_GENERATED", {
                    "pseudo_df": pseudo_df,
                    "plg": plg
                })

            # ======= 4. CONTRASTIVE REPRESENTATION LEARNING =======
            rl = RepresentationLearner(cfg)
            rl.train(labeled_df, pseudo_df, debug=args.debug)
            logging.info("Representation learning complete")

            if "contrastive" in breakpoints:
                debug_breakpoint("CONTRASTIVE_LEARNING_DONE", {
                    "rl": rl,
                    "embeddings": rl.get_embeddings(labeled_df['text'].tolist())
                })

        # ============= 5. EVALUATION =============
        evaluate_model(model if not args.skip_train else None, test_df)
        logging.info("Evaluation complete")

        if "eval" in breakpoints:
            debug_breakpoint("EVALUATION_COMPLETE", {
                "test_df": test_df,
                "model": model if not args.skip_train else None
            })

    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
        print(f"Critical error: {str(e)}")
        ipdb.post_mortem()  # Enter debug mode on error
        raise


if __name__ == "__main__":
    # Environment verification
    print("=== Environment Check ===")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    # Check GPU availability
    if 'torch' in sys.modules:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch not imported - GPU check skipped")

    main()