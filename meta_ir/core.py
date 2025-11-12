import os
import pandas as pd
from meta_ir.trainer import train_all
from meta_ir.predictor import run_predictions
from meta_ir.executor import execute_all


class MetaIR:
    """
    MetaIR Framework
    ----------------
    Unified API for training meta-models, generating recommendations,
    and evaluating them on a given dataset.

    Methods:
    --------
    - fit(meta_base_path): trains all meta-models (Independent, Model-First, Strategy-First)
    - predict(meta_features_path): generates recommendations based on meta-features
    - evaluate(dataset_path): executes and evaluates recommendations on real data
    """

    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        self.trained = False
        self.recommended = False
        self.results_path = os.path.join(base_dir, "evaluation_results_all.csv")

    # -------------------------------
    # STEP 1: train meta-models
    # -------------------------------
    def fit(self, meta_base_path):
        print("Fitting MetaIR meta-models...")
        train_all(meta_base_path, save_dir=self.base_dir)
        self.trained = True
        print("All meta-models trained successfully.")
        return self

    # -------------------------------
    # STEP 2: generate recommendations
    # -------------------------------
    def predict(self, meta_features_path):
        if not self.trained:
            print("⚠️ Warning: Meta-models not trained. Calling fit() first is recommended.")
        print("Generating recommendations...")
        run_predictions(meta_features_path, model_dir=self.base_dir)
        self.recommended = True
        print("Recommendations generated successfully.")
        return self

    # -------------------------------
    # STEP 3: execute & evaluate recommendations
    # -------------------------------
    def evaluate(self, dataset_path):
        if not self.recommended:
            print("⚠️ Warning: No recommendations found. Calling predict() first is recommended.")
        print("Executing and evaluating recommendations...")
        execute_all(dataset_path, self.base_dir, self.results_path)
        print(f"Evaluation complete. Results saved to: {self.results_path}")
        return pd.read_csv(self.results_path)
