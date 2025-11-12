"""
meta_ir_predictor.py
--------------------

META-IR Predictor for meta-learning-based recommendation in
information retrieval regression tasks.

This version:
- Imports custom model classes from meta_ir_trainer.py (fix for pickle loading)
- Uses absolute paths (fix for empty DataFrame)
- Validates dataset before prediction
- Compatible with Colab environment

Author: Juscimara Avelino
"""

import os
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import OneHotEncoder

# Import model classes from trainer to enable loading of custom pickle objects
try:
    from meta_ir_trainer import IndependentModel, ModelFirst, StrategyFirst
    print("Imported model classes from meta_ir_trainer.py")
except Exception as e:
    print(f"Warning: could not import model classes ({e}). You might need to run meta_ir_trainer.py first.")


class META_IR_Predictor:
    def __init__(self, model_paths):
        self.model_paths = model_paths
        self.models = {}

        for mode, path in model_paths.items():
            try:
                self.models[mode] = load(path)
                print(f"Model '{mode}' successfully loaded from {path}")
            except Exception as e:
                print(f"Failed to load model '{mode}' from {path}: {e}")

    def _check_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas.DataFrame.")
        if X.empty:
            raise ValueError("Input DataFrame is empty. Check if the CSV path is correct.")
        return X

    def predict(self, X, mode='independent'):
        X = self._check_input(X)

        if mode not in self.models:
            raise ValueError(f"Invalid mode. Choose one of: {list(self.models.keys())}")

        model = self.models[mode]
        print(f"Generating predictions using mode '{mode}'...")

        if mode == 'independent':
            pred = model.predict(X)
            pred_df = pd.DataFrame(pred, columns=['model', 'strategy'])

        elif mode == 'model_first':
            pred = model.predict(X)
            pred_df = pd.DataFrame(pred, columns=['strategy'])
            if hasattr(model, 'model_predictor'):
                pred_df['model'] = model.model_predictor.predict(X)

        elif mode == 'strategy_first':
            pred = model.predict(X)
            pred_df = pd.DataFrame(pred, columns=['model'])
            if hasattr(model, 'strategy_predictor'):
                pred_df['strategy'] = model.strategy_predictor.predict(X)

        print("Recommendations successfully generated.")
        return pred_df


if __name__ == "__main__":
    base_dir = "/content/MetaIR-Framework"
    data_path = os.path.join(base_dir, "data", "machineCPU_meta.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Meta-features CSV not found at {data_path}. Make sure extraction step ran.")

    print(f"Loading meta-features from: {data_path}")
    X_new = pd.read_csv(data_path)

    if X_new.shape[1] > 1:
        X_new = X_new.drop(X_new.columns[0], axis=1)

    print(f"Meta-features loaded. Shape: {X_new.shape}")

    model_paths = {
        'independent': os.path.join(base_dir, "meta_independent.pkl"),
        'model_first': os.path.join(base_dir, "meta_model_first.pkl"),
        'strategy_first': os.path.join(base_dir, "meta_strategy_first.pkl"),
    }

    meta_pred = META_IR_Predictor(model_paths)

    if not meta_pred.models:
        raise RuntimeError("No models were loaded. Run meta_ir_trainer.py first to train and save them.")

    output_dir = base_dir
    os.makedirs(output_dir, exist_ok=True)

    for mode in ['independent', 'model_first', 'strategy_first']:
        if mode not in meta_pred.models:
            print(f"Skipping '{mode}' (model not available)")
            continue
        try:
            preds = meta_pred.predict(X_new, mode=mode)
            out_path = os.path.join(output_dir, f"recommendation_{mode}.csv")
            preds.to_csv(out_path, index=False)
            print(f"Saved recommendations for {mode} to: {out_path}")
        except Exception as e:
            print(f"Failed to predict using mode '{mode}': {e}")
