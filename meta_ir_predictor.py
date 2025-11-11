meta_ir_predictor.py
--------------------

Simplified META-IR Predictor class for meta-learning-based recommendation.

This version is designed for *inference only* — assuming the models have
already been trained (using Independent Training, Model-First, or
Strategy-First approaches) and saved to disk.

Author: Juscimara Avelino
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from joblib import load
import os


class META_IR_Predictor:
    """
    META_IR_Predictor
    -----------------
    A lightweight class for recommending models and strategies in
    imbalanced regression tasks based on meta-features.

    This version assumes that all meta-models are pre-trained and stored
    as `.pkl` files.

    Parameters
    ----------
    model_paths : dict
        Dictionary mapping each training mode to the path of its saved model.
        Example:
        {
            'independent': 'meta_independent.pkl',
            'model_first': 'meta_model_first.pkl',
            'strategy_first': 'meta_strategy_first.pkl'
        }

    Attributes
    ----------
    models : dict
        Loaded sklearn-compatible models for each mode.

    Methods
    -------
    predict(X, mode='independent')
        Generate recommendations using one of the available trained models.
    """

    def __init__(self, model_paths):
        self.model_paths = model_paths
        self.models = {}

        # Load all available trained models
        for mode, path in model_paths.items():
            try:
                self.models[mode] = load(path)
                print(f"Model '{mode}' successfully loaded from {path}.")
            except Exception as e:
                print(f"Failed to load model '{mode}' from {path}: {e}")

    def _check_input(self, X):
        """Validate that the input is a non-empty pandas DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas.DataFrame containing meta-features.")
        if X.empty:
            raise ValueError("Input DataFrame is empty.")
        return X

    def predict(self, X, mode='independent'):
        """
        Generate recommendations for a given dataset based on pre-trained models.

        Parameters
        ----------
        X : pandas.DataFrame
            A DataFrame containing meta-features extracted from a dataset.
        mode : str
            One of the following prediction modes:
                - 'independent'
                - 'model_first'
                - 'strategy_first'

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the recommended 'model' and/or 'strategy'.
        """
        X = self._check_input(X)

        if mode not in self.models:
            raise ValueError(f"Invalid mode. Choose one of: {list(self.models.keys())}")

        model = self.models[mode]
        print(f"Generating predictions using mode '{mode}'...")

        # Depending on how the models were saved, adjust outputs accordingly
        if mode == 'independent':
            pred = model.predict(X)
            pred_df = pd.DataFrame(pred, columns=['model', 'strategy'])

        elif mode == 'model_first':
            pred = model.predict(X)
            pred_df = pd.DataFrame(pred, columns=['strategy'])

            # If an additional model predictor exists (optional)
            if hasattr(model, 'model_predictor'):
                pred_df['model'] = model.model_predictor.predict(X)

        elif mode == 'strategy_first':
            pred = model.predict(X)
            pred_df = pd.DataFrame(pred, columns=['model'])

            # If an additional strategy predictor exists (optional)
            if hasattr(model, 'strategy_predictor'):
                pred_df['strategy'] = model.strategy_predictor.predict(X)

        print("Recommendations successfully generated.")
        return pred_df


# ===========================================================
# Example usage with CSV output
# ===========================================================
if __name__ == "__main__":
    # Paths to trained meta-models
    model_paths = {
        'independent': 'meta_independent.pkl',
        'model_first': 'meta_model_first.pkl',
        'strategy_first': 'meta_strategy_first.pkl'
    }

    # Initialize the predictor
    meta_pred = META_IR_Predictor(model_paths)

    # Load dataset to extract meta-features
    X_new = pd.read_csv("/content/machineCPU.csv")

    # Drop the first column (ID or dataset name)
    X_new = X_new.drop(X_new.columns[0], axis=1)

    # Folder to save recommendations
    output_dir = "/content/"
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save recommendations for all modes
    for mode in ['independent', 'model_first', 'strategy_first']:
        predictions = meta_pred.predict(X_new, mode=mode)
        print(f"\nPredicted recommendation ({mode}):")
        print(predictions)

        # Save to CSV
        output_path = os.path.join(output_dir, f"recommendation_{mode}.csv")
        predictions.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
