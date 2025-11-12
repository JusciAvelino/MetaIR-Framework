"""
meta_ir_predictor.py
--------------------

META-IR Predictor for inference.

"""

import os
import io
import requests
import pandas as pd
import numpy as np
from joblib import load
import traceback

# Import the classes so pickle can find them when loading objects that reference them.
# This assumes meta_ir_trainer.py is in the same folder or on PYTHONPATH.
try:
    from meta_ir_trainer import IndependentModel, ModelFirst, StrategyFirst
    _trainer_imported = True
except Exception as e:
    # If import fails, we still continue — dill fallback may help, but best is to have trainer available.
    print("Warning: could not import classes from meta_ir_trainer.py:", e)
    _trainer_imported = False

# Try to import dill for fallback loading of pickles if needed
try:
    import dill
    _have_dill = True
except Exception:
    _have_dill = False

class META_IR_Predictor:
    def __init__(self, model_paths):
        self.model_paths = model_paths
        self.models = {}

        for mode, path in model_paths.items():
            if not os.path.exists(path):
                print(f"Model file for mode '{mode}' not found at path: {path}")
                continue

            # First try joblib.load (most common)
            try:
                self.models[mode] = load(path)
                print(f"✅ Model '{mode}' successfully loaded from {path} (joblib).")
                continue
            except Exception as e_joblib:
                print(f"joblib.load failed for '{mode}' ({path}): {e_joblib}")

            # If joblib failed, try dill (if available)
            if _have_dill:
                try:
                    with open(path, 'rb') as f:
                        self.models[mode] = dill.load(f)
                    print(f"Model '{mode}' successfully loaded from {path} (dill fallback).")
                    continue
                except Exception as e_dill:
                    print(f"dill.load also failed for '{mode}' ({path}): {e_dill}")

            # If we arrive here, loading failed
            print(f"Failed to load model '{mode}' from {path}. See errors above.")

        if not self.models:
            print("ERROR: No models were loaded. Check that the .pkl files exist and that meta_ir_trainer.py is available.")
        else:
            print("Loaded models:", list(self.models.keys()))

    def _check_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas.DataFrame containing meta-features.")
        if X.empty:
            raise ValueError("Input DataFrame is empty.")
        return X

    def predict(self, X, mode='independent'):
        X = self._check_input(X)

        if mode not in self.models:
            raise ValueError(f"Invalid mode. Choose one of: {list(self.models.keys())}")

        model = self.models[mode]
        print(f"Generating predictions using mode '{mode}'...")

        # Support both the custom objects defined in trainer and scikit-learn-like objects
        if mode == 'independent':
            # If the loaded object has a predict that returns a DataFrame, accept it
            pred = model.predict(X)
            if isinstance(pred, pd.DataFrame):
                pred_df = pred
            else:
                pred_df = pd.DataFrame(pred, columns=['model', 'strategy'])

        elif mode == 'model_first':
            pred = model.predict(X)
            if isinstance(pred, pd.DataFrame):
                pred_df = pred
            else:
                pred_df = pd.DataFrame(pred, columns=['strategy'])
                if hasattr(model, 'model_predictor'):
                    pred_df['model'] = model.model_predictor.predict(X)

        elif mode == 'strategy_first':
            pred = model.predict(X)
            if isinstance(pred, pd.DataFrame):
                pred_df = pred
            else:
                pred_df = pd.DataFrame(pred, columns=['model'])
                if hasattr(model, 'strategy_predictor'):
                    pred_df['strategy'] = model.strategy_predictor.predict(X)

        else:
            raise ValueError("Unknown mode.")

        print("Recommendations successfully generated.")
        return pred_df


# ===========================================================
# Example usage with GitHub dataset
# ===========================================================
if __name__ == "__main__":
    import io
    import requests

    # Load meta-features dataset locally instead of downloading from GitHub
    local_path = "/content/MetaIR-Framework/data/machineCPU_meta.csv"
    print(f"Loading dataset locally from: {local_path}")
    
    if not os.path.exists(local_path):
        raise FileNotFoundError(f" Local file not found: {local_path}\nDid you run the meta-feature extraction step?")
    else:
        X_new = pd.read_csv(local_path)
        print(f" Dataset loaded locally. Shape: {X_new.shape}")

    # Drop first column if it's an ID/name
    if X_new.shape[1] > 1:
        X_new = X_new.drop(X_new.columns[0], axis=1)

    # Paths to trained meta-models (adjust if your .pkl are in models/ or other folder)
    model_paths = {
        'independent': 'meta_independent.pkl',
        'model_first': 'meta_model_first.pkl',
        'strategy_first': 'meta_strategy_first.pkl'
    }

    # If the .pkl files are in a models/ directory use that path e.g. 'models/meta_independent.pkl'
    # model_paths = {k: os.path.join('models', v) for k,v in model_paths.items()}

    meta_pred = META_IR_Predictor(model_paths)

    # If no models loaded, try to instruct the user
    if not meta_pred.models:
        print("\\nNo models loaded. Try running the trainer first:")
        print("  !python meta_ir_trainer.py")
        raise SystemExit("Aborting: no models available for prediction.")

    output_dir = "/content"
    os.makedirs(output_dir, exist_ok=True)

    for mode in ['independent', 'model_first', 'strategy_first']:
        if mode not in meta_pred.models:
            print(f"Skipping mode '{mode}' (model not loaded).")
            continue
        try:
            preds = meta_pred.predict(X_new, mode=mode)
            out_path = os.path.join(output_dir, f"recommendation_{mode}.csv")
            preds.to_csv(out_path, index=False)
            print(f"Saved recommendations for {mode} -> {out_path}")
        except Exception as e:
            print(f"Failed to predict for mode '{mode}': {e}")
            traceback.print_exc()
