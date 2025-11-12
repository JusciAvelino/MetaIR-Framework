"""
Train three meta-models:
  1. Independent Training
  2. Model-First Training
  3. Strategy-First Training

This version trains each model once on the entire meta-base and correctly saves them as .pkl files.

Author: Juscimara Avelino
"""

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder



class IndependentModel:
    """Independent model that predicts both model and strategy separately."""
    def __init__(self, model_pred, strat_pred):
        self.model_predictor = model_pred
        self.strategy_predictor = strat_pred

    def predict(self, X):
        model_pred = self.model_predictor.predict(X)
        strat_pred = self.strategy_predictor.predict(X)
        return pd.DataFrame({'model': model_pred, 'strategy': strat_pred})


class ModelFirst:
    """Model-First approach: predict model first, then strategy."""
    def __init__(self, model_pred, strat_pred, encoder):
        self.model_predictor = model_pred
        self.strategy_predictor = strat_pred
        self.encoder = encoder

    def predict(self, X):
        model_pred = self.model_predictor.predict(X)
        X_temp = X.copy()
        X_temp['y_model'] = model_pred
        encoded = self.encoder.transform(X_temp[['y_model']])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(['y_model']))
        X_encoded = pd.concat([X_temp.drop('y_model', axis=1), encoded_df], axis=1)
        strat_pred = self.strategy_predictor.predict(X_encoded)
        return pd.DataFrame({'model': model_pred, 'strategy': strat_pred})


class StrategyFirst:
    """Strategy-First approach: predict strategy first, then model."""
    def __init__(self, strat_pred, model_pred, encoder):
        self.strategy_predictor = strat_pred
        self.model_predictor = model_pred
        self.encoder = encoder

    def predict(self, X):
        strat_pred = self.strategy_predictor.predict(X)
        X_temp = X.copy()
        X_temp['y_strategy'] = strat_pred
        encoded = self.encoder.transform(X_temp[['y_strategy']])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(['y_strategy']))
        X_encoded = pd.concat([X_temp.drop('y_strategy', axis=1), encoded_df], axis=1)
        model_pred = self.model_predictor.predict(X_encoded)
        return pd.DataFrame({'model': model_pred, 'strategy': strat_pred})


# ============================================================
# Training functions
# ============================================================

def independent_training(meta_base):
    print("Training Independent Model...")

    X = meta_base.select_dtypes(exclude=['object']).copy()
    y_model = meta_base['model']
    y_strategy = meta_base['strategy']

    model_classifier = RandomForestClassifier(random_state=42)
    strategy_classifier = RandomForestClassifier(random_state=42)

    model_classifier.fit(X, y_model)
    strategy_classifier.fit(X, y_strategy)

    print("Independent model trained.")
    return IndependentModel(model_classifier, strategy_classifier)


def model_first_training(meta_base):
    print("Training Model-First Meta-Model...")

    X = meta_base.select_dtypes(exclude=['object']).copy()
    y_model = meta_base['model']
    y_strategy = meta_base['strategy']

    # Step 1: Train model predictor
    model_classifier = RandomForestClassifier(random_state=42)
    model_classifier.fit(X, y_model)
    model_pred = model_classifier.predict(X)

    # Step 2: Encode model predictions and train strategy predictor
    X_temp = X.copy()
    X_temp['y_model'] = model_pred
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(X_temp[['y_model']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['y_model']))
    X_encoded = pd.concat([X_temp.drop('y_model', axis=1), encoded_df], axis=1)

    strategy_classifier = RandomForestClassifier(random_state=42)
    strategy_classifier.fit(X_encoded, y_strategy)

    print("Model-First training completed.")
    return ModelFirst(model_classifier, strategy_classifier, encoder)


def strategy_first_training(meta_base):
    print("Training Strategy-First Meta-Model...")

    X = meta_base.select_dtypes(exclude=['object']).copy()
    y_model = meta_base['model']
    y_strategy = meta_base['strategy']

    # Step 1: Train strategy predictor
    strategy_classifier = RandomForestClassifier(random_state=42)
    strategy_classifier.fit(X, y_strategy)
    strat_pred = strategy_classifier.predict(X)

    # Step 2: Encode predicted strategy and train model predictor
    X_temp = X.copy()
    X_temp['y_strategy'] = strat_pred
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(X_temp[['y_strategy']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['y_strategy']))
    X_encoded = pd.concat([X_temp.drop('y_strategy', axis=1), encoded_df], axis=1)

    model_classifier = RandomForestClassifier(random_state=42)
    model_classifier.fit(X_encoded, y_model)

    print("Strategy-First training completed.")
    return StrategyFirst(strategy_classifier, model_classifier, encoder)


# ============================================================
# Main script
# ============================================================

if __name__ == "__main__":
    import requests
    import io

    github_csv_url = "https://raw.githubusercontent.com/JusciAvelino/MetaIR-Framework/main/data/MetaBase.csv"

    print(f"Downloading meta-base from GitHub: {github_csv_url}")

    try:
        response = requests.get(github_csv_url)
        response.raise_for_status()  # garante que deu certo
        meta_base = pd.read_csv(io.StringIO(response.text))
        print(f"Meta-base loaded from GitHub. Shape: {meta_base.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to load MetaBase.csv from GitHub: {e}")

    # Verifica colunas obrigatórias
    if not {'model', 'strategy'}.issubset(meta_base.columns):
        raise ValueError("The meta-base must contain 'model' and 'strategy' columns.")

    # Treina os meta-modelos
    model_independent = independent_training(meta_base)
    model_model_first = model_first_training(meta_base)
    model_strategy_first = strategy_first_training(meta_base)

    # Salva todos os modelos
    dump(model_independent, "meta_independent.pkl")
    dump(model_model_first, "meta_model_first.pkl")
    dump(model_strategy_first, "meta_strategy_first.pkl")

    print("\nAll meta-models successfully trained and saved:")
    print("   - meta_independent.pkl")
    print("   - meta_model_first.pkl")
    print("   - meta_strategy_first.pkl")

def train_all(meta_base_path="/content/MetaIR-Framework/data/MetaBase.csv", save_dir="."):
    meta_base = pd.read_csv(meta_base_path)
    model_independent = independent_training(meta_base)
    model_model_first = model_first_training(meta_base)
    model_strategy_first = strategy_first_training(meta_base)

    dump(model_independent, os.path.join(save_dir, "meta_independent.pkl"))
    dump(model_model_first, os.path.join(save_dir, "meta_model_first.pkl"))
    dump(model_strategy_first, os.path.join(save_dir, "meta_strategy_first.pkl"))
    print("All meta-models trained and saved.")
