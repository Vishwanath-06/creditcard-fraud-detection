from data_processing import load_data, preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os

def train_models():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=150),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9
        )
    }

    os.makedirs("../models", exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"../models/{name}.joblib")
        print(f"Saved model: {name}")

if __name__ == "__main__":
    train_models()
