import joblib
from data_processing import load_data, preprocess_data
from sklearn.metrics import classification_report, roc_auc_score

def evaluate():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model_names = ["logistic_regression", "random_forest", "xgboost"]

    for name in model_names:
        model = joblib.load(f"../models/{name}.joblib")
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        print("\n===========", name.upper(), "===========")
        print(classification_report(y_test, preds))
        print("AUC:", roc_auc_score(y_test, proba))

if __name__ == "__main__":
    evaluate()
