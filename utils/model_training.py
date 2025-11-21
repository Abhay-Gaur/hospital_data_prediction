# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import mean_absolute_error
# import joblib, os

# def train_and_save_model(X_train, X_test, y_train, y_test, model_name="Recovery_Status", model_path="models/trained_model.pkl"):
#     models = {
#         "LinearRegression": LinearRegression(),
#         "DecisionTree": DecisionTreeRegressor(),
#         "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
#         "GradientBoosting": GradientBoostingRegressor(),
#     }

#     model = models.get(model_name, RandomForestRegressor())
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     mae = mean_absolute_error(y_test, y_pred)

#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     joblib.dump(model, model_path)
#     return model, mae

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def train_and_save_model(X_train, X_test, y_train, y_test, model_choice, encoders):
    # Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    # Select model
    model = models.get(model_choice)
    if model is None:
        raise ValueError(f"❌ Model '{model_choice}' not found! Available: {list(models.keys())}")

    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Ensure folder exists
    os.makedirs("models", exist_ok=True)

    # Save model & encoders
    joblib.dump(model, "models/trained_model.pkl")
    joblib.dump(encoders, "models/label_encoders.pkl")

    return model, acc
