# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

# Use local folder (works in GitHub Actions)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Iris_Classification")

# Start MLflow run
with mlflow.start_run() as run:
    
    # Load data
    data = pd.read_csv("iris.csv")

    X = data.drop("species", axis=1)
    y = data["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Save model locally (for DVC requirement)
    joblib.dump(model, "model.pkl")

    print(f"✅ Accuracy: {accuracy}")
    print(f"✅ Run ID: {run.info.run_id}")

    # SAVE RUN ID to file (IMPORTANT FIX)
    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)