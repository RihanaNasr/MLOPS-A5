import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Use Linux-friendly or relative path for MLflow tracking
tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or "./mlruns"
mlflow.set_tracking_uri(tracking_uri)

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train a simple classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# Log metrics and model
with mlflow.start_run() as run:
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    run_id = run.info.run_id

# Save run ID for deploy job
with open("model_info.txt", "w") as f:
    f.write(run_id)

print(f"Accuracy: {acc}")
print(f"Run ID: {run_id}")