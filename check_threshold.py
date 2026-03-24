import os
import mlflow

# --- CONFIG ---
FORCE_FAIL = True  # Set to True to simulate low accuracy
THRESHOLD = 0.85

# Read Run ID
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

# Fetch accuracy from MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
client = mlflow.MlflowClient()
run = client.get_run(run_id)
accuracy = float(run.data.metrics.get("accuracy", 0.0))

# Force fail for testing if needed
if FORCE_FAIL:
    accuracy = 0.5

print(f"ℹ️ Accuracy for Run ID {run_id}: {accuracy:.4f}")

# Check threshold
if accuracy >= THRESHOLD:
    print(f"✅ Accuracy above threshold ({THRESHOLD}) — ready to deploy!")
else:
    print(f"❌ Accuracy below threshold ({THRESHOLD}) — failing pipeline.")
    exit(1)  # This will fail the GitHub Action