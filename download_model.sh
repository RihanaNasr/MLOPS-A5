# download_model.sh
#!/bin/bash
echo "Downloading model $RUN_ID..."
mlflow models download -m "runs:/$RUN_ID/model" -d /app/model