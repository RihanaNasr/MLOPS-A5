# Use a lightweight Python base
FROM python:3.10-slim

# Accept a Run ID argument
ARG RUN_ID
ENV RUN_ID=$RUN_ID

# Set working directory
WORKDIR /app

# Copy repository files (optional)
COPY . .

# Simulate downloading the model
RUN echo "Downloading model with Run ID: $RUN_ID"

# Default command
CMD ["python3", "-c", "print(f'Model ready for Run ID: {RUN_ID}')"]