# Simple Dockerfile for the assignment
FROM python:3.10-slim

ARG RUN_ID

WORKDIR /app

# Simulate downloading model
RUN echo "Downloading model for RUN_ID=$RUN_ID"

CMD ["echo", "Container ready for RUN_ID=$RUN_ID"]