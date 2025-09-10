# --- Stage 1: The "Builder" ---
# Use a full Python image to build our dependencies
FROM python:3.11 AS builder

WORKDIR /app

# Create a virtual environment within the builder
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the requirements file and install dependencies into the venv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: The "Final" Image ---
# Start from a lightweight, slim Python image for the final product
FROM python:3.11-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the rest of the application's code
COPY . .

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Make port 7860 available
EXPOSE 7860

# Define environment variable for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run app.py when the container launches
CMD ["python", "app.py"]

