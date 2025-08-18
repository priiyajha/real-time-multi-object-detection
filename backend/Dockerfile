# This Dockerfile sets up the Python backend with necessary libraries.
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy backend files from the build context (./backend) to the workdir (/app)
COPY . .

# Copy models from the models subdirectory
COPY models /app/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    aiohttp==3.9.1 \
    aiortc==1.5.0 \
    onnxruntime==1.16.3 \
    opencv-python-headless

# Expose the server port
EXPOSE 8000

# Set the entry point to run the server directly with python
CMD ["python", "server.py"]
