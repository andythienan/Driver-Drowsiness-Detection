FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies for OpenCV, MediaPipe, and WebRTC/FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY app.py .
COPY svm_model.joblib .
COPY scaler.joblib .

# Streamlit config via environment variables
ENV STREAMLIT_SERVER_PORT=8502 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# Expose Streamlit port (changed from default 8501 to 8502)
EXPOSE 8502

# Default command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]


