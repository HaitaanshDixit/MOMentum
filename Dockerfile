FROM python:3.11.9-slim

# Installing system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Setting working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install openai-whisper --no-build-isolation && \
    pip install -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords averaged_perceptron_tagger

# Copy all project files
COPY . .

# Create necessary directories
RUN mkdir -p output uploads_temp vector_store

# Expose port 7860 (HuggingFace Spaces default)
EXPOSE 7860

# Set environment variables
ENV DEPLOYMENT_MODE=full
ENV WHISPER_MODEL=base
ENV MAX_FILE_SIZE_MB=500

# Start the app on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]