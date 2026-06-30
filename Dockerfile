FROM python:3.11.9-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install openai-whisper --no-build-isolation && \
    pip install -r requirements.txt

RUN python -m nltk.downloader punkt stopwords averaged_perceptron_tagger

# Predownload models during build so they're baked into the image
# avoids slow downloads on first request after a cold start which is 10 min atleast
RUN python -c "import whisper; whisper.load_model('base')"
RUN python -c "from transformers import pipeline; pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

RUN mkdir -p output uploads_temp vector_store

# Expose port 7860
EXPOSE 7860

ENV DEPLOYMENT_MODE=full
ENV WHISPER_MODEL=base
ENV MAX_FILE_SIZE_MB=500

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]