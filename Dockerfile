FROM python:3.11-slim

WORKDIR /app

# Minimal system deps for any package that still needs to build from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NOTE: no local embedding model to pre-download anymore - embeddings now call Hugging
# Face's hosted Inference API at runtime (see src/embeddings.py), so there are no model
# weights to bake into the image.

COPY . .

# Hugging Face Spaces (Docker SDK) expects port 7860; Render/other platforms assign
# their own port via $PORT. Use $PORT if set, otherwise fall back to 7860.
EXPOSE 7860

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PORT=7860

CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true
