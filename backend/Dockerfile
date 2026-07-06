# Free HuggingFace Space (Docker SDK, CPU basic tier — $0).
# Python 3.11 has torch wheels; the local machine's 3.14 does not.
FROM python:3.11-slim

WORKDIR /app

# HF Spaces run as a non-root user (uid 1000); cache model weights there.
ENV HF_HOME=/app/.cache \
    TRANSFORMERS_CACHE=/app/.cache \
    PORT=7860

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fast_detect.py app.py ./

# Warm the model into the image cache so the first request isn't a cold download.
RUN python -c "from fast_detect import _load; _load()"

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
