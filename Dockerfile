FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces expects the app to listen on $PORT (default 7860).
ENV PORT=7860 \
    MAX_COACH_ITER=400 \
    MAX_BOT_MCTS_ITER=400 \
    DISABLE_LOG_SAVE=1

EXPOSE 7860

CMD ["sh", "-c", "gunicorn -w 1 -t 180 -b 0.0.0.0:${PORT:-7860} webapp:app"]
