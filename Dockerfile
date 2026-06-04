FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces expects the app to listen on $PORT (default 7860).
# SUPABASE_URL / SUPABASE_KEY are set as Space secrets, not baked in here.
# AZ_SIMS = PUCT simulations/move for the AlphaZero opponent (4-player games).
# Lower = faster on HF's CPU; the net is strong even at modest sims.
ENV PORT=7860 \
    MAX_COACH_ITER=400 \
    BOT_TIME_BUDGET_S=0.5 \
    AZ_SIMS=80 \
    DISABLE_LOG_SAVE=1

EXPOSE 7860

CMD ["sh", "-c", "gunicorn -w 1 -t 180 -b 0.0.0.0:${PORT:-7860} webapp:app"]
