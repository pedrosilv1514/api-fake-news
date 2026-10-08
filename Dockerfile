FROM python:3.14-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir poetry==2.5.1
ENV POETRY_VIRTUALENVS_CREATE=false
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-interaction
# Only legacy stopwords are provisioned at build time; never at API startup.
RUN python -c "import nltk; assert nltk.download('stopwords', download_dir='/usr/local/share/nltk_data', raise_on_error=True)"

FROM python:3.14-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/share/nltk_data /usr/local/share/nltk_data
ENV PYTHONUNBUFFERED=1 ENABLE_BIAS_MODEL=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
