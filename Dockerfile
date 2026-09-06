FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["sh", "-c", "python -m src.utils.docker_utils && streamlit run app.py --server.address=0.0.0.0 --server.port=8501"]