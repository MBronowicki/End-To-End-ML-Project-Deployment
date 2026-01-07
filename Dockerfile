FROM python:3.11-slim

WORKDIR /app

RUN apt-get update -y && apt-get install awscli -y \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/
CMD ["python3", "app.py"]