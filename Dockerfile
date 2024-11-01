FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/weights /app/input /app/output

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py .

ENTRYPOINT ["python", "inference.py"]