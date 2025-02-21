FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && \
    apt-get install -y gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instalar dependências Python com índices múltiplos
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

COPY . .

CMD ["python", "-u", "agent.py"]
