# Usa un'immagine base di Python 3.9 slim per mantenere il container leggero
FROM python:3.9-slim

# Imposta variabili d'ambiente
# PYTHONDONTWRITEBYTECODE: Previene la scrittura di file .pyc
# PYTHONUNBUFFERED: Assicura che l'output di Python sia inviato direttamente al terminale (utile per i log)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Imposta la directory di lavoro nel container
WORKDIR /app

# Installa le dipendenze di sistema necessarie (es. compilatori per alcune librerie Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia il file requirements.txt e installa le dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copia il resto del codice dell'applicazione
COPY . .

# Espone la porta 5000 per Flask
EXPOSE 5000

# Avvia l'applicazione usando Gunicorn come server WSGI di produzione
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "app:app"]