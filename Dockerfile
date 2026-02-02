# Fase 1: Usa un'immagine Python ufficiale come base
FROM python:3.9-slim

# Imposta la directory di lavoro all'interno del container
WORKDIR /app

# Copia il file delle dipendenze
COPY requirements.txt .

# Installa le dipendenze
# --no-cache-dir riduce la dimensione dell'immagine
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il codice dell'applicazione nella directory di lavoro
COPY . .

# Esponi la porta su cui Gunicorn sarà in ascolto
#EXPOSE 5000
EXPOSE 10000

# Comando per avviare l'applicazione quando il container parte
#CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "app:app"]
CMD ["sh", "-c", "gunicorn", "--workers", "2", "--bind", "0.0.0.0:$PORT app:app"]