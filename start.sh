#!/bin/sh

# Imposta il numero di worker.
# Usa la variabile d'ambiente GUNICORN_WORKERS se è definita,
# altrimenti calcola il valore di default con la formula raccomandata: (2 * numero di core CPU) + 1.
WORKERS=${GUNICORN_WORKERS:-$(($(nproc) * 2 + 1))}

# Imposta il timeout (default 300s) per evitare che Gunicorn uccida il processo durante l'ottimizzazione
TIMEOUT=${GUNICORN_TIMEOUT:-300}

echo "INFO: Avvio di Gunicorn con ${WORKERS} workers."

# Usa 'exec' per sostituire il processo della shell con quello di Gunicorn.
# Questo assicura che i segnali (es. SIGTERM da 'docker stop') vengano passati correttamente.
exec gunicorn --workers ${WORKERS} --timeout ${TIMEOUT} --bind 0.0.0.0:5000 app:app