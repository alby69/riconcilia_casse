# Guida al Deploy su Render — CashRec

Questa guida descrive come distribuire la versione **Web App** di CashRec su **Render.com** utilizzando Docker.

## Deploy tramite Render Blueprint (`render.yaml`)

1. Fai il forking / push del repository su GitHub (`https://github.com/alby69/cashrec`).
2. Accedi a [Render Dashboard](https://dashboard.render.com/).
3. Clicca su **New +** e seleziona **Blueprint**.
4. Collega il tuo repository GitHub `cashrec`.
5. Render rileverà automaticamente il file `render.yaml` nella root.
6. Clicca su **Apply** per avviare il deploy.

## Deploy Manuale su Render (Docker)

1. Nel dashboard di Render, seleziona **New +** → **Web Service**.
2. Collega il repository GitHub.
3. Configura i seguenti parametri:
   - **Name**: `cashrec`
   - **Environment**: `Docker`
   - **Dockerfile Path**: `./docker/Dockerfile`
   - **Docker Context**: `.`
   - **Region**: Frankfurt (o preferita)
   - **Branch**: `main`
4. Aggiungi la variabile d'ambiente:
   - `PORT`: `80` (o assegnata automaticamente da Render)
5. Clicca su **Create Web Service**.

## Verifica del Deploy

Una volta completato il deploy, l'applicazione sarà raggiungibile all'URL assegnato da Render (es. `https://cashrec.onrender.com`).
