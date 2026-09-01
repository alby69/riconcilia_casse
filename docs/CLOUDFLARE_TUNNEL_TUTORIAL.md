# Pubblicare l'applicazione su Raspberry Pi con Cloudflare Tunnel

Questa guida spiega come esporre in modo sicuro l'applicazione web (eseguita su un Raspberry Pi) su internet tramite **Cloudflare Tunnel**. Questo metodo non richiede un IP pubblico né complicate configurazioni firewall.

## ✨ Vantaggi principali

- **Nessun IP pubblico necessario**: il Raspberry Pi resta nascosto da internet.
- **Sicurezza di default**: tutto il traffico è crittografato con HTTPS e Cloudflare protegge anche da attacchi DDoS.
- **URL stabile**: ottieni un indirizzo permanente e raggiungibile da ovunque.
- **Gratuito**: il servizio Tunnel base di Cloudflare è gratuito.

## 📋 Prerequisiti

1. **Raspberry Pi**: consigliato un Raspberry Pi 4 o successivo, con una versione a 64 bit di Raspberry Pi OS.
2. **Docker e Docker Compose**: devono essere installati sul Raspberry Pi.
3. **Account Cloudflare**: è necessario un account gratuito.
4. **(Opzionale) Dominio**: se possiedi un dominio, puoi gestirlo tramite Cloudflare per creare un URL personalizzato (es. `riconciliazione.tuodominio.com`). In caso contrario, Cloudflare fornirà un URL gratuito e casuale `trycloudflare.com`.

---

## 🚀 Guida passo per passo

### Passo 1: prepara l'applicazione sul Raspberry Pi

1. **Apri un terminale** sul Raspberry Pi (direttamente o tramite SSH).

2. **Clona il repository del progetto**:
   ```bash
   git clone <URL_DEL_TUO_REPOSITORY>
   cd riconcilia_casse
   ```
   *(Sostituisci `<URL_DEL_TUO_REPOSITORY>` con l'indirizzo reale del tuo repository)*

3. **Compila e avvia l'applicazione con Docker Compose**:
   Questo comando crea l'immagine Docker appositamente per l'architettura ARM del tuo Pi e avvia il servizio web in background.
   ```bash
   docker compose up -d --build
   ```

4. **Verifica che l'applicazione funzioni in locale**:
   Apri un browser *sul Raspberry Pi* e vai su `http://localhost:5000`. Dovresti vedere l'interfaccia web dell'applicazione. Questo conferma che il container funziona correttamente.

### Passo 2: configura Cloudflare Tunnel

1. **Installa `cloudflared` sul Raspberry Pi**:
   Segui le istruzioni ufficiali Cloudflare per scaricare e installare il demone `cloudflared`. Per Raspberry Pi OS a 64 bit, scegli di solito il pacchetto **Debian** per architettura **arm64**.

   Esempio di comando:
   ```bash
   wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
   sudo dpkg -i cloudflared-linux-arm64.deb
   ```

2. **Autentica `cloudflared`**:
   Questo comando apre una finestra del browser. Accedi al tuo account Cloudflare e seleziona un dominio da autorizzare (oppure autorizza solo l'account se non hai un dominio).
   ```bash
   cloudflared tunnel login
   ```
   Una volta autorizzato, viene salvato un file di certificato nella cartella `~/.cloudflared/`.

3. **Crea un tunnel**:
   Dai al tunnel un nome facile da ricordare. Questo comando registra il tunnel su Cloudflare e crea un file di credenziali.
   ```bash
   cloudflared tunnel create riconciliazione-app
   ```
   Annota l'**UUID del tunnel** e il percorso del file di credenziali (`.json`) mostrati nell'output: ti serviranno nel passo successivo.

### Passo 3: configura e avvia il tunnel

1. **Crea un file di configurazione**:
   Devi dire a `cloudflared` dove inviare il traffico in ingresso. Crea un file di configurazione nella cartella `~/.cloudflared/`.
   ```bash
   nano ~/.cloudflared/config.yml
   ```

2. **Aggiungi questo contenuto a `config.yml`**:
   Sostituisci `<UUID-Del-Tuo-Tunnel>` con l'UUID del passo precedente.

   ```yaml
   tunnel: <UUID-Del-Tuo-Tunnel>
   credentials-file: /home/pi/.cloudflared/<UUID-Del-Tuo-Tunnel>.json

   ingress:
     # Questa regola inoltra il traffico alla tua applicazione web locale
     - hostname: riconciliazione.tuodominio.com # <-- IMPORTANTE: cambia questo!
       service: http://localhost:5000

     # Regola di fallback: restituisce un errore 404 per tutto il resto del traffico
     - service: http_status:404
   ```

   **Configurazione del nome host:**
   - **Se hai un dominio**: sostituisci `riconciliazione.tuodominio.com` con il sottodominio e il dominio che vuoi usare.
   - **Se non hai un dominio**: puoi omettere la riga `hostname`. Cloudflare assegnerà un URL casuale `*.trycloudflare.com` quando avvii il tunnel.

3. **Punta il DNS al tunnel (solo se usi un tuo dominio)**:
   Questo comando crea i record DNS necessari nel tuo account Cloudflare per collegare il nome host scelto al tunnel.
   ```bash
   cloudflared tunnel route dns riconciliazione-app riconciliazione.tuodominio.com
   ```

4. **Avvia il tunnel come servizio di sistema**:
   Eseguire `cloudflared` come servizio di sistema garantisce che si avvii automaticamente all'accensione del Raspberry Pi.

   ```bash
   # Installa il servizio
   sudo cloudflared service install

   # Avvia il servizio
   sudo systemctl start cloudflared

   # (Opzionale) Controlla lo stato del servizio
   sudo systemctl status cloudflared
   ```

---

## ✅ Fatto!

La tua applicazione è ora raggiungibile in modo sicuro da qualsiasi punto di internet all'URL configurato (es. `https://riconciliazione.tuodominio.com`). Cloudflare gestisce automaticamente il certificato HTTPS.

Per vedere i log del tunnel, puoi usare:
```bash
journalctl -u cloudflared -f
```