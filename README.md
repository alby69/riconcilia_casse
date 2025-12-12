# Servizio Web di Riconciliazione Contabile

Questo progetto è un'applicazione web basata su Flask che fornisce un servizio di riconciliazione contabile. Gli utenti possono caricare un file Excel contenente movimenti di "Dare" e "Avere", e il sistema restituisce un nuovo file Excel con i movimenti riconciliati, le statistiche e i dettagli delle operazioni.

Il progetto originale è stato refattorizzato per passare da un'architettura a script batch a un'architettura client-server, più flessibile e accessibile tramite browser.

## ✨ Caratteristiche Principali

- **Interfaccia Web Semplice**: Un'interfaccia pulita per caricare i file direttamente dal browser.
- **Elaborazione in Memoria**: I file vengono processati interamente in memoria per garantire la massima velocità e sicurezza, senza la necessità di salvare file temporanei sul disco del server.
- **Report Dettagliati**: L'output è un file Excel multi-foglio che include:
  - Abbinamenti trovati (1-a-1 e combinazioni multiple).
  - Movimenti non riconciliati.
  - Statistiche complete sull'elaborazione.
  - Un riepilogo dei parametri utilizzati.
- **Sicurezza per l'Uso Concorrente**: L'architettura è "stateless", il che significa che ogni richiesta utente è isolata. Più utenti possono usare il servizio contemporaneamente senza che i loro dati si sovrappongano.
- **Pronto per la Produzione**: Include istruzioni per l'avvio con un server WSGI di produzione come Gunicorn.

## ⚙️ Installazione

1.  **Prerequisiti**: Assicurati di avere Python 3.9 o superiore installato.

2.  **Clona il Repository (se necessario)**:
    ```bash
    git clone <URL_DEL_TUO_REPOSITORY>
    cd riconcilia_casse
    ```

3.  **Crea un Ambiente Virtuale**: È una buona pratica isolare le dipendenze del progetto.
    ```bash
    python -m venv .venv
    ```

4.  **Attiva l'Ambiente Virtuale**:
    - Su macOS/Linux:
      ```bash
      source .venv/bin/activate
      ```
    - Su Windows:
      ```bash
      .venv\Scripts\activate
      ```

5.  **Installa le Dipendenze**: Installa tutte le librerie necessarie, inclusa Flask.
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Utilizzo

Puoi avviare l'applicazione in due modalità: una per lo sviluppo e il test, l'altra ottimizzata per la produzione.

### 1. Modalità di Sviluppo

Questa modalità è ideale per testare l'applicazione in locale. Utilizza il server integrato di Flask, che è semplice da avviare ma può gestire una sola richiesta alla volta.

**Avvio del server:**
```bash
python app.py
```
Dopo aver eseguito il comando, vedrai un output simile a questo:
```
 * Running on http://127.0.0.1:5000
```
Apri il tuo browser e vai all'indirizzo **http://127.0.0.1:5000** per usare l'applicazione.

### 2. Modalità di Produzione (Consigliata)

Per un uso reale con più utenti, è necessario un server WSGI come **Gunicorn**, in grado di gestire più richieste contemporaneamente.

**a. Installa Gunicorn:**
```bash
pip install gunicorn
```

**b. Avvia il server con Gunicorn:**
```bash
gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
```
- `--workers 4`: Avvia 4 "operai". Questo significa che il server può processare fino a 4 richieste utente in parallelo. Puoi adattare questo numero in base alla CPU e alla RAM del tuo server.
- `--bind 0.0.0.0:5000`: Rende l'applicazione accessibile da altre macchine sulla stessa rete all'indirizzo IP del server sulla porta 5000.
- `app:app`: Indica a Gunicorn di eseguire l'oggetto `app` che si trova all'interno del file `app.py`.

## 📖 Come Usare l'Interfaccia Web

1.  **Avvia il server** usando uno dei metodi descritti sopra.
2.  **Apri il browser** all'indirizzo del server (es. `http://127.0.0.1:5000`).
3.  **Carica il file**:
    - Clicca su "Scegli file".
    - Seleziona un file Excel (`.xlsx` o `.xls`).
    - **Requisito**: Il file deve contenere obbligatoriamente le tre colonne `Data`, `Dare`, `Avere`.
4.  **Elabora**: Clicca sul pulsante "Elabora File".
5.  **Scarica il risultato**: Dopo pochi istanti, il browser scaricherà automaticamente il file Excel elaborato, con un nome simile a `Riconciliato_tuofile.xlsx`.
