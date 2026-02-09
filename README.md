# Accounting Reconciliation Web Service

This project provides a powerful and flexible accounting reconciliation service, accessible via a web interface or as a batch processing script. It allows users to upload financial data, apply sophisticated matching algorithms, and generate detailed reports.

## ✨ Key Features

- **Intuitive Web Interface**: A clean, tab-organized UI for uploading files and customizing processing settings.
- **Multiple Algorithms**: Supports various reconciliation algorithms, including "Simple" (1-to-1) and "Subset Sum" (N-to-1), selectable by the user.
- **Dynamic Configuration**: Allows real-time modification of key parameters like tolerance, time windows, and search strategies directly from the browser.
- **Secure In-Memory Processing**: Files are processed in memory to ensure speed and data privacy, without permanently storing sensitive data.
- **Detailed Excel Reports**: The output is a multi-sheet Excel file that includes:
  - A **Summary** sheet with parameters and high-level results.
  - A sheet with all **Matches** found.
  - Sheets for **Unreconciled Debit** and **Unreconciled Credit** transactions.
  - Detailed processing **Statistics**.
  - A **Monthly Balance** analysis with a visual chart.
- **Batch Processing**: A command-line script (`batch.py`) to process multiple files automatically.
- **Parameter Optimizer**: A script (`optimizer.py`) to find the optimal reconciliation parameters for a given dataset.
- **Production & Docker Ready**: Containerized with Docker and ready for production deployment using a Gunicorn WSGI server.
- **Modular Architecture**: Core logic separated from reporting and UI for better maintainability.

## ⚙️ Installation

1.  **Prerequisites**: Python 3.9+ and Git.

2.  **Clone the Repository**:
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd accounting-reconciliation
    ```

3.  **Create and Activate a Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
    ```

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🐳 Usage with Docker

The recommended way to run the application is with Docker Compose, which simplifies building the image and managing containers.

### Using Docker Compose

1.  **Build and Start the Service**:
    ```bash
    docker compose up -d --build
    ```
    The application will be accessible at `http://localhost:5000`.

2.  **View Logs**:
    ```bash
    docker compose logs -f
    ```

3.  **Stop the Service**:
    ```bash
    docker compose down
    ```

## 🚀 How to Use

### Web Application

The web app is ideal for interactive, single-file processing.

#### 1. Start the Application

**Development Mode (for local testing):**
Uses the Flask development server, suitable for a single user.
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
pip install gunicorn  # Già incluso in requirements.txt
```

**b. Avvia il server con Gunicorn:**
```bash
python batch_processor.py
python batch.py
```

### Output Generato

Per ogni file elaborato:
```
output/
├── risultato_supermercato_A.xlsx  ← Foglio Excel completo
├── risultato_supermercato_B.xlsx
└── logs/
    ├── batch_log_[timestamp].json     ← Log dettagliato JSON
    └── riepilogo_[timestamp].csv      ← Tabella riepilogo
```

### Esempio Output Console

```
╔════════════════════════════════════════════════════════════╗
║          BATCH PROCESSOR - ELABORAZIONE MULTIPLA           ║
╚════════════════════════════════════════════════════════════╝

🔍 Ricerca file in: input
✓ Trovati 3 file da elaborare

⚙️  CONFIGURAZIONE:
   - Tolleranza: €0.01
   - Finestra temporale: ±30 giorni
   - Max combinazioni: 6
   - Output: output/

[1/3] ============================================================
📂 Elaborazione: supermercato_A.xlsx
============================================================
  ⏳ Caricamento dati...
  ✓ Caricati 850 movimenti
  Movimenti DARE: 320, AVERE: 530
  ⏳ Riconciliazione in corso...

  ✅ COMPLETATO
  📊 DARE riconciliati: 305/320 (95.3%)
  📊 AVERE utilizzati: 498/530 (94.0%)
  💾 Salvato in: risultato_supermercato_A.xlsx

[2/3] ============================================================
...

============================================================
📊 RIEPILOGO GLOBALE
============================================================
✓ File elaborati con successo: 3/3
✗ File con errori: 0
⏱  Tempo totale: 45.3 secondi (0.8 minuti)

📈 STATISTICHE AGGREGATE:
   - Totale movimenti DARE: 1,240
   - DARE riconciliati: 1,180 (95.2%)

💾 Log salvato in: output/logs/
============================================================
```

### File Log JSON

Il file `batch_log_[timestamp].json` contiene:

```json
{
  "timestamp": "2025-01-15T14:30:22",
  "configurazione": {
    "tolleranza": 0.01,
    "giorni_finestra": 30,
    "max_combinazioni": 6
  },
  "risultati": [
    {
      "file": "supermercato_A.xlsx",
      "successo": true,
      "statistiche": {
        "totale_dare": 320,
        "dare_riconciliati": 305,
        "percentuale_dare": 95.3,
        "output_file": "output/risultato_supermercato_A.xlsx"
      }
    }
  ]
}
```

### File Riepilogo CSV

Tabella analisi rapida per Excel/analisi:

| File | Successo | totale_dare | dare_riconciliati | percentuale_dare |
|------|----------|-------------|-------------------|------------------|
| supermercato_A.xlsx | True | 320 | 305 | 95.3 |
| supermercato_B.xlsx | True | 450 | 430 | 95.6 |
| supermercato_C.xlsx | True | 470 | 445 | 94.7 |

### Pattern Avanzati

**Elabora solo file specifici:**
```python
'pattern': 'supermercato_*.xlsx'  # Solo file che iniziano con "supermercato_"
'pattern': '*_gennaio_2025.xlsx'  # Solo file di gennaio
'pattern': 'store_[0-9]*.xlsx'    # Solo store con numero
```

**Cartelle separate per periodo:**
```python
config = {
    'cartella_input': 'dati/gennaio_2025',
    'cartella_output': 'risultati/gennaio_2025',
}
```

### Automazione con Cron/Task Scheduler

**Linux/Mac (cron):**
```bash
# Esegui ogni giorno alle 2:00 AM
0 2 * * * cd /percorso/progetto && python batch_processor.py >> batch.log 2>&1
```

**Windows (Task Scheduler):**
```
Azione: Avvia programma
Programma: python.exe
Argomenti: batch_processor.py
Cartella di avvio: C:\percorso\progetto
```

### Gestione Errori

Se alcuni file falliscono, il batch continua con gli altri:

```
[2/5] ============================================================
📂 Elaborazione: file_con_colonne_errate.xlsx
============================================================
  ❌ ERRORE: Il file deve contenere le colonne: Data, Dare, Avere

[3/5] ============================================================
📂 Elaborazione: file_ok.xlsx
============================================================
  ✅ COMPLETATO
...

⚠️  FILE CON ERRORI:
   - file_con_colonne_errate.xlsx: Nome colonna 'Incasso' non trovato. Controllare il file di configurazione.
```

### Integrazione con Script Esterni

**Esportare solo statistiche:**
```python
from batch_processor import BatchProcessor

processor = BatchProcessor(config)
processor.elabora_tutti()

# Accedi a statistiche
for stat in processor.stats_globali:
    print(f"{stat['file']}: {stat['statistiche']['percentuale_dare']}%")
```

**Callback personalizzati:**
```python
class BatchProcessorCustom(BatchProcessor):
    def elabora_file(self, file_path):
        risultato = super().elabora_file(file_path)
        
        # Invia email se errori
        if not risultato['successo']:
            self.invia_alert(risultato['file'], risultato['errore'])
        
        return risultato
```

---

## ⚙️ Configurazione via File JSON (Consigliato)

Per una maggiore flessibilità, è possibile gestire tutti i parametri tramite un file esterno `config.json` senza modificare il codice Python.

**1. Crea un file `config.json`** nella stessa cartella degli script con questo contenuto:

```json
{
  "tolleranza": 0.01,
  "giorni_finestra": 30,
  "max_combinazioni": 6,
  "cartella_input": "input",
  "cartella_output": "output",
  "pattern": [
    "*.xlsx",
    "*.csv"
  ]
}
```

**2. Esegui lo script `batch.py`**: Lo script rileverà automaticamente il file `config.json` e utilizzerà i valori specificati. Se il file non viene trovato, verranno utilizzati i parametri di default.

---

## 📞 Supporto

Per domande o problemi, contatta: [tua-email@esempio.com]

---

## 📂 Project Structure & File Explanation

Here is an overview of the key files in the project and how they interact:

### Core Logic
- **`core.py`**: The heart of the application. Contains the `ReconciliationEngine` class which implements the reconciliation algorithms (Subset Sum, Progressive Balance) and manages the data flow.
- **`reporting.py`**: Handles the generation of the multi-sheet Excel report. It extracts data from the engine and formats it into a user-friendly Excel file with charts and statistics, adhering to the Single Responsibility Principle.
- **`config.json`**: The central configuration file. Defines parameters like tolerance, column mapping, and algorithm choice.

### Interfaces
- **`app.py`**: The Flask web application. Manages the web interface, file uploads, and API endpoints. It instantiates `ReconciliationEngine` to process uploaded files.
- **`riconciliazione.py`**: The batch processing script. It reads all files in the `input/` folder and processes them sequentially using the settings in `config.json`. Ideal for bulk processing.
- **`main.py`**: A command-line wrapper for single-file execution. Useful for integration with other tools or specific one-off runs.
- **`optimizer.py`**: An advanced script using `Optuna` to automatically find the best parameters (tolerance, window, etc.) for a specific dataset to maximize the reconciliation rate.

### Infrastructure & Docs
- **`docker-compose.yml`**: Defines the Docker service configuration, including volume mounts for live code updates.
- **`templates/index.html`**: The frontend HTML/JS for the web interface.
- **`doc/`**: Folder containing documentation and tutorials (e.g., Git guide).

### How they connect
1.  **User Input**: The user interacts via Web (`app.py`) or Batch Script (`riconciliazione.py`).
2.  **Configuration**: Both interfaces load settings from `config.json`.
3.  **Processing**: The input data is passed to `core.py` (`ReconciliationEngine`).
4.  **Reporting**: Once processed, `core.py` delegates the creation of the Excel file to `reporting.py`.

---

## 📜 Changelog

### v3.0.0 (Versione Corrente)
- **Docker**: Aggiunto supporto ufficiale con `Dockerfile` e `.dockerignore`.
- **Core Engine**: Riscrittura completa del motore (`core.py`) utilizzando **Pandas DataFrame** per performance elevate.
- **Testing**: Nuova suite di test in `tests/` con script di automazione `run_tests.sh`.
- **Refactoring**: Spostamento utility in `tools/` e pulizia del codice.
- **Logging**: Controllo granulare sul salvataggio dei log (parametro `save_log`).
- **Best Fit**: Migliorata la logica di abbinamento parziale con opzione per disabilitarla.

### v1.1.0 (2025-01-15)
- ✨ Aggiunto Batch Processor per elaborazione multipla
- 📊 Log JSON e CSV dettagliati
- 🚀 Ottimizzazioni performance per file grandi

### v1.0.0 (2025-01-14)
- 🎉 Release iniziale
- ✅ Algoritmo riconciliazione base
- 📄 Export Excel multi-foglio

---

**Versione**: 3.0.0
**Ultimo aggiornamento**: Gennaio 2026
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
