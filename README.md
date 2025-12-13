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
python batch_processor.py
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

## 📜 Changelog

### v1.1.0 (2025-01-15)
- ✨ Aggiunto Batch Processor per elaborazione multipla
- 📊 Log JSON e CSV dettagliati
- 🚀 Ottimizzazioni performance per file grandi

### v1.0.0 (2025-01-14)
- 🎉 Release iniziale
- ✅ Algoritmo riconciliazione base
- 📄 Export Excel multi-foglio

---

**Versione**: 1.1.0  
**Ultimo aggiornamento**: Novembre 2025
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
