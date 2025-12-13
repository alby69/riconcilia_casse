# Servizio Web di Riconciliazione Contabile

Questo progetto è un'applicazione web basata su Flask che fornisce un servizio di riconciliazione contabile. Gli utenti possono caricare un file Excel contenente movimenti di "Dare" e "Avere", e il sistema restituisce un nuovo file Excel con i movimenti riconciliati, le statistiche e i dettagli delle operazioni.

Il progetto originale è stato refattorizzato per passare da un'architettura a script batch a un'architettura client-server, più flessibile e accessibile tramite browser.

## ✨ Caratteristiche Principali

- **Interfaccia Web Semplice**: Un'interfaccia pulita per caricare i file direttamente dal browser.
- **Configurazione via Web**: Una nuova scheda "Configurazione" permette di visualizzare e modificare i parametri dell'algoritmo di riconciliazione in tempo reale.
- **Elaborazione in Memoria**: I file vengono processati interamente in memoria per garantire la massima velocità e sicurezza.
- **Report Dettagliati**: L'output è un file Excel multi-foglio che include abbinamenti, movimenti non riconciliati, statistiche e un riepilogo dei parametri.
- **Sicurezza per l'Uso Concorrente**: Architettura "stateless" che isola ogni richiesta, permettendo a più utenti di usare il servizio contemporaneamente.
- **Pronto per la Produzione**: Include istruzioni per l'avvio con un server WSGI di produzione come Gunicorn.

## ⚙️ Installazione

1.  **Prerequisiti**: Assicurati di avere Python 3.9 o superiore installato.

2.  **Clona il Repository**:
    ```bash
    git clone <URL_DEL_TUO_REPOSITORY>
    cd riconcilia_casse
    ```

3.  **Crea un Ambiente Virtuale**:
    ```bash
    python -m venv .venv
    ```

4.  **Attiva l'Ambiente Virtuale**:
    - Su macOS/Linux: `source .venv/bin/activate`
    - Su Windows: `.venv\Scripts\activate`

5.  **Installa le Dipendenze**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Nota: per alcuni sistemi operativi potrebbero essere necessari strumenti di compilazione aggiuntivi. Consulta la documentazione delle singole librerie in caso di errori).*

## 🚀 Utilizzo

Puoi avviare l'applicazione in due modalità: sviluppo (per test locali) o produzione (consigliata per l'uso reale).

### 1. Modalità di Sviluppo
Utilizza il server integrato di Flask, semplice da avviare ma in grado di gestire una sola richiesta alla volta.

**Avvio del server:**
```bash
python app.py
```
Apri il browser e vai all'indirizzo **http://127.0.0.1:5000** per usare l'applicazione.

### 2. Modalità di Produzione (Consigliata)
Per un uso con più utenti, è necessario un server WSGI come **Gunicorn** (su macOS/Linux).

**a. Installa Gunicorn:**
```bash
pip install gunicorn
```

**b. Avvia il server con Gunicorn:**
```bash
gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
```
- `--workers 4`: Avvia 4 processi "operai" per gestire fino a 4 richieste in parallelo.
- `--bind 0.0.0.0:5000`: Rende l'applicazione accessibile da altre macchine sulla rete.

## 📖 Come Usare l'Interfaccia Web

1.  **Avvia il server** usando uno dei metodi descritti sopra.
2.  **Apri il browser** all'indirizzo del server (es. `http://127.0.0.1:5000`).
3.  L'interfaccia è divisa in due schede: **Elaborazione** e **Configurazione**.

### Scheda "Elaborazione"

Qui puoi caricare i file da analizzare:
1.  **Carica il file**: Clicca su "Scegli file" e seleziona un file Excel (`.xlsx` o `.xls`). Il file deve contenere le colonne `Data`, `Dare`, `Avere`.
2.  **Elabora**: Clicca su "Elabora File".
3.  **Scarica il risultato**: Al termine dell'elaborazione, apparirà un riepilogo e un pulsante per scaricare il report Excel.

### Scheda "Configurazione"

Questa scheda ti permette di personalizzare il comportamento dell'algoritmo di riconciliazione.
1.  **Visualizza i Parametri**: La pagina mostra i valori attuali caricati dal file `config.json`.
2.  **Modifica i Valori**: Aggiorna i campi del modulo (es. `Tolleranza`, `Giorni Finestra`, ecc.).
3.  **Salva**: Clicca su "Salva Configurazione". Le modifiche verranno scritte nel file `config.json` e usate per le elaborazioni successive.

---

## ⚙️ Configurazione via File JSON

L'applicazione utilizza un file `config.json` per gestire i parametri. La configurazione può essere modificata sia manualmente, editando il file, sia tramite l'interfaccia web.

**Struttura del file `config.json`:**
```json
{
  "tolleranza": 0.01,
  "giorni_finestra": 10,
  "max_combinazioni": 6,
  "cartella_input": "input",
  "cartella_output": "output",
  "pattern": [
    "*.xlsx",
    "*.csv"
  ],
  "residui": {
    "attiva": true,
    "soglia_importo": 100,
    "giorni_finestra": 90
  }
}
```

---

## 📞 Supporto

Per domande o problemi, apri una issue su GitHub.

---

## 📜 Changelog

### v2.0.0 (2025-12-13)
- ✨ **Interfaccia di Configurazione Web**: Aggiunta una nuova scheda nell'interfaccia utente per visualizzare e modificare i parametri dell'algoritmo direttamente dal browser. Le modifiche vengono salvate nel file `config.json`.
- 🎨 **Restyling UI**: Migliorata la struttura della pagina con una navigazione a schede.

### v1.1.0 (2025-01-15)
- ✨ Aggiunto Batch Processor per elaborazione multipla (`batch.py`).
- 📊 Log JSON e CSV dettagliati per il processore batch.

### v1.0.0 (2025-01-14)
- 🎉 Release iniziale del servizio web.
- ✅ Algoritmo di riconciliazione base e export Excel.

---

**Versione**: 2.0.0  
**Ultimo aggiornamento**: Dicembre 2025
