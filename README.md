# Sistema di Riconciliazione Contabile Automatico

Questo progetto fornisce un sistema automatico per la riconciliazione di movimenti contabili (Dare/Avere) da file Excel o CSV. Il sistema è progettato per essere robusto, flessibile e ottimizzato, grazie a un processo di elaborazione in batch che include una fase di ottimizzazione automatica dei parametri per ogni singolo file.

## ✨ Caratteristiche Principali

- **Elaborazione in Batch**: Processa automaticamente tutti i file presenti nella cartella `input/`.
- **Conversione Automatica a Feather**: I file Excel (`.xlsx`, `.xls`) vengono automaticamente convertiti nel formato ad alte prestazioni Apache Feather per accelerare drasticamente i tempi di lettura durante le elaborazioni.
- **Ottimizzazione Adattiva dei Parametri (`optimizer.py`)**: Per ogni file, il sistema non usa parametri fissi, ma esegue un'ottimizzazione automatica per trovare la configurazione migliore.
  - **Range Dinamici**: Invece di cercare i parametri in range ampi e fissi (es. `giorni_finestra` da 1 a 100), l'ottimizzatore crea un intervallo di ricerca mirato attorno ai valori di partenza (es. `50 ± 30%`, quindi tra 35 e 65).
  - **Approccio Evolutivo**: I parametri ottimali trovati vengono salvati. Nelle esecuzioni successive, l'ottimizzazione partirà da questa base migliore, affinando la ricerca in aree sempre più promettenti. Questo rende il processo più veloce ed efficace a ogni ciclo.
- **Supporto Multi-Formato**: Gestisce file Excel moderni (`.xlsx`), legacy (`.xls`) e CSV (`.csv`).
- **Architettura Modulare (KISS)**: La logica di business è incapsulata nel "motore" (`core.py`), mentre gli altri script agiscono come orchestratori o wrapper, seguendo la filosofia *Keep It Simple, Stupid*.
- **Report Dettagliati**: Per ogni file elaborato, genera una cartella dedicata in `output/` contenente:
 - **Ottimizzazioni di Performance Avanzate**: Il motore di riconciliazione (`core.py`) impiega `dask.bag` per l'elaborazione parallela dei movimenti, algoritmi ricorsivi ottimizzati con memoization (caching) e pruning per la ricerca efficiente di combinazioni, e una gestione accurata degli indici dei movimenti già utilizzati per prevenire ricalcoli.
 - **Feedback in Tempo Reale**: Durante l'elaborazione, viene mostrato un avanzamento a console per tenere traccia del progresso.
 - **Report Dettagliati**: Per ogni file elaborato, genera una cartella dedicata in `output/` contenente:
  - Un file di configurazione (`config.json`) con i parametri ottimali trovati.
  - Un report Excel (`risultato_*.xlsx`) con i dettagli degli abbinamenti, i movimenti non riconciliati e statistiche complete.
- **Riepilogo Aggregato**: Al termine del processo batch, stampa a console un riepilogo globale con le statistiche aggregate di tutti i file elaborati.

## ⚙️ Prerequisiti e Installazione

1.  **Python**: Assicurati di avere Python 3.8 o superiore installato.
2.  **Creazione e Attivazione dell'Ambiente Virtuale (Consigliato)**: È buona norma creare un ambiente virtuale per isolare le dipendenze del progetto. Apri il terminale nella cartella principale del progetto ed esegui:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Su Windows: .venv\Scripts\activate
    ```
3.  **Installazione Dipendenze**: Con l'ambiente virtuale attivato, installa tutte le librerie necessarie tramite il file `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## 📁 Struttura del Progetto

```
/
├── input/                  # Cartella per i file di input da analizzare
├── output/                 # Cartella principale per tutti i risultati
│   └── [nome_file]/        # Sottocartella per ogni file elaborato
│       ├── config.json     # Configurazione ottimizzata per questo file
│       └── risultato_...   # Report Excel dettagliato
├── batch.py                # ✅ Script principale per avviare l'elaborazione
├── convert_to_feather.py   # Utility per convertire Excel in Feather per performance
├── profile_core.py         # Script per la profilazione delle performance di core.py
├── main.py                 # Esecutore della singola riconciliazione
├── core.py                 # Motore di riconciliazione (logica di business)
├── optimizer.py            # Ottimizzatore dei parametri
├── config.json             # File di configurazione di base
├── requirements.txt        # Lista delle dipendenze Python
└── README.md               # Questo file
```

## 🚀 Funzionamento del Programma

L'intero sistema è orchestrato dallo script `batch.py`. Quando viene eseguito, segue un flusso di lavoro preciso e robusto per ogni file trovato nella cartella `input/`.

### Diagramma delle Relazioni tra i File

Il diagramma seguente illustra come i vari componenti interagiscono tra loro.

```mermaid
graph TD
    subgraph "Utente"
        A[Avvio: python batch.py]
        P[Profilazione: python profile_core.py]
    end

    subgraph "Processo Batch (batch.py)"
        B{Trova file in 'input/'}
        C[Crea cartella in 'output/']
        D[Copia e aggiorna config.json locale]
        E[Chiama optimizer.py]
        F[Chiama main.py con config ottimizzata]
        G[Aggrega statistiche globali]
    end

    subgraph "Ottimizzazione (optimizer.py)"
        H[Legge config locale]
        I{Esegue N simulazioni}
        J[Scrive parametri migliori in config locale]
    end

    subgraph "Esecuzione (main.py)"
        K[Legge config locale ottimizzata]
        L[Chiama il motore di riconciliazione]
        M[Stampa statistiche JSON]
    end

    subgraph "Motore (core.py)"
        N[Classe RiconciliatoreContabile]
        O[Metodo .run()]
    end

    A --> B
    B -- Per ogni file --> C
    B -- Per ogni file --> X
    X -- Input Feather/CSV --> C
    C --> D
    D --> E
    E --> H
    H -- Chiama --> N
    I -- Usa --> O
    I --> J
    E -- Ritorna a --> F
    E -- Ritorna a --> F -- Input Feather --> K
    F --> K
    K --> L
    L -- Chiama --> O
    O -- Ritorna stats --> M
    F -- Ritorna a --> G
    P --> N
    P --> O
```

### Descrizione del Flusso

1.  **Avvio (`batch.py`)**: L'utente lancia `python ./batch.py`. Lo script legge il `config.json` di base.
2.  **Ciclo sui File**: Per ogni file in `input/` (es. `cassa.xlsx`):
    1.  **Setup**: Crea una cartella dedicata `output/cassa/`.
    2.  **Configurazione Locale**: Copia il `config.json` di base in `output/cassa/` e lo aggiorna con i percorsi di input e output specifici per `cassa.xlsx`.
    3.  **Ottimizzazione Adattiva (`optimizer.py`)**: `batch.py` esegue `optimizer.py`, passandogli il percorso del `config.json` locale.
        - L'ottimizzatore (`optimizer.py`) legge i parametri di partenza dal `config.json` locale.
        - **Crea dinamicamente un intervallo di ricerca** attorno a questi valori (es. `giorni_finestra: 50` diventa un range di ricerca tra `35` e `65`).
        - Utilizzando la libreria `Optuna`, esegue decine di simulazioni in parallelo per testare diverse combinazioni di parametri all'interno di questo intervallo mirato.
        - Trovata la combinazione migliore, **aggiorna il file `output/cassa/config.json`** con i parametri ottimali.
    4.  **Esecuzione Finale (`main.py`)**: `batch.py` esegue `main.py` usando il `config.json` locale, che ora contiene i parametri ottimizzati.
        - `main.py` istanzia il motore `RiconciliatoreContabile` con i parametri ottimali e lancia il processo finale.
        - Il motore esegue la riconciliazione e salva il report Excel dettagliato in `output/cassa/risultato_cassa.xlsx`.
        - Il motore di riconciliazione (`core.py`) ora beneficia di ottimizzazioni come l'elaborazione parallela tramite Dask e algoritmi di ricerca combinatoria con memoization e pruning.
        - `main.py` stampa le statistiche finali in formato JSON.
    5.  **Aggregazione**: `batch.py` cattura le statistiche JSON e le aggiunge a un riepilogo globale.
3.  **Riepilogo Finale**: Al termine del ciclo, `batch.py` stampa un riepilogo aggregato di tutti i file elaborati con successo.

## 🛠️ Utilizzo

1.  Popola la cartella `input/` con i file Excel o CSV che desideri analizzare.
2.  Esegui lo script principale dal terminale:
    ```bash
    python ./batch.py
    ```
3.  Attendi il completamento del processo. I risultati per ogni file saranno disponibili nelle rispettive sottocartelle dentro `output/`.

### Esecuzione in Modalità Sequenziale (per CPU meno potenti)

L'ottimizzatore dei parametri (`optimizer.py`) è progettato per sfruttare tutti i core della CPU ed eseguire le simulazioni in parallelo. Questo accelera notevolmente la ricerca dei parametri migliori.

Tuttavia, su macchine con risorse limitate o in caso di problemi di stabilità, l'esecuzione parallela intensiva potrebbe causare blocchi o un'eccessiva lentezza del sistema. In questi casi, è possibile forzare l'ottimizzatore a lavorare in modalità **sequenziale** (un test alla volta) usando il flag `--sequential-optimizer`.

Esegui lo script con il flag in questo modo:

```bash
python ./batch.py --sequential-optimizer
```

Questa modalità sarà più lenta, ma garantirà una maggiore stabilità su sistemi con poche risorse.

### Esecuzione Iniziale (Prima Ottimizzazione)

La prima volta che esegui il sistema su un nuovo file, è fondamentale capire come funziona l'ottimizzazione automatica:

- **Esplorazione Ampia**: Durante la prima esecuzione, il sistema ignora i valori di partenza nel `config.json` principale e utilizza i range definiti nel `config_optimizer.json` per esplorare un'ampia gamma di possibilità. Questo assicura che, anche se i valori iniziali sono errati, il sistema troverà una combinazione di parametri promettente.
- **Flag `--first-run`**: Per indicare che è la prima esecuzione, devi aggiungere il flag `--first-run` quando lanci `optimizer.py` da `batch.py`. Questo flag attiva la modalità di esplorazione ampia.

Esempio di esecuzione (il flag `--first-run` è gestito automaticamente da `batch.py`):

```bash
python ./batch.py
```

## 🔧 Configurazione (`config.json`)

Il file `config.json` alla radice del progetto serve come modello di base per tutte le elaborazioni. I parametri più importanti sono:

- `tolleranza`: La massima differenza di importo accettata per un abbinamento.
- `giorni_finestra`: La finestra temporale (in giorni) per la ricerca di abbinamenti.
- `max_combinazioni`: Il numero massimo di movimenti da combinare per trovare un abbinamento.
- `sorting_strategy`: Come ordinare i movimenti prima della riconciliazione (`date` o `amount`).
- `search_direction`: Direzione della ricerca temporale (`future_only`, `past_only`, `both`).

Questi parametri verranno ottimizzati automaticamente per ogni file, ma i valori di base forniscono un punto di partenza per l'ottimizzatore.