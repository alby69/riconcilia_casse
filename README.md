# 🧾 Sistema di Riconciliazione Contabile

Sistema automatico per la quadratura di movimenti contabili (es. incassi vs versamenti) tramite algoritmi di matching ottimizzati.

---

## 📋 Indice

- [Descrizione del Problema](#-descrizione-del-problema)
- [Architettura del Progetto](#-architettura-del-progetto)
- [Come Funziona l'Algoritmo](#-come-funziona-lalgoritmo)
- [Installazione](#-installazione)
- [Utilizzo](#-utilizzo)
- [Parametri di Configurazione](#-parametri-di-configurazione)
- [Formato File Input](#-formato-file-input)
- [Output Generato](#-output-generato)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Descrizione del Problema

L'obiettivo è riconciliare due tipi di movimenti contabili, tipicamente versamenti e incassi.

- **AVERE**: Versamenti registrati (es. 100€ versato in banca)
- **DARE**: Incassi di cassa che dovrebbero corrispondere al versamento

**Obiettivo**: Trovare quali incassi (`DARE`), anche combinati tra loro, corrispondono a ciascun versamento (`AVERE`).

### Esempio pratico

```
DARE:  100€  (versamento del 15/01)

AVERE disponibili:
- 40€  (incasso del 13/01)
- 30€  (incasso del 14/01)  
- 30€  (incasso del 15/01)
- 25€  (incasso del 16/01)

Soluzione: 40 + 30 + 30 = 100€ ✓
```

---

## 🧠 Come Funziona l'Algoritmo

Il sistema utilizza un approccio **multi-step** ottimizzato per massimizzare velocità e accuratezza:

### 1. **Pre-processamento**
```
1. Carica file Excel
2. Separa movimenti DARE da AVERE
3. Ordina per importo (decrescente) → ottimizzazione greedy
4. Aggiungi flag "usato" per evitare duplicati
```

### 2. **Strategia di Matching (per ogni DARE)**

#### Step A: Match Esatto 1:1 
*Complessità: O(n)*

```python
Cerca un singolo AVERE che corrisponda esattamente al DARE
Condizioni:
- |AVERE - DARE| ≤ tolleranza
- Data AVERE compresa tra la data del DARE e `giorni_finestra` giorni successivi
- AVERE non già utilizzato
```

**Veloce ed efficiente** - risolve ~70% dei casi reali

#### Step B: Combinazioni Multiple
*Complessità: O(n^k) con k limitato*

Se il match 1:1 fallisce, cerca combinazioni di più AVERE:

```python
for n in range(2, max_combinazioni):
    for combinazione in combinations(avere_disponibili, n):
        if somma(combinazione) == DARE:
            return combinazione
```

**Ottimizzazioni applicate:**
- Filtra per finestra temporale → riduce spazio di ricerca
- Esclude valori > DARE → impossibili da usare
- Limita a max 10.000 combinazioni per evitare esplosione
- Greedy ordering → prova prima valori grandi

### 3. **Gestione Stato Globale**

```
Quando trovato un match:
1. Marca DARE come "usato"
2. Marca tutti gli AVERE coinvolti come "usati"
3. Registra abbinamento nei risultati
4. Passa al prossimo DARE non riconciliato
```

### 4. **Algoritmo Completo (Pseudocodice)**

```
RICONCILIAZIONE(dare_list, avere_list):
    
    # Ordina per importo decrescente
    dare_list = ORDINA_DESC(dare_list)
    avere_list = ORDINA_DESC(avere_list)
    
    results = []
    
    FOR EACH dare IN dare_list:
        IF dare.usato:
            CONTINUE
        
        # Filtra AVERE per finestra temporale
        avere_finestra = FILTRA_PER_DATA(avere_list, dare.data, ±giorni_finestra)
        avere_finestra = FILTRA_NON_USATI(avere_finestra)
        
        # Step 1: Prova match esatto
        match = TROVA_MATCH_ESATTO(dare, avere_finestra)
        
        # Step 2: Prova combinazioni
        IF match == NULL:
            match = TROVA_COMBINAZIONI(dare, avere_finestra, max_k=6)
        
        # Step 3: Registra risultato
        IF match != NULL:
            MARCA_USATO(dare)
            MARCA_USATO(match.avere_items)
            results.ADD(match)
    
    RETURN results
```

---

## 💻 Installazione

### Requisiti

- Python 3.7+
- Librerie: `pandas`, `openpyxl`, `tqdm`

### Setup

```bash
# 1. Clona o scarica lo script
git clone https://github.com/tuo-repo/riconciliazione-contabile.git
cd riconciliazione-contabile

# 2. Installa dipendenze (incluso tqdm per la barra di avanzamento)
pip install pandas openpyxl tqdm

# 3. Verifica installazione
python riconciliazione.py --help
```

---

## 🚀 Utilizzo

### Uso Base

1. **Prepara il file Excel** con le colonne richieste (vedi sotto)

2. **Modifica il nome del file nello script:**

```python
# Riga ~231 in riconciliazione.py
file_input = "movimenti_gennaio.xlsx"  # <-- TUO FILE
```

3. **Esegui:**

```bash
python riconciliazione.py
```

4. **Trova i risultati** in `riconciliazione_risultati.xlsx`

### Uso Avanzato (Personalizzazione Parametri)

```python
# Modifica righe ~237-241
riconciliatore = RiconciliatoreContabile(
    tolleranza=0.05,        # Accetta differenze fino a 5 centesimi
    giorni_finestra=45,     # Cerca match entro ±45 giorni
    max_combinazioni=8      # Prova fino a 8 elementi combinati
)
```

---

## ⚙️ Parametri di Configurazione

| Parametro | Default | Descrizione | Quando modificarlo |
|-----------|---------|-------------|-------------------|
| `tolleranza` | 0.01 | Differenza massima accettabile in € | Aumenta se hai arrotondamenti sistematici |
| `giorni_finestra` | 30 | Giorni prima/dopo per cercare match | Aumenta se versamenti molto ritardati |
| `max_combinazioni` | 6 | Max elementi da combinare | Riduci se troppo lento (file enormi) |

### 📊 Impatto dei Parametri

```
tolleranza = 0.00  →  Match solo perfetti (più rigoroso)
tolleranza = 0.10  →  Accetta differenze di 10 centesimi (più flessibile)

giorni_finestra = 7   →  Solo match nella stessa settimana (veloce)
giorni_finestra = 90  →  Cerca match anche a 3 mesi di distanza (lento)

max_combinazioni = 3  →  Solo coppie/triple (velocissimo)
max_combinazioni = 10 →  Fino a 10 elementi (molto lento su grandi dataset)
```

---

## 📄 Formato File Input

### Struttura Excel Richiesta

Il file Excel deve contenere **esattamente** queste 3 colonne:

| Data | Dare | Avere |
|------|------|-------|
| 2025-01-15 | 100.00 | |
| 2025-01-14 | | 40.00 |
| 2025-01-15 | | 30.00 |
| 2025-01-16 | | 30.00 |

### Regole

✅ **Fare:**
- Una riga per ogni movimento
- Date in formato standard (YYYY-MM-DD, DD/MM/YYYY, ecc.)
- Importi numerici (anche con decimali)
- Lasciare vuoto il campo non utilizzato (se DARE, lascia Avere vuoto)

❌ **Evitare:**
- Testo negli importi (es. "€ 100")
- Date testuali (es. "15 gennaio")
- Celle unite
- Formule Excel (meglio valori)

### Esempio File Completo

```
Data        | Dare    | Avere
------------|---------|--------
01/01/2025  | 150.00  |
02/01/2025  |         | 50.00
02/01/2025  |         | 100.00
05/01/2025  | 200.50  |
05/01/2025  |         | 200.50
08/01/2025  | 75.30   |
09/01/2025  |         | 25.00
10/01/2025  |         | 50.30
```

---

## 📊 Output Generato

Il file `riconciliazione_risultati.xlsx` contiene **4 fogli**:

### 1. **Abbinamenti** ✓
Tutti i match trovati

| Indice DARE | Data DARE | Importo DARE | N° Avere | Indici AVERE | Importi AVERE | Somma AVERE | Differenza |
|-------------|-----------|--------------|----------|--------------|---------------|-------------|------------|
| 0 | 2025-01-15 | 100.00 | 3 | 5, 12, 18 | 40.00, 30.00, 30.00 | 100.00 | 0.00 |

### 2. **DARE non riconciliati** ⚠️
Versamenti senza corrispondenza trovata

| Indice | Data | Importo |
|--------|------|---------|
| 45 | 2025-01-20 | 250.00 |

### 3. **AVERE non utilizzati** ⚠️
Movimenti di cassa non abbinati

| Indice | Data | Importo |
|--------|------|---------|
| 23 | 2025-01-18 | 15.50 |

### 4. **Statistiche** 📈
Riepilogo performance

| Metrica | Valore |
|---------|--------|
| Totale DARE | 150 |
| DARE riconciliati | 142 |
| % DARE riconciliati | 94.7% |
| AVERE utilizzati | 328 |
| % AVERE utilizzati | 89.3% |

---

## 💡 Esempi Pratici

### Esempio 1: Match Semplice

**Input:**
```
DARE:  50€  (05/01)
AVERE: 50€  (05/01)
```

**Output:**
```
✓ Match esatto 1:1 trovato
```

### Esempio 2: Combinazione Multipla

**Input:**
```
DARE:  100€  (10/01)
AVERE: 
  - 60€  (09/01)
  - 25€  (10/01)
  - 15€  (11/01)
```

**Output:**
```
✓ Match combinato: 60 + 25 + 15 = 100€
Utilizzati 3 elementi in 3 giorni diversi
```

### Esempio 3: Con Tolleranza

**Input:**
```
DARE:  100.00€  (15/01)
AVERE: 99.99€   (15/01)
Tolleranza: 0.01€
```

**Output:**
```
✓ Match con tolleranza: diff = 0.01€
```

---

## 🚄 Ottimizzazione per File Grandi

### File < 1.000 righe
```python
# Configurazione default - funziona perfettamente
tolleranza=0.01
giorni_finestra=30
max_combinazioni=6
```

### File 1.000 - 10.000 righe
```python
# Riduci finestra temporale
tolleranza=0.01
giorni_finestra=15        # ← ridotto
max_combinazioni=5        # ← ridotto
```

### File > 10.000 righe
```python
# Configurazione veloce
tolleranza=0.02
giorni_finestra=7         # ← solo settimana corrente
max_combinazioni=4        # ← max quartetti
```

### Performance Attese

| Righe | Tempo elaborazione | RAM |
|-------|-------------------|-----|
| 500 | ~5 secondi | 50 MB |
| 5.000 | ~45 secondi | 150 MB |
| 50.000 | ~8 minuti | 500 MB |

---

## 🔧 Troubleshooting

### ❌ Errore: "File non trovato"

```bash
# Verifica nome file e percorso
ls -la *.xlsx

# Oppure specifica percorso completo
file_input = "/percorso/completo/movimenti.xlsx"
```

### ❌ Errore: "Il file deve contenere le colonne: Data, Dare, Avere"

Controlla che le colonne abbiano **esattamente** questi nomi (case-sensitive).

Fix:
```
❌ data  →  ✅ Data
❌ DARE  →  ✅ Dare
❌ avere →  ✅ Avere
```

### ⚠️ Pochi match trovati (< 50%)

**Cause comuni:**
1. Finestra temporale troppo stretta → aumenta `giorni_finestra`
2. Tolleranza troppo bassa → aumenta `tolleranza` a 0.05
3. Dati incompleti → verifica file Excel

### 🐌 Script troppo lento

**Soluzioni:**
1. Riduci `max_combinazioni` a 3-4
2. Riduci `giorni_finestra` a 7-14
3. Pre-filtra dati per periodo (es. solo gennaio)

### 📉 RAM insufficiente

Per file giganti (> 100.000 righe):

```python
# Processa a blocchi mensili
for mese in range(1, 13):
    df_mese = df[df['Data'].dt.month == mese]
    # ... elabora singolo mese
```

---

## 🤝 Contributi

Suggerimenti e miglioramenti sono benvenuti!

---

## 📜 Licenza

MIT License - Uso libero per scopi commerciali e non

---

---

## 🚀 Elaborazione Batch (Multipli File)

### Quando usare il Batch Processor

Ideale per:
- ✅ Elaborare file di più punti vendita contemporaneamente
- ✅ Processare dati mensili di tutti i supermercati
- ✅ Automazione notturna/schedulata
- ✅ Analisi comparative tra diversi store

### Setup Cartelle

```
progetto/
├── batch_processor.py
├── riconciliazione.py
├── input/                    ← Metti qui i file da elaborare
│   ├── supermercato_A.xlsx
│   ├── supermercato_B.xlsx
│   └── supermercato_C.xlsx
└── output/                   ← I risultati vengono generati qui
    ├── Riepilogo_Batch_[timestamp].xlsx
    └── logs/
        ├── batch_log_20250115_143022.json
        └── riepilogo_20250115_143022.csv
```

### Uso Base

**1. Crea le cartelle:**
```bash
mkdir input output
```

**2. Copia i file Excel nella cartella `input/`**

**3. Configura parametri** (opzionale - modifica righe ~280-286):
```python
config = {
    'tolleranza': 0.01,
    'giorni_finestra': 30,
    'max_combinazioni': 6,
    'cartella_input': 'input',
    'cartella_output': 'output',
    'pattern': '*.xlsx'  # Elabora tutti gli .xlsx
}
```

**4. Esegui:**
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
📂 Elaborazione: file_corrotto.xlsx
============================================================
  ❌ ERRORE: Il file deve contenere le colonne: Data, Dare, Avere

[3/5] ============================================================
📂 Elaborazione: file_ok.xlsx
============================================================
  ✅ COMPLETATO
...

⚠️  FILE CON ERRORI:
   - file_corrotto.xlsx: Il file deve contenere le colonne...
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