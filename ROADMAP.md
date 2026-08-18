# 🗺️ Roadmap di Miglioramento Progetto Riconciliazione Casse

Questo documento descrive la roadmap strategica ed esecutiva per l'evoluzione e il consolidamento del progetto **CashRec (Riconciliazione Casse)**.

---

## 🎯 Obiettivi Principali
1. **Profilo Predefinito Operatore Punto Vendita**: Ottimizzare i parametri di default per l'operatore di cassa (versamenti relativi ad incassi di 1-5 giorni prima, ricerca retroattiva `past_only`, finestra max 5 giorni, tolleranza 50€).
2. **Single Source of Truth (SSOT)**: Unificare tutte le configurazioni predefinite in `config.json` e sincronizzarle con `core.py`, `app.py`, `main.py` e `batch.py`.
3. **Persistenza & Gestione Profili**: Consentire il salvataggio e il caricamento di profili di configurazione dalla Web UI e la modifica persistente di `config.json`.
4. **Hardening del Motore di Riconciliazione**: Eliminare side-effect su dizionari, rendere deterministica la scelta dei match esatti basandosi sulla prossimità temporale.
5. **Sicurezza Web & Pulizia File**: Gestione sicura delle chiavi segrete via `.env`, limiti upload, e auto-cleanup dei file generati in `output/` e `log/`.
6. **Igiene del Codice & CI**: Aggiornare la suite di unit test, eliminare i file legacy duplicati e aggiungere workflow di CI automatizzato.

---

## 🚀 Fasi di Implementazione

### Fase 1: Centralizzazione Configurazioni (SSOT & Profilo Operatore)
- [x] Configurazione di `config.json` con default allineati al profilo "Operatore Punto Vendita":
  - `algorithm`: `progressive_balance`
  - `days_window`: 5
  - `tolerance`: 50.0 €
  - `search_direction`: `past_only`
- [x] Sincronizzazione di `ReconciliationEngine.__init__` in `core.py` con i default SSOT.
- [x] Refactoring di `app.py`, `main.py`, `batch.py` per leggere dinamicamente la configurazione centralizzata da `config.json`.

### Fase 2: Hardening del Motore (Core Engine)
- [x] Scelta deterministica in `_find_matches`: in presenza di più match 1-a-1 esatti, scegliere la transazione più vicina per data anziché la prima in lista.
- [x] Risoluzione mutazione in-place dei dizionari candidati in `_find_matches` / `_find_combinations_recursive_py` per evitare alterazioni durante il backtracking.
- [x] Verifica e ottimizzazione della modalità `algorithm="auto"`.
- [x] Nel `progressive_balance` la finestra temporale ora rispetta la `search_direction` (`_calculate_time_window`): con `past_only` un versamento può essere agganciato solo a incassi **precedenti o dello stesso giorno**, mai a incassi successivi (prima la finestra era simmetrica ±days_window anche con `past_only`).

### Fase 3: Gestione Profili & Interfaccia Web
- [x] Endpoint backend in Flask (`/api/config`, `/api/profiles`) per lettura, salvataggio e applicazione profili.
- [x] Interfaccia Web UI (`templates/index.html`): selettore di profili, salvataggio nuove impostazioni e ripristino default.

### Fase 4: Sicurezza Web & Gestione Risorse
- [x] Lettura dinamica di `SECRET_KEY` da ambiente (`os.environ`) con fallback sicuro.
- [x] Configurazione limite massimo upload (`MAX_CONTENT_LENGTH = 50MB`).
- [x] Routine automatica di pulizia file temporanei vecchi (>24h) in `output/` and `log/`.
- [x] Creazione di `.env.example` e verifica presenza `.env` in `.gitignore`.

### Fase 5: Suite di Test, Consolidation & CI
- [x] Aggiornamento ed estensione della suite di test in `tests/test_riconciliazione.py` per l'API corrente (inclusi test per la finestra `past_only` a 5 giorni e abbinamento per data più vicina).
- [x] Rimozione / consolidamento script legacy a livello root (`test_riconciliazione.py` e `riconciliazione.py`).
- [x] Creazione del workflow GitHub Actions (`.github/workflows/ci.yml`).

### Fase 6: Documentazione
- [x] Aggiornamento di `README.md` e delle guide in `docs/` con le nuove funzionalità dei profili e le istruzioni aggiornate.

### Fase 8: App Standalone Client-Side in HTML/JS (`app/cashrec.html`)
- [x] Creazione della cartella `app/` e realizzazione dell'applicazione web monofile standalone `cashrec.html`.
- [x] Implementazione dell'interfaccia grafica identica (Bootstrap 5, FontAwesome, supporto tema chiaro/scuro, drag & drop, mapping colonne e profili).
- [x] Inserimento delle librerie `SheetJS` e `ExcelJS` da CDN per la lettura/scrittura dei file Excel direttamente nel browser client-side.
- [x] Replicazione del motore di riconciliazione (`JSReconciliationEngine`) con aritmica in centesimi, algoritmi `progressive_balance`, `subset_sum`, `greedy_amount_first`, supporto `valuta_date`, `past_only` window e `smart residual recovery`.
- [x] Generazione completa dei report Excel multi-foglio (`Summary`, `MANUAL`, `Matches`, `Anomalie`, `Unused DEBIT`, `Unreconciled CREDIT`, `Original` con colorazione a 3 gruppi, split righe e subtotali mensili, `Monthly Balance`).
- [x] Gestione profili e configurazioni salvate in `LocalStorage`.

### Fase 7: Quadratura Visiva nel foglio Original (Color Grouping & Delta)
- [x] Colorazione delle celle delle colonne `Debit` e `Credit` nel foglio `Original` in base ai gruppi di abbinamento del foglio `Matches`, con ciclo di 3 colori (es. `D(3,4) A(7)` → celle dello stesso colore). **Incassi scomposti su più versamenti**: la riga originale mostra la quota consumata dal primo versamento; sotto di essa viene inserita una nuova riga (stessa data, `Saldo Prog.` vuoto) per ogni quota residua, ciascuna col colore del proprio gruppo — così ogni gruppo ha celle tutte dello stesso colore (es. cella 4 di 2777,00 → riga originale 464,50 colore 1 `D(3,4)_A(7)` + riga inserita 2312,50 colore 2 `D(4,5,6)_A(9)`). I totali Debit/Credit del foglio restano invariati.
- [x] Aggiunta delle colonne `Gruppo` (Transaction ID es. `D(3,4)_A(7)`) e `Difference` (Δ in €) nel foglio `Original` per visualizzare il delta di ogni gruppo e ricollegarlo al foglio `Matches` (cella `Difference` colorata con tonalità più scura dello stesso gruppo, evidenziata in rosso se Δ > 0).
- [x] Analisi della colonna `Saldo Prog.` (contanti progressivi della cassa del punto vendita): verifica di coerenza tra saldo dichiarato e cassa teorica cumulata (Dare − Avere) con colonna di controllo nel foglio `Original`, KPI di cassa iniziale/finale e anomalie nella `Summary`, e legenda esplicativa nel foglio `Original`.
- [x] **Importi residui nel progressive balance**: negli `debit_amounts` dei match viene registrato solo l'importo **effettivamente consumato** (residuo) di ciascun incasso, non l'importo originale. Un incasso già parzialmente impegnato da un versamento precedente entra nei match successivi con il solo residuo disponibile (es. cella 4 di 2777,00 impegnata per 464,50 in `D(3,4)_A(7)` → residuo 2312,50 in `D(4,5,6)_A(9)`). `total_debit`, `difference` e il "Saldo Assorbito" del bilanciamento mensile risultano così corretti.
- [x] **Indici post-split**: i Transaction ID (es. `D(3,4)_A(9)`) e le colonne `debit_indices`/`credit_indices` del foglio `Matches` ora riferiscono i **numeri di riga del foglio `Original` dopo lo sdoppiamento** (header escluso), non più le righe del file sorgente. Ogni riga appartiene a un gruppo i cui indici includono il proprio numero di riga (verificato: 0 violazioni su LA_PERLA). Implementato in `reporting.py` con `_build_original_layout` (layout con `portion_rows` orig_index→righe foglio per ciascuna porzione) e `_remap_matches_to_sheet_rows` (aggiorna Transaction ID + colonne display chiamato da `generate_report`; `_create_matches_sheet` visualizza le colonne display senza offset +2).
- [x] **Totali mensili nel foglio `Original`**: al cambio mese viene inserita una riga di subtotale con `Date` = fine mese, `Debit`/`Credit` = somme del mese, `Difference` = squadratura mensile (Dare − Avere, con segno) e `Gruppo` = `TOTALE MESE <NomeMese> <Anno>` (es. `TOTALE MESE Gennaio 2026`), stilizzata in grassetto su fondo grigio. I totali sono inseriti direttamente in `_build_original_layout` così anche gli indici post-split dei mesi successivi restano allineati (0 violazioni verificate).
