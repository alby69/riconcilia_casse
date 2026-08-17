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
