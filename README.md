# CashRec — Riconciliazione Casse

CashRec confronta gli **incassi di cassa** di un punto vendita con i **versamenti in banca**, li abbina tra loro e genera un report Excel che evidenzia le differenze da verificare.

## 📖 Documentazione

- **[Manuale Utente](./docs/MANUALE_UTENTE.md)** — guida semplice per chi usa l'applicazione tutti i giorni (in italiano, integrata anche come help nella web UI e nell'app standalone).
- **[Tutorial Cloudflare Tunnel](./docs/CLOUDFLARE_TUNNEL_TUTORIAL.md)** — come esporre l'applicazione su internet in modo sicuro tramite Cloudflare Tunnel (Raspberry Pi).

## ✨ Funzionalità principali

- **Web UI (Flask)**: interfaccia per caricare il file Excel, modificare i parametri, gestire i profili e scaricare il report.
- **App standalone (`app/cashrec.html`)**: un unico file HTML/JS che gira interamente nel browser, senza server né installazione. I dati non lasciano mai il computer.
- **Elaborazione batch**: script `batch.py` per processare più file automaticamente.
- **Tre algoritmi di riconciliazione**: `progressive_balance` (profilo operatore, default), `subset_sum`, `greedy_amount_first`.
- **Profilo Operatore Punto Vendita**: default preconfigurati per la cassa quotidiana (versamenti abbinati a incassi di 1–5 giorni prima, direzione `past_only`, tolleranza 50 €).
- **SSOT (Single Source of Truth)**: tutte le configurazioni sono centralizzate in `config.json`, condivise da CLI, web UI, app standalone e batch.
- **Gestione profili**: salva, carica ed elimina profili di configurazione dalla Web UI o via REST API.
- **Recupero residui**: recupera automaticamente le differenze dai blocchi forzati.
- **Multi-negozio**: colonna opzionale *Codice Negozio* per abbinamenti prioritari all'interno dello stesso negozio.
- **Data Valuta**: gestisce i passaggi di fine anno (versamenti di gennaio che si riferiscono a dicembre).
- **Report Excel dettagliato**: fogli Summary, Matches, Anomalie, Original, Quadratura Mensile, Unused DEBIT e Unreconciled CREDIT, con importi in euro (`#,##0.00 €`), colori per stato e totali mensili.

## 📚 Come funzionano gli algoritmi

### Concetti base

- **Dare / DEBIT** — denaro entrato in cassa (scontrini, fatture, ricevute).
- **Avere / CREDIT** — denaro versato in banca.
- **Abbinamento (match)** — l'unione di uno o più incassi con un versamento che li copre.
- **Tolleranza** — differenza massima (in €) accettata come "uguale".

### Progressive Balance (profilo operatore, default)

Simula il comportamento umano: procede cronologicamente e abbina ogni versamento con gli incassi presenti **negli N giorni prima** (di default 5).

```
1. Data_Analisi = Data_Valuta (se presente) altrimenti Data Registrazione
2. Ordinamento per Data_Analisi crescente
3. Per ogni versamento (CREDIT):
   - cerca gli incassi (DEBIT) non usati dentro la finestra di giorni nel passato
   - totale DEBIT >= CREDIT  → match (con uso parziale se serve)
   - totale DEBIT < CREDIT entro la tolleranza → match tollerato
   - totale DEBIT < CREDIT oltre la tolleranza → ANOMALIA (il residuo NON passa al versamento successivo)
4. I movimenti usati vengono marcati
```

Esempio: versamento di **150 €** del 10/01 con `days_window=5` e `past_only`:
- cerca incassi dal 05/01 al 10/01;
- trova 100 € + 50 € = 150 € → **MATCH** ✅;
- trova solo 80 € → **ANOMALIA** di 70 € (residuo non trasferito).

### Subset Sum

Cerca abbinamenti per **combinazioni** di importi, in 3 passate: aggregazione di molti incassi su un versamento, scomposizione su più versamenti, recupero residui con finestra estesa.

### Greedy Amount First

Ordina i movimenti per importo decrescente e abbina prima gli importi più grandi.

## ⚙️ Installazione e test

Prerequisiti: **Python 3.9+** e **Git**.

```bash
git clone <URL_DEL_REPOSITORY>
cd riconcilia_casse
pip install -r requirements.txt
./run_tests.sh
```

## 🚀 Utilizzo

### Web UI (server Flask)

```bash
python app.py
# http://localhost:5001
```

### App standalone (100% nel browser)

Apri `app/cashrec.html` con un doppio clic (o trascinala nel browser). Nessuna installazione richiesta.

### Docker

```bash
docker compose up -d --build
# http://localhost:5000
```

### CLI su un singolo file

```bash
python main.py --config config.json
```

### API principali

| Endpoint | Descrizione |
|---|---|
| `GET /api/config` | Configurazione corrente |
| `POST /api/config` | Aggiorna la configurazione |
| `GET /api/profiles` | Elenco profili |
| `POST /api/profiles` | Salva un profilo |
| `DELETE /api/profiles/<nome>` | Elimina un profilo |

## 🔧 Parametri di configurazione (`config.json`)

### Mappatura colonne (file in stile SAN SEVERO)

| Campo | Colonna del file |
|---|---|
| Data | `Data Reg.` |
| Dare (Incassi) | `Dare` |
| Avere (Versamenti) | `Avere` |
| Data Valuta | `Data Val.` |
| Codice Negozio | (opzionale) |

### Parametri comuni

| Parametro | Default (Operatore) | Descrizione |
|---|---|---|
| `algorithm` | `progressive_balance` | Strategia di riconciliazione |
| `tolerance` | `50.0 €` | Differenza massima accettata |
| `days_window` | `5 giorni` | Finestra temporale di ricerca |
| `search_direction` | `past_only` | Direzione di ricerca (`past_only`, `future_only`, `both`) |
| `max_combinations` | `10` | Massimo numero di elementi combinati (subset_sum) |
| `residual_threshold` | `50.0 €` | Soglia per il recupero dei residui |
| `residual_days_window` | `5 giorni` | Finestra estesa per i residui |
| `handover_days` | `5` | **Giorni Finestra Lasca (handover)**: giorni del mese successivo riportati al mese precedente nella Quadratura Mensile (es. versamenti di gennaio da attribuire a dicembre) |

## 📂 Struttura del progetto

```
├── core.py              # ReconciliationEngine (logica e algoritmi)
├── reporting.py         # Generazione del report Excel
├── app.py               # Web UI Flask e REST API
├── main.py              # CLI su singolo file
├── batch.py             # Elaborazione batch
├── config.json          # Configurazione (Single Source of Truth)
├── profiles.json        # Profili di configurazione salvati
├── templates/           # Template della Web UI
├── app/                 # App standalone (cashrec.html e asset)
├── tests/               # Suite di unit test
├── tools/               # Script di supporto per sviluppatori
└── .github/workflows/   # Workflow CI
```

## 📜 Changelog

### v5.3 (September 2026)
- **Rimozione "Ottimizza Parametri"**: il pulsante non faceva nulla di sostanziale (restituiva i valori di default) e il backend richiedeva la libreria `optuna` non installata. Eliminato da Web UI, app standalone, `app.py`, `optimizer.py` e `config.json`. I parametri si impostano manualmente nelle Impostazioni Avanzate.
- **Giorni Finestra Lasca (handover)**: campo ora presente e valorizzato (default 5) in Web UI, app standalone e default `config.json`; aggiunto anche ai profili predefiniti.
- **Documentazione semplificata**: ridotti i file a 2 guide (manuale utente + tutorial Cloudflare Tunnel) e tradotto tutto il resto in italiano.

### v5.2 (September 2026)
- **Quadratura Mensile**: riepilogo mensile semplificato (Mese, Dare, Avere, Δ, Cumulato, Stato OK/Controllare) al posto del vecchio "Monthly Balance", che sommava importi su basi incoerenti. Rimosso il fuorviante "Vers. Non Agganciati".
- **Original Sheet**: aggiunta la colonna `Data Valuta` (competenza dell'aggancio Dare/Avere) e formato euro `#,##0.00 €` (`.` migliaia, `,` decimali).
- **Default Column Mapping**: `Data Reg.`→Data, `Dare`→Dare, `Avere`→Avere, `Data Val.`→Data Valuta (Codice Negozio opzionale), in `config.json` e nelle UI.

### v5.1 (March 2026)
- **Default Operatore Punto Vendita**: parametri ottimizzati per la cassa quotidiana.
- **Single Source of Truth**: configurazione centralizzata in `config.json`.
- **Gestione profili**: salvataggio e applicazione di profili nominativi da Web UI e REST API.
- **Hardening del motore**: abbinamenti esatti deterministici per prossimità di data, elaborazione senza side-effect.
- **Sicurezza**: chiavi segrete da ambiente, limiti di dimensione upload, pulizia automatica dei report generati, workflow CI.