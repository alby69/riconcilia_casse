# ROADMAP — CashRec Standalone

> **Documento di lavoro per Google Jules.** Questo file descrive (A) il refactoring da eseguire subito per trasformare il progetto in un'app 100% client-side, e (B) la roadmap di evoluzione futura, organizzata per fasi con priorità e impatto UX/UI. Segui le fasi in ordine. Dopo ogni fase apri una PR separata, verifica che `app/cashrec.html` funzioni ancora aprendolo da file system (nessun server), e aggiorna la sezione "Stato" in fondo a questo documento.

---

## 0. Contesto e obiettivo

Il repository `riconcilia_casse` nasce come servizio web Flask + CLI + Docker, ma include già da tempo una versione standalone completa e funzionante in `app/cashrec.html`: un singolo file HTML che usa `xlsx.js` ed `ExcelJS` via CDN, esegue tutta la logica di riconciliazione nel browser e salva configurazione/profili in `localStorage`. **Non contatta mai un backend.**

**Obiettivo di questa roadmap:** eliminare tutta la parte server/Docker/Python e mantenere solo `app/cashrec.html` come prodotto, poi far evolvere quest'unico file (o il piccolo bundle in cui eventualmente verrà diviso) in un'app più robusta e con un'interfaccia più curata.

**Vincolo non negoziabile per tutte le fasi:** i risultati della riconciliazione (match, anomalie, quadratura mensile) devono restare identici a quelli attuali. Qualunque refactoring del motore di calcolo deve essere accompagnato da test che confrontino l'output prima/dopo su almeno i file di esempio disponibili in `input/` (se presenti) o su casi sintetici costruiti ad hoc.

---

## FASE 0 — Refactoring "Standalone Only" (da eseguire per prima)

### 0.1 File e cartelle da **rimuovere**

Tutto ciò che serve solo al server Flask, al CLI Python o a Docker:

| Percorso | Motivo |
|---|---|
| `app.py` | Server Flask + REST API (`/api/config`, `/api/profiles`), non più necessario |
| `main.py` | CLI per singolo file (Python) |
| `batch.py` | Elaborazione batch (Python) |
| `core.py` | Motore di riconciliazione Python — la logica equivalente vive già in JS dentro `app/cashrec.html` |
| `reporting.py` | Generazione report Excel lato server |
| `requirements.txt` | Dipendenze Python |
| `config.json` | Config lato server (l'app standalone usa `localStorage`) |
| `profiles.json` | Profili lato server (l'app standalone gestisce i profili in `localStorage`) |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | Containerizzazione non più necessaria |
| `deploy.sh`, `start.sh` | Script di avvio/deploy del server |
| `.env`, `.env.example` | Variabili Flask/Gunicorn (`SECRET_KEY`, `GUNICORN_WORKERS`, ecc.) |
| `templates/` (intera cartella) | Template Flask (`index.html` server-side) |
| `tests/` (intera cartella, `test_riconciliazione.py`) | Test del motore Python |
| `tools/algorithms.py` | Utility Python legata al motore server |
| `tools/analizza_log.py` | Analisi log del server Flask |
| `tools/convert_to_feather.py` | Conversione dati per uso interno server |
| `static/` (intera cartella) | Asset serviti da Flask (`url_for('static', ...)`), usati solo da `templates/index.html`. `app/cashrec.html` ha già logo e banner inline in SVG, quindi non dipende da questi file |
| `docs/CLOUDFLARE_TUNNEL_TUTORIAL.md` | Guida per esporre il server Flask su internet: non ha più senso senza server |
| `app_output.log`, `.DS_Store` | File generati/di sistema, non versionabili |
| `.github/workflows/ci.yml` | Pipeline CI basata su Python/pytest — va sostituita (vedi 0.3) |

### 0.2 File da **mantenere**

| Percorso | Note |
|---|---|
| `app/cashrec.html` | Il prodotto. Nessuna modifica funzionale in questa fase, solo eventuali fix ai link interni rotti dopo la rimozione di file collegati |
| `docs/MANUALE_UTENTE.md` | Manuale utente, già iniettato in `cashrec.html` tra i marcatori `<!-- HELP_MANUAL_START -->` / `<!-- HELP_MANUAL_END -->` |
| `tools/generate_help.py` | Script di sviluppo che rigenera l'help inline in `cashrec.html` a partire da `docs/MANUALE_UTENTE.md`. Non è un server: gira una tantum, offline, lato manutentore. Aggiornalo per puntare **solo** a `app/cashrec.html` (rimuovi il riferimento a `templates/index.html` che non esiste più) |
| `README.md` | Da riscrivere (vedi 0.4) |
| `.gitignore` | Da semplificare (rimuovi le voci relative a Python/venv/`.env`, tieni `.DS_Store`, `.idea/`, `.vscode/`, `node_modules/` se in Fase 3 si introduce un build step) |

Facoltativo: se una delle immagini in `static/img/` (es. `cashrec-banner-it.png`) viene usata come social preview del repo su GitHub, spostala in una nuova cartella `assets/` prima di cancellare `static/`.

### 0.3 Struttura finale attesa

```
riconcilia_casse/
├── README.md
├── ROADMAP.md
├── .gitignore
├── app/
│   └── cashrec.html
├── docs/
│   └── MANUALE_UTENTE.md
├── assets/                     # opzionale, solo se serve un'immagine per il social preview
│   └── cashrec-banner-it.png
└── tools/
    └── generate_help.py        # opzionale, solo dev-time
```

Aggiorna `.github/workflows/ci.yml` (o rimuovilo se non si vuole ancora CI) sostituendo la pipeline Python con una pipeline minima adatta a un progetto HTML/JS statico, ad es.: validazione HTML (`html-validate` o `htmlhint`), e in Fase 4 test end-to-end con Playwright. Non serve più `setup-python`, `pip install`, `pytest`.

### 0.4 README.md — nuovi contenuti

Riscrivi `README.md` mantenendo tono e lingua italiana attuali, ma:

1. Rimuovi ogni riferimento a Flask, Docker, `docker compose`, endpoint REST, CLI Python, `batch.py`, `optimizer.py`.
2. La sezione "Utilizzo" deve contenere **solo**: *"Apri `app/cashrec.html` con un doppio clic, oppure trascinalo nel browser. Nessuna installazione, nessun server, i dati non lasciano mai il tuo computer."*
3. Aggiorna "Struttura del progetto" con l'albero minimale di cui sopra.
4. **Sposta l'intera sezione `## 📜 Changelog` (storico versioni v3.0 → v5.3) in questo file ROADMAP.md**, nella sezione "Storico Versioni" qui sotto, e lascia nel README solo un link: `Per lo storico delle versioni e la roadmap di sviluppo, vedi [ROADMAP.md](./ROADMAP.md)`.
5. Tieni la spiegazione degli algoritmi (Progressive Balance, Subset Sum, Greedy Amount First) e i parametri di configurazione: sono ancora validi, l'app standalone li implementa 1:1.

### 0.5 Checklist di accettazione Fase 0

- [ ] `app/cashrec.html` si apre da file system (`file://`) e completa un ciclo upload → elaborazione → download report senza errori in console.
- [ ] Nessun file Python, `Dockerfile`, `docker-compose.yml` rimane nel repo.
- [ ] `docs/MANUALE_UTENTE.md` risulta ancora correttamente iniettato nell'help modal di `cashrec.html`.
- [ ] `README.md` non contiene più istruzioni Docker/Flask/CLI.
- [ ] Storico versioni presente in `ROADMAP.md` (sezione 1 qui sotto), rimosso dal README.
- [ ] CI (se mantenuta) non referenzia più Python/pytest.

---

## 1. Storico Versioni (spostato dal README)

### v5.3 (Settembre 2026)
- **Rimozione "Ottimizza Parametri"**: il pulsante non faceva nulla di sostanziale (restituiva i valori di default) e il backend richiedeva la libreria `optuna` non installata. Eliminato da Web UI, app standalone, `app.py`, `optimizer.py` e `config.json`. I parametri si impostano manualmente nelle Impostazioni Avanzate.
- **Giorni Finestra Lasca (handover)**: campo ora presente e valorizzato (default 5) in Web UI, app standalone e default `config.json`; aggiunto anche ai profili predefiniti.
- **Documentazione semplificata**: ridotti i file a 2 guide (manuale utente + tutorial Cloudflare Tunnel) e tradotto tutto il resto in italiano.

### v5.2 (Settembre 2026)
- **Quadratura Mensile**: riepilogo mensile semplificato (Mese, Dare, Avere, Δ, Cumulato, Stato OK/Controllare) al posto del vecchio "Monthly Balance", che sommava importi su basi incoerenti. Rimosso il fuorviante "Vers. Non Agganciati".
- **Original Sheet**: aggiunta la colonna `Data Valuta` (competenza dell'aggancio Dare/Avere) e formato euro `#,##0.00 €` (`.` migliaia, `,` decimali).
- **Default Column Mapping**: `Data Reg.`→Data, `Dare`→Dare, `Avere`→Avere, `Data Val.`→Data Valuta (Codice Negozio opzionale), in `config.json` e nelle UI.

### v5.1 (Marzo 2026)
- **Default Operatore Punto Vendita**: parametri ottimizzati per la cassa quotidiana.
- **Single Source of Truth**: configurazione centralizzata in `config.json`.
- **Gestione profili**: salvataggio e applicazione di profili nominativi da Web UI e REST API.
- **Hardening del motore**: abbinamenti esatti deterministici per prossimità di data, elaborazione senza side-effect.
- **Sicurezza**: chiavi segrete da ambiente, limiti di dimensione upload, pulizia automatica dei report generati, workflow CI.

### v5.0
- **Data Valuta**: gestione dei passaggi di fine anno (versamenti di gennaio riferiti a dicembre).
- **Data Analisi**: colonna calcolata automaticamente (valuta se presente, altrimenti data di registrazione).
- **Column Mapping**: supporto completo per nomi di colonna personalizzati via web UI.

### v4.0
- **Smart Residual Recovery**: nuova fase che recupera le differenze dai blocchi forzati.
- **Capienza Logic**: supporto per abbinamenti stile GDO (credito ≥ debiti).
- **Multi-Store Support**: nuovo parametro `store_id_column` per abbinamenti a livello di negozio.

### v3.1.0
- Optimizer con `sorting_strategy`.
- Ottimizzazione Docker/Gunicorn.
- Miglioramenti al grafico Monthly Performance.

### v3.0.0
- Riscrittura completa con Pandas.
- Logica Best Fit.
- Supporto Docker.

> **Nota:** a partire dalla Fase 0 di questa roadmap, le voci di changelog relative a Docker/Flask/optimizer sono da considerarsi storiche: descrivono l'evoluzione del progetto quando includeva ancora un backend, oggi rimosso.

---

## 2. Roadmap di miglioramento (dopo la Fase 0)

Le fasi seguenti sono indipendenti tra loro quanto a implementazione, ma numerate per priorità consigliata. Ogni fase è pensata per essere una PR a sé stante.

### Matrice priorità / impatto

| # | Fase | Priorità | Impatto UX/UI | Effort stimato |
|---|---|---|---|---|
| 1 | Robustezza del motore e gestione errori | **P0 — Alta** | Basso (invisibile se tutto va bene) | Medio |
| 2 | File di grandi dimensioni senza bloccare il browser (Web Worker) | **P0 — Alta** | Medio | Medio |
| 3 | Refresh visivo dell'interfaccia (design system proprio) | **P1 — Alta** | **Alto** | Medio |
| 4 | Dashboard risultati con grafici e stati leggibili | **P1 — Alta** | **Alto** | Medio |
| 5 | Micro-interazioni, feedback e stati vuoti/di caricamento | **P1 — Media-Alta** | **Alto** | Basso |
| 6 | Test automatici (unit + e2e) | **P1 — Alta** | Nullo (qualità interna) | Medio-Alto |
| 7 | Cronologia elaborazioni e report in locale (IndexedDB) | **P2 — Media** | Medio | Medio |
| 8 | PWA installabile / uso offline garantito | **P2 — Media** | Medio | Basso-Medio |
| 9 | Import CSV oltre a Excel, drag&drop multi-file | **P2 — Media** | Medio | Basso |
| 10 | Export PDF del riepilogo, oltre al report Excel | **P2 — Media** | Medio | Medio |
| 11 | Accessibilità (WCAG) e responsive mobile reale | **P2 — Media** | Medio | Medio |
| 12 | Modularizzazione del codice con build step (Vite) | **P3 — Bassa** | Nullo (solo manutenibilità) | Alto |
| 13 | i18n (IT/EN) e temi colore aggiuntivi | **P3 — Bassa** | Basso-Medio | Basso |

---

### Fase 1 — Robustezza del motore e gestione errori *(P0)*

Obiettivo: far sì che l'app non si "rompa in silenzio" su file reali imperfetti.

- Validazione esplicita del file caricato: colonne mancanti, formati data non riconosciuti, celle vuote/testo dove ci si aspetta un numero → messaggi di errore chiari invece di `NaN` silenziosi o eccezioni non gestite.
- `try/catch` attorno a tutte le fasi (parsing, matching, generazione report) con un pannello errori leggibile (non solo `console.error`).
- Limiti realistici e comunicati all'utente: numero massimo di righe gestibili senza rallentare troppo il browser (misurare empiricamente, es. con dataset da 5k/20k/50k righe).
- Log delle scelte del motore (già presente parzialmente nel pannello log) reso consultabile e scaricabile come file `.txt` per il debug da parte dell'utente.

### Fase 2 — File di grandi dimensioni via Web Worker *(P0)*

Obiettivo: evitare che l'interfaccia si "congeli" durante l'elaborazione di file grandi (gli algoritmi Subset Sum in particolare sono combinatoriamente pesanti).

- Spostare parsing Excel + motore di riconciliazione in un `Web Worker` dedicato, con la UI che resta reattiva e mostra una progress bar reale (non solo uno spinner indeterminato).
- Aggiungere un indicatore di progresso a step (Lettura file → Parsing → Pass 1/2/3 → Generazione report) coerente con le fasi già presenti nel log.
- Possibilità di annullare un'elaborazione in corso.

### Fase 3 — Refresh visivo dell'interfaccia *(P1, impatto alto)*

L'attuale UI (Bootstrap 5 + palette blu/teal, dark mode già presente) è pulita ma "da template". Proposte concrete, incrementali e senza stravolgere ciò che già funziona bene:

- **Identità visiva propria**: sostituire gradualmente le classi Bootstrap generiche con un piccolo design system custom (variabili CSS già presenti in `:root` — da estendere con scala tipografica, spaziature e raggio angoli coerenti) così l'app non "sembra un altro sito Bootstrap qualunque".
- **Tipografia**: valutare un font più distintivo per i titoli (es. una serif o una sans geometrica per il logo "CashRec") mantenendo una sans neutra molto leggibile per numeri e tabelle.
- **Area di upload**: stato più "vivo" — anteprima nome file/dimensione/numero righe rilevate prima di avviare l'elaborazione, icona che cambia in base al tipo di file, animazione di conferma al drop.
- **Risultati**: oggi il risultato principale è presumibilmente "scarica il file Excel". Aggiungere un riepilogo visivo *dentro l'app* (vedi Fase 4) così l'utente ha un feedback immediato senza dover aprire Excel.
- **Dark mode**: già implementata via `data-theme` — rifinire i contrasti (in particolare `.log-container` e stati di errore) e aggiungere transizione automatica in base a `prefers-color-scheme` al primo avvio.
- **Coerenza con il logo**: il logo SVG inline (cerchio + spunta) e il banner "Simplify your accounts, liberate your business." sono un buon punto di partenza: riusarli come base per favicon, social preview e header dell'help modal.

### Fase 4 — Dashboard risultati con grafici *(P1, impatto alto)*

- Dopo l'elaborazione, mostrare **in pagina** (non solo nel file Excel) un riepilogo: % di righe riconciliate, importo totale abbinato vs residuo, numero di anomalie, mini-grafico della Quadratura Mensile (es. con Chart.js, già coerente con l'uso di librerie via CDN come `xlsx`/`exceljs`).
- Tabella filtrabile delle anomalie/non riconciliati direttamente in pagina, con possibilità di ordinare per importo o data, prima ancora di scaricare l'Excel.
- Badge di stato colorati (Match esatto / Match con tolleranza / Capienza / Forzato / Anomalia) coerenti con la legenda colori già usata nel report Excel.

### Fase 5 — Micro-interazioni e stati vuoti *(P1)*

- Sostituire eventuali `alert()`/`confirm()` nativi con toast/modali coerenti con il design system.
- Skeleton loader durante l'elaborazione invece di un'area vuota.
- Empty state illustrato quando non è stato ancora caricato nessun file.
- Animazioni leggere (200–300ms, easing coerente con quanto già presente in `.upload-area:hover`) on cambi di stato, non decorative fine a sé stesse.

### Fase 6 — Test automatici *(P1, qualità)*

- Estrarre la logica di matching (Progressive Balance, Subset Sum, Greedy) dagli script inline in moduli JS testabili.
- Unit test con **Vitest** (o Jest) che replicano gli scenari già documentati nel README (es. l'esempio "versamento 150€, incassi 100+50 → match"; "solo 80€ trovati → anomalia 70€").
- Test di regressione: stessi file di input usati storicamente per i test Python in `tests/test_riconciliazione.py` (da riprodurre come fixture JSON/CSV), per garantire che i risultati numerici non cambino durante i refactoring successivi.
- Test e2e leggeri con Playwright: apertura del file HTML, upload di un file di esempio, verifica che il download del report venga generato.

### Fase 7 — Cronologia elaborazioni locali *(P2)*

- Salvare in `IndexedDB` (più adatto di `localStorage` per volumi maggiori) un elenco delle ultime N elaborazioni: nome file, data, riepilogo risultati, configurazione usata.
- Permettere di riaprire un riepilogo passato senza dover rielaborare il file.

### Fase 8 — PWA installabile / offline *(P2)*

- `manifest.json` + service worker minimale per rendere `cashrec.html` installabile come app desktop/mobile e utilizzabile anche senza connessione dopo il primo caricamento (le CDN di `xlsx`/`exceljs` andrebbero cacheate o vendorizzate localmente).
- Icona app basata sul logo SVG esistente.

### Fase 9 — Import CSV e drag&drop multi-file *(P2)*

- Supporto a `.csv` oltre a `.xlsx`/`.xls` (rilevamento automatico separatore `,`/`;`).
- Drag&drop di più file in sequenza per elaborazioni batch client-side (oggi il batch esiste solo lato Python in `batch.py`, da rimuovere in Fase 0: qui se ne ricrea l'equivalente client-side se serve davvero).

### Fase 10 — Export PDF del riepilogo *(P2)*

- Generazione di un PDF "one-pager" con KPI principali (già disponibile via librerie client-side tipo `jsPDF`), utile per condividere rapidamente l'esito senza allegare l'intero Excel.

### Fase 11 — Accessibilità e mobile reale *(P2)*

- Contrasto colori AA/AAA su entrambi i temi, focus visibile su tutti i controlli da tastiera, `aria-label` su icone-pulsante (help, tema, upload).
- Verifica layout su viewport stretti (oggi Bootstrap dovrebbe già dare una base responsive, ma va testato con tabelle/risultati reali, che tendono a rompere il layout mobile).

### Fase 12 — Modularizzazione con build step *(P3, solo se il file diventa difficile da mantenere)*

- Se `cashrec.html` continua a crescere oltre le ~2000 righe attuali, valutare **Vite** per separare HTML/CSS/JS in moduli e produrre comunque un singolo file `dist/cashrec.html` finale (mantenendo la promessa "un file, nessuna installazione" per l'utente finale).
- Da fare *solo dopo* le Fasi 1, 2 e 6: senza test automatici, spezzare il file rischia di introdurre regressioni difficili da individuare.

### Fase 13 — i18n e temi *(P3)*

- Toggle IT/EN per l'interfaccia (il pubblico attuale è italiano/GDO, ma non costa molto prevedere l'inglese).
- Temi colore alternativi oltre a light/dark (es. un tema "alto contrasto").

---

## 3. Come procedere (istruzioni operative per Google Jules)

1. Esegui **solo** la Fase 0 nella prima PR. Non toccare la logica interna di `cashrec.html`, solo i file di contorno (server/Docker/docs) e il `README.md`.
2. Verifica manualmente (o con uno script headless) che `app/cashrec.html` funzioni ancora aprendolo direttamente da filesystem.
3. Per le fasi successive (sezione 2), procedi **una alla volta**, in ordine di priorità (P0 → P1 → P2 → P3), aprendo una PR per fase con una checklist di verifica funzionale nella descrizione.
4. Prima di ogni modifica alla logica di calcolo (Fasi 1, 2, 6, 12), assicurati che esista un test di regressione che confronti l'output "prima" e "dopo" sullo stesso file di input.
5. Aggiorna la tabella nella sezione 4 di questo documento man mano che le fasi vengono completate.

---

## 4. Stato di avanzamento

| Fase | Stato | Note |
|---|---|---|
| Fase 0 — Refactoring standalone-only | ✅ Completato | Rimosso backend Flask/Docker/Python, `app/cashrec.html` 100% standalone |
| Fase 1 — Robustezza motore | ⬜ Da fare | |
| Fase 2 — Web Worker | ⬜ Da fare | |
| Fase 3 — Refresh visivo | ⬜ Da fare | |
| Fase 4 — Dashboard risultati | ⬜ Da fare | |
| Fase 5 — Micro-interazioni | ⬜ Da fare | |
| Fase 6 — Test automatici | ⬜ Da fare | |
| Fase 7 — Cronologia locale | ⬜ Da fare | |
| Fase 8 — PWA offline | ⬜ Da fare | |
| Fase 9 — CSV / multi-file | ⬜ Da fare | |
| Fase 10 — Export PDF | ⬜ Da fare | |
| Fase 11 — Accessibilità | ⬜ Da fare | |
| Fase 12 — Modularizzazione | ⬜ Da fare | |
| Fase 13 — i18n / temi | ⬜ Da fare | |
