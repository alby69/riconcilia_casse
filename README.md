# CashRec — Riconciliazione Casse

CashRec confronta gli **incassi di cassa** di un punto vendita con i **versamenti in banca**, li abbina tra loro e genera un report Excel che evidenzia le differenze da verificare.

Il progetto è in **Dual Mode**:
1. **Modalità Standalone (`app/cashrec.html`)**: un singolo file HTML autocontenuto che gira 100% nel browser, senza server né installazione.
2. **Modalità Web App (`web/`)**: versione deployabile su **Render.com** tramite Docker containerizzato Nginx.

---

## 📖 Documentazione e Roadmap

- **[Manuale Utente](./docs/MANUALE_UTENTE.md)** — guida semplice per chi usa l'applicazione tutti i giorni (in italiano, integrata anche come help nell'app standalone).
- **[Specifiche Tecniche di Refactoring](./docs/SPECIFICHE.md)** — specifiche architetturali Dual Mode (Standalone + Web App).
- **[Guida al Deploy su Render](./docs/DEPLOY.md)** — istruzioni per pubblicare la Web App su Render via Blueprint o Docker.
- **[Guida allo Sviluppo](./docs/DEVELOPMENT.md)** — informazioni sull'ambiente dev, modularizzazione e test E2E Playwright.
- **[ROADMAP e Storico Versioni](./ROADMAP.md)** — per lo storico delle versioni e la roadmap di sviluppo.

---

## ✨ Funzionalità principali

- **Dual Mode**:
  - **Standalone**: `app/cashrec.html` funziona con un semplice doppio click (`file://`), 100% offline, privacy-first.
  - **Web App**: `web/index.html` servita via Docker/Nginx su Render con moduli JS/CSS separati.
- **Tre algoritmi di riconciliazione**: `progressive_balance` (profilo operatore, default), `subset_sum`, `greedy_amount_first`.
- **Profilo Operatore Punto Vendita**: default preconfigurati per la cassa quotidiana (versamenti abbinati a incassi di 1–5 giorni prima, direzione `past_only`, tolleranza 50 €).
- **Gestione profili**: salva, carica ed elimina profili di configurazione salvati in `localStorage`.
- **Multi-formato & Multi-file**: supporto per file `.xlsx`, `.xls` e `.csv` con rilevamento automatico del separatore.
- **IndexedDB History**: salva le ultime 10 elaborazioni localmente nel browser (`CashRecDB`).
- **PWA & Offline**: Web App Manifest e Service Worker (`sw.js`) per installabilità e caching offline.
- **Report Excel & PDF**: generazione client-side con ExcelJS e html2pdf.js.
- **Accessibilità & Temi**: temi Chiaro, Scuro e Alto Contrasto, selettore lingua IT/EN e visualizzazione responsive.

---

## 🚀 Utilizzo

### Modalità Standalone
Apri `app/cashrec.html` con un doppio clic, oppure trascinalo nel browser. Nessuna installazione, nessun server, i dati non lasciano mai il tuo computer.

### Modalità Web App
Avvia il container Docker locale:
```bash
docker compose -f docker/docker-compose.yml up
```
L'applicazione sarà accessibile su `http://localhost:8080`.

---

## 🔧 Parametri di configurazione

| Campo | Colonna del file |
|---|---|
| Data | `Data Reg.` |
| Dare (Incassi) | `Dare` |
| Avere (Versamenti) | `Avere` |
| Data Valuta | `Data Val.` |
| Codice Negozio | (opzionale) |

---

## 📂 Struttura del progetto

```
cashrec/
├── README.md
├── ROADMAP.md
├── package.json
├── render.yaml
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx.conf
├── app/
│   ├── cashrec.html
│   ├── manifest.json
│   └── sw.js
├── web/
│   ├── index.html
│   └── static/
│       ├── css/
│       │   └── cashrec.css
│       └── js/
│           ├── config.js
│           ├── engine.js
│           ├── parser.js
│           ├── reporter.js
│           ├── worker.js
│           ├── history.js
│           ├── pwa.js
│           ├── ui.js
│           └── main.js
├── docs/
│   ├── MANUALE_UTENTE.md
│   ├── SPECIFICHE.md
│   ├── DEPLOY.md
│   └── DEVELOPMENT.md
└── tools/
    └── generate_help.py
```
