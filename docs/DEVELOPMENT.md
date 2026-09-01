# Guida allo Sviluppo — CashRec

## Prerequisiti
- **Node.js** (v20+)
- **npm** (v10+)
- **Docker** (opzionale, per test container Nginx locale)

## Installazione Dipendenze
```bash
npm ci
```

## Esecuzione Test End-to-End (Playwright)
I test automatizzati con Playwright verificano la riconciliazione Excel/CSV, il cambio temi e lingua, la cronologia IndexedDB, la dashboard visiva e l'interruzione dell'elaborazione.

```bash
# Installazione browser Playwright (se necessario)
npx playwright install

# Esecuzione test E2E
npm test
```

## Sviluppo e Struttura Moduli

### 1. Standalone (`app/cashrec.html`)
- Mantiene l'intero codice JS/CSS inline per garantire il funzionamento 100% offline tramite semplice doppio click (`file://`).

### 2. Web App (`web/`)
- Mantiene la stessa logica di `app/cashrec.html` ma modularizzata per manutenibilità web e deploy containerizzato.
- `web/static/js/config.js`: parametri di default e traduzioni
- `web/static/js/engine.js`: motore di riconciliazione `JSReconciliationEngine`
- `web/static/js/parser.js`: parsing Excel/CSV e formatters
- `web/static/js/reporter.js`: generatore di report ExcelJS `JSExcelReporter`
- `web/static/js/worker.js`: Web Worker inline / dedicato
- `web/static/js/history.js`: gestione IndexedDB `CashRecDB`
- `web/static/js/ui.js`: componenti UI, Toast, Modali, Grafici Chart.js
- `web/static/js/main.js`: entry point della Web App

## Generazione Guida Utente
Se modifichi `docs/MANUALE_UTENTE.md`, puoi aggiornare la documentazione inline in `app/cashrec.html` eseguendo:
```bash
python3 tools/generate_help.py
```
