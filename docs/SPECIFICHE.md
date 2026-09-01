# Specifiche di Refactoring — CashRec
## Dual Mode: Standalone HTML + Web App su Render

---

## 1. Panoramica
CashRec è un'applicazione 100% client-side per la riconciliazione automatica degli incassi di cassa con i versamenti bancari.

Il refactoring v6.0 trasforma la struttura del progetto per supportare due modalità operative (Dual Mode):
1. **Modalità Standalone (`app/cashrec.html`)**: un singolo file HTML autocontenuto, senza server, completamente offline e standalone (`file://` protocol).
2. **Modalità Web App (`web/`)**: versione deployabile su Render.com servita via Nginx containerizzato Docker, con asset CSS/JS modularizzati in `web/static/`.

---

## 2. Struttura del Repository

```
cashrec/
├── README.md                     ← Guida principale bilingue
├── LICENSE                       ← Licenza MIT
├── package.json                  ← Dipendenze e script test E2E Playwright
├── render.yaml                   ← Blueprint deploy automatico Render
├── .github/
│   └── workflows/
│       └── ci.yml                ← CI GitHub Actions (test + Docker build)
├── docker/
│   ├── Dockerfile                ← Dockerfile Nginx
│   ├── docker-compose.yml        ← Docker Compose per dev locale
│   └── nginx.conf                ← Config Nginx (gzip, cache, SPA, $PORT)
├── app/
│   ├── cashrec.html              ← VERSIONE STANDALONE (100% client-side)
│   ├── manifest.json             ← PWA manifest standalone
│   └── sw.js                     ← Service Worker standalone
├── web/
│   ├── index.html                ← Entry point Web App
│   └── static/
│       ├── css/
│       │   └── cashrec.css       ← CSS estratto
│       └── js/
│           ├── config.js         ← Default e traduzioni i18n
│           ├── engine.js         ← JSReconciliationEngine
│           ├── parser.js         ← Parser Excel/CSV e valute
│           ├── reporter.js       ← JSExcelReporter
│           ├── worker.js         ← Web Worker
│           ├── history.js        ← CashRecDB IndexedDB
│           ├── pwa.js            ← ServiceWorker registration
│           ├── ui.js             ← UI Helpers, Toast, Modal, Dashboard, Chart
│           └── main.js           ← Entry point JS
├── docs/
│   ├── MANUALE_UTENTE.md         ← Guida utente finale
│   ├── SPECIFICHE.md             ← Specifiche tecniche di refactoring
│   ├── DEPLOY.md                 ← Istruzioni deploy Render
│   └── DEVELOPMENT.md            ← Guida per gli sviluppatori
└── tools/
    └── generate_help.py          ← Script dev per iniezione help
```
