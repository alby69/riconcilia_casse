# CashRec — Riconciliazione Casse

CashRec confronta gli **incassi di cassa** di un punto vendita con i **versamenti in banca**, li abbina tra loro e genera un report Excel che evidenzia le differenze da verificare. Funziona al 100% nel tuo browser, in modalità completamente offline e standalone.

## 📖 Documentazione e Roadmap

- **[Manuale Utente](./docs/MANUALE_UTENTE.md)** — guida semplice per chi usa l'applicazione tutti i giorni (in italiano, integrata anche come help nell'app standalone).
- **[ROADMAP e Storico Versioni](./ROADMAP.md)** — per lo storico delle versioni e la roadmap di sviluppo, vedi [ROADMAP.md](./ROADMAP.md).

## ✨ Funzionalità principali

- **App standalone (`app/cashrec.html`)**: un unico file HTML/JS che gira interamente nel browser, senza server né installazione. I dati non lasciano mai il computer.
- **Tre algoritmi di riconciliazione**: `progressive_balance` (profilo operatore, default), `subset_sum`, `greedy_amount_first`.
- **Profilo Operatore Punto Vendita**: default preconfigurati per la cassa quotidiana (versamenti abbinati a incassi di 1–5 giorni prima, direzione `past_only`, tolleranza 50 €).
- **Gestione profili**: salva, carica ed elimina profili di configurazione salvati in `localStorage`.
- **Recupero residui**: recupera automaticamente le differenze dai blocchi forzati.
- **Multi-negozio**: colonna opzionale *Codice Negozio* per abbinamenti prioritari all'interno dello stesso negozio.
- **Data Valuta**: gestisce i passaggi di fine anno (versamenti di gennaio che si riferiscono a dicembre).
- **Report Excel dettagliato**: fogli Summary, Matches, Anomalie, Original, Quadratura Mensile, Unused DEBIT e Unreconciled CREDIT, con importi in euro (`#,##0.00 €`), colori per stato e totali mensili.

## 🚀 Utilizzo

Apri `app/cashrec.html` con un doppio clic, oppure trascinalo nel browser. Nessuna installazione, nessun server, i dati non lasciano mai il tuo computer.

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

## 🔧 Parametri di configurazione

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
riconcilia_casse/
├── README.md
├── ROADMAP.md
├── .gitignore
├── app/
│   └── cashrec.html
├── docs/
│   └── MANUALE_UTENTE.md
├── assets/
│   └── cashrec-banner-it.png
└── tools/
    └── generate_help.py
```
