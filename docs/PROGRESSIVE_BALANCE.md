# Algoritmo Progressive Balance

## Panoramica

L'algoritmo **Progressive Balance** è progettato per simulare il comportamento umano nella riconciliazione contabile. Ordina cronologicamente le transazioni e cerca di abbinare ogni versamento (Credit) con gli incassi (Debit) corrispondenti.

## Logica di Funzionamento

### 1. Preparazione dei Dati

- **Data Analisi**: Per ogni movimento, viene calcolata una data di riferimento:
  - Se presente la **Data Valuta**, viene usata quella
  - Altrimenti viene usata la **Data Registrazione**

- **Ordinamento**: I movimenti vengono ordinati per Data Analisi crescente

### 2. Processo di Matching

Per ogni **versamento (Credit)**:

1. **Ricerca Incassi**: Cerca tutti gli incassi (Debit) disponibili nella finestra temporale di ±5 giorni dalla data del versamento

2. **Verifica Periodo**: 
   - Se il versamento è di un mese/anno **precedente** rispetto agli incassi disponibili → **NON** viene agganciato
   - Se il versamento è di un mese/anno **successivo** (dopo l'ultimo incasso) → viene segnalato come "senza incassi"

3. **Abbinamento**:
   - Somma gli incassi disponibili nella finestra
   - Se la somma copre il versamento → crea un match
   - Se la somma è insufficiente ma entro tolleranza → match con tolleranza
   - Se la somma è insufficiente e oltre tolleranza → **ANOMALY**

4. **Residuo**: Il residuo **NON** viene trasferito al versamento successivo

## Parametri di Configurazione

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `tolerance` | 50.0 € | Differenza massima accettata per un match |
| `days_window` | 5 giorni | Finestra temporale per la ricerca incassi (±5 giorni) |
| `search_direction` | both | Direzione di ricerca (past_only, future_only, both) |
| `residual_threshold` | 50.0 € | Soglia per il recupero residui |
| `residual_days_window` | 5 giorni | Finestra per il recupero residui |

## Interpretazione dei Risultati

### Tipi di Match nel Foglio Excel

#### ✅ Match Normali
```
Match: 3D vs 1C
```
- **Significato**: 3 Debit abbinati a 1 Credit
- **Interpretazione**: Abbinamento corretto

```
Match with tolerance (+X.XX€)
```
- **Significato**: Match con differenza entro la tolleranza
- **Interpretazione**: Abbinamento accettabile, differenza minima

#### ⚠️ ANOMALY
```
ANOMALY: X.XX€ non coperti (differenza oltre tolleranza)
```
- **Significato**: Il versamento aveva incassi disponibili ma la somma non copriva completamente il importo e la differenza supera la tolleranza
- **Interpretazione**: **Irregolarità contabile** - verificare la causa (errore di registrazione, movimento mancante, ecc.)

#### ❌ VERSAMENTO MESE PRECEDENTE
```
VERSAMENTO MESE PRECEDENTE: X.XX€ (non agganciato - periodo precedente)
```
- **Significato**: Versamento riferito a un mese/anno precedente rispetto agli incassi disponibili
- **Interpretazione**: Normale per inizio anno - i versamenti di dicembre sono stati registrati a gennaio

#### ❌ VERSAMENTO SENZA INCASSI
```
VERSAMENTO SENZA INCASSI: X.XX€ (mese/anno successivo o senza dati)
```
- **Significato**: Non ci sono incassi disponibili nella finestra temporale (il versamento è dopo l'ultimo incasso)
- **Interpretazione**: Versamenti di mesi futuri rispetto agli incassi caricati - sono "non agganciabili" con i dati attuali

## Esempi Pratici

### Esempio 1: Match Normale
```
Credit (versamento): 01/01/2026 - €500
Debit (incassi): 28/12/2025 - €200, 30/12/2025 - €300
```
- Risultato: **Match** (€200 + €300 = €500)

### Esempio 2: Versamento Mese Precedente
```
Credit: 30/12/2025 (Data Valuta) - €184.15
Debit disponibili: solo da gennaio 2026
```
- Risultato: **VERSAMENTO MESE PRECEDENTE** (non agganciabile)

### Esempio 3: Anomalia
```
Credit: 15/01/2026 - €500
Debit disponibili: €350 (nella finestra)
Differenza: €150 (oltre tolleranza 50€)
```
- Risultato: **ANOMALY: 150.00€ non coperti**

### Esempio 4: Versamento Senza Incassi
```
Credit: 10/03/2026 - €1.000
Ultimo Debit disponibile: 28/02/2026
```
- Risultato: **VERSAMENTO SENZA INCASSI** (nessun incasso nella finestra)

## Differenza con Subset Sum

| Aspetto | Progressive Balance | Subset Sum |
|---------|-------------------|------------|
| Velocità | Veloce | Lento |
| Logica | Sequenziale | Combinatorio |
| Residuo | Non trasferito | Trasferito |
| Finestra temporale | ±5 giorni | Configurabile |
| Utilizzo | GDO Retail | Casi complessi |

## Best Practices

1. **Verificare le ANOMALY**: Rappresentano irregolarità da investigare
2. **Controllare i periodi**: I "VERSAMENTO MESE PRECEDENTE" sono normali a inizio anno
3. **I "VERSAMENTO SENZA INCASSI"**: Indicano che mancano incassi nei periodi successivi
4. **Usare l'ottimizzatore**: Per trovare i parametri migliori per i tuoi dati
