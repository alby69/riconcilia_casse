# Analisi Implementazione Progressive Balance

## Obiettivo
Implementare l'algoritmo di Progressive Balance come descritto dall'utente:

1. Creare colonna `Data_Analisi` = `Data_Valuta` se presente, altrimenti `Data`
2. Ordinare tutto per `Data_Analisi` crescente
3. Processare i CREDIT (Avere) in ordine, cercando i DEBIT (Dare) non usati
4. Sottrare e portare avanti il residuo

## Logica Descritta dall'Utente

Esempio con file RISCA_2025_NEW:

1. **Primo versamento**: Riga 14, Avere=4910, Data_Valuta=02/01/2026
   - Risalgo agli incassi: primo Dare non usato con Data >= 02/01/2026 è Riga 3, Dare=7610
   - 7610 - 4910 = 2700 residuo incasso
   - Trovo versamento successivo con stessa Data_Valuta (02/01): Riga 17, Avere=2700
   - 2700 - 2700 = 0 ✓

2. **Secondo versamento**: Riga 15, Avere=12715, Data_Valuta=03/01/2026
   - Primo incasso non usato (dopo Riga 3 già usata): Riga 4, Dare=1035
   - 12715 - 1035 = 11680 residuo
   - Incasso successivo: Riga 5, Dare=10480
   - 11680 - 10480 = 1200 residuo (non agganciato)

3. **Terzo versamento**: Riga 16, Avere=8405, Data_Valuta=04/01/2026
   - Primo incasso non usato: Riga 6, Dare=5970
   - 8405 - 5970 = 2435 residuo
   - Incasso successivo: Riga 7, Dare=1310
   - 2435 - 1310 = 1125 residuo

## Problemi Attuali con l'Implementazione

### Issue 1: Duplicazione Credit
Quando un debito viene "spacchettato" (usato parzialmente), il credit viene aggiunto alla lista match ogni iterazione invece che solo una volta.

### Issue 2: Reset Lists
Le liste `current_match_debits`, `current_match_credits`, ecc. vengono resettate ma il `remaining_credit` non viene azzerato, causando calcoli errati.

### Issue 3: Logic Main Loop
Il loop principale continua ad aggiungere credits anche quando dovrebbe fermarsi.

## Codice Corrente (Parziale)

```python
def _reconcile_progressive_balance(self, verbose=True):
    df_debit = self.debit_df[~self.debit_df['orig_index'].isin(self.used_debit_indices)].copy()
    df_credit = self.credit_df[~self.credit_df['orig_index'].isin(self.used_credit_indices)].copy()
    
    df_debit['analysis_date'] = df_debit['Date']
    df_credit['analysis_date'] = df_credit.get('valuta_date', df_credit['Date'])
    df_credit['analysis_date'] = df_credit['analysis_date'].combine_first(df_credit['Date'])
    
    df_debit = df_debit.sort_values(by=['analysis_date', 'orig_index'])
    df_credit = df_credit.sort_values(by=['analysis_date', 'orig_index'])
    
    debit_rows = df_debit.to_dict('records')
    credit_rows = df_credit.to_dict('records')
    
    n_debit = len(debit_rows)
    n_credit = len(credit_rows)
    
    debit_remaining = {i: debit_rows[i]['Debit'] for i in range(n_debit)}
    
    matches = []
    
    debit_idx = 0
    credit_idx = 0
    
    remaining_credit = 0
    
    current_match_debits = []
    current_match_credits = []
    current_debit_amounts = []
    current_credit_amounts = []
    
    while credit_idx < n_credit:
        credit_amount = credit_rows[credit_idx]['Credit']
        credit_orig_idx = credit_rows[credit_idx]['orig_index']
        
        # BUG: Manca controllo se debit_idx >= n_debit
        
        remaining_credit += credit_amount
        current_match_credits.append(credit_orig_idx)  # BUG: Aggiunto ogni iterazione
        current_credit_amounts.append(credit_amount)
        
        while remaining_credit > 0 and debit_idx < n_debit:
            if debit_remaining[debit_idx] <= 0:
                debit_idx += 1
                continue
            
            debit_amount = debit_remaining[debit_idx]
            debit_orig_idx = debit_rows[debit_idx]['orig_index']
            
            if debit_amount <= remaining_credit:
                current_match_debits.append(debit_orig_idx)
                current_debit_amounts.append(debit_amount)
                remaining_credit -= debit_amount
                debit_remaining[debit_idx] = 0
                debit_idx += 1
            else:
                current_match_debits.append(debit_orig_idx)
                current_debit_amounts.append(remaining_credit)
                debit_remaining[debit_idx] = debit_amount - remaining_credit
                remaining_credit = 0  # BUG: Dovrebbe anche chiudere il match e resettare
        
        # BUG: Se remaining_credit > 0 ma debit_idx >= n_debit, i credit successivi vengono aggiunti lo stesso
        
        if remaining_credit == 0:
            # Crea match
            ...
            current_match_debits = []
            current_match_credits = []
            current_debit_amounts = []
            current_credit_amounts = []
        
        credit_idx += 1
```

## Prossimi Passi

1. **Correggere la logica di match**: Un match dovrebbe chiudersi solo quando `remaining_credit == 0` OPPURE quando `debit_idx >= n_debit`

2. **Gestire correttamente i credit multipli**: Quando un debito viene usato parzialmente, non bisogna aggiungere tutti i credit al match finale - solo quelli che hanno contribuito a pareggiare

3. **Aggiungere controllo early exit**: Quando `debit_idx >= n_debit`, uscire dal loop principale

4. **Resettare remaining_credit**: Dopo aver creato un match, azzerare `remaining_credit`

## Test Case

File: `input/RISCA_2025_NEW.xlsx`
- 116 Debit movements
- 76 Credit movements
- Credits con Data Valuta: 30/12/2025, 31/12/2025, poi date nel 2026

## Riferimenti

- Expected result: `input/RISCA_2025_NEW_result.xlsx`
- Il file atteso usa "Pass 1: Receipt Aggregation (Many DEBIT -> 1 CREDIT) [with Best Fit]" NON Progressive Balance
- L'utente vuole che Progressive Balance segua la logica descritta sopra
