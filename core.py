import pandas as pd
from itertools import combinations
from collections import deque
from datetime import datetime
import warnings
import numpy as np

warnings.filterwarnings('ignore')
from tqdm import tqdm
import sys
from openpyxl.styles import PatternFill # Necessario per la colorazione del report
from openpyxl.chart import BarChart, Reference # Necessario per i grafici

# --- MODIFICA: Gestione di Numba come dipendenza opzionale ---
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Definisci un decoratore fittizio se Numba non è disponibile
    def jit(signature_or_function=None, locals={}, cache=False, pipeline_class=None, boundscheck=None, **options):
        def decorator(func):
            return func
        return decorator

class RiconciliatoreContabile:
    """Contiene la logica di business per la riconciliazione."""

    def __init__(self, tolleranza=0.01, giorni_finestra=7, max_combinazioni=10, soglia_residui=100, giorni_finestra_residui=30, sorting_strategy="date", search_direction="past_only", column_mapping=None):
        self.tolleranza = tolleranza
        self.giorni_finestra = giorni_finestra
        self.max_combinazioni = max_combinazioni
        self.soglia_residui = soglia_residui
        self.giorni_finestra_residui = giorni_finestra_residui
        self.sorting_strategy = sorting_strategy
        self.search_direction = search_direction
        # AGGIUNTA: Imposta la mappatura delle colonne, con un default se non fornita.
        self.column_mapping = column_mapping or {'Data': 'Data', 'Dare': 'Dare', 'Avere': 'Avere'}
        
        # Stato interno che verrà popolato durante l'esecuzione
        self.dare_df = self.avere_df = self.df_abbinamenti = None
        self.dare_non_util = self.avere_non_riconc = self.original_df = None

        # Ottimizzazione: Usare set per tenere traccia degli indici usati
        self.used_dare_indices = set()
        self.used_avere_indices = set()
        
        # Contatore per generare nuovi ID univoci per i residui
        self.max_id_counter = 0

    def carica_file(self, file_path):
        """Carica un file Excel o CSV in un DataFrame."""
        # Parametri comuni per la lettura di CSV/Excel con formato europeo
        common_read_params = {
            'decimal': ',',
            'thousands': '.'
        }

        if str(file_path).endswith('.csv'):
            df = pd.read_csv(file_path, decimal=',', thousands='.')
        elif str(file_path).endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            # Questo ramo dovrebbe essere raggiunto solo se non è un feather e non è un CSV.
            # Per i file Excel, convert_to_feather.py dovrebbe aver già gestito il parsing.
            df = pd.read_excel(file_path, decimal=',', thousands='.', engine='openpyxl')

        # --- MODIFICA: Gestione dinamica dei nomi delle colonne ---
        # Inverti la mappa per rinominare: {'Nome Colonna Sorgente': 'Nome Interno'} -> {'Nome Interno': 'Nome Colonna Sorgente'}
        source_col_names = self.column_mapping.keys()
        
        # Controlla se le colonne sorgente definite nella configurazione esistono nel file
        if not set(source_col_names).issubset(df.columns):
            missing_cols = set(source_col_names) - set(df.columns)
            raise ValueError(f"Il file di input non contiene le colonne sorgente specificate nella configurazione: {', '.join(missing_cols)}")

        # Rinomina le colonne del DataFrame usando la mappatura per standardizzarle ai nomi interni ('Data', 'Dare', 'Avere')
        df.rename(columns=self.column_mapping, inplace=True)

        # Dopo la lettura, assicurati che 'Data' sia datetime e 'Dare'/'Avere' siano numerici.
        # Questo è un fallback nel caso in cui i parametri di lettura non siano stati sufficienti
        # o se il DataFrame proviene da una fonte già pre-caricata (es. dall'ottimizzatore).
        df['Data'] = pd.to_datetime(df['Data'], errors='coerce', dayfirst=True)
        df.dropna(subset=['Data'], inplace=True) # Rimuove righe con date non valide

        # --- CHECK DATE FUTURE ---
        today = datetime.now()
        future_rows = df[df['Data'] > today]
        if not future_rows.empty:
            print(f"\n⚠️  ATTENZIONE: Trovati {len(future_rows)} movimenti con data futura (rispetto a {today.strftime('%d/%m/%Y')})!")
            print(f"    Esempio: {future_rows.iloc[0]['Data'].strftime('%d/%m/%Y')} (Riga {future_rows.index[0] + 2})")

        # --- PULIZIA IMPORTI ROBUSTA ---
        # Rimuove simboli di valuta e spazi che potrebbero far fallire la conversione
        for col in ['Dare', 'Avere']:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace('€', '', regex=False) \
                                             .str.replace(' ', '', regex=False)
                # Gestione manuale formato italiano se necessario (1.000,00 -> 1000.00)
                # Si applica solo se la stringa contiene virgola e non è già stata parsata
                df[col] = df[col].apply(lambda x: x.replace('.', '').replace(',', '.') if isinstance(x, str) and ',' in x else x)

        # Converti 'Dare' e 'Avere' in numerico. Se sono già stati letti correttamente,
        # questa operazione sarà veloce. Se sono ancora stringhe, pd.to_numeric li gestirà.
        df['Dare'] = pd.to_numeric(df['Dare'], errors='coerce')
        df['Avere'] = pd.to_numeric(df['Avere'], errors='coerce')

        # Riempi i valori non numerici con 0 PRIMA di convertire in centesimi
        df[['Dare', 'Avere']] = df[['Dare', 'Avere']].fillna(0)

        # --- OTTIMIZZAZIONE: Converti in interi (centesimi) per evitare errori di floating point ---
        # Moltiplichiamo per 100 e arrotondiamo per sicurezza, poi convertiamo in interi.
        df['Dare'] = (df['Dare'] * 100).round().astype(int)
        df['Avere'] = (df['Avere'] * 100).round().astype(int)

        df['indice_orig'] = df.index
        
        # Inizializza il contatore ID con il massimo indice esistente
        if not df.empty:
            self.max_id_counter = df.index.max()
            
        return df

    def _separa_movimenti(self, df):
        """Separa il DataFrame in movimenti DARE e AVERE."""
        self.dare_df = df[df['Dare'] != 0][['indice_orig', 'Data', 'Dare']].copy()
        self.avere_df = df[df['Avere'] != 0][['indice_orig', 'Data', 'Avere']].copy()
       
        if self.sorting_strategy == "date":
            self.dare_df = self.dare_df.sort_values('Data', ascending=True)
            self.avere_df = self.avere_df.sort_values('Data', ascending=True)
        elif self.sorting_strategy == "amount":
            self.dare_df = self.dare_df.sort_values('Dare', ascending=False)
            self.avere_df = self.avere_df.sort_values('Avere', ascending=False)
        else:
            raise ValueError(f"Strategia di ordinamento non valida: '{self.sorting_strategy}'. Usare 'date' o 'amount'.")
            
        return self.dare_df, self.avere_df

    def _trova_abbinamenti(self, dare_row, avere_candidati_np, avere_indices_map, giorni_finestra, max_combinazioni, enable_best_fit=False):
        """Logica interna per trovare un abbinamento per un singolo DARE. Riceve candidati pre-filtrati per data."""
        dare_importo = dare_row['Dare']
        dare_data = dare_row['Data']

        # --- OTTIMIZZAZIONE CORRETTA: Filtra la lista di dizionari ---
        # avere_candidati_np è ora una lista di dizionari, non un array numpy
        # I candidati sono già pre-filtrati per data e per indici non usati. Filtriamo solo per importo.
        candidati_avere = [
            c for c in avere_candidati_np if c['Avere'] <= dare_importo + self.tolleranza
        ]
        
        if not candidati_avere:
            return None
        
        # 1. Cerca match esatto 1-a-1
        match_esatto_list = [c for c in candidati_avere if abs(c['Avere'] - dare_importo) <= self.tolleranza]
        if match_esatto_list:
            best_match = match_esatto_list[0] # Prende il primo match esatto trovato
            return {
                'dare_indices': [dare_row['indice_orig']],
                'dare_date': [dare_data],
                'dare_importi': [dare_importo],
                'avere_indices': [best_match['indice_orig']],
                'avere_date': [best_match['Data']],
                'avere_importi': [best_match['Avere']],
                'somma_avere': best_match['Avere'],
                'differenza': abs(dare_importo - best_match['Avere']),
                'tipo_match': '1-a-1'
            }

        # 2. Cerca combinazioni multiple in modo ottimizzato
        candidati_avere = sorted(candidati_avere, key=lambda x: x['Avere'], reverse=True)
        
        # Aggiunta cache per la memoization
        cache = {}
        somma_totale_candidati = sum(c['Avere'] for c in candidati_avere)
        
        match = None
        if NUMBA_AVAILABLE:
            # --- OTTIMIZZAZIONE NUMBA ---
            candidati_np_numba = np.array([(c['Avere'], c['indice_orig']) for c in candidati_avere], dtype=np.int64)
            match_indices = _numba_find_combination(dare_importo, candidati_np_numba, max_combinazioni, self.tolleranza)
            if len(match_indices) > 0:
                match = [c for c in candidati_avere if c['indice_orig'] in match_indices]
        else:
            # --- FALLBACK A PYTHON PURO ---
            for c in candidati_avere: c['Dare'] = c.pop('Avere') # Adatta per la funzione generica
            match = self._trova_combinazioni_ricorsivo_py(dare_importo, candidati_avere, max_combinazioni, self.tolleranza)
            if match:
                for c in match: c['Avere'] = c.pop('Dare') # Ripristina

        if match:
            return {
                'dare_indices': [dare_row['indice_orig']],
                'dare_date': [dare_data],
                'dare_importi': [dare_importo],
                'avere_indices': [m['indice_orig'] for m in match],
                'avere_date': [m['Data'] for m in match],
                'avere_importi': [m['Avere'] for m in match],
                'somma_avere': sum(m['Avere'] for m in match),
                'differenza': abs(dare_importo - sum(m['Avere'] for m in match)),
                'tipo_match': f'Combinazione {len(match)}'
            }
        
        return None

    def _trova_combinazioni_ricorsivo_py(self, target, candidati, max_combinazioni, tolleranza):
        """Funzione iterativa (basata su stack) per il subset-sum. Versione Python pura."""
        stack = deque([(0, 0, [])]) # (start_index, somma_parziale, percorso_parziale)

        while stack:
            idx, current_sum, current_path = stack.pop()

            # --- CONDIZIONE DI SUCCESSO ---
            if abs(target - current_sum) <= tolleranza and len(current_path) > 1:
                return current_path

            # --- CONDIZIONI DI PRUNING (POTATURA) ---
            if len(current_path) >= max_combinazioni or idx >= len(candidati):
                continue

            # --- RAMO 1: Escludi il candidato corrente ---
            # Continua l'esplorazione dal prossimo candidato.
            stack.append((idx + 1, current_sum, current_path))

            # --- RAMO 2: Includi il candidato corrente ---
            candidato = candidati[idx]
            new_sum = current_sum + candidato['Dare'] # La funzione generica usa 'Dare'
            
            # Pruning: non includere se la nuova somma sfora già troppo
            if new_sum > target + tolleranza:
                continue

            new_path = current_path + [candidato]
            
            # --- CONDIZIONE DI SUCCESSO (anche dopo l'aggiunta) ---
            if abs(target - new_sum) <= tolleranza and len(new_path) > 1:
                return new_path

            # Continua l'esplorazione includendo l'elemento corrente
            stack.append((idx + 1, new_sum, new_path))

        return None # Nessuna combinazione trovata

    def _trova_abbinamenti_dare(self, avere_row, dare_candidati_np, dare_indices_map, giorni_finestra, max_combinazioni, enable_best_fit=False):
        """Logica per trovare combinazioni di DARE che corrispondono a un AVERE. Riceve candidati pre-filtrati."""
        avere_importo = avere_row['Avere']
        avere_data = avere_row['Data']

        # I candidati sono già pre-filtrati per data e indici non usati. Filtriamo solo per importo.
        candidati_dare_list = [c for c in dare_candidati_np if c['Dare'] <= avere_importo + self.tolleranza]

        if not candidati_dare_list:
            return None

        # Cerca combinazioni multiple di DARE in modo ottimizzato
        candidati_da_modificare = [c.copy() for c in candidati_dare_list]
        candidati_da_modificare = sorted(candidati_da_modificare, key=lambda x: x['Dare'], reverse=True)

        match = None
        is_partial = False
        
        if NUMBA_AVAILABLE:
            candidati_np_numba = np.array([(c['Dare'], c['indice_orig']) for c in candidati_da_modificare], dtype=np.int64)
            # Tentativo 1: Match Esatto
            match_indices = _numba_find_combination(avere_importo, candidati_np_numba, max_combinazioni, self.tolleranza)
            
            if len(match_indices) > 0:
                match = [c for c in candidati_da_modificare if c['indice_orig'] in match_indices]
            elif enable_best_fit:
                # Tentativo 2: Best Fit (Partial Match)
                # Cerca la combinazione che riempie meglio il versamento senza superarlo
                match_indices = _numba_find_best_fit_combination(avere_importo, candidati_np_numba, max_combinazioni, self.tolleranza)
                if len(match_indices) > 0:
                    match = [c for c in candidati_da_modificare if c['indice_orig'] in match_indices]
                    is_partial = True
        else:
            match = self._trova_combinazioni_ricorsivo_py(avere_importo, candidati_da_modificare, max_combinazioni, self.tolleranza)

        if match:
            somma_dare = sum(m['Dare'] for m in match)
            differenza = abs(avere_importo - somma_dare)
            
            # Se è un best fit parziale, calcoliamo il residuo
            residuo = 0
            if is_partial and differenza > self.tolleranza:
                residuo = differenza
            
            return {
                'dare_indices': [m['indice_orig'] for m in match],
                'dare_date': [m['Data'] for m in match],
                'dare_importi': [m['Dare'] for m in match],
                'avere_indices': [avere_row['indice_orig']],
                'avere_date': [avere_data],
                'avere_importi': [avere_importo],
                'somma_dare': somma_dare,
                'differenza': differenza,
                'tipo_match': f'Combinazione DARE {len(match)}' + (' (Best Fit)' if is_partial else ''),
                'residuo': residuo if is_partial else 0
            }
        return None

    def _esegui_passata_riconciliazione_dare(self, dare_df, avere_df, giorni_finestra, max_combinazioni, abbinamenti_list, title, verbose=True, enable_best_fit=False):
        """Esegue una passata cercando combinazioni di DARE per abbinare AVERE (ottimizzata con NumPy)."""
        # Filtra gli AVERE che non sono ancora stati usati
        avere_da_processare = avere_df[~avere_df['indice_orig'].isin(self.used_avere_indices)].copy() if avere_df is not None and not avere_df.empty else pd.DataFrame()

        self._esegui_passata_generica(
            df_da_processare=avere_da_processare,
            df_candidati=dare_df,
            col_da_processare='Avere',
            col_candidati='Dare',
            used_indices_candidati=self.used_dare_indices,
            giorni_finestra=giorni_finestra,
            max_combinazioni=max_combinazioni,
            abbinamenti_list=abbinamenti_list,
            title=title,
            search_direction=self.search_direction, # Usa la direzione principale dalla configurazione
            find_function=self._trova_abbinamenti_dare,
            verbose=verbose,
            enable_best_fit=enable_best_fit
        )

    def _esegui_passata_generica(self, df_da_processare, df_candidati, col_da_processare, col_candidati, used_indices_candidati, giorni_finestra, max_combinazioni, abbinamenti_list, title, search_direction, find_function, verbose=True, enable_best_fit=False):
        """
        Funzione helper generica che esegue una passata di riconciliazione.
        Questa funzione astrae la logica comune tra le passate DARE->AVERE e AVERE->DARE.
        """
        if df_da_processare is None or df_da_processare.empty:
            return

        if verbose:
            print(f"\n{title} (Direzione: {search_direction})...")

        # Prepara le liste di record una sola volta
        records_da_processare = df_da_processare.to_dict('records')
        records_candidati = sorted(df_candidati.to_dict('records'), key=lambda x: x['Data']) if df_candidati is not None else []

        matches = []
        total_records = len(records_da_processare)
        processed_count = 0
        
        # Lista per raccogliere i nuovi movimenti residui generati dallo splitting
        nuovi_residui = []

        for record_row in records_da_processare:
            if verbose:
                processed_count += 1
                percentuale = (processed_count / total_records) * 100
                sys.stdout.write(f"\r   - Avanzamento: {percentuale:.1f}% ({processed_count}/{total_records})")
                sys.stdout.flush()

            # Pre-filtra i candidati per finestra temporale
            min_data, max_data = self._calcola_finestra_temporale(record_row['Data'], giorni_finestra, search_direction)
            
            candidati_prefiltrati = [
                c for c in records_candidati
                if min_data <= c['Data'] <= max_data and c['indice_orig'] not in used_indices_candidati
            ]

            if candidati_prefiltrati:
                match = find_function(record_row, candidati_prefiltrati, None, giorni_finestra, max_combinazioni, enable_best_fit=enable_best_fit)
                if match:
                    # Gestione dello split (Best Fit)
                    residuo = match.get('residuo', 0)
                    if residuo > 0:
                        # Crea un nuovo movimento per il residuo
                        nuovo_movimento = self._crea_movimento_residuo(record_row, residuo, col_da_processare)
                        nuovi_residui.append(nuovo_movimento)

                        # Aggiorna l'importo della riga originale per riflettere solo la parte riconciliata
                        # Questo corregge le statistiche evitando la duplicazione degli importi (Originale + Residuo)
                        idx_orig = record_row['indice_orig']
                        new_amount = record_row[col_da_processare] - residuo
                        
                        if col_da_processare == 'Avere':
                            self.avere_df.loc[self.avere_df['indice_orig'] == idx_orig, 'Avere'] = new_amount
                            # FIX REPORT: Aggiorna anche il match per mostrare solo la parte usata nell'Excel
                            match['avere_importi'] = [new_amount]
                            match['somma_avere'] = new_amount
                            match['differenza'] = abs(match.get('somma_dare', 0) - new_amount)
                        elif col_da_processare == 'Dare':
                            self.dare_df.loc[self.dare_df['indice_orig'] == idx_orig, 'Dare'] = new_amount
                            # FIX REPORT
                            match['dare_importi'] = [new_amount]
                            match['somma_dare'] = new_amount
                            match['differenza'] = abs(match.get('somma_avere', 0) - new_amount)

                    # --- FIX CRITICO: REGISTRAZIONE IMMEDIATA ---
                    # Registra subito l'abbinamento per marcare gli indici come usati ed evitare
                    # che vengano riutilizzati nella stessa passata (Double Spending).
                    match['pass_name'] = title
                    self._registra_abbinamento(match, abbinamenti_list)
                    matches.append(match) # Mantiene la lista solo per il conteggio finale

        if verbose: print(f"\n   - Registrati {len(matches)} abbinamenti.")
            
        # Aggiungi i residui generati al DataFrame originale per essere processati nelle passate successive
        if nuovi_residui:
            if verbose: print(f"   - Generati {len(nuovi_residui)} movimenti residui da split (Best Fit).")
            df_residui = pd.DataFrame(nuovi_residui)
            
            if col_da_processare == 'Avere':
                self.avere_df = pd.concat([self.avere_df, df_residui], ignore_index=True)
                # Assicurati che i tipi siano corretti
                self.avere_df['Avere'] = self.avere_df['Avere'].astype(int)
            elif col_da_processare == 'Dare':
                self.dare_df = pd.concat([self.dare_df, df_residui], ignore_index=True)
                self.dare_df['Dare'] = self.dare_df['Dare'].astype(int)
                
        if verbose: sys.stdout.write("\n   ✓ Completato.\n")

    def _crea_movimento_residuo(self, record_originale, importo_residuo, col_tipo):
        """Crea un dizionario rappresentante il movimento residuo."""
        self.max_id_counter += 1
        nuovo_id = self.max_id_counter
        
        nuovo_movimento = record_originale.copy()
        nuovo_movimento['indice_orig'] = nuovo_id
        nuovo_movimento[col_tipo] = importo_residuo
        # Nota: 'usato' sarà False (o NaN che verrà trattato come False) di default quando aggiunto al DF
        
        return nuovo_movimento

    def _calcola_finestra_temporale(self, data_riferimento, giorni_finestra, search_direction):
        """Calcola la finestra temporale (min_data, max_data) in base alla direzione di ricerca."""
        if search_direction == "future_only":
            min_data = data_riferimento
            max_data = data_riferimento + pd.Timedelta(days=giorni_finestra)
        elif search_direction == "past_only":
            min_data = data_riferimento - pd.Timedelta(days=giorni_finestra)
            max_data = data_riferimento
        elif search_direction == "both":
            min_data = data_riferimento - pd.Timedelta(days=giorni_finestra)
            max_data = data_riferimento + pd.Timedelta(days=giorni_finestra)
        else:
            raise ValueError(f"Direzione di ricerca temporale non valida: '{search_direction}'. Usare 'both', 'future_only' o 'past_only'.")
        return min_data, max_data

    def _esegui_passata_riconciliazione(self, dare_df, avere_df, giorni_finestra, max_combinazioni, abbinamenti_list, title, verbose=True):
        """Esegue una passata di riconciliazione e aggiorna i DataFrame (ottimizzata con NumPy)."""
        # Filtra i DARE che non sono ancora stati usati
        dare_da_processare = dare_df[~dare_df['indice_orig'].isin(self.used_dare_indices)].copy() if dare_df is not None and not dare_df.empty else pd.DataFrame()
        
        # --- FIX: Inversione logica della direzione per la passata DARE->AVERE ---
        # Se la strategia globale è "past_only" (DARE prima di AVERE), 
        # quando partiamo da DARE dobbiamo cercare AVERE nel futuro ("future_only").
        direction_for_pass2 = self.search_direction
        if self.search_direction == "past_only":
            direction_for_pass2 = "future_only"
        elif self.search_direction == "future_only":
            direction_for_pass2 = "past_only"
        
        self._esegui_passata_generica(
            df_da_processare=dare_da_processare,
            df_candidati=avere_df,
            col_da_processare='Dare',
            col_candidati='Avere',
            used_indices_candidati=self.used_avere_indices,
            giorni_finestra=giorni_finestra,
            max_combinazioni=max_combinazioni,
            abbinamenti_list=abbinamenti_list,
            title=title,
            search_direction=direction_for_pass2, # Usa la direzione corretta (invertita)
            find_function=self._trova_abbinamenti,
            verbose=verbose
        )

    def _riconcilia(self, verbose=True):
        """Orchestra il processo di riconciliazione in più passate."""
        
        abbinamenti = []

        # --- Passata 1: Combinazione DARE per AVERE (Molti Incassi -> 1 Versamento) ---
        # Questa è la logica "Umana": cerco quali incassi passati compongono questo versamento.
        # Priorità massima dopo i match 1-a-1 impliciti.
        # ABILITATO BEST FIT: Se non trova match esatto, cerca di riempire il versamento parzialmente.
        self._esegui_passata_riconciliazione_dare(
            self.dare_df, self.avere_df,
            self.giorni_finestra,
            self.max_combinazioni,
            abbinamenti,
            "Passata 1: Aggregazione Incassi (Molti DARE -> 1 AVERE) [con Best Fit]",
            verbose,
            enable_best_fit=True
        )

        # --- Passata 2: Riconciliazione Standard Inversa (1 Incasso -> Molti Versamenti) ---
        # Utile se un incasso molto grande viene versato in più tranche (meno comune ma possibile).
        self._esegui_passata_riconciliazione(
            self.dare_df, self.avere_df,
            self.giorni_finestra,
            self.max_combinazioni,
            abbinamenti,
            "Passata 2: Versamenti Frazionati (1 DARE -> Molti AVERE)",
            verbose
        )

        # --- Passata 3: Analisi Residui (Finestra allargata) ---
        # Tenta di recuperare ciò che è rimasto fuori con una finestra più ampia
        self._esegui_passata_riconciliazione_dare(
            self.dare_df, self.avere_df,
            self.giorni_finestra_residui,
            self.max_combinazioni,
            abbinamenti,
            f"Passata 3: Recupero Residui (Finestra estesa: {self.giorni_finestra_residui}gg)",
            verbose,
            enable_best_fit=False
        )

        # AGGIUNTA: Aggiorna le colonne 'usato' nei DataFrame originali una sola volta alla fine
        self.dare_df['usato'] = self.dare_df['indice_orig'].isin(self.used_dare_indices)
        self.avere_df['usato'] = self.avere_df['indice_orig'].isin(self.used_avere_indices)

        # Colonne attese nel DataFrame finale
        final_columns = [
            'ID Transazione', 'dare_indices', 'dare_date', 'dare_importi', 
            'avere_data', 'num_avere', 'avere_indices', 'avere_importi', 
            'somma_avere', 'differenza', 'tipo_match', 'pass_name'
        ]

        # Creazione del DataFrame finale degli abbinamenti
        if abbinamenti:
            df_abbinamenti = pd.DataFrame(abbinamenti)
            # Gestione colonne mancanti (es. 'somma_dare' vs 'somma_avere')
            if 'somma_dare' in df_abbinamenti.columns and 'somma_avere' not in df_abbinamenti.columns:
                df_abbinamenti['somma_avere'] = df_abbinamenti['somma_dare']
            
            # --- MODIFICA: Creazione dell'ID Transazione con nuovo formato D(..)_A(..) ---
            df_abbinamenti['ID Transazione'] = df_abbinamenti.apply(
                lambda row: "D({})_A({})".format(
                    ','.join(map(str, [i + 2 for i in row['dare_indices']])),
                    ','.join(map(str, [i + 2 for i in row['avere_indices']]))
                ), axis=1
            )
            df_abbinamenti['sort_date'] = df_abbinamenti['dare_date'].apply(lambda x: x[0] if isinstance(x, list) else x)
            df_abbinamenti['sort_importo'] = df_abbinamenti['dare_importi'].apply(lambda x: sum(x) if isinstance(x, list) else x)
            df_abbinamenti = df_abbinamenti.sort_values(by=['sort_date', 'sort_importo'], ascending=[True, False]).drop(columns=['sort_date', 'sort_importo'])
            df_abbinamenti = df_abbinamenti.reindex(columns=final_columns) # Assicura che tutte le colonne esistano
        else:
            df_abbinamenti = pd.DataFrame(columns=final_columns)
            
        return self.dare_df, self.avere_df, df_abbinamenti

    def _registra_abbinamento(self, match, abbinamenti_list): # Rimosso dare_df, avere_df dagli argomenti
        """Marca gli elementi come 'usati' e registra l'abbinamento."""
        if not match:
            return

        dare_indices_orig = match.get('dare_indices', [])
        avere_indices_orig = match.get('avere_indices', [])

        # Aggiungi gli indici ai set di indici usati
        self.used_dare_indices.update(dare_indices_orig)
        self.used_avere_indices.update(avere_indices_orig)

        # Aggiungi ai risultati formattati
        # Ensure 'pass_name' is included
        avere_dates = match.get('avere_date')
        abbinamenti_list.append({
            'dare_indices': dare_indices_orig,
            'dare_date': match.get('dare_date', []),
            'dare_importi': match.get('dare_importi', []),
            'avere_data': min(avere_dates) if avere_dates else None,
            'num_avere': len(avere_indices_orig),
            'avere_indices': avere_indices_orig,
            'avere_importi': match.get('avere_importi', []),
            'somma_avere': match.get('somma_avere', match.get('somma_dare', 0)),
            'differenza': match.get('differenza', 0),
            'tipo_match': match.get('tipo_match', 'N/D'),
            'pass_name': match.get('pass_name', 'N/D')
        })

    def _calcola_quadratura_mensile(self):
        """Calcola le statistiche aggregate per mese per identificare sbilanci periodici."""
        if self.dare_df is None or self.avere_df is None:
            return pd.DataFrame()

        # Helper per raggruppare
        def aggrega(df, col_valore):
            if df.empty:
                return pd.DataFrame()
            temp = df.copy()
            # Assicuriamoci che Data sia datetime
            temp['Data'] = pd.to_datetime(temp['Data'])
            temp['Mese'] = temp['Data'].dt.to_period('M')
            
            # Raggruppa per mese
            gruppo = temp.groupby('Mese')
            
            totale = gruppo[col_valore].sum()
            usato = temp[temp['usato']].groupby('Mese')[col_valore].sum()
            
            res = pd.DataFrame({
                f'Totale {col_valore}': totale,
                f'Usato {col_valore}': usato
            })
            return res.fillna(0)

        stats_dare = aggrega(self.dare_df, 'Dare') # Colonne: Totale Dare, Usato Dare
        stats_avere = aggrega(self.avere_df, 'Avere') # Colonne: Totale Avere, Usato Avere

        # Unione dei due dataframe (outer join per coprire tutti i mesi)
        stats = pd.merge(stats_dare, stats_avere, left_index=True, right_index=True, how='outer').fillna(0)
        
        # Calcolo dei Delta (ancora in centesimi)
        stats['Delta DARE (Non Usato)'] = stats['Totale Dare'] - stats['Usato Dare']
        stats['Delta AVERE (Non Riconc.)'] = stats['Totale Avere'] - stats['Usato Avere']
        
        # Sbilancio netto del mese (quello che avanza in DARE meno quello che avanza in AVERE)
        stats['Sbilancio (Delta DARE - Delta AVERE)'] = stats['Delta DARE (Non Usato)'] - stats['Delta AVERE (Non Riconc.)']

        # Ordina per mese
        stats = stats.sort_index()
        
        # Formatta l'indice (Period) in stringa
        stats.index = stats.index.astype(str)
        stats.index.name = 'Mese'
        
        return stats.reset_index()

    def _verifica_quadratura_totali(self, tot_dare_orig, tot_avere_orig, verbose=True):
        """Verifica che il totale finale (usato + residuo) corrisponda al totale originale."""
        if self.dare_df is None or self.avere_df is None:
            return

        tot_dare_final = self.dare_df['Dare'].sum()
        tot_avere_final = self.avere_df['Avere'].sum()
        
        diff_dare = tot_dare_final - tot_dare_orig
        diff_avere = tot_avere_final - tot_avere_orig
        
        if verbose:
            print("\n🔍 Verifica Quadratura Totali (Originale vs Finale):")
            print(f"   DARE:  {tot_dare_orig/100:,.2f} € (Orig) vs {tot_dare_final/100:,.2f} € (Fin) -> Delta: {diff_dare/100:,.2f} €")
            print(f"   AVERE: {tot_avere_orig/100:,.2f} € (Orig) vs {tot_avere_final/100:,.2f} € (Fin) -> Delta: {diff_avere/100:,.2f} €")
            
        if abs(diff_dare) > 1 or abs(diff_avere) > 1:
             print(f"⚠️  ATTENZIONE: Rilevata discrepanza nei totali! DARE: {diff_dare}, AVERE: {diff_avere}", file=sys.stderr)
        elif verbose:
             print("   ✅ Quadratura confermata: Nessuna perdita di importi durante lo split.")

    def _crea_report_excel(self, output_file, original_df):
        """Salva i risultati in un file Excel multi-foglio."""
        # --- Foglio Originale ---
        # Crea una copia per evitare SettingWithCopyWarning e riconverti in euro
        df_originale_report = original_df.copy()

        # --- AGGIUNTA: Ordina per data per calcolare il Saldo Progressivo ---
        if 'Data' in df_originale_report.columns:
             df_originale_report.sort_values(by=['Data', 'indice_orig'], inplace=True)
        
        # Riconverte gli importi da centesimi a float per la visualizzazione
        if 'Dare' in df_originale_report.columns:
            df_originale_report['Dare'] = df_originale_report['Dare'] / 100
        if 'Avere' in df_originale_report.columns:
            df_originale_report['Avere'] = df_originale_report['Avere'] / 100

        # --- AGGIUNTA: Calcolo Saldo Progressivo ---
        if 'Dare' in df_originale_report.columns and 'Avere' in df_originale_report.columns:
            df_originale_report['Saldo Progressivo'] = (df_originale_report['Dare'] - df_originale_report['Avere']).cumsum()

        # --- MODIFICA: Formatta la data e rimuovi la colonna indice_orig ridondante ---
        if 'Data' in df_originale_report.columns:
            df_originale_report['Data'] = pd.to_datetime(df_originale_report['Data']).dt.strftime('%d/%m/%Y')
            if 'indice_orig' in df_originale_report.columns and 'usato' not in df_originale_report.columns:
                df_originale_report.drop(columns=['indice_orig'], inplace=True)


        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # --- Foglio Abbinamenti ---
            df_abbinamenti_excel = self.df_abbinamenti.copy()
            if not df_abbinamenti_excel.empty:
                # --- MODIFICA: Semplificata e corretta la formattazione delle liste per l'output Excel ---
                # Funzione per formattare correttamente le liste di indici
                def format_index_list(index_list):
                    if not isinstance(index_list, list): return index_list
                    # FIX: Aggiunge 2 per allineare l'indice 0-based di pandas con la riga 2 di Excel
                    return ', '.join(map(str, [i + 2 for i in index_list]))

                # Funzione per formattare correttamente un singolo valore numerico (es. somma_avere, differenza)
                def format_currency_value(value):
                    if pd.isna(value): return ''
                    # Divide per 100 e formatta con 2 decimali, usando la virgola come separatore
                    return f"{value/100:.2f}".replace('.', ',')

                # Funzione per formattare correttamente le liste di importi (numeri float) con virgola
                # La funzione originale usava già i centesimi, quindi basta sostituire il punto con la virgola.
                # Ho modificato la funzione format_list per includere la sostituzione del punto con la virgola.
                # La divisione per 100 è già presente.

                def format_list(data, is_float=False):
                    if not isinstance(data, list): return data
                    items = [f"{i/100:.2f}".replace('.', ',') for i in data] if is_float else data
                    return ', '.join(map(str, items))

                for col in ['dare_indices', 'avere_indices']: df_abbinamenti_excel[col] = df_abbinamenti_excel[col].apply(format_index_list)
                for col in ['dare_importi', 'avere_importi']: df_abbinamenti_excel[col] = df_abbinamenti_excel[col].apply(lambda x: format_list(x, is_float=True))
                df_abbinamenti_excel['dare_date'] = df_abbinamenti_excel['dare_date'].apply(lambda x: ', '.join([d.strftime('%d/%m/%y') for d in x]) if isinstance(x, list) else x.strftime('%d/%m/%y'))
                df_abbinamenti_excel['avere_data'] = pd.to_datetime(df_abbinamenti_excel['avere_data']).dt.strftime('%d/%m/%y')

                # Applica la formattazione per somma_avere e differenza
                df_abbinamenti_excel['somma_avere'] = df_abbinamenti_excel['somma_avere'].apply(format_currency_value)
                df_abbinamenti_excel['differenza'] = df_abbinamenti_excel['differenza'].apply(format_currency_value)

            df_abbinamenti_excel.to_excel(writer, sheet_name='Abbinamenti', index=False)

            # --- COLORAZIONE RIGHE PER PASSATA ---
            ws = writer.sheets['Abbinamenti']
            
            # Definisci i colori (Pastello)
            fill_pass1 = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid") # Verde Chiaro (Passata 1)
            fill_pass2 = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid") # Giallo Chiaro (Passata 2)
            fill_pass3 = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid") # Rosso Chiaro (Passata 3/Residui)
            
            # Applica i colori iterando sulle righe
            # Nota: df_abbinamenti_excel ha le stesse righe del foglio Excel (header è riga 1)
            if 'pass_name' in df_abbinamenti_excel.columns:
                for i, row in df_abbinamenti_excel.iterrows():
                    pass_name = str(row['pass_name'])
                    fill = None
                    if "Passata 1" in pass_name: fill = fill_pass1
                    elif "Passata 2" in pass_name: fill = fill_pass2
                    elif "Passata 3" in pass_name: fill = fill_pass3
                    
                    if fill:
                        excel_row = i + 2 # +2 perché Excel è 1-based e c'è l'header
                        for col in range(1, len(df_abbinamenti_excel.columns) + 1):
                            ws.cell(row=excel_row, column=col).fill = fill

            # --- Fogli Non Riconciliati ---
            if not self.dare_non_util.empty:
                df_dare_report = self.dare_non_util[['indice_orig', 'Data', 'Dare']].copy()
                # FIX: Aggiunge 2 per allineare l'indice con la riga di Excel
                df_dare_report['indice_orig'] = df_dare_report['indice_orig'] + 2
                df_dare_report['Data'] = pd.to_datetime(df_dare_report['Data']).dt.strftime('%d/%m/%y')
                df_dare_report['Dare'] = df_dare_report['Dare'] / 100.0
                df_dare_report.rename(columns={'indice_orig': 'Indice Riga', 'Dare': 'Importo'}).to_excel(writer, sheet_name='DARE non utilizzati', index=False)
            else:
                pd.DataFrame(columns=['Indice Riga', 'Data', 'Importo']).to_excel(writer, sheet_name='DARE non utilizzati', index=False)
            if not self.avere_non_riconc.empty:
                df_avere_report = self.avere_non_riconc[['indice_orig', 'Data', 'Avere']].copy()
                # FIX: Aggiunge 2 per allineare l'indice con la riga di Excel
                df_avere_report['indice_orig'] = df_avere_report['indice_orig'] + 2
                df_avere_report['Data'] = pd.to_datetime(df_avere_report['Data']).dt.strftime('%d/%m/%y')
                df_avere_report['Avere'] = df_avere_report['Avere'] / 100.0
                df_avere_report.rename(columns={'indice_orig': 'Indice Riga', 'Avere': 'Importo'}).to_excel(writer, sheet_name='AVERE non riconciliati', index=False)
            else:
                pd.DataFrame(columns=['Indice Riga', 'Data', 'Importo']).to_excel(writer, sheet_name='AVERE non riconciliati', index=False)

            # --- Foglio con i dati originali ---
            df_originale_report.to_excel(writer, sheet_name='Originale', index=False)

            # --- Foglio Statistiche ---
            stats = self.get_stats()
            if stats and self.dare_df is not None and self.avere_df is not None:
                def format_eur(value): return f"{value:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
                
                df_incassi = pd.DataFrame({
                    'TOT': [stats.get('Totale Incassi (DARE)'), format_eur(self.dare_df['Dare'].sum() / 100)],
                    'USATI': [stats.get('Incassi (DARE) utilizzati'), format_eur(self.dare_df[self.dare_df['usato']]['Dare'].sum() / 100)],
                    '% USATI': [stats.get('% Incassi (DARE) utilizzati (Num)'), f"{stats.get('_raw_perc_dare_importo', 0):.2f}%"],
                    'Delta': [stats.get('Incassi (DARE) non utilizzati'), format_eur(stats.get('_raw_importo_dare_non_util', 0))]
                }, index=['Numero', 'Importo'])

                df_versamenti = pd.DataFrame({
                    'TOT': [stats.get('Totale Versamenti (AVERE)'), format_eur(self.avere_df['Avere'].sum() / 100)],
                    'USATI': [stats.get('Versamenti (AVERE) riconciliati'), format_eur(self.avere_df[self.avere_df['usato']]['Avere'].sum() / 100)],
                    '% USATI': [stats.get('% Versamenti (AVERE) riconciliati (Num)'), f"{stats.get('_raw_perc_avere_importo', 0):.2f}%"],
                    'Delta': [stats.get('Versamenti (AVERE) non riconciliati'), format_eur(stats.get('_raw_importo_avere_non_riconc', 0))]
                }, index=['Numero', 'Importo'])

                delta_conteggio = stats.get('Incassi (DARE) non utilizzati', 0) - stats.get('Versamenti (AVERE) non riconciliati', 0)
                df_confronto = pd.DataFrame({
                    'Delta Conteggio': [delta_conteggio],
                    'Delta Importo (€)': [stats.get('Delta finale (DARE - AVERE)')]
                }, index=['Incassi vs Versamenti'])
                
                # Aggiungi info sullo sbilancio strutturale
                df_strutturale = pd.DataFrame({
                    'Info': ['Differenza presente nei dati originali (DARE - AVERE)'],
                    'Importo': [stats.get('Sbilancio Strutturale (Origine)')]
                })

                df_incassi.to_excel(writer, sheet_name='Statistiche', startrow=2)
                df_versamenti.to_excel(writer, sheet_name='Statistiche', startrow=8)
                df_confronto.to_excel(writer, sheet_name='Statistiche', startrow=14)
                df_strutturale.to_excel(writer, sheet_name='Statistiche', startrow=18, index=False)

            sheet_stats = writer.sheets['Statistiche']
            sheet_stats.cell(row=1, column=1, value="Riepilogo Incassi (DARE)")
            sheet_stats.cell(row=7, column=1, value="Riepilogo Versamenti (AVERE)")
            sheet_stats.cell(row=13, column=1, value="Confronto Sbilancio Finale")
            sheet_stats.cell(row=17, column=1, value="Analisi Sbilancio Strutturale (Dati Iniziali)")

            # --- AGGIUNTA: Foglio Riepilogo Parametri ---
            params_data = {
                "Parametro": [
                    "Tolleranza",
                    "Finestra Temporale (giorni)",
                    "Max Combinazioni per Match",
                    "Soglia Avvio Analisi Residui",
                    "Finestra Temporale Residui (giorni)",
                    "Strategia di Ordinamento Iniziale",
                    "Direzione Ricerca Temporale",
                    "Mappatura Colonne di Input"
                ],
                "Valore Utilizzato": [
                    f"{self.tolleranza / 100:.2f}".replace('.', ','), # Mostra in euro
                    self.giorni_finestra,
                    self.max_combinazioni,
                    f"{self.soglia_residui / 100:.2f}".replace('.', ','), # Mostra in euro
                    self.giorni_finestra_residui,
                    self.sorting_strategy,
                    self.search_direction,
                    str(self.column_mapping)
                ]
            }
            df_params = pd.DataFrame(params_data)
            df_params.to_excel(writer, sheet_name='Riepilogo Parametri', index=False)

            # --- AGGIUNTA: Foglio Quadratura Mensile ---
            df_mensile = self._calcola_quadratura_mensile()
            if not df_mensile.empty:
                # Converti da centesimi a Euro (float) per permettere al grafico di funzionare
                cols_to_convert = [c for c in df_mensile.columns if c != 'Mese']
                for col in cols_to_convert:
                    df_mensile[col] = df_mensile[col] / 100.0
                
                df_mensile.to_excel(writer, sheet_name='Quadratura Mensile', index=False)
                
                ws = writer.sheets['Quadratura Mensile']
                
                # Applica formattazione valuta alle celle (perché ora sono numeri puri)
                for col_idx, col_name in enumerate(df_mensile.columns, start=1):
                    if col_name != 'Mese':
                        for row in range(2, len(df_mensile) + 2):
                            cell = ws.cell(row=row, column=col_idx)
                            cell.number_format = '#,##0.00 €'
                
                # Adatta larghezza colonne (A-H)
                for idx, col in enumerate(df_mensile.columns):
                    max_len = min(50, max(df_mensile[col].astype(str).map(len).max() if not df_mensile.empty else 0, len(str(col)))) + 2
                    ws.column_dimensions[chr(65 + idx)].width = max_len

                # --- GRAFICO SBILANCI ---
                try:
                    chart = BarChart()
                    chart.type = "col"
                    chart.style = 10
                    chart.title = "Composizione Sbilancio Mensile (Dare vs Avere)"
                    chart.y_axis.title = "Importo (€)"
                    chart.x_axis.title = "Mese"
                    
                    # Colonne: Mese, Delta DARE, Delta AVERE
                    col_mese_idx = df_mensile.columns.get_loc('Mese') + 1
                    col_dare_idx = df_mensile.columns.get_loc('Delta DARE (Non Usato)') + 1
                    col_avere_idx = df_mensile.columns.get_loc('Delta AVERE (Non Riconc.)') + 1
                    
                    # Seleziona le due colonne dei delta (sono adiacenti)
                    data = Reference(ws, min_col=col_dare_idx, max_col=col_avere_idx, min_row=1, max_row=len(df_mensile)+1)
                    cats = Reference(ws, min_col=col_mese_idx, min_row=2, max_row=len(df_mensile)+1)
                    
                    chart.add_data(data, titles_from_data=True)
                    chart.set_categories(cats)
                    chart.shape = 4
                    chart.width = 25 # Larghezza in cm
                    chart.height = 12 # Altezza in cm
                    
                    ws.add_chart(chart, "K2") # Posiziona il grafico a destra
                except Exception as e:
                    print(f"Impossibile creare il grafico: {e}")

    def get_stats(self):
        """Calcola e restituisce un dizionario completo di statistiche."""
        if self.dare_df is None or self.avere_df is None or 'usato' not in self.dare_df.columns or 'usato' not in self.avere_df.columns: return {}

        num_dare_tot = len(self.dare_df)
        num_dare_usati = int(self.dare_df['usato'].sum()) # Ora la colonna 'usato' esiste
        imp_dare_tot = self.dare_df['Dare'].sum() # in cents
        imp_dare_usati = self.dare_df[self.dare_df['usato']]['Dare'].sum() # in cents

        num_avere_tot = len(self.avere_df)
        num_avere_usati = int(self.avere_df['usato'].sum()) # Ora la colonna 'usato' esiste
        imp_avere_tot = self.avere_df['Avere'].sum() # in cents
        imp_avere_usati = self.avere_df[self.avere_df['usato']]['Avere'].sum() # in cents

        # Ricalcola dare_non_util e avere_non_riconc basandosi sulla colonna 'usato' aggiornata
        importo_dare_non_util = (self.dare_non_util['Dare'].sum() / 100) if self.dare_non_util is not None and not self.dare_non_util.empty else 0
        importo_avere_non_riconc = (self.avere_non_riconc['Avere'].sum() / 100) if self.avere_non_riconc is not None and not self.avere_non_riconc.empty else 0

        sbilancio_strutturale = imp_dare_tot - imp_avere_tot

        return {
            'Totale Incassi (DARE)': num_dare_tot,
            'Incassi (DARE) utilizzati': num_dare_usati,
            '% Incassi (DARE) utilizzati (Num)': f"{(num_dare_usati / num_dare_tot * 100) if num_dare_tot > 0 else 0:.1f}%",
            '% Incassi (DARE) coperti (Vol)': f"{(imp_dare_usati / imp_dare_tot * 100) if imp_dare_tot > 0 else 0:.1f}%",
            'Incassi (DARE) non utilizzati': num_dare_tot - num_dare_usati,
            
            'Totale Versamenti (AVERE)': num_avere_tot,
            'Versamenti (AVERE) riconciliati': num_avere_usati,
            '% Versamenti (AVERE) riconciliati (Num)': f"{(num_avere_usati / num_avere_tot * 100) if num_avere_tot > 0 else 0:.1f}%",
            '% Versamenti (AVERE) coperti (Vol)': f"{(imp_avere_usati / imp_avere_tot * 100) if imp_avere_tot > 0 else 0:.1f}%",
            'Versamenti (AVERE) non riconciliati': num_avere_tot - num_avere_usati,

            'Delta finale (DARE - AVERE)': f"{(importo_dare_non_util - importo_avere_non_riconc):,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."),
            'Sbilancio Strutturale (Origine)': f"{(sbilancio_strutturale / 100):,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."),
            
            # Valori grezzi per aggregazioni e calcoli interni
            '_raw_importo_dare_non_util': importo_dare_non_util,
            '_raw_importo_avere_non_riconc': importo_avere_non_riconc,
            '_raw_perc_dare_importo': (imp_dare_usati / imp_dare_tot * 100) if imp_dare_tot > 0 else 0,
            '_raw_perc_avere_importo': (imp_avere_usati / imp_avere_tot * 100) if imp_avere_tot > 0 else 0,
        }

    def run(self, input_file, output_file=None, verbose=True):
        """
        Metodo pubblico principale per eseguire l'intero processo di riconciliazione.
        """
        if not NUMBA_AVAILABLE and verbose:
            # Stampa un avviso se Numba non è disponibile
            print("\n⚠️  ATTENZIONE: Libreria 'numba' non trovata. Esecuzione in modalità non ottimizzata (più lenta).")
            print("   Per performance migliori, installala con: pip install numba\n")
        try:
           # Reset degli indici usati per ogni esecuzione, importante per l'ottimizzatore
            self.used_dare_indices = set()
            self.used_avere_indices = set()

            # --- MODIFICA: Gestione flessibile dell'input ---
            # L'ottimizzatore passa un DataFrame per efficienza, main.py passa un path.
            if isinstance(input_file, pd.DataFrame):
                if verbose: print("1. Utilizzo del DataFrame pre-caricato.")
                # L'input è già un df processato, lo usiamo direttamente
                df = input_file
            else:
                if verbose: print(f"1. Caricamento e validazione del file: {input_file}")
                df = self.carica_file(input_file)

            # Calcolo totali originali per verifica quadratura
            tot_dare_orig = df['Dare'].sum()
            tot_avere_orig = df['Avere'].sum()

            if verbose: print("2. Separazione e ordinamento movimenti DARE/AVERE...")
            self._separa_movimenti(df)

            if verbose: print("3. Avvio passate di riconciliazione...")
            # _riconcilia ora restituisce i DataFrame aggiornati con la colonna 'usato'
            self.dare_df, self.avere_df, self.df_abbinamenti = self._riconcilia(verbose=verbose)

            # Verifica quadratura
            self._verifica_quadratura_totali(tot_dare_orig, tot_avere_orig, verbose=verbose)

            # --- CHECK SBILANCIO STRUTTURALE ---
            diff_strutturale = tot_dare_orig - tot_avere_orig
            if verbose and abs(diff_strutturale) > 100: # > 1 euro
                 print(f"\n⚖️  ANALISI DATI INIZIALI: Rilevato sbilancio strutturale!")
                 print(f"    Totale DARE (Incassi):    {tot_dare_orig/100:,.2f} €")
                 print(f"    Totale AVERE (Versamenti): {tot_avere_orig/100:,.2f} €")
                 print(f"    Differenza all'origine:    {diff_strutturale/100:,.2f} € (Questo importo non potrà mai essere riconciliato)")

            # Calcola i dataframe dei non utilizzati, necessari per report e statistiche
            self.dare_non_util = self.dare_df[~self.dare_df['usato']].copy()
            self.avere_non_riconc = self.avere_df[~self.avere_df['usato']].copy()

            if verbose: print("4. Calcolo statistiche finali...")
            stats = self.get_stats()

            # Se viene fornito un file di output, salva i risultati
            if output_file:
                if verbose: print(f"5. Generazione report Excel in: {output_file}")
                self._crea_report_excel(output_file, df)
                if verbose: print("✓ Report Excel creato con successo.")

            if verbose: print("\n🎉 Riconciliazione completata con successo!")
            return stats

        except (FileNotFoundError, ValueError, IndexError) as e:
            # Gestisce tutti gli errori noti (file non trovato, colonne mancanti, file corrotto)
            print(f"\n❌ ERRORE CRITICO durante l'elaborazione di '{input_file}': {e}", file=sys.stderr)
            return None
        except Exception as e:
            # Gestisce qualsiasi altro errore imprevisto
            print(f"\n❌ ERRORE IMPREVISTO: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return None

# --- FUNZIONE COMPILATA CON NUMBA ---
# Questa funzione vive al di fuori della classe per essere compilata correttamente da Numba.
# @jit è il decoratore che compila la funzione in codice macchina.
# nopython=True assicura che non ci sia fallback all'interprete Python, garantendo massima velocità.
@jit(nopython=True) # Questo decoratore sarà quello reale di Numba o quello fittizio
def _numba_find_combination(target, candidati_np, max_combinazioni, tolleranza):
    """
    Trova una combinazione di candidati la cui somma si avvicina al target.
    Questa versione è ottimizzata per Numba e opera su array NumPy.

    Args:
        target (int): L'importo da raggiungere.
        candidati_np (np.array): Array 2D dove ogni riga è [importo, indice_originale].
        max_combinazioni (int): Numero massimo di elementi nella combinazione.
        tolleranza (int): Margine di errore accettabile per la somma.

    Returns:
        np.array: Un array degli indici originali della combinazione trovata, o un array vuoto.
    """
    # Stack: (candidate_index, current_sum, level)
    # Inizializza con i candidati di primo livello
    stack = []
    n_candidati = len(candidati_np)
    
    # Iteriamo in ordine inverso per pushare nello stack, così processiamo prima i candidati più grandi (indice 0)
    for i in range(n_candidati - 1, -1, -1):
        val = candidati_np[i, 0]
        if val <= target + tolleranza:
            stack.append((i, val, 1))
            
    path = np.full(max_combinazioni, -1, dtype=np.int64)

    while len(stack) > 0:
        idx, current_sum, level = stack.pop()
        path[level-1] = idx
        
        # Check match esatto
        if abs(target - current_sum) <= tolleranza:
             result_indices = np.full(level, 0, dtype=np.int64)
             for k in range(level):
                 result_indices[k] = candidati_np[path[k], 1]
             return result_indices
             
        if level >= max_combinazioni:
            continue
            
        # Pruning: Se anche aggiungendo i valori più grandi rimasti non arriviamo al target
        remaining_slots = max_combinazioni - level
        if idx + 1 < n_candidati:
            # Stima ottimistica: usiamo il prossimo valore più grande per tutti gli slot rimasti
            max_add = candidati_np[idx+1, 0] * remaining_slots
            if current_sum + max_add < target - tolleranza:
                continue
        elif current_sum < target - tolleranza:
            # Nessun candidato rimasto e non siamo al target
            continue

        # Genera figli: prova candidati successivi
        # Push in ordine inverso (dal più piccolo al più grande) per esplorare prima i grandi
        for i in range(n_candidati - 1, idx, -1):
            val = candidati_np[i, 0]
            new_sum = current_sum + val
            if new_sum <= target + tolleranza:
                 stack.append((i, new_sum, level + 1))

    # Se lo stack si svuota, non è stata trovata nessuna combinazione.
    return np.empty(0, dtype=np.int64)

@jit(nopython=True)
def _numba_find_best_fit_combination(target, candidati_np, max_combinazioni, tolleranza):
    """
    Trova la combinazione di candidati che massimizza la somma <= target (Best Fit / Knapsack).
    Non cerca la somma esatta, ma quella che si avvicina di più senza superare il target.
    """
    # Stack: (candidate_index, current_sum, level)
    stack = []
    n_candidati = len(candidati_np)
    
    for i in range(n_candidati - 1, -1, -1):
        val = candidati_np[i, 0]
        if val <= target + tolleranza:
            stack.append((i, val, 1))
    
    path = np.full(max_combinazioni, -1, dtype=np.int64)
    
    # Variabili per tracciare la soluzione migliore trovata finora
    best_sum = 0
    best_path_len = 0
    best_path = np.full(max_combinazioni, -1, dtype=np.int64)
    
    # Soglia minima per considerare un best fit utile (es. riempire almeno l'1% del target)
    min_fill_threshold = target * 0.01

    while len(stack) > 0:
        idx, current_sum, level = stack.pop()
        path[level-1] = idx
        
        # Se la somma corrente è migliore di quella trovata finora, aggiorna il best fit
        if current_sum > best_sum:
            best_sum = current_sum
            best_path_len = level
            # Copia il percorso corrente nel best_path
            for k in range(level):
                best_path[k] = path[k]
                
            # Se abbiamo trovato un match quasi perfetto (entro la tolleranza), possiamo fermarci
            if abs(target - best_sum) <= tolleranza:
                break

        if level >= max_combinazioni:
            continue
            
        remaining_slots = max_combinazioni - level
        
        # Pruning Upper Bound
        if idx + 1 < n_candidati:
             max_potential = current_sum + candidati_np[idx+1, 0] * remaining_slots
             if max_potential <= best_sum:
                 continue
        else:
             continue

        for i in range(n_candidati - 1, idx, -1):
            val = candidati_np[i, 0]
            new_sum = current_sum + val
            
            if new_sum > target + tolleranza:
                continue
                
            # Local Pruning
            if new_sum + (val * (remaining_slots - 1)) <= best_sum:
                continue
                
            stack.append((i, new_sum, level + 1))

    # Se abbiamo trovato una soluzione valida
    if best_path_len > 0 and best_sum >= min_fill_threshold:
        result_indices = np.full(best_path_len, 0, dtype=np.int64)
        for k in range(best_path_len):
            result_indices[k] = candidati_np[best_path[k], 1]
        return result_indices

    return np.empty(0, dtype=np.int64)