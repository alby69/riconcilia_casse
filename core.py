import pandas as pd
from itertools import combinations
from collections import deque
from datetime import datetime
import warnings
import numpy as np

warnings.filterwarnings('ignore')
from tqdm import tqdm
import sys
from openpyxl.styles import PatternFill, Alignment, Font # Necessario per la colorazione e formattazione
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

def _robust_currency_parser(value):
    """
    Converte in modo robusto una stringa o un numero in un formato numerico standard per pd.to_numeric.
    Questa funzione helper viene usata da `carica_file`.
    """
    # Se è già un numero, è a posto.
    if isinstance(value, (int, float)):
        return value
    # Se non è una stringa, non possiamo farci nulla.
    if not isinstance(value, str):
        return None # Verrà convertito in NaN
    
    # Pulisci la stringa da spazi e simbolo euro
    cleaned_str = str(value).strip().replace('€', '').replace(' ', '')
    
    # Caso 1: Formato italiano completo (es. "1.234,56")
    if '.' in cleaned_str and ',' in cleaned_str:
        return cleaned_str.replace('.', '').replace(',', '.')
    # Caso 2: Formato italiano con solo decimali (es. "1234,56")
    elif ',' in cleaned_str:
        return cleaned_str.replace(',', '.')
    # Caso 3: Formato senza virgole (es. "1234" o "1234.56"). Lasciamo il punto.
    return cleaned_str

class RiconciliatoreContabile:
    """Contiene la logica di business per la riconciliazione."""

    def __init__(self, tolleranza=0.01, giorni_finestra=7, max_combinazioni=10, soglia_residui=100.0, giorni_finestra_residui=30, sorting_strategy="date", search_direction="past_only", column_mapping=None, algorithm="subset_sum", use_numba=True, ignore_tolerance=False, enable_best_fit=True):
        """
        Inizializza il riconciliatore con i parametri di configurazione.

        Args:
            tolleranza (float): Differenza massima accettata tra importi (default 0.01).
            giorni_finestra (int): Finestra temporale di ricerca in giorni (default 7).
            max_combinazioni (int): Numero massimo di movimenti combinabili (default 10).
            soglia_residui (float): Importo minimo per considerare un movimento nella fase residui (default 100.0).
            giorni_finestra_residui (int): Finestra temporale estesa per la fase residui (default 30).
            sorting_strategy (str): Strategia di ordinamento ('date' o 'amount').
            search_direction (str): Direzione di ricerca ('past_only', 'future_only', 'both').
            column_mapping (dict): Mappatura nomi colonne (es. {'Data': 'MyDate', ...}).
            algorithm (str): Algoritmo da usare ('subset_sum', 'progressive_balance', 'all').
            use_numba (bool): Se True, utilizza l'accelerazione Numba se disponibile.
            ignore_tolerance (bool): Se True, forza la chiusura dei blocchi nel saldo progressivo anche se non quadrano.
            enable_best_fit (bool): Se True, abilita la logica di abbinamento parziale (splitting).
        """
        # FIX: Converte i valori in euro (float) a centesimi (int) per coerenza interna
        self.tolleranza = int(tolleranza * 100)
        self.giorni_finestra = giorni_finestra
        self.max_combinazioni = max_combinazioni
        # FIX: Converte i valori in euro (float) a centesimi (int)
        self.soglia_residui = int(soglia_residui * 100)
        self.giorni_finestra_residui = giorni_finestra_residui
        self.sorting_strategy = sorting_strategy
        self.search_direction = search_direction
        self.algorithm = algorithm
        # AGGIUNTA: Imposta la mappatura delle colonne, con un default se non fornita.
        self.column_mapping = column_mapping or {'Data': 'Data', 'Dare': 'Dare', 'Avere': 'Avere'}
        
        # Flag per abilitare/disabilitare Numba
        self.use_numba = use_numba and NUMBA_AVAILABLE
        
        # Flag per forzare la chiusura dei blocchi in Saldo Progressivo anche se non quadrano (su timeout finestra)
        self.ignore_tolerance = ignore_tolerance

        # AGGIUNTA: Flag per abilitare la logica di best-fit (splitting)
        self.enable_best_fit = enable_best_fit

        # Stato interno che verrà popolato durante l'esecuzione
        self.dare_df = self.avere_df = self.df_abbinamenti = None
        self.dare_non_util = self.avere_non_riconc = self.original_df = None

        # Ottimizzazione: Usare set per tenere traccia degli indici usati
        self.used_dare_indices = set()
        self.used_avere_indices = set()
        
        # Contatore per generare nuovi ID univoci per i residui
        self.max_id_counter = 0

    def carica_file(self, file_path):
        """
        Carica un file Excel, CSV o Feather in un DataFrame Pandas standardizzato.

        Gestisce la lettura, la rinomina delle colonne secondo il mapping configurato,
        la conversione delle date e la pulizia degli importi.

        Args:
            file_path (str): Percorso del file da caricare.

        Returns:
            pd.DataFrame: DataFrame con colonne standardizzate ('Data', 'Dare', 'Avere', 'indice_orig').
        """
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
        # Applica il parser robusto a ogni cella, poi converte l'intera colonna.
        for col in ['Dare', 'Avere']:
            df[col] = pd.to_numeric(df[col].apply(_robust_currency_parser), errors='coerce')

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
        if self.use_numba:
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
        
        if self.use_numba:
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
        
        Itera su ogni riga di `df_da_processare` e cerca corrispondenze in `df_candidati`
        utilizzando la `find_function` fornita. Gestisce la logica di finestra temporale,
        registrazione match e splitting (Best Fit).

        Args:
            df_da_processare (pd.DataFrame): DataFrame principale da scorrere.
            df_candidati (pd.DataFrame): DataFrame dove cercare le combinazioni.
            col_da_processare (str): Nome colonna importo nel DF principale ('Dare' o 'Avere').
            col_candidati (str): Nome colonna importo nel DF candidati.
            used_indices_candidati (set): Set di indici già usati da escludere.
            giorni_finestra (int): Finestra temporale per la ricerca.
            max_combinazioni (int): Max elementi combinabili.
            abbinamenti_list (list): Lista dove appendere i match trovati.
            title (str): Titolo della passata per il logging.
            search_direction (str): Direzione temporale ('past_only', 'future_only', 'both').
            find_function (callable): Funzione che implementa la logica di matching specifica.
            verbose (bool): Se True, stampa log.
            enable_best_fit (bool): Se True, abilita la logica di splitting per match parziali.
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

    def _riconcilia_subset_sum(self, verbose=True):
        """
        Esegue la riconciliazione basata sulla ricerca di combinazioni (Subset Sum).

        Orchestra tre passate successive:
        1. Molti DARE -> 1 AVERE (con Best Fit opzionale).
        2. 1 DARE -> Molti AVERE (Versamenti frazionati).
        3. Recupero residui con finestra temporale estesa.

        Args:
            verbose (bool): Se True, stampa l'avanzamento.

        Returns:
            list: Lista di dizionari rappresentanti gli abbinamenti trovati.
        """
        
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
            enable_best_fit=self.enable_best_fit
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

        return abbinamenti

    def _finalizza_abbinamenti(self, abbinamenti):
        """Crea il DataFrame finale, genera gli ID e ordina i risultati."""
        # Colonne attese nel DataFrame finale
        final_columns = [
            'ID Transazione', 'dare_indices', 'dare_date', 'dare_importi', 
            'avere_data', 'num_avere', 'avere_indices', 'avere_importi', 
            'somma_avere', 'differenza', 'giorni_diff', 'tipo_match', 'pass_name'
        ]

        # Creazione del DataFrame finale degli abbinamenti
        if abbinamenti:
            df_abbinamenti = pd.DataFrame(abbinamenti)
            # Gestione colonne mancanti (es. 'somma_dare' vs 'somma_avere')
            if 'somma_dare' in df_abbinamenti.columns and 'somma_avere' not in df_abbinamenti.columns:
                df_abbinamenti['somma_avere'] = df_abbinamenti['somma_dare']
            
            # Calcolo differenza giorni (Avere - Dare)
            df_abbinamenti['giorni_diff'] = df_abbinamenti.apply(
                lambda row: (row['avere_data'] - min(row['dare_date'])).days 
                if isinstance(row['dare_date'], list) and len(row['dare_date']) > 0 and pd.notnull(row['avere_data']) 
                else None, axis=1
            )

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
            
        return df_abbinamenti

    def _riconcilia_saldo_progressivo(self, verbose=True):
        """
        Algoritmo di riconciliazione basato sul saldo progressivo sequenziale (Two Pointers).

        Simula un operatore che scorre le liste ordinate cronologicamente e chiude
        un blocco quando la somma progressiva di DARE e AVERE coincide.

        Args:
            verbose (bool): Se True, stampa l'avanzamento.

        Returns:
            list: Lista di dizionari rappresentanti gli abbinamenti (blocchi) trovati.
        """
        from datetime import timedelta # Assicurati che sia importato
        if verbose:
            print("\nAvvio riconciliazione con algoritmo 'Saldo Progressivo' (Sequenziale)...")

        # 1. Prepara i dati: Filtra non usati e Ordina per Data
        # Nota: Usiamo copie per non modificare i df originali durante l'iterazione
        df_dare_temp = self.dare_df[~self.dare_df['indice_orig'].isin(self.used_dare_indices)].copy()
        df_avere_temp = self.avere_df[~self.avere_df['indice_orig'].isin(self.used_avere_indices)].copy()
        
        # Ordinamento per data: è fondamentale e intenzionale per questo algoritmo.
        # L'algoritmo simula un saldo che progredisce nel tempo, quindi ignora la 'sorting_strategy'
        # globale (es. per importo) e forza sempre un ordinamento cronologico.
        df_dare_temp.sort_values(by=['Data', 'indice_orig'], inplace=True)
        df_avere_temp.sort_values(by=['Data', 'indice_orig'], inplace=True)

        dare_rows = df_dare_temp.to_dict('records')
        avere_rows = df_avere_temp.to_dict('records')
        
        n_dare = len(dare_rows)
        n_avere = len(avere_rows)
        
        i = 0 # Puntatore Dare
        j = 0 # Puntatore Avere
        
        cum_dare = 0
        cum_avere = 0
        
        start_i = 0
        start_j = 0
        
        abbinamenti = []
        
        if verbose:
            print(f"   - Analisi sequenziale su {n_dare} movimenti Dare e {n_avere} movimenti Avere...")

        # Loop principale (Two Pointers)
        while i < n_dare or j < n_avere:
            # Verifica se abbiamo raggiunto un punto di pareggio (con almeno un movimento processato nel blocco corrente)
            diff = cum_dare - cum_avere
            
            # --- LOGICA DI RESET MIGLIORATA (Block Duration) ---
            # Calcola la durata del blocco accumulato finora
            # Data inizio: minimo tra il primo DARE e il primo AVERE del blocco corrente
            start_date_dare = dare_rows[start_i]['Data'] if start_i < n_dare else None
            start_date_avere = avere_rows[start_j]['Data'] if start_j < n_avere else None
            
            # Data fine: massimo tra l'ultimo DARE e AVERE processati (i-1, j-1) o correnti
            curr_date_dare = dare_rows[i]['Data'] if i < n_dare else (dare_rows[i-1]['Data'] if i > 0 else None)
            curr_date_avere = avere_rows[j]['Data'] if j < n_avere else (avere_rows[j-1]['Data'] if j > 0 else None)
            
            valid_starts = [d for d in [start_date_dare, start_date_avere] if d is not None]
            valid_ends = [d for d in [curr_date_dare, curr_date_avere] if d is not None]
            
            should_reset = False
            if valid_starts and valid_ends:
                block_duration = (max(valid_ends) - min(valid_starts)).days
                if block_duration > self.giorni_finestra and (cum_dare > 0 or cum_avere > 0):
                    should_reset = True

            if should_reset:
                if self.ignore_tolerance:
                    # --- FORZA CHIUSURA (Accetta errore) ---
                    # Se l'utente ha scelto di ignorare la tolleranza (o meglio, di forzare su timeout),
                    # chiudiamo il blocco così com'è.
                    block_dare = dare_rows[start_i:i]
                    block_avere = avere_rows[start_j:j]
                    match = {
                        'dare_indices': [r['indice_orig'] for r in block_dare],
                        'dare_date': [r['Data'] for r in block_dare],
                        'dare_importi': [r['Dare'] for r in block_dare],
                        'avere_indices': [r['indice_orig'] for r in block_avere],
                        'avere_date': [r['Data'] for r in block_avere],
                        'avere_importi': [r['Avere'] for r in block_avere],
                        'somma_avere': cum_avere,
                        'differenza': abs(diff),
                        'tipo_match': f'Forzato (Timeout {self.giorni_finestra}gg)',
                        'pass_name': 'Saldo Progressivo (Forzato)'
                    }
                    self._registra_abbinamento(match, abbinamenti)
                    # Reset e continua
                    start_i = i
                    start_j = j
                    cum_dare = 0
                    cum_avere = 0
                else:
                    # --- RESET STANDARD (Salta blocco errato) ---
                    # Abbandona il blocco corrente che non quadra e riparti fresco.
                    # Questo permette di trovare i match successivi invece di trascinare l'errore.
                    start_i = i
                    start_j = j
                    cum_dare = 0
                    cum_avere = 0
            
            # Condizione di Match: Differenza zero (entro tolleranza) e abbiamo avanzato almeno uno dei puntatori
            if abs(diff) <= self.tolleranza and (i > start_i or j > start_j):
                # --- BLOCCO QUADRATO TROVATO ---
                block_dare = dare_rows[start_i:i]
                block_avere = avere_rows[start_j:j]
                
                match = {
                    'dare_indices': [r['indice_orig'] for r in block_dare],
                    'dare_date': [r['Data'] for r in block_dare],
                    'dare_importi': [r['Dare'] for r in block_dare],
                    'avere_indices': [r['indice_orig'] for r in block_avere],
                    'avere_date': [r['Data'] for r in block_avere],
                    'avere_importi': [r['Avere'] for r in block_avere],
                    'somma_avere': cum_avere, # O cum_dare, sono uguali
                    'differenza': abs(diff),
                    'tipo_match': f'Saldo Progressivo (Seq. {len(block_dare)}D vs {len(block_avere)}A)',
                    'pass_name': 'Saldo Progressivo'
                }
                self._registra_abbinamento(match, abbinamenti)

                # Reset per il prossimo blocco (ripartiamo da zero per evitare accumulo di errori)
                start_i = i
                start_j = j
                cum_dare = 0
                cum_avere = 0
                
                # Se abbiamo finito entrambi, usciamo
                if i == n_dare and j == n_avere:
                    break
            
            # --- LOGICA DI AVANZAMENTO (GREEDY) ---
            # Decidiamo quale puntatore avanzare per cercare di pareggiare i conti.
            
            can_advance_dare = i < n_dare
            can_advance_avere = j < n_avere
            
            if can_advance_dare and can_advance_avere:
                # Se Dare è indietro come importo, aggiungiamo Dare
                if cum_dare < cum_avere:
                    cum_dare += dare_rows[i]['Dare']
                    i += 1
                # Se Avere è indietro come importo, aggiungiamo Avere
                elif cum_avere < cum_dare:
                    cum_avere += avere_rows[j]['Avere']
                    j += 1
                else:
                    # Se gli importi sono uguali (inizio blocco o importi zero), avanziamo quello con data antecedente
                    date_dare = dare_rows[i]['Data']
                    date_avere = avere_rows[j]['Data']
                    
                    if date_dare <= date_avere:
                        cum_dare += dare_rows[i]['Dare']
                        i += 1
                    else:
                        cum_avere += avere_rows[j]['Avere']
                        j += 1
                        
            elif can_advance_dare:
                # Possiamo avanzare solo Dare
                cum_dare += dare_rows[i]['Dare']
                i += 1
            elif can_advance_avere:
                # Possiamo avanzare solo Avere
                cum_avere += avere_rows[j]['Avere']
                j += 1
            else:
                # Non possiamo avanzare nessuno dei due, ma non abbiamo matchato (caso residuo finale non quadrato)
                break

        if verbose:
            print(f"   - Trovati {len(abbinamenti)} blocchi bilanciati.")
            
        return abbinamenti

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
        stats = pd.merge(stats_dare, stats_avere, left_index=True, right_index=True, how='outer')
        
        # --- NUOVO: Calcolo dello sbilancio assorbito nei match ---
        sbilancio_assorbito = pd.DataFrame()
        if self.df_abbinamenti is not None and not self.df_abbinamenti.empty:
            df_temp_abbinamenti = self.df_abbinamenti.copy()
            
            # La data di riferimento per il mese è la data del primo DARE nel blocco
            df_temp_abbinamenti['Mese'] = df_temp_abbinamenti['dare_date'].apply(
                lambda x: x[0].to_period('M') if isinstance(x, list) and x else None
            )
            df_temp_abbinamenti.dropna(subset=['Mese'], inplace=True)
            
            # Calcola la differenza con segno (DARE - AVERE) per ogni blocco
            df_temp_abbinamenti['somma_dare'] = df_temp_abbinamenti['dare_importi'].apply(lambda x: sum(x) if isinstance(x, list) else 0)
            df_temp_abbinamenti['sbilancio_blocco'] = df_temp_abbinamenti['somma_dare'] - df_temp_abbinamenti['somma_avere']
            
            sbilancio_assorbito = df_temp_abbinamenti.groupby('Mese')['sbilancio_blocco'].sum().to_frame('Sbilancio Assorbito (in match)')

        if not sbilancio_assorbito.empty:
            stats = pd.merge(stats, sbilancio_assorbito, left_index=True, right_index=True, how='outer')

        stats = stats.fillna(0)
        
        # Calcolo dei Delta (ancora in centesimi)
        stats['DARE non abbinati'] = stats['Totale Dare'] - stats['Usato Dare']
        stats['AVERE non abbinati'] = stats['Totale Avere'] - stats['Usato Avere']
        
        # Sbilancio netto dei soli movimenti non abbinati
        stats['Sbilancio Residui (DARE - AVERE)'] = stats['DARE non abbinati'] - stats['AVERE non abbinati']

        # Sbilancio Finale del Mese
        if 'Sbilancio Assorbito (in match)' not in stats.columns:
            stats['Sbilancio Assorbito (in match)'] = 0
            
        stats['Sbilancio Finale Mese'] = stats['Sbilancio Residui (DARE - AVERE)'] + stats['Sbilancio Assorbito (in match)']

        # Riorganizza le colonne per chiarezza
        stats = stats[[
            'Totale Dare', 'Usato Dare', 'DARE non abbinati',
            'Totale Avere', 'Usato Avere', 'AVERE non abbinati',
            'Sbilancio Residui (DARE - AVERE)',
            'Sbilancio Assorbito (in match)',
            'Sbilancio Finale Mese'
        ]]

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

    def _crea_foglio_manuale(self, writer):
        """Crea il foglio 'MANUALE' con la spiegazione dell'algoritmo e dei parametri."""
        ws = writer.book.create_sheet("MANUALE", 0) # Create as the first sheet

        # Stili
        title_font = Font(bold=True, size=14)
        header_font = Font(bold=True, size=12)
        
        # Contenuti
        manual_content = {}
        if self.algorithm == 'subset_sum':
            manual_content = {
                "title": "Algoritmo: Subset Sum (Ricerca Combinazioni)",
                "description": [
                    ("Descrizione Generale", "Questo algoritmo tenta di risolvere il 'problema della somma di un sottoinsieme'. Per ogni movimento su un lato (es. un versamento in AVERE), cerca una combinazione di uno o più movimenti sull'altro lato (es. incassi in DARE) la cui somma corrisponda all'importo del movimento di partenza, entro una data tolleranza e finestra temporale."),
                    ("Logica di Funzionamento", "Il processo avviene in più passate:\n1. **Aggregazione Incassi (Molti DARE -> 1 AVERE)**: Simula la logica umana di raggruppare più incassi per formare un unico versamento. Cerca anche 'best fit' parziali, generando residui.\n2. **Versamenti Frazionati (1 DARE -> Molti AVERE)**: Gestisce il caso meno comune in cui un grande incasso viene versato in più tranche.\n3. **Recupero Residui**: Esegue una passata finale con una finestra temporale più ampia per tentare di abbinare i movimenti rimasti."),
                ],
                "params": [
                    ("Tolleranza", f"{self.tolleranza / 100:.2f} €", "Il margine di errore massimo accettato tra la somma dei movimenti combinati e l'importo target."),
                    ("Finestra Temporale", f"{self.giorni_finestra} giorni", "L'intervallo di giorni (prima, dopo o entrambi) in cui cercare i movimenti candidati per un abbinamento."),
                    ("Max Combinazioni", f"{self.max_combinazioni}", "Il numero massimo di movimenti che possono essere combinati per formare un singolo abbinamento."),
                    ("Direzione Ricerca", f"{self.search_direction}", "Specifica se cercare candidati solo nel passato ('past_only'), solo nel futuro ('future_only') o in entrambe le direzioni ('both') rispetto alla data del movimento target."),
                    ("Soglia Analisi Residui", f"{self.soglia_residui / 100:.2f} €", "L'importo minimo che un movimento deve avere per essere considerato nella passata di recupero residui."),
                    ("Finestra Temporale Residui", f"{self.giorni_finestra_residui} giorni", "La finestra temporale, solitamente più ampia, usata specificamente per la passata di recupero residui."),
                ]
            }
        elif self.algorithm == 'progressive_balance':
            manual_content = {
                "title": "Algoritmo: Saldo Progressivo (Bilanciamento Continuo)",
                "description": [
                    ("Descrizione Generale", "Questo algoritmo simula il comportamento di un operatore che tenta di quadrare i conti scorrendo cronologicamente le liste. Somma progressivamente gli incassi e i versamenti e, appena i due totali coincidono, chiude il blocco e riparte da zero."),
                    ("Logica di Funzionamento", "1. Ordina separatamente Incassi (Dare) e Versamenti (Avere) per data.\n2. Mantiene due totali progressivi separati.\n3. Se il totale Dare è inferiore al totale Avere, aggiunge il prossimo incasso per 'recuperare'.\n4. Se il totale Avere è inferiore, aggiunge il prossimo versamento.\n5. Quando i totali si equivalgono (differenza zero), il gruppo di movimenti accumulati viene considerato riconciliato.\n6. **Reset/Forzatura**: Se il blocco accumulato supera la durata della finestra temporale senza quadrare, viene resettato (o forzato se l'opzione è attiva) per isolare l'errore e permettere la riconciliazione dei movimenti successivi."),
                    ("Casi d'uso ideali", "Ideale per situazioni in cui gli incassi vengono versati in blocco o viceversa, ma senza una corrispondenza 1-a-1 immediata. Gestisce naturalmente finestre temporali variabili tra incasso e versamento."),
                ],
                "params": [
                    ("Tolleranza", f"{self.tolleranza / 100:.2f} €", "Il margine di errore massimo per considerare il saldo progressivo 'azzerato' e quindi identificare un blocco di transazioni bilanciate."),
                ]
            }
            
        # Aggiunta di tutti i parametri comuni al report manuale
        common_params = [
             ("Strategia Ordinamento", self.sorting_strategy, "Criterio usato per ordinare i movimenti prima dell'elaborazione (es. Data)."),
             ("Direzione Ricerca", self.search_direction, "Direzione temporale preferenziale per gli abbinamenti."),
             ("Ottimizzazione Numba", "Attiva" if self.use_numba else "Disattiva", "Indica se è stato usato il motore di calcolo accelerato."),
             ("Mappatura Colonne", str(self.column_mapping), "Nomi delle colonne nel file originale mappate su Data/Dare/Avere."),
             ("Forza Chiusura su Timeout", "Sì" if self.ignore_tolerance else "No", "Se Sì, accetta blocchi non quadrati se superano la finestra temporale.")
        ]
        if 'params' in manual_content:
            manual_content['params'].extend(common_params)
        else:
            manual_content['params'] = common_params

        # Scrittura sul foglio
        row_cursor = 1
        ws.cell(row=row_cursor, column=1, value=manual_content.get('title')).font = title_font
        row_cursor += 2

        for header, text in manual_content.get('description', []):
            ws.cell(row=row_cursor, column=1, value=header).font = header_font
            row_cursor += 1
            cell = ws.cell(row=row_cursor, column=1, value=text)
            cell.alignment = Alignment(wrap_text=True, vertical='top')
            ws.merge_cells(start_row=row_cursor, start_column=1, end_row=row_cursor, end_column=5)
            row_cursor += 2
        
        ws.cell(row=row_cursor, column=1, value="Parametri Utilizzati in questa Esecuzione").font = title_font
        row_cursor += 1
        ws.cell(row=row_cursor, column=1, value="Parametro").font = header_font
        ws.cell(row=row_cursor, column=2, value="Valore").font = header_font
        ws.cell(row=row_cursor, column=3, value="Significato").font = header_font
        row_cursor += 1

        for name, value, desc in manual_content.get('params', []):
            ws.cell(row=row_cursor, column=1, value=name)
            ws.cell(row=row_cursor, column=2, value=value)
            cell = ws.cell(row=row_cursor, column=3, value=desc)
            cell.alignment = Alignment(wrap_text=True, vertical='top')
            ws.merge_cells(start_row=row_cursor, start_column=3, end_row=row_cursor, end_column=5)
            row_cursor += 1

        # Adatta larghezza colonne
        ws.column_dimensions['A'].width = 30
        ws.column_dimensions['B'].width = 20
        ws.column_dimensions['C'].width = 80

    def _crea_report_excel(self, output_file, original_df):
        """Salva i risultati in un file Excel multi-foglio."""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # --- FOGLIO MANUALE (NUOVO) ---
            self._crea_foglio_manuale(writer)

            # --- Foglio Abbinamenti ---
            if self.df_abbinamenti is not None and not self.df_abbinamenti.empty:
                df_abbinamenti_excel = self.df_abbinamenti.copy()
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

                # --- COLORAZIONE RIGHE PER PASSATA (MODIFICATA) ---
                ws = writer.sheets['Abbinamenti']
                
                fill_pass1 = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid") # Verde
                fill_pass2 = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid") # Giallo
                fill_pass3 = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid") # Rosso
                fill_progressive = PatternFill(start_color="DDEBF7", end_color="DDEBF7", fill_type="solid") # Blu chiaro

                if 'pass_name' in df_abbinamenti_excel.columns:
                    for i, row in df_abbinamenti_excel.iterrows():
                        pass_name = str(row['pass_name'])
                        fill = None
                        if self.algorithm == 'subset_sum':
                            if "Passata 1" in pass_name: fill = fill_pass1
                            elif "Passata 2" in pass_name: fill = fill_pass2
                            elif "Passata 3" in pass_name: fill = fill_pass3
                        elif self.algorithm == 'progressive_balance':
                            fill = fill_progressive
                        
                        if fill:
                            excel_row = i + 2
                            for col in range(1, len(df_abbinamenti_excel.columns) + 1):
                                ws.cell(row=excel_row, column=col).fill = fill

            # --- Fogli Non Riconciliati ---
            if self.dare_non_util is not None and not self.dare_non_util.empty:
                df_dare_report = self.dare_non_util[['indice_orig', 'Data', 'Dare']].copy()
                # FIX: Aggiunge 2 per allineare l'indice con la riga di Excel
                df_dare_report['indice_orig'] = df_dare_report['indice_orig'] + 2
                df_dare_report['Data'] = pd.to_datetime(df_dare_report['Data']).dt.strftime('%d/%m/%y')
                df_dare_report['Dare'] = df_dare_report['Dare'] / 100.0
                df_dare_report.rename(columns={'indice_orig': 'Indice Riga', 'Dare': 'Importo'}).to_excel(writer, sheet_name='DARE non utilizzati', index=False)

            if self.avere_non_riconc is not None and not self.avere_non_riconc.empty:
                df_avere_report = self.avere_non_riconc[['indice_orig', 'Data', 'Avere']].copy()
                # FIX: Aggiunge 2 per allineare l'indice con la riga di Excel
                df_avere_report['indice_orig'] = df_avere_report['indice_orig'] + 2
                df_avere_report['Data'] = pd.to_datetime(df_avere_report['Data']).dt.strftime('%d/%m/%y')
                df_avere_report['Avere'] = df_avere_report['Avere'] / 100.0
                df_avere_report.rename(columns={'indice_orig': 'Indice Riga', 'Avere': 'Importo'}).to_excel(writer, sheet_name='AVERE non riconciliati', index=False)

            # --- Foglio con i dati originali ---
            df_originale_report = original_df.copy()
            if 'Data' in df_originale_report.columns:
                 df_originale_report.sort_values(by=['Data', 'indice_orig'], inplace=True)
            if 'Dare' in df_originale_report.columns: df_originale_report['Dare'] = df_originale_report['Dare'] / 100
            if 'Avere' in df_originale_report.columns: df_originale_report['Avere'] = df_originale_report['Avere'] / 100
            if 'Dare' in df_originale_report.columns and 'Avere' in df_originale_report.columns:
                df_originale_report['Saldo Progressivo'] = (df_originale_report['Dare'] - df_originale_report['Avere']).cumsum()
            if 'Data' in df_originale_report.columns:
                df_originale_report['Data'] = pd.to_datetime(df_originale_report['Data']).dt.strftime('%d/%m/%Y')
                if 'indice_orig' in df_originale_report.columns and 'usato' not in df_originale_report.columns:
                    df_originale_report.drop(columns=['indice_orig'], inplace=True)
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

            # --- AGGIUNTA: Foglio Quadratura Mensile ---
            df_mensile = self._calcola_quadratura_mensile()
            if not df_mensile.empty:
                # Converti da centesimi a Euro (float) per permettere al grafico di funzionare
                cols_to_convert = [c for c in df_mensile.columns if c != 'Mese']
                for col in cols_to_convert:
                    df_mensile[col] = df_mensile[col] / 100.0
                
                # --- NUOVO: Colonna helper per il grafico ---
                # Per visualizzare correttamente, gli AVERE non abbinati devono essere negativi.
                df_mensile['AVERE non abbinati (Grafico)'] = -df_mensile['AVERE non abbinati']
                
                df_mensile.to_excel(writer, sheet_name='Quadratura Mensile', index=False)
                
                ws = writer.sheets['Quadratura Mensile']
                
                # Applica formattazione valuta alle celle (perché ora sono numeri puri)
                for col_idx, col_name in enumerate(df_mensile.columns, start=1):
                    if col_name != 'Mese' and '(Grafico)' not in col_name: # Escludi colonna helper
                        for row in range(2, len(df_mensile) + 2):
                            cell = ws.cell(row=row, column=col_idx)
                            cell.number_format = '#,##0.00 €'
                
                # Adatta larghezza colonne
                for idx, col in enumerate(df_mensile.columns):
                    if '(Grafico)' not in col: # Non adattare la colonna nascosta
                        max_len = min(50, max(df_mensile[col].astype(str).map(len).max() if not df_mensile.empty else 0, len(str(col)))) + 2
                        ws.column_dimensions[chr(65 + idx)].width = max_len

                # --- GRAFICO SBILANCI (MODIFICATO) ---
                try:
                    chart = BarChart()
                    chart.type = "col"
                    chart.style = 10
                    chart.title = "Composizione Sbilancio Mensile"
                    chart.y_axis.title = "Importo (€)"
                    chart.x_axis.title = "Mese"
                    chart.grouping = "stacked"
                    chart.overlap = 100
                    
                    # Colonne: Mese, DARE non abbinati, AVERE non abbinati (Grafico), Sbilancio Assorbito
                    col_mese_idx = df_mensile.columns.get_loc('Mese') + 1
                    col_dare_idx = df_mensile.columns.get_loc('DARE non abbinati') + 1
                    col_avere_grafico_idx = df_mensile.columns.get_loc('AVERE non abbinati (Grafico)') + 1
                    col_assorbito_idx = df_mensile.columns.get_loc('Sbilancio Assorbito (in match)') + 1
                    
                    cats = Reference(ws, min_col=col_mese_idx, min_row=2, max_row=len(df_mensile)+1)
                    
                    # Serie 1: DARE non abbinati
                    data_dare = Reference(ws, min_col=col_dare_idx, min_row=1, max_row=len(df_mensile)+1)
                    chart.add_data(data_dare, titles_from_data=True)
                    chart.series[0].title = "DARE non abbinati"
                    
                    # Serie 2: AVERE non abbinati (negativi)
                    data_avere = Reference(ws, min_col=col_avere_grafico_idx, min_row=1, max_row=len(df_mensile)+1)
                    chart.add_data(data_avere, titles_from_data=True)
                    chart.series[1].title = "AVERE non abbinati" # Il titolo è positivo per la legenda
                    
                    # Serie 3: Sbilancio Assorbito
                    data_assorbito = Reference(ws, min_col=col_assorbito_idx, min_row=1, max_row=len(df_mensile)+1)
                    chart.add_data(data_assorbito, titles_from_data=True)
                    chart.series[2].title = "Sbilancio Assorbito (in match)"

                    chart.set_categories(cats)
                    chart.shape = 4
                    chart.width = 25 # Larghezza in cm
                    chart.height = 12 # Altezza in cm
                    
                    # Posiziona il grafico sotto i dati, con un paio di righe di margine
                    chart_anchor = f"A{len(df_mensile) + 4}"
                    ws.add_chart(chart, chart_anchor)
                    
                    # Nascondi la colonna helper per il grafico
                    ws.column_dimensions[chr(65 + col_avere_grafico_idx - 1)].hidden = True
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
        Esegue l'intero processo di riconciliazione.

        1. Carica (o riceve) i dati.
        2. Separa i movimenti in DARE e AVERE.
        3. Esegue gli algoritmi di riconciliazione configurati.
        4. Genera statistiche e report.

        Args:
            input_file (str or pd.DataFrame): Percorso del file di input o DataFrame già caricato.
            output_file (str, optional): Percorso dove salvare il report Excel. Se None, non salva.
            verbose (bool): Se True, stampa log dettagliati su console.

        Returns:
            dict: Dizionario contenente le statistiche finali della riconciliazione.
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
            
            all_abbinamenti = []

            # --- SCELTA DELL'ALGORITMO ---
            # Se 'all', esegue prima il saldo progressivo (per i blocchi) poi il subset sum (per i residui)
            algorithms_to_run = []
            if self.algorithm == 'all':
                algorithms_to_run = ['progressive_balance', 'subset_sum']
            elif self.algorithm == 'progressive_balance':
                algorithms_to_run = ['progressive_balance']
            else: # subset_sum o default
                algorithms_to_run = ['subset_sum']

            for algo in algorithms_to_run:
                if algo == 'progressive_balance':
                    all_abbinamenti.extend(self._riconcilia_saldo_progressivo(verbose=verbose))
                elif algo == 'subset_sum':
                    all_abbinamenti.extend(self._riconcilia_subset_sum(verbose=verbose))

            # Finalizzazione comune
            self.dare_df['usato'] = self.dare_df['indice_orig'].isin(self.used_dare_indices)
            self.avere_df['usato'] = self.avere_df['indice_orig'].isin(self.used_avere_indices)
            self.df_abbinamenti = self._finalizza_abbinamenti(all_abbinamenti)

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