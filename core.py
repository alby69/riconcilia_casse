import pandas as pd
from itertools import combinations
from collections import deque
from datetime import datetime
import warnings
import numpy as np

warnings.filterwarnings('ignore')
from tqdm import tqdm
import sys

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

    def __init__(self, tolleranza=0.01, giorni_finestra=30, max_combinazioni=6, soglia_residui=100, giorni_finestra_residui=60, sorting_strategy="date", search_direction="both", column_mapping=None):
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
        return df

    def _separa_movimenti(self, df):
        """Separa il DataFrame in movimenti DARE e AVERE."""
        self.dare_df = df[df['Dare'] != 0][['indice_orig', 'Data', 'Dare']].copy()
        self.avere_df = df[df['Avere'] != 0][['indice_orig', 'Data', 'Avere']].copy()
       
        if self.sorting_strategy == "date":
            return self.dare_df.sort_values('Data', ascending=True), self.avere_df.sort_values('Data', ascending=True)
        elif self.sorting_strategy == "amount":
            return self.dare_df.sort_values('Dare', ascending=False), self.avere_df.sort_values('Avere', ascending=False)
        else:
            raise ValueError(f"Strategia di ordinamento non valida: '{self.sorting_strategy}'. Usare 'date' o 'amount'.")

    def _trova_abbinamenti(self, dare_row, avere_candidati_np, avere_indices_map, giorni_finestra, max_combinazioni):
        """Logica interna per trovare un abbinamento per un singolo DARE."""
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

    def _trova_abbinamenti_dare(self, avere_row, dare_candidati_np, dare_indices_map, giorni_finestra, max_combinazioni):
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
        if NUMBA_AVAILABLE:
            candidati_np_numba = np.array([(c['Dare'], c['indice_orig']) for c in candidati_da_modificare], dtype=np.int64)
            match_indices = _numba_find_combination(avere_importo, candidati_np_numba, max_combinazioni, self.tolleranza)
            if len(match_indices) > 0:
                match = [c for c in candidati_da_modificare if c['indice_orig'] in match_indices]
        else:
            match = self._trova_combinazioni_ricorsivo_py(avere_importo, candidati_da_modificare, max_combinazioni, self.tolleranza)

        if match:
            return {
                'dare_indices': [m['indice_orig'] for m in match],
                'dare_date': [m['Data'] for m in match],
                'dare_importi': [m['Dare'] for m in match],
                'avere_indices': [avere_row['indice_orig']],
                'avere_date': [avere_data],
                'avere_importi': [avere_importo],
                'somma_dare': sum(m['Dare'] for m in match),
                'differenza': abs(avere_importo - sum(m['Dare'] for m in match)),
                'tipo_match': f'Combinazione DARE {len(match)}'
            }
        return None

    def _esegui_passata_riconciliazione_dare(self, dare_df, avere_df, giorni_finestra, max_combinazioni, abbinamenti_list, title, verbose=True):
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
            search_direction="past_only", # La passata AVERE->DARE cerca solo nel passato
            find_function=self._trova_abbinamenti_dare,
            verbose=verbose
        )

    def _esegui_passata_generica(self, df_da_processare, df_candidati, col_da_processare, col_candidati, used_indices_candidati, giorni_finestra, max_combinazioni, abbinamenti_list, title, search_direction, find_function, verbose=True):
        """
        Funzione helper generica che esegue una passata di riconciliazione.
        Questa funzione astrae la logica comune tra le passate DARE->AVERE e AVERE->DARE.
        """
        if df_da_processare is None or df_da_processare.empty:
            return

        if verbose:
            print(f"\n{title}...")

        # Prepara le liste di record una sola volta
        records_da_processare = df_da_processare.to_dict('records')
        records_candidati = sorted(df_candidati.to_dict('records'), key=lambda x: x['Data']) if df_candidati is not None else []

        matches = []
        total_records = len(records_da_processare)
        processed_count = 0

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
                match = find_function(record_row, candidati_prefiltrati, None, giorni_finestra, max_combinazioni)
                if match:
                    matches.append(match)

        if verbose: print(f"\n   - Trovati {len(matches)} potenziali abbinamenti. Registrazione in corso...")
        for match in matches:
            match['pass_name'] = title
            self._registra_abbinamento(match, abbinamenti_list)
        if verbose: sys.stdout.write("\n   ✓ Completato.\n")

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
            search_direction=self.search_direction, # Usa la direzione principale
            find_function=self._trova_abbinamenti,
            verbose=verbose
        )

    def _riconcilia(self, dare_df, avere_df, verbose=True):
        """Orchestra il processo di riconciliazione in più passate."""
        
        abbinamenti = []

        # --- Passata 1: Riconciliazione Standard (DARE -> AVERE) ---
        self._esegui_passata_riconciliazione(
            dare_df, avere_df,
            self.giorni_finestra,
            self.max_combinazioni,
            abbinamenti,
            "Inizio Passata 1: Riconciliazione Standard (DARE -> AVERE)",
            verbose
        )

        # --- Passata 2: Riconciliazione Residui (DARE -> AVERE con finestra più ampia) ---
        self._esegui_passata_riconciliazione(
            dare_df, avere_df, self.giorni_finestra_residui, self.max_combinazioni, abbinamenti,
            f"Inizio Passata 2: Analisi Residui (Finestra: {self.giorni_finestra_residui}gg)", verbose
        )

        # --- Passata 3: Combinazione DARE per AVERE (AVERE -> DARE) ---
        # Eseguita come ultima risorsa per abbinare gli AVERE rimanenti.
        self._esegui_passata_riconciliazione_dare(
            dare_df, avere_df,
            self.giorni_finestra,
            self.max_combinazioni,
            abbinamenti,
            "Inizio Passata 3: Combinazione DARE per AVERE",
            verbose
        )

        # AGGIUNTA: Aggiorna le colonne 'usato' nei DataFrame originali una sola volta alla fine
        dare_df['usato'] = dare_df['indice_orig'].isin(self.used_dare_indices)
        avere_df['usato'] = avere_df['indice_orig'].isin(self.used_avere_indices)

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
            
        return dare_df, avere_df, df_abbinamenti

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

    def _crea_report_excel(self, output_file, original_df):
        """Salva i risultati in un file Excel multi-foglio."""
        # --- Foglio Originale ---
        # Crea una copia per evitare SettingWithCopyWarning e riconverti in euro
        df_originale_report = original_df.copy()
        
        # Riconverte gli importi da centesimi a float per la visualizzazione
        if 'Dare' in df_originale_report.columns:
            df_originale_report['Dare'] = df_originale_report['Dare'] / 100
        if 'Avere' in df_originale_report.columns:
            df_originale_report['Avere'] = df_originale_report['Avere'] / 100
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

            # --- Fogli Non Riconciliati ---
            if not self.dare_non_util.empty:
                df_dare_report = self.dare_non_util[['indice_orig', 'Data', 'Dare']].copy()
                # FIX: Aggiunge 2 per allineare l'indice con la riga di Excel
                df_dare_report['indice_orig'] = df_dare_report['indice_orig'] + 2
                df_dare_report['Data'] = pd.to_datetime(df_dare_report['Data']).dt.strftime('%d/%m/%y')
                df_dare_report['Dare'] = df_dare_report['Dare'].apply(lambda x: f"{x/100:.2f}".replace('.', ','))
                df_dare_report.rename(columns={'indice_orig': 'Indice Riga', 'Dare': 'Importo'}).to_excel(writer, sheet_name='DARE non utilizzati', index=False)
            else:
                pd.DataFrame(columns=['Indice Riga', 'Data', 'Importo']).to_excel(writer, sheet_name='DARE non utilizzati', index=False)
            if not self.avere_non_riconc.empty:
                df_avere_report = self.avere_non_riconc[['indice_orig', 'Data', 'Avere']].copy()
                # FIX: Aggiunge 2 per allineare l'indice con la riga di Excel
                df_avere_report['indice_orig'] = df_avere_report['indice_orig'] + 2
                df_avere_report['Data'] = pd.to_datetime(df_avere_report['Data']).dt.strftime('%d/%m/%y')
                df_avere_report['Avere'] = df_avere_report['Avere'].apply(lambda x: f"{x/100:.2f}".replace('.', ','))
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
                    '% USATI': [stats.get('% Incassi (DARE) utilizzati'), f"{stats.get('_raw_perc_dare_importo', 0):.2f}%"],
                    'Delta': [stats.get('Incassi (DARE) non utilizzati'), format_eur(stats.get('_raw_importo_dare_non_util', 0))]
                }, index=['Numero', 'Importo'])

                df_versamenti = pd.DataFrame({
                    'TOT': [stats.get('Totale Versamenti (AVERE)'), format_eur(self.avere_df['Avere'].sum() / 100)],
                    'USATI': [stats.get('Versamenti (AVERE) riconciliati'), format_eur(self.avere_df[self.avere_df['usato']]['Avere'].sum() / 100)],
                    '% USATI': [stats.get('% Versamenti (AVERE) riconciliati'), f"{stats.get('_raw_perc_avere_importo', 0):.2f}%"],
                    'Delta': [stats.get('Versamenti (AVERE) non riconciliati'), format_eur(stats.get('_raw_importo_avere_non_riconc', 0))]
                }, index=['Numero', 'Importo'])

                delta_conteggio = stats.get('Incassi (DARE) non utilizzati', 0) - stats.get('Versamenti (AVERE) non riconciliati', 0)
                df_confronto = pd.DataFrame({
                    'Delta Conteggio': [delta_conteggio],
                    'Delta Importo (€)': [stats.get('Delta finale (DARE - AVERE)')]
                }, index=['Incassi vs Versamenti'])

                df_incassi.to_excel(writer, sheet_name='Statistiche', startrow=2)
                df_versamenti.to_excel(writer, sheet_name='Statistiche', startrow=8)
                df_confronto.to_excel(writer, sheet_name='Statistiche', startrow=14)

            sheet_stats = writer.sheets['Statistiche']
            sheet_stats.cell(row=1, column=1, value="Riepilogo Incassi (DARE)")
            sheet_stats.cell(row=7, column=1, value="Riepilogo Versamenti (AVERE)")
            sheet_stats.cell(row=13, column=1, value="Confronto Sbilancio Finale")

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

        return {
            'Totale Incassi (DARE)': num_dare_tot,
            'Incassi (DARE) utilizzati': num_dare_usati,
            '% Incassi (DARE) utilizzati': f"{(num_dare_usati / num_dare_tot * 100) if num_dare_tot > 0 else 0:.1f}%",
            'Incassi (DARE) non utilizzati': num_dare_tot - num_dare_usati,
            
            'Totale Versamenti (AVERE)': num_avere_tot,
            'Versamenti (AVERE) riconciliati': num_avere_usati,
            '% Versamenti (AVERE) riconciliati': f"{(num_avere_usati / num_avere_tot * 100) if num_avere_tot > 0 else 0:.1f}%",
            'Versamenti (AVERE) non riconciliati': num_avere_tot - num_avere_usati,

            'Delta finale (DARE - AVERE)': f"{(importo_dare_non_util - importo_avere_non_riconc):,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."),
            
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

            if verbose: print("2. Separazione e ordinamento movimenti DARE/AVERE...")
            dare_df, avere_df = self._separa_movimenti(df)

            if verbose: print("3. Avvio passate di riconciliazione...")
            # _riconcilia ora restituisce i DataFrame aggiornati con la colonna 'usato'
            self.dare_df, self.avere_df, self.df_abbinamenti = self._riconcilia(dare_df, avere_df, verbose=verbose)

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
    # Stack per l'esplorazione iterativa, per evitare limiti di ricorsione di Python.
    # Contiene tuple: (indice_partenza, somma_corrente, numero_elementi_nel_percorso)
    stack = [(0, 0, 0)]
    
    # Array per tenere traccia del percorso della combinazione corrente (indici dei candidati)
    path = np.full(max_combinazioni, -1, dtype=np.int64)

    while len(stack) > 0:
        start_index, current_sum, path_len = stack.pop()

        # Esplora i candidati a partire dall'indice corrente
        for i in range(start_index, len(candidati_np)):
            new_sum = current_sum + candidati_np[i, 0]
            
            # Se la somma è nel range di tolleranza e abbiamo più di un elemento, abbiamo trovato una soluzione.
            if abs(target - new_sum) <= tolleranza and path_len + 1 > 1:
                path[path_len] = i
                # Estrai gli indici originali dalla combinazione trovata e restituiscili
                result_indices = np.full(path_len + 1, 0, dtype=np.int64)
                for k in range(path_len + 1):
                    result_indices[k] = candidati_np[path[k], 1]
                return result_indices

            # Se non abbiamo superato il limite di combinazioni e la somma è ancora inferiore al target,
            # aggiungi un nuovo stato allo stack per continuare l'esplorazione.
            if path_len + 1 < max_combinazioni and new_sum < target + tolleranza:
                path[path_len] = i
                stack.append((i + 1, new_sum, path_len + 1))

    # Se lo stack si svuota, non è stata trovata nessuna combinazione.
    return np.empty(0, dtype=np.int64)