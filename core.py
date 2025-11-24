import pandas as pd
from itertools import combinations
from tqdm import tqdm
import sys

class RiconciliatoreContabile:
    """Contiene la logica di business per la riconciliazione."""

    def __init__(self, tolleranza=0.01, giorni_finestra=30, max_combinazioni=6, soglia_residui=100, giorni_finestra_residui=60, sorting_strategy="date", search_direction="both"):
        self.tolleranza = tolleranza
        self.giorni_finestra = giorni_finestra
        self.max_combinazioni = max_combinazioni
        self.soglia_residui = soglia_residui
        self.giorni_finestra_residui = giorni_finestra_residui
        self.sorting_strategy = sorting_strategy
        self.search_direction = search_direction


    def carica_file(self, file_path):
        """Carica un file Excel o CSV in un DataFrame."""
        if str(file_path).endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        required_cols = {'Data', 'Dare', 'Avere'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Il file deve contenere le colonne: {', '.join(required_cols)}")        
        
        # Converte la colonna 'Data', trasformando qualsiasi valore non valido (es. testo) in NaT (Not a Time).
        # Questo rende il caricamento robusto a righe di intestazione/piè di pagina nel file.
        df['Data'] = pd.to_datetime(df['Data'], errors='coerce', dayfirst=True)
        
        # Rimuove le righe dove la data non è stata riconosciuta (quelle diventate NaT).
        df.dropna(subset=['Data'], inplace=True)
        
        df[['Dare', 'Avere']] = df[['Dare', 'Avere']].fillna(0)
        df['indice_orig'] = df.index
        return df

    def separa_movimenti(self, df):
        """Separa il DataFrame in movimenti DARE e AVERE."""
        dare_df = df[df['Dare'] != 0][['indice_orig', 'Data', 'Dare']].copy()
        avere_df = df[df['Avere'] != 0][['indice_orig', 'Data', 'Avere']].copy()
        
        dare_df['usato'] = False
        avere_df['usato'] = False

        if self.sorting_strategy == "date":
            return dare_df.sort_values('Data', ascending=True), avere_df.sort_values('Data', ascending=True)
        elif self.sorting_strategy == "amount":
            return dare_df.sort_values('Dare', ascending=False), avere_df.sort_values('Avere', ascending=False)
        else:
            raise ValueError(f"Strategia di ordinamento non valida: '{self.sorting_strategy}'. Usare 'date' o 'amount'.")

    def _trova_abbinamenti(self, dare_row, avere_df, giorni_finestra, max_combinazioni):
        """Logica interna per trovare un abbinamento per un singolo DARE."""
        dare_importo = dare_row['Dare']
        dare_data = dare_row['Data']

        # Calcola la finestra temporale in base alla direzione di ricerca
        min_data_window, max_data_window = self._calcola_finestra_temporale(dare_data, giorni_finestra, self.search_direction)

        
        # Filtra AVERE per finestra temporale e importo
        min_data = dare_data - pd.Timedelta(days=giorni_finestra)
        max_data = dare_data + pd.Timedelta(days=giorni_finestra)

        candidati_avere = avere_df[
            (~avere_df['usato']) &
            (avere_df['Data'].between(min_data_window, max_data_window)) &
            (avere_df['Avere'] <= dare_importo + self.tolleranza)
        ].copy()

        if candidati_avere.empty:
            return None
        
        # 1. Cerca match esatto 1-a-1
        match_esatto = candidati_avere[abs(candidati_avere['Avere'] - dare_importo) <= self.tolleranza]
        if not match_esatto.empty:
            best_match = match_esatto.iloc[0]
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
        candidati_avere = candidati_avere.sort_values('Avere', ascending=False).to_dict('records')
        
        match = self._trova_combinazioni_ricorsivo(
            target=dare_importo,
            candidati=candidati_avere,
            max_combinazioni=max_combinazioni,
            parziale=[],
            start_index=0
        )

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

    def _trova_combinazioni_ricorsivo(self, target, candidati, max_combinazioni, parziale, start_index):
        """Funzione ricorsiva ottimizzata per il subset-sum con pruning."""
        somma_parziale = sum(c['Avere'] for c in parziale)

        # Condizione di successo
        if abs(target - somma_parziale) <= self.tolleranza and len(parziale) > 1:
            return parziale

        # Condizioni di pruning (potatura)
        if len(parziale) >= max_combinazioni or start_index >= len(candidati) or somma_parziale > target + self.tolleranza:
            return None

        for i in range(start_index, len(candidati)):
            candidato = candidati[i]
            
            # Aggiungi il candidato alla soluzione parziale
            parziale.append(candidato)
            
            # Chiamata ricorsiva
            risultato = self._trova_combinazioni_ricorsivo(target, candidati, max_combinazioni, parziale, i + 1)
            if risultato:
                return risultato
            
            # Backtrack: rimuovi il candidato e prova il prossimo
            parziale.pop()
            
        return None

    def _trova_abbinamenti_dare(self, avere_row, dare_df, giorni_finestra, max_combinazioni):
        """Logica per trovare combinazioni di DARE che corrispondono a un AVERE."""
        avere_importo = avere_row['Avere']
        avere_data = avere_row['Data']

        # Calcola la finestra temporale in base alla direzione di ricerca
        # Per la combinazione DARE per AVERE, è più logico che i DARE siano precedenti o uguali all'AVERE.
        # Quindi, anche se search_direction è 'both' o 'future_only', qui forziamo 'past_only' per i DARE.
        # Questo perché i DARE sono gli incassi che "formano" il versamento (AVERE).
        min_data_window, max_data_window = self._calcola_finestra_temporale(avere_data, giorni_finestra, "past_only")

        candidati_dare = dare_df[
            (~dare_df['usato']) &
            (dare_df['Data'].between(min_data_window, max_data_window)) &
            (dare_df['Dare'] <= avere_importo + self.tolleranza)
        ].copy()

        if candidati_dare.empty:
            return None

        # Cerca combinazioni multiple di DARE in modo ottimizzato
        candidati_dare_list = candidati_dare.sort_values('Dare', ascending=False).to_dict('records')
        
        # Riusiamo la stessa logica ricorsiva, adattandola per i DARE
        # Nota: rinominiamo 'Avere' in 'Dare' per la funzione generica
        for c in candidati_dare_list:
            c['Avere'] = c.pop('Dare')

        match = self._trova_combinazioni_ricorsivo(
            target=avere_importo,
            candidati=candidati_dare_list,
            max_combinazioni=max_combinazioni,
            parziale=[],
            start_index=0
        )

        if match:
            return {
                'dare_indices': [m['indice_orig'] for m in match],
                'dare_date': [m['Data'] for m in match],
                'dare_importi': [m['Avere'] for m in match], # 'Avere' qui è corretto perché lo abbiamo rinominato
                'avere_indices': [avere_row['indice_orig']],
                'avere_date': [avere_data],
                'avere_importi': [avere_importo],
                'somma_dare': sum(m['Avere'] for m in match),
                'differenza': abs(avere_importo - sum(m['Avere'] for m in match)),
                'tipo_match': f'Combinazione DARE {len(match)}'
            }
        return None

    def _esegui_passata_riconciliazione_dare(self, dare_df, avere_df, giorni_finestra, max_combinazioni, abbinamenti_list, title, verbose=True):
        """Esegue una passata cercando combinazioni di DARE per abbinare AVERE."""
        avere_da_processare = avere_df[~avere_df['usato']].copy()

        if avere_da_processare.empty:
            return

        if verbose:
            print(f"\n{title}...")

        total_avere = len(avere_da_processare)
        processed_count = 0

        for index, avere_row in avere_da_processare.iterrows():
            if verbose:
                processed_count += 1
                percentuale = (processed_count / total_avere) * 100
                # Aggiorna l'output sulla stessa riga
                sys.stdout.write(f"\r   - Avanzamento: {percentuale:.1f}% ({processed_count}/{total_avere})")
                sys.stdout.flush()

            match = self._trova_abbinamenti_dare(avere_row, dare_df, giorni_finestra, max_combinazioni)
            if match:
                match['pass_name'] = title # Add pass name
                self._registra_abbinamento(dare_df, avere_df, match, abbinamenti_list)

        if verbose:
            sys.stdout.write("\n   ✓ Completato.\n") # Vai a capo alla fine

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
        """Esegue una passata di riconciliazione e aggiorna i DataFrame."""
        
        dare_da_processare = dare_df[~dare_df['usato']].copy()
        
        if dare_da_processare.empty:
            return

        if verbose:
            print(f"\n{title}...")
        
        total_dare = len(dare_da_processare)
        processed_count = 0

        for index, dare_row in dare_da_processare.iterrows():
            if verbose:
                processed_count += 1
                percentuale = (processed_count / total_dare) * 100
                # Aggiorna l'output sulla stessa riga
                sys.stdout.write(f"\r   - Avanzamento: {percentuale:.1f}% ({processed_count}/{total_dare})")
                sys.stdout.flush()
            match = self._trova_abbinamenti(dare_row, avere_df, giorni_finestra, max_combinazioni)
            if match:
                match['pass_name'] = title # Add pass name
                # Correzione del bug: registra l'abbinamento e marca subito i movimenti come usati.
                self._registra_abbinamento(dare_df, avere_df, match, abbinamenti_list)
        
        if verbose:
            sys.stdout.write("\n   ✓ Completato.\n")

    def riconcilia(self, dare_df, avere_df, verbose=True):
        """Orchestra il processo di riconciliazione in più passate."""
        
        abbinamenti = []

        # --- Passata 1: Riconciliazione Standard ---
        self._esegui_passata_riconciliazione(
            dare_df, avere_df,
            self.giorni_finestra,
            self.max_combinazioni,
            abbinamenti,
            "Inizio Passata 1: Riconciliazione Standard",
            verbose
        )

        # --- Passata 2: Riconciliazione Residui ---
        # Seleziona i DARE non usati sopra una certa soglia
        # Questa passata ora viene eseguita prima della combinazione dei DARE per massimizzare l'efficacia.
        dare_residui = dare_df[(~dare_df['usato']) & (dare_df['Dare'] >= self.soglia_residui)]
        if not dare_residui.empty:
            self._esegui_passata_riconciliazione(
                dare_df, avere_df,
                self.giorni_finestra_residui, # Usa la finestra temporale più ampia
                self.max_combinazioni,
                abbinamenti,
                f"Inizio Passata 2: Analisi Residui > {self.soglia_residui}€ (Finestra: {self.giorni_finestra_residui}gg)",
                verbose
            )

        # --- Passata 3: Combinazione DARE per AVERE ---
        # Eseguita come ultima risorsa per abbinare gli AVERE rimanenti.
        self._esegui_passata_riconciliazione_dare(
            dare_df, avere_df,
            self.giorni_finestra,
            self.max_combinazioni,
            abbinamenti,
            "Inizio Passata 3: Combinazione DARE per AVERE",
            verbose
        )

        # Creazione del DataFrame finale degli abbinamenti
        if abbinamenti:
            df_abbinamenti = pd.DataFrame(abbinamenti)
            # Gestione colonne mancanti (es. 'somma_dare' vs 'somma_avere')
            if 'somma_dare' in df_abbinamenti.columns and 'somma_avere' not in df_abbinamenti.columns:
                df_abbinamenti['somma_avere'] = df_abbinamenti['somma_dare']
            
            # Per ordinare correttamente, estrai un valore singolo dalle colonne che potrebbero contenere liste.
            # Usiamo la prima data e la somma degli importi come chiavi di ordinamento.
            df_abbinamenti['sort_date'] = df_abbinamenti['dare_date'].apply(lambda x: x[0] if isinstance(x, list) else x)
            df_abbinamenti['sort_importo'] = df_abbinamenti['dare_importi'].apply(lambda x: sum(x) if isinstance(x, list) else x)
            
            df_abbinamenti = df_abbinamenti.sort_values(by=['sort_date', 'sort_importo'], ascending=[True, False]).drop(columns=['sort_date', 'sort_importo'])
            df_abbinamenti = df_abbinamenti[[
                'dare_indices', 'dare_date', 'dare_importi', 'avere_data', 'num_avere', 'avere_indices', 
                'avere_importi', 'somma_avere', 'differenza', 'tipo_match', 'pass_name' # Assicura che pass_name sia mantenuto
            ]]
        else:
            df_abbinamenti = pd.DataFrame(columns=[
                'dare_indices', 'dare_date', 'dare_importi', 'avere_data', 'num_avere', 'avere_indices', 
                'avere_importi', 'somma_avere', 'differenza', 'tipo_match', 'pass_name'
            ])
            
        return dare_df, avere_df, df_abbinamenti

    def _registra_abbinamento(self, dare_df, avere_df, match, abbinamenti_list):
        """Marca gli elementi come 'usati' e registra l'abbinamento."""
        dare_indices_orig = match['dare_indices']
        avere_indices_orig = match['avere_indices']

        dare_df.loc[dare_df['indice_orig'].isin(dare_indices_orig), 'usato'] = True
        avere_df.loc[avere_df['indice_orig'].isin(avere_indices_orig), 'usato'] = True

        # Aggiungi ai risultati formattati
        # Ensure 'pass_name' is included
        abbinamenti_list.append({
            'dare_indices': match.get('dare_indices', []),
            'dare_date': match.get('dare_date', []),
            'dare_importi': match.get('dare_importi', []),
            'avere_data': min(match['avere_date']) if match.get('avere_date') else None,
            'num_avere': len(match.get('avere_indices', [])),
            'avere_indices': match.get('avere_indices', []),
            'avere_importi': match.get('avere_importi', []),
            'somma_avere': match.get('somma_avere', match.get('somma_dare', 0)),
            'differenza': match.get('differenza', 0),
            'tipo_match': match.get('tipo_match', 'N/D'),
            'pass_name': match.get('pass_name', 'N/D')
        })