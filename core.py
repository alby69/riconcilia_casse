import pandas as pd
from itertools import combinations
from pathlib import Path
from datetime import datetime
import warnings

import dask.bag as db

warnings.filterwarnings('ignore')
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
        
        # Stato interno che verrà popolato durante l'esecuzione
        self.dare_df = self.avere_df = self.df_abbinamenti = None
        self.dare_non_util = self.avere_non_riconc = None

        # Ottimizzazione: Usare set per tenere traccia degli indici usati
        self.used_dare_indices = set()
        self.used_avere_indices = set()

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
            (~avere_df['indice_orig'].isin(self.used_avere_indices)) & # Modifica qui
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
        
        # Aggiunta cache per la memoization
        cache = {}
        somma_totale_candidati = sum(c['Avere'] for c in candidati_avere)
        match = self._trova_combinazioni_ricorsivo(
            target=dare_importo,
            candidati=candidati_avere,
            max_combinazioni=max_combinazioni,
            parziale=[],
            start_index=0,
            somma_parziale=0.0,
            cache=cache,
            remaining_sum=somma_totale_candidati
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

    def _trova_combinazioni_ricorsivo(self, target, candidati, max_combinazioni, parziale, start_index, somma_parziale, cache, remaining_sum):
        """Funzione ricorsiva ottimizzata per il subset-sum con pruning."""
        # --- OTTIMIZZAZIONE: MEMOIZATION ---
        cache_key = (start_index, round(somma_parziale, 2)) # Arrotonda per evitare problemi di precisione float
        if cache_key in cache:
            return cache[cache_key]

        # Condizione di successo
        if abs(target - somma_parziale) <= self.tolleranza and len(parziale) > 1:
            return parziale

        # Condizioni di pruning (potatura)
        if len(parziale) >= max_combinazioni or start_index >= len(candidati) or somma_parziale > target + self.tolleranza:
            return None
        
        # --- NUOVA OTTIMIZZAZIONE: PRUNING AGGRESSIVO ---
        # Se la somma parziale più tutto ciò che rimane non può raggiungere il target, fermati.
        if somma_parziale + remaining_sum < target - self.tolleranza:
            return None

        for i in range(start_index, len(candidati)):
            candidato = candidati[i]
            
            # La somma rimanente per la prossima chiamata ricorsiva sarà quella attuale meno il candidato corrente.
            next_remaining_sum = remaining_sum - candidato['Avere']
            # Aggiungi il candidato alla soluzione parziale
            parziale.append(candidato)
            
            # Chiamata ricorsiva
            risultato = self._trova_combinazioni_ricorsivo(
                target, 
                candidati, 
                max_combinazioni, 
                parziale, i + 1, 
                somma_parziale + candidato['Avere'],
                cache,
                next_remaining_sum
            )
            if risultato:
                # Non salvare nella cache il percorso di successo, solo i fallimenti
                return risultato
            
            # Backtrack: rimuovi il candidato e prova il prossimo
            parziale.pop()
        
        # Se nessun percorso da qui ha portato a una soluzione, salva il fallimento nella cache
        cache[cache_key] = None
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
            (~dare_df['indice_orig'].isin(self.used_dare_indices)) & # Modifica qui
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

        cache = {} # Cache anche qui
        somma_totale_candidati = sum(c['Avere'] for c in candidati_dare_list)
        match = self._trova_combinazioni_ricorsivo(
            target=avere_importo,
            candidati=candidati_dare_list,
            max_combinazioni=max_combinazioni,
            parziale=[],
            start_index=0,
            somma_parziale=0.0,
            cache=cache,
            remaining_sum=somma_totale_candidati
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
        avere_da_processare = avere_df[~avere_df['indice_orig'].isin(self.used_avere_indices)].copy() # Modifica qui

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
                match['pass_name'] = title
                self._registra_abbinamento(match, abbinamenti_list) # Modifica qui

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
        # Filtra i DARE che non sono ancora stati usati
        dare_da_processare = dare_df[~dare_df['indice_orig'].isin(self.used_dare_indices)].copy() # Modifica qui
        
        if dare_da_processare.empty:
            return

        if verbose:
            print(f"\n{title}...")
        
        # --- OTTIMIZZAZIONE CON DASK ---
        # Converti il DataFrame in una lista di record per Dask
        dare_records = dare_da_processare.to_dict('records')

        # Crea un Dask Bag e mappa la funzione di abbinamento in parallelo
        bag = db.from_sequence(dare_records)
        
        # Usiamo una lambda per adattare la firma della funzione
        matches = bag.map(lambda dare_row: self._trova_abbinamenti(dare_row, avere_df, giorni_finestra, max_combinazioni)).compute()

        # Filtra i risultati nulli e registra gli abbinamenti trovati
        # Questo blocco rimane sequenziale, ma è molto veloce.
        valid_matches = [m for m in matches if m is not None]
        
        if verbose: print(f"\n   - Trovati {len(valid_matches)} potenziali abbinamenti. Registrazione in corso...")
        
        for match in valid_matches:
            if match: # Add pass name
                match['pass_name'] = title
                self._registra_abbinamento(match, abbinamenti_list) # Modifica qui
        if verbose:
            sys.stdout.write("\n   ✓ Completato.\n")

    def _riconcilia(self, dare_df, avere_df, verbose=True):
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
        # Filtra i DARE che non sono ancora stati usati e che superano la soglia
        dare_residui = dare_df[
            (~dare_df['indice_orig'].isin(self.used_dare_indices)) & # Modifica qui
            (dare_df['Dare'] >= self.soglia_residui)
        ]
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

        # AGGIUNTA: Aggiorna le colonne 'usato' nei DataFrame originali una sola volta alla fine
        dare_df['usato'] = dare_df['indice_orig'].isin(self.used_dare_indices)
        avere_df['usato'] = avere_df['indice_orig'].isin(self.used_avere_indices)

        # Colonne attese nel DataFrame finale
        final_columns = [
            'dare_indices', 'dare_date', 'dare_importi', 'avere_data', 'num_avere', 'avere_indices', 
            'avere_importi', 'somma_avere', 'differenza', 'tipo_match', 'pass_name'
        ]

        # Creazione del DataFrame finale degli abbinamenti
        if abbinamenti:
            df_abbinamenti = pd.DataFrame(abbinamenti)
            # Gestione colonne mancanti (es. 'somma_dare' vs 'somma_avere')
            if 'somma_dare' in df_abbinamenti.columns and 'somma_avere' not in df_abbinamenti.columns:
                df_abbinamenti['somma_avere'] = df_abbinamenti['somma_dare']
            df_abbinamenti['sort_date'] = df_abbinamenti['dare_date'].apply(lambda x: x[0] if isinstance(x, list) else x)
            df_abbinamenti['sort_importo'] = df_abbinamenti['dare_importi'].apply(lambda x: sum(x) if isinstance(x, list) else x)
            df_abbinamenti = df_abbinamenti.sort_values(by=['sort_date', 'sort_importo'], ascending=[True, False]).drop(columns=['sort_date', 'sort_importo'])
            df_abbinamenti = df_abbinamenti.reindex(columns=final_columns) # Assicura che tutte le colonne esistano
        else:
            df_abbinamenti = pd.DataFrame(columns=final_columns)
            
        return dare_df, avere_df, df_abbinamenti

    def _registra_abbinamento(self, match, abbinamenti_list): # Rimosso dare_df, avere_df dagli argomenti
        """Marca gli elementi come 'usati' e registra l'abbinamento."""
        dare_indices_orig = match['dare_indices']
        avere_indices_orig = match['avere_indices']

        # Aggiungi gli indici ai set di indici usati
        self.used_dare_indices.update(dare_indices_orig)
        self.used_avere_indices.update(avere_indices_orig)

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

    def _crea_report_excel(self, output_file):
        """Salva i risultati in un file Excel multi-foglio."""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # --- Foglio Abbinamenti ---
            df_abbinamenti_excel = self.df_abbinamenti.copy()
            if not df_abbinamenti_excel.empty:
                def format_list(data, is_float=False):
                    if not isinstance(data, list): return data
                    items = [i + 2 for i in data] if not is_float else [f"{i:.2f}" for i in data]
                    return ', '.join(map(str, items))
                
                for col in ['dare_indices', 'avere_indices']: df_abbinamenti_excel[col] = df_abbinamenti_excel[col].apply(lambda x: format_list(x))
                for col in ['dare_importi', 'avere_importi']: df_abbinamenti_excel[col] = df_abbinamenti_excel[col].apply(lambda x: format_list(x, is_float=True))
                df_abbinamenti_excel['dare_date'] = df_abbinamenti_excel['dare_date'].apply(lambda x: ', '.join([d.strftime('%d/%m/%y') for d in x]) if isinstance(x, list) else x.strftime('%d/%m/%y'))
                df_abbinamenti_excel['avere_data'] = pd.to_datetime(df_abbinamenti_excel['avere_data']).dt.strftime('%d/%m/%y')
            
            df_abbinamenti_excel.to_excel(writer, sheet_name='Abbinamenti', index=False)

            # --- Fogli Non Riconciliati ---
            if not self.dare_non_util.empty:
                df_dare_report = self.dare_non_util[['indice_orig', 'Data', 'Dare']].copy()
                df_dare_report['Data'] = pd.to_datetime(df_dare_report['Data']).dt.strftime('%d/%m/%y')
                df_dare_report.rename(columns={'indice_orig': 'Indice Riga', 'Dare': 'Importo'}).to_excel(writer, sheet_name='DARE non utilizzati', index=False)
            else:
                pd.DataFrame(columns=['Indice Riga', 'Data', 'Importo']).to_excel(writer, sheet_name='DARE non utilizzati', index=False)

            if not self.avere_non_riconc.empty:
                df_avere_report = self.avere_non_riconc[['indice_orig', 'Data', 'Avere']].copy()
                df_avere_report['Data'] = pd.to_datetime(df_avere_report['Data']).dt.strftime('%d/%m/%y')
                df_avere_report.rename(columns={'indice_orig': 'Indice Riga', 'Avere': 'Importo'}).to_excel(writer, sheet_name='AVERE non riconciliati', index=False)
            else:
                pd.DataFrame(columns=['Indice Riga', 'Data', 'Importo']).to_excel(writer, sheet_name='AVERE non riconciliati', index=False)

            # --- Foglio Statistiche ---
            stats = self.get_stats()
            def format_eur(value): return f"{value:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")

            df_incassi = pd.DataFrame({
                'TOT': [stats['Totale Incassi (DARE)'], format_eur(self.dare_df['Dare'].sum())],
                'USATI': [stats['Incassi (DARE) utilizzati'], format_eur(self.dare_df[self.dare_df['usato']]['Dare'].sum())],
                '% USATI': [stats['% Incassi (DARE) utilizzati'], f"{stats['_raw_perc_dare_importo']:.2f}%"],
                'Delta': [stats['Incassi (DARE) non utilizzati'], format_eur(stats['_raw_importo_dare_non_util'])]
            }, index=['Numero', 'Importo'])

            df_versamenti = pd.DataFrame({
                'TOT': [stats['Totale Versamenti (AVERE)'], format_eur(self.avere_df['Avere'].sum())],
                'USATI': [stats['Versamenti (AVERE) riconciliati'], format_eur(self.avere_df[self.avere_df['usato']]['Avere'].sum())],
                '% USATI': [stats['% Versamenti (AVERE) riconciliati'], f"{stats['_raw_perc_avere_importo']:.2f}%"],
                'Delta': [stats['Versamenti (AVERE) non riconciliati'], format_eur(stats['_raw_importo_avere_non_riconc'])]
            }, index=['Numero', 'Importo'])

            df_confronto = pd.DataFrame({
                'Delta Conteggio': [stats['Incassi (DARE) non utilizzati'] - stats['Versamenti (AVERE) non riconciliati']],
                'Delta Importo (€)': [stats['Delta finale (DARE - AVERE)']]
            }, index=['Incassi vs Versamenti'])

            df_incassi.to_excel(writer, sheet_name='Statistiche', startrow=2)
            df_versamenti.to_excel(writer, sheet_name='Statistiche', startrow=8)
            df_confronto.to_excel(writer, sheet_name='Statistiche', startrow=14)

            sheet_stats = writer.sheets['Statistiche']
            sheet_stats.cell(row=1, column=1, value="Riepilogo Incassi (DARE)")
            sheet_stats.cell(row=7, column=1, value="Riepilogo Versamenti (AVERE)")
            sheet_stats.cell(row=13, column=1, value="Confronto Sbilancio Finale")

    def get_stats(self):
        """Calcola e restituisce un dizionario completo di statistiche."""
        if self.dare_df is None or self.avere_df is None or 'usato' not in self.dare_df.columns: return {}

        num_dare_tot = len(self.dare_df)
        num_dare_usati = int(self.dare_df['usato'].sum()) # Ora la colonna 'usato' esiste
        imp_dare_tot = self.dare_df['Dare'].sum()
        imp_dare_usati = self.dare_df[self.dare_df['usato']]['Dare'].sum()

        num_avere_tot = len(self.avere_df)
        num_avere_usati = int(self.avere_df['usato'].sum()) # Ora la colonna 'usato' esiste
        imp_avere_tot = self.avere_df['Avere'].sum()
        imp_avere_usati = self.avere_df[self.avere_df['usato']]['Avere'].sum()

        # Ricalcola dare_non_util e avere_non_riconc basandosi sulla colonna 'usato' aggiornata
        importo_dare_non_util = self.dare_non_util['Dare'].sum() if self.dare_non_util is not None else 0
        importo_avere_non_riconc = self.avere_non_riconc['Avere'].sum() if self.avere_non_riconc is not None else 0

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
        try:
            # Reset degli indici usati per ogni esecuzione, importante per l'ottimizzatore
            self.used_dare_indices = set()
            self.used_avere_indices = set()

            # --- MODIFICA: Gestione flessibile dell'input ---
            # L'ottimizzatore passa un DataFrame per efficienza, main.py passa un path.
            if isinstance(input_file, pd.DataFrame):
                if verbose: print("1. Utilizzo del DataFrame pre-caricato.")
                df = input_file.copy() # Usa una copia per evitare effetti collaterali
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
                self._crea_report_excel(output_file)
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