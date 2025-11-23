"""
Core del sistema di riconciliazione.
Contiene la classe RiconciliatoreContabile con tutta la logica di matching.
"""

import pandas as pd
from datetime import timedelta
from itertools import combinations
import math
from tqdm import tqdm

class RiconciliatoreContabile:
    """Classe per riconciliare movimenti Dare e Avere"""
    
    def __init__(self, tolleranza=0.01, giorni_finestra=30, max_combinazioni=6, soglia_residui=100, giorni_finestra_residui=60):
        self.tolleranza = tolleranza
        self.giorni_finestra = giorni_finestra
        self.giorni_finestra_residui = giorni_finestra_residui
        self.max_combinazioni = max_combinazioni
        self.soglia_residui = soglia_residui
        self.risultati = []

    def carica_file(self, file_path):
        """Carica file Excel o CSV in base all'estensione."""
        p = pd.read_excel(file_path) if str(file_path).endswith(('.xlsx', '.xls')) else pd.read_csv(file_path, sep=';', decimal=',')
        
        colonne_richieste = ['Data', 'Dare', 'Avere']
        if not all(col in p.columns for col in colonne_richieste):
            raise ValueError(f"Il file deve contenere le colonne: {colonne_richieste}")

        p['Data'] = pd.to_datetime(p['Data'], dayfirst=True, errors='coerce')
        p['Dare'] = pd.to_numeric(p['Dare'], errors='coerce').fillna(0)
        p['Avere'] = pd.to_numeric(p['Avere'], errors='coerce').fillna(0)
        p['indice_orig'] = p.index
        return p

    def separa_movimenti(self, df):
        dare = df[df['Dare'] > 0].copy()
        avere = df[df['Avere'] > 0].copy()
        dare['usato'] = False
        avere['usato'] = False
        dare = dare.sort_values(['Data', 'Dare'], ascending=[True, False]).reset_index(drop=True)
        avere = avere.sort_values(['Data', 'Avere'], ascending=[True, False]).reset_index(drop=True)
        return dare, avere

    def trova_match_esatto(self, avere_row, dare_df):
        importo_avere = avere_row['Avere']
        data_avere = avere_row['Data']
        dare_finestra = dare_df[
            (dare_df['Data'] >= data_avere - timedelta(days=self.giorni_finestra)) &
            (dare_df['Data'] <= data_avere + timedelta(days=self.giorni_finestra)) &
            (~dare_df['usato'])
        ]
        match = dare_finestra[abs(dare_finestra['Dare'] - importo_avere) <= self.tolleranza]
        
        if not match.empty:
            if len(match) > 1:
                match = match.copy()
                match.loc[:, 'costo_temporale'] = (data_avere - match['Data']).dt.days.abs()
                best_match_1_to_1 = match.sort_values('costo_temporale').iloc[0]
                return [best_match_1_to_1.name], best_match_1_to_1['Dare']
            else:
                return [match.index[0]], match.iloc[0]['Dare']
        return None, None

    def trova_combinazioni(self, avere_row, dare_df, giorni_finestra_override=None):
        importo_avere = avere_row['Avere']
        data_avere = avere_row['Data']
        finestra = giorni_finestra_override if giorni_finestra_override is not None else self.giorni_finestra
        
        dare_finestra = dare_df[
            (dare_df['Data'] >= data_avere - timedelta(days=finestra)) &
            (dare_df['Data'] <= data_avere + timedelta(days=finestra)) &
            (~dare_df['usato'])
        ].copy()

        if dare_finestra.empty: return None, None
        dare_finestra = dare_finestra[dare_finestra['Dare'] <= importo_avere + self.tolleranza]
        if dare_finestra.empty: return None, None

        best_match = {'indices': None, 'somma': None, 'costo_temporale': float('inf'), 'costo_importo': float('inf')}

        for n in range(2, min(self.max_combinazioni + 1, len(dare_finestra) + 1)):
            indices = dare_finestra.index.tolist()
            num_combinations = math.comb(len(indices), n)
            if num_combinations > 10000: continue

            for combo in combinations(indices, n):
                combo_df = dare_df.loc[list(combo)]
                somma = combo_df['Dare'].sum()
                costo_importo = abs(somma - importo_avere)

                if costo_importo <= self.tolleranza:
                    costo_temporale = (data_avere - combo_df['Data']).dt.days.abs().mean()
                    if costo_temporale < best_match['costo_temporale'] or \
                       (costo_temporale == best_match['costo_temporale'] and costo_importo < best_match['costo_importo']):
                        best_match.update({'indices': list(combo), 'somma': somma, 'costo_temporale': costo_temporale, 'costo_importo': costo_importo})

        return best_match['indices'], best_match['somma']

    def trova_match_parziale(self, avere_row, dare_df):
        importo_avere = avere_row['Avere']
        data_avere = avere_row['Data']
        dare_finestra = dare_df[
            (dare_df['Data'] >= data_avere - timedelta(days=self.giorni_finestra)) &
            (dare_df['Data'] <= data_avere + timedelta(days=self.giorni_finestra)) &
            (~dare_df['usato']) &
            (dare_df['Dare'] > importo_avere - self.tolleranza)
        ].copy()
        
        if not dare_finestra.empty:
            dare_da_spezzare = dare_finestra.sort_values('Dare').iloc[0]
            indice_da_spezzare = dare_da_spezzare.name
            importo_originale = dare_df.loc[indice_da_spezzare, 'Dare']
            residuo = importo_originale - importo_avere
            return [indice_da_spezzare], importo_avere, residuo
        return None, None, None

    def _registra_match(self, avere_row, match_indices, somma, dare_df, avere_df, nuovi_dare, residuo=None):
        avere_df.at[avere_row.name, 'usato'] = True
        dare_df.loc[match_indices, 'usato'] = True
        
        if residuo is not None and residuo > self.tolleranza:
            dare_originale = dare_df.loc[match_indices[0]]
            nuovo_dare_row = dare_originale.copy()
            nuovo_dare_row['Dare'] = residuo
            nuovo_dare_row['usato'] = False
            nuovi_dare.append(nuovo_dare_row)

        self.risultati.append({
            'avere_idx': avere_row['indice_orig'], 'avere_data': avere_row['Data'], 'avere_importo': avere_row['Avere'],
            'dare_indices': [dare_df.loc[i, 'indice_orig'] for i in match_indices],
            'dare_date': [dare_df.loc[i, 'Data'] for i in match_indices],
            'dare_importi': [dare_df.loc[i, 'Dare'] for i in match_indices],
            'somma_dare': somma, 'differenza': avere_row['Avere'] - somma, 'num_elementi': len(match_indices)
        })

    def _riconcilia_residui_greedy(self, dare_df, avere_df):
        """Nuovo algoritmo greedy per i residui, più veloce e meno esigente in memoria."""
        print("\n  🧹 Avvio fase 2: Riconciliazione dei residui (Greedy)...")
        dare_residui = dare_df[(~dare_df['usato']) & (dare_df['Dare'] < self.soglia_residui)].sort_values('Dare', ascending=False).copy()
        avere_da_riconciliare = avere_df[~avere_df['usato']].sort_values('Avere', ascending=False).copy()

        if dare_residui.empty or avere_da_riconciliare.empty:
            print("  - Nessun residuo o versamento da analizzare.")
            return

        for idx_avere, avere_row in tqdm(avere_da_riconciliare.iterrows(), total=len(avere_da_riconciliare), desc="  Analisi residui AVERE", unit=" mov", ncols=100):
            if avere_df.loc[idx_avere, 'usato']: continue

            importo_target = avere_row['Avere']
            somma_corrente = 0
            combinazione_corrente = []

            # Scansione "ingorda" dei DARE residui
            for idx_dare, dare_row in dare_residui[~dare_residui['usato']].iterrows():
                if somma_corrente + dare_row['Dare'] <= importo_target + self.tolleranza:
                    somma_corrente += dare_row['Dare']
                    combinazione_corrente.append(idx_dare)
            
            # Verifica se la combinazione trovata è valida
            if abs(somma_corrente - importo_target) <= self.tolleranza and combinazione_corrente:
                self._registra_match(avere_row, combinazione_corrente, somma_corrente, dare_df, avere_df, [])
                dare_residui.loc[combinazione_corrente, 'usato'] = True # Aggiorna lo stato nel pool locale

    def riconcilia(self, dare_df, avere_df, verbose=True):
        if verbose:
            print(f"  - Versamenti (AVERE) da riconciliare: {len(avere_df)}")
            print(f"  - Incassi (DARE) disponibili: {len(dare_df)}")
            
        self.risultati = []
        nuovi_dare = []

        # --- FASE 1: RICONCILIAZIONE PRINCIPALE ---
        for idx, avere_row in tqdm(avere_df.iterrows(), total=len(avere_df), desc="  Riconciliazione AVERE", unit=" mov", ncols=100):
            if avere_row['usato']: continue

            match_indices, somma, residuo = None, None, None
            
            match_indices, somma = self.trova_match_esatto(avere_row, dare_df)
            if match_indices is None:
                match_indices, somma = self.trova_combinazioni(avere_row, dare_df)
            if match_indices is None:
                match_indices, somma, residuo = self.trova_match_parziale(avere_row, dare_df)

            if match_indices is not None:
                self._registra_match(avere_row, match_indices, somma, dare_df, avere_df, nuovi_dare, residuo)

        # --- FASE 2: RICONCILIAZIONE RESIDUI (con nuovo algoritmo) ---
        self._riconcilia_residui_greedy(dare_df, avere_df)

        # Aggiungi i DARE "spezzati" al dataframe principale
        if nuovi_dare:
            dare_df = pd.concat([dare_df, pd.DataFrame(nuovi_dare)], ignore_index=True)
        
        df_abbinamenti = pd.DataFrame(self.risultati) if self.risultati else pd.DataFrame()
        return dare_df, avere_df, df_abbinamenti