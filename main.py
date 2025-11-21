"""
Sistema di Riconciliazione Contabile - Dare/Avere
Autore: Sistema di matching per quadratura casse supermercati
"""

import pandas as pd
import numpy as np
from itertools import combinations
from datetime import timedelta
import openpyxl
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class RiconciliatoreContabile:
    """
    Classe per riconciliare movimenti Dare e Avere con algoritmo ottimizzato
    """
    
    def __init__(self, tolleranza=0.01, giorni_finestra=30, max_combinazioni=6):
        """
        Parametri:
        - tolleranza: differenza massima accettabile (es. 0.01 per 1 centesimo)
        - giorni_finestra: giorni prima/dopo per cercare match
        - max_combinazioni: numero massimo di elementi da combinare
        """
        self.tolleranza = tolleranza
        self.giorni_finestra = giorni_finestra
        self.max_combinazioni = max_combinazioni
        self.risultati = []
        
    def carica_excel(self, file_path):
        """Carica file Excel con colonne: Data, Dare, Avere"""
        df = pd.read_excel(file_path)
        
        # Verifica colonne richieste
        colonne_richieste = ['Data', 'Dare', 'Avere']
        if not all(col in df.columns for col in colonne_richieste):
            raise ValueError(f"Il file deve contenere le colonne: {colonne_richieste}")
        
        # Converti Data in datetime
        df['Data'] = pd.to_datetime(df['Data'])
        
        # Riempi NaN con 0 e converti in float
        df['Dare'] = pd.to_numeric(df['Dare'], errors='coerce').fillna(0)
        df['Avere'] = pd.to_numeric(df['Avere'], errors='coerce').fillna(0)
        
        # Aggiungi indice originale per tracciamento
        df['indice_orig'] = df.index
        
        return df
    
    def separa_movimenti(self, df):
        """Separa movimenti Dare e Avere in due DataFrame"""
        dare = df[df['Dare'] > 0].copy().reset_index(drop=True)
        avere = df[df['Avere'] > 0].copy().reset_index(drop=True)
        
        # Aggiungi flag per tracciare se già usato
        dare['usato'] = False
        avere['usato'] = False
        
        # Ordina per importo decrescente (ottimizzazione greedy)
        dare = dare.sort_values('Dare', ascending=False).reset_index(drop=True)
        avere = avere.sort_values('Avere', ascending=False).reset_index(drop=True)
        
        return dare, avere
    
    def trova_match_esatto(self, dare_row, avere_df):
        """Cerca match esatto 1:1"""
        importo_dare = dare_row['Dare']
        
        # Filtra per finestra temporale
        data_dare = dare_row['Data']
        avere_finestra = avere_df[
            (avere_df['Data'] >= data_dare - timedelta(days=self.giorni_finestra)) &
            (avere_df['Data'] <= data_dare + timedelta(days=self.giorni_finestra)) &
            (~avere_df['usato'])
        ]
        
        # Cerca match esatto (con tolleranza)
        match = avere_finestra[
            abs(avere_finestra['Avere'] - importo_dare) <= self.tolleranza
        ]
        
        if not match.empty:
            return [match.index[0]], match.iloc[0]['Avere']
        
        return None, None
    
    def trova_combinazioni(self, dare_row, avere_df):
        """Cerca combinazioni di più movimenti Avere che sommano al Dare"""
        importo_dare = dare_row['Dare']
        data_dare = dare_row['Data']
        
        # Filtra per finestra temporale e non usati
        avere_finestra = avere_df[
            (avere_df['Data'] >= data_dare - timedelta(days=self.giorni_finestra)) &
            (avere_df['Data'] <= data_dare + timedelta(days=self.giorni_finestra)) &
            (~avere_df['usato'])
        ].copy()
        
        if avere_finestra.empty:
            return None, None
        
        # Ottimizzazione: escludi valori troppo grandi
        avere_finestra = avere_finestra[
            avere_finestra['Avere'] <= importo_dare + self.tolleranza
        ]
        
        if avere_finestra.empty:
            return None, None
        
        # Prova combinazioni di lunghezza crescente
        for n in range(2, min(self.max_combinazioni + 1, len(avere_finestra) + 1)):
            # Ottimizzazione: limita numero di combinazioni da testare
            indices = avere_finestra.index.tolist()
            
            # Se troppe combinazioni, usa approccio greedy
            if len(list(combinations(indices, n))) > 10000:
                continue
            
            for combo in combinations(indices, n):
                somma = avere_df.loc[list(combo), 'Avere'].sum()
                
                if abs(somma - importo_dare) <= self.tolleranza:
                    return list(combo), somma
        
        return None, None
    
    def riconcilia(self, dare_df, avere_df):
        """Esegue riconciliazione completa"""
        print(f"\n{'='*60}")
        print(f"AVVIO RICONCILIAZIONE")
        print(f"{'='*60}")
        print(f"Movimenti DARE: {len(dare_df)}")
        print(f"Movimenti AVERE: {len(avere_df)}")
        print(f"Totale DARE: €{dare_df['Dare'].sum():,.2f}")
        print(f"Totale AVERE: €{avere_df['Avere'].sum():,.2f}")
        print(f"{'='*60}\n")
        
        self.risultati = []
        dare_riconciliati = 0
        avere_utilizzati = 0
        
        for idx, dare_row in dare_df.iterrows():
            if dare_row['usato']:
                continue
            
            # Prova match esatto
            match_indices, somma = self.trova_match_esatto(dare_row, avere_df)
            
            # Se non trovato, prova combinazioni
            if match_indices is None:
                match_indices, somma = self.trova_combinazioni(dare_row, avere_df)
            
            # Se trovato match, registralo
            if match_indices is not None:
                dare_df.at[idx, 'usato'] = True
                avere_df.loc[match_indices, 'usato'] = True
                
                self.risultati.append({
                    'dare_idx': dare_row['indice_orig'],
                    'dare_data': dare_row['Data'],
                    'dare_importo': dare_row['Dare'],
                    'avere_indices': [avere_df.loc[i, 'indice_orig'] for i in match_indices],
                    'avere_date': [avere_df.loc[i, 'Data'] for i in match_indices],
                    'avere_importi': [avere_df.loc[i, 'Avere'] for i in match_indices],
                    'somma_avere': somma,
                    'differenza': dare_row['Dare'] - somma,
                    'num_elementi': len(match_indices)
                })
                
                dare_riconciliati += 1
                avere_utilizzati += len(match_indices)
                
                # Progress
                if dare_riconciliati % 10 == 0:
                    print(f"Riconciliati: {dare_riconciliati}/{len(dare_df)} DARE")
        
        print(f"\n{'='*60}")
        print(f"RISULTATI")
        print(f"{'='*60}")
        print(f"✓ DARE riconciliati: {dare_riconciliati}/{len(dare_df)} ({dare_riconciliati/len(dare_df)*100:.1f}%)")
        print(f"✓ AVERE utilizzati: {avere_utilizzati}/{len(avere_df)} ({avere_utilizzati/len(avere_df)*100:.1f}%)")
        print(f"✗ DARE non riconciliati: {len(dare_df) - dare_riconciliati}")
        print(f"✗ AVERE non utilizzati: {len(avere_df) - avere_utilizzati}")
        print(f"{'='*60}\n")
        
        return dare_df, avere_df
    
    def esporta_risultati(self, output_path, dare_df, avere_df):
        """Esporta risultati in Excel con più fogli"""
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Foglio 1: Riepilogo abbinamenti
            if self.risultati:
                df_abbinamenti = []
                for r in self.risultati:
                    df_abbinamenti.append({
                        'Indice DARE': r['dare_idx'],
                        'Data DARE': r['dare_data'],
                        'Importo DARE': r['dare_importo'],
                        'N° Avere': r['num_elementi'],
                        'Indici AVERE': ', '.join(map(str, r['avere_indices'])),
                        'Date AVERE': ', '.join([d.strftime('%Y-%m-%d') for d in r['avere_date']]),
                        'Importi AVERE': ', '.join([f"{i:.2f}" for i in r['avere_importi']]),
                        'Somma AVERE': r['somma_avere'],
                        'Differenza': r['differenza']
                    })
                
                pd.DataFrame(df_abbinamenti).to_excel(writer, sheet_name='Abbinamenti', index=False)
            
            # Foglio 2: DARE non riconciliati
            dare_non_riconc = dare_df[~dare_df['usato']][['indice_orig', 'Data', 'Dare']]
            dare_non_riconc.columns = ['Indice', 'Data', 'Importo']
            dare_non_riconc.to_excel(writer, sheet_name='DARE non riconciliati', index=False)
            
            # Foglio 3: AVERE non utilizzati
            avere_non_util = avere_df[~avere_df['usato']][['indice_orig', 'Data', 'Avere']]
            avere_non_util.columns = ['Indice', 'Data', 'Importo']
            avere_non_util.to_excel(writer, sheet_name='AVERE non utilizzati', index=False)
            
            # Foglio 4: Statistiche
            stats = {
                'Metrica': [
                    'Totale DARE', 'Totale AVERE', 'DARE riconciliati', 
                    'AVERE utilizzati', 'DARE non riconciliati', 'AVERE non utilizzati',
                    '% DARE riconciliati', '% AVERE utilizzati'
                ],
                'Valore': [
                    len(dare_df), len(avere_df), 
                    dare_df['usato'].sum(), avere_df['usato'].sum(),
                    (~dare_df['usato']).sum(), (~avere_df['usato']).sum(),
                    f"{dare_df['usato'].sum()/len(dare_df)*100:.1f}%",
                    f"{avere_df['usato'].sum()/len(avere_df)*100:.1f}%"
                ]
            }
            pd.DataFrame(stats).to_excel(writer, sheet_name='Statistiche', index=False)
        
        print(f"✓ Risultati esportati in: {output_path}")


def main():
    """Funzione principale - Esempio d'uso"""
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║   SISTEMA DI RICONCILIAZIONE CONTABILE SUPERMERCATI       ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # CONFIGURAZIONE
    file_input = "movimenti.xlsx"  # <-- MODIFICA CON IL TUO FILE
    file_output = "riconciliazione_risultati.xlsx"
    
    # Parametri algoritmo
    riconciliatore = RiconciliatoreContabile(
        tolleranza=0.01,           # 1 centesimo di tolleranza
        giorni_finestra=30,        # Cerca match entro ±30 giorni
        max_combinazioni=6         # Max 6 elementi da combinare
    )
    
    try:
        # Carica dati
        print(f"📂 Caricamento file: {file_input}")
        df = riconciliatore.carica_excel(file_input)
        print(f"✓ Caricati {len(df)} movimenti")
        
        # Separa Dare e Avere
        dare_df, avere_df = riconciliatore.separa_movimenti(df)
        
        # Esegui riconciliazione
        dare_df, avere_df = riconciliatore.riconcilia(dare_df, avere_df)
        
        # Esporta risultati
        riconciliatore.esporta_risultati(file_output, dare_df, avere_df)
        
        print("\n✅ PROCESSO COMPLETATO CON SUCCESSO!\n")
        
    except FileNotFoundError:
        print(f"\n❌ ERRORE: File '{file_input}' non trovato!")
        print("Crea un file Excel con colonne: Data, Dare, Avere")
    except Exception as e:
        print(f"\n❌ ERRORE: {str(e)}")


if __name__ == "__main__":
    main()
