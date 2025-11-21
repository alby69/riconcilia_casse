"""
Batch Processor - Elaborazione multipla file Excel
Processa automaticamente tutti i file di una cartella o lista
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import sys
import os

# Importa la classe dal file principale
# Se lo script è nella stessa cartella:
# from riconciliazione import RiconciliatoreContabile

# Altrimenti copia la classe qui (per semplicità)
class RiconciliatoreContabile:
    """Classe per riconciliare movimenti Dare e Avere"""
    
    def __init__(self, tolleranza=0.01, giorni_finestra=30, max_combinazioni=6):
        self.tolleranza = tolleranza
        self.giorni_finestra = giorni_finestra
        self.max_combinazioni = max_combinazioni
        self.risultati = []
        
    def carica_excel(self, file_path):
        df = pd.read_excel(file_path)
        colonne_richieste = ['Data', 'Dare', 'Avere']
        if not all(col in df.columns for col in colonne_richieste):
            raise ValueError(f"Il file deve contenere le colonne: {colonne_richieste}")
        # df['Data'] = pd.to_datetime(df['Data'])

        df['Data'] = pd.to_datetime(
            df['Data'],
            format=None,                     # permette parsing flessibile
            dayfirst=True,                   # interpreta 01/02/2025 come 1 febbraio
            errors='coerce'                  # converte errori in NaT
        )
        df['Dare'] = pd.to_numeric(df['Dare'], errors='coerce').fillna(0)
        df['Avere'] = pd.to_numeric(df['Avere'], errors='coerce').fillna(0)
        df['indice_orig'] = df.index
        return df
    
    def separa_movimenti(self, df):
        dare = df[df['Dare'] > 0].copy().reset_index(drop=True)
        avere = df[df['Avere'] > 0].copy().reset_index(drop=True)
        dare['usato'] = False
        avere['usato'] = False
        dare = dare.sort_values('Dare', ascending=False).reset_index(drop=True)
        avere = avere.sort_values('Avere', ascending=False).reset_index(drop=True)
        return dare, avere
    
    def trova_match_esatto(self, dare_row, avere_df):
        from datetime import timedelta
        importo_dare = dare_row['Dare']
        data_dare = dare_row['Data']
        avere_finestra = avere_df[
            (avere_df['Data'] >= data_dare - timedelta(days=self.giorni_finestra)) &
            (avere_df['Data'] <= data_dare + timedelta(days=self.giorni_finestra)) &
            (~avere_df['usato'])
        ]
        match = avere_finestra[abs(avere_finestra['Avere'] - importo_dare) <= self.tolleranza]
        if not match.empty:
            return [match.index[0]], match.iloc[0]['Avere']
        return None, None
    
    def trova_combinazioni(self, dare_row, avere_df):
        from itertools import combinations
        from datetime import timedelta
        importo_dare = dare_row['Dare']
        data_dare = dare_row['Data']
        avere_finestra = avere_df[
            (avere_df['Data'] >= data_dare - timedelta(days=self.giorni_finestra)) &
            (avere_df['Data'] <= data_dare + timedelta(days=self.giorni_finestra)) &
            (~avere_df['usato'])
        ].copy()
        if avere_finestra.empty:
            return None, None
        avere_finestra = avere_finestra[avere_finestra['Avere'] <= importo_dare + self.tolleranza]
        if avere_finestra.empty:
            return None, None
        for n in range(2, min(self.max_combinazioni + 1, len(avere_finestra) + 1)):
            indices = avere_finestra.index.tolist()
            if len(list(combinations(indices, n))) > 10000:
                continue
            for combo in combinations(indices, n):
                somma = avere_df.loc[list(combo), 'Avere'].sum()
                if abs(somma - importo_dare) <= self.tolleranza:
                    return list(combo), somma
        return None, None
    
    def riconcilia(self, dare_df, avere_df, verbose=True):
        if verbose:
            print(f"  Movimenti DARE: {len(dare_df)}, AVERE: {len(avere_df)}")
        self.risultati = []
        for idx, dare_row in dare_df.iterrows():
            if dare_row['usato']:
                continue
            match_indices, somma = self.trova_match_esatto(dare_row, avere_df)
            if match_indices is None:
                match_indices, somma = self.trova_combinazioni(dare_row, avere_df)
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
        return dare_df, avere_df
    
    def esporta_risultati(self, output_path, dare_df, avere_df):
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
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
            dare_non_riconc = dare_df[~dare_df['usato']][['indice_orig', 'Data', 'Dare']]
            dare_non_riconc.columns = ['Indice', 'Data', 'Importo']
            dare_non_riconc.to_excel(writer, sheet_name='DARE non riconciliati', index=False)
            avere_non_util = avere_df[~avere_df['usato']][['indice_orig', 'Data', 'Avere']]
            avere_non_util.columns = ['Indice', 'Data', 'Importo']
            avere_non_util.to_excel(writer, sheet_name='AVERE non utilizzati', index=False)
            stats = {
                'Metrica': ['Totale DARE', 'Totale AVERE', 'DARE riconciliati', 
                           'AVERE utilizzati', 'DARE non riconciliati', 'AVERE non utilizzati',
                           '% DARE riconciliati', '% AVERE utilizzati'],
                'Valore': [len(dare_df), len(avere_df), dare_df['usato'].sum(), 
                          avere_df['usato'].sum(), (~dare_df['usato']).sum(), 
                          (~avere_df['usato']).sum(),
                          f"{dare_df['usato'].sum()/len(dare_df)*100:.1f}%",
                          f"{avere_df['usato'].sum()/len(avere_df)*100:.1f}%"]
            }
            pd.DataFrame(stats).to_excel(writer, sheet_name='Statistiche', index=False)


class BatchProcessor:
    """Elabora automaticamente multipli file Excel"""
    
    def __init__(self, config=None):
        """
        config: dizionario con parametri (opzionale)
        {
            'tolleranza': 0.01,
            'giorni_finestra': 30,
            'max_combinazioni': 6,
            'cartella_input': 'input/',
            'cartella_output': 'output/',
            'pattern': '*.xlsx'
        }
        """
        self.config = config or {}
        self.tolleranza = self.config.get('tolleranza', 0.01)
        self.giorni_finestra = self.config.get('giorni_finestra', 30)
        self.max_combinazioni = self.config.get('max_combinazioni', 6)
        self.cartella_input = Path(self.config.get('cartella_input', 'input'))
        self.cartella_output = Path(self.config.get('cartella_output', 'output'))
        self.pattern = self.config.get('pattern', '*.xlsx')
        
        # Statistiche globali
        self.stats_globali = []
        
    def crea_cartelle(self):
        """Crea cartelle output se non esistono"""
        self.cartella_output.mkdir(exist_ok=True)
        (self.cartella_output / 'logs').mkdir(exist_ok=True)
        
    def trova_files(self):
        """Trova tutti i file Excel nella cartella input"""
        if not self.cartella_input.exists():
            raise FileNotFoundError(f"Cartella '{self.cartella_input}' non trovata!")
        
        files = list(self.cartella_input.glob(self.pattern))
        
        # Escludi file temporanei e già elaborati
        files = [f for f in files if not f.name.startswith('~') 
                 and not f.name.startswith('risultato_')]
        
        return sorted(files)
    
    def elabora_file(self, file_path):
        """Elabora singolo file"""
        nome_file = file_path.stem
        print(f"\n{'='*60}")
        print(f"📂 Elaborazione: {file_path.name}")
        print(f"{'='*60}")
        
        risultato = {
            'file': file_path.name,
            'timestamp': datetime.now().isoformat(),
            'successo': False,
            'errore': None,
            'statistiche': {}
        }
        
        try:
            # Inizializza riconciliatore
            riconciliatore = RiconciliatoreContabile(
                tolleranza=self.tolleranza,
                giorni_finestra=self.giorni_finestra,
                max_combinazioni=self.max_combinazioni
            )
            
            # Carica e processa
            print(f"  ⏳ Caricamento dati...")
            df = riconciliatore.carica_excel(file_path)
            
            print(f"  ✓ Caricati {len(df)} movimenti")
            
            dare_df, avere_df = riconciliatore.separa_movimenti(df)
            
            print(f"  ⏳ Riconciliazione in corso...")
            dare_df, avere_df = riconciliatore.riconcilia(dare_df, avere_df, verbose=True)
            
            # Salva risultati
            output_file = self.cartella_output / f"risultato_{nome_file}.xlsx"
            riconciliatore.esporta_risultati(output_file, dare_df, avere_df)
            
            # Statistiche
            dare_riconc = dare_df['usato'].sum()
            avere_util = avere_df['usato'].sum()
            
            risultato['successo'] = True
            risultato['statistiche'] = {
                'totale_dare': len(dare_df),
                'totale_avere': len(avere_df),
                'dare_riconciliati': int(dare_riconc),
                'avere_utilizzati': int(avere_util),
                'percentuale_dare': round(dare_riconc/len(dare_df)*100, 1) if len(dare_df) > 0 else 0,
                'percentuale_avere': round(avere_util/len(avere_df)*100, 1) if len(avere_df) > 0 else 0,
                'output_file': str(output_file)
            }
            
            print(f"\n  ✅ COMPLETATO")
            print(f"  📊 DARE riconciliati: {dare_riconc}/{len(dare_df)} ({risultato['statistiche']['percentuale_dare']}%)")
            print(f"  📊 AVERE utilizzati: {avere_util}/{len(avere_df)} ({risultato['statistiche']['percentuale_avere']}%)")
            print(f"  💾 Salvato in: {output_file.name}")
            
        except Exception as e:
            risultato['errore'] = str(e)
            print(f"  ❌ ERRORE: {e}")
        
        return risultato
    
    def elabora_tutti(self):
        """Elabora tutti i file trovati"""
        print("""
╔════════════════════════════════════════════════════════════╗
║          BATCH PROCESSOR - ELABORAZIONE MULTIPLA           ║
╚════════════════════════════════════════════════════════════╝
        """)
        
        # Setup
        self.crea_cartelle()
        
        # Trova files
        print(f"🔍 Ricerca file in: {self.cartella_input}")
        files = self.trova_files()
        
        if not files:
            print(f"⚠️  Nessun file trovato con pattern '{self.pattern}'")
            return
        
        print(f"✓ Trovati {len(files)} file da elaborare\n")
        
        # Mostra configurazione
        print(f"⚙️  CONFIGURAZIONE:")
        print(f"   - Tolleranza: €{self.tolleranza}")
        print(f"   - Finestra temporale: ±{self.giorni_finestra} giorni")
        print(f"   - Max combinazioni: {self.max_combinazioni}")
        print(f"   - Output: {self.cartella_output}/")
        
        # Elabora ogni file
        inizio = datetime.now()
        
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}]", end=" ")
            risultato = self.elabora_file(file_path)
            self.stats_globali.append(risultato)
        
        # Riepilogo finale
        fine = datetime.now()
        durata = (fine - inizio).total_seconds()
        
        self.stampa_riepilogo(durata)
        self.salva_log()
        
    def stampa_riepilogo(self, durata):
        """Stampa riepilogo globale"""
        print(f"\n\n{'='*60}")
        print(f"📊 RIEPILOGO GLOBALE")
        print(f"{'='*60}")
        
        successi = sum(1 for s in self.stats_globali if s['successo'])
        errori = len(self.stats_globali) - successi
        
        print(f"✓ File elaborati con successo: {successi}/{len(self.stats_globali)}")
        print(f"✗ File con errori: {errori}")
        print(f"⏱  Tempo totale: {durata:.1f} secondi ({durata/60:.1f} minuti)")
        
        if successi > 0:
            # Statistiche aggregate
            totale_dare = sum(s['statistiche'].get('totale_dare', 0) 
                            for s in self.stats_globali if s['successo'])
            totale_dare_riconc = sum(s['statistiche'].get('dare_riconciliati', 0) 
                                   for s in self.stats_globali if s['successo'])
            
            print(f"\n📈 STATISTICHE AGGREGATE:")
            print(f"   - Totale movimenti DARE: {totale_dare:,}")
            print(f"   - DARE riconciliati: {totale_dare_riconc:,} ({totale_dare_riconc/totale_dare*100:.1f}%)")
        
        # File con errori
        if errori > 0:
            print(f"\n⚠️  FILE CON ERRORI:")
            for stat in self.stats_globali:
                if not stat['successo']:
                    print(f"   - {stat['file']}: {stat['errore']}")
        
        print(f"\n💾 Log salvato in: {self.cartella_output}/logs/")
        print(f"{'='*60}\n")
    
    def salva_log(self):
        """Salva log dettagliato in JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.cartella_output / 'logs' / f'batch_log_{timestamp}.json'
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'configurazione': {
                'tolleranza': self.tolleranza,
                'giorni_finestra': self.giorni_finestra,
                'max_combinazioni': self.max_combinazioni,
                'cartella_input': str(self.cartella_input),
                'cartella_output': str(self.cartella_output)
            },
            'risultati': self.stats_globali
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Salva anche CSV per analisi rapida
        if self.stats_globali:
            csv_data = []
            for stat in self.stats_globali:
                row = {
                    'File': stat['file'],
                    'Successo': stat['successo'],
                    'Errore': stat.get('errore', ''),
                }
                if stat['successo']:
                    row.update(stat['statistiche'])
                csv_data.append(row)
            
            csv_file = self.cartella_output / 'logs' / f'riepilogo_{timestamp}.csv'
            pd.DataFrame(csv_data).to_csv(csv_file, index=False)


def main():
    """Funzione principale"""
    
    # CONFIGURAZIONE - MODIFICA QUI
    config = {
        'tolleranza': 0.01,
        'giorni_finestra': 30,
        'max_combinazioni': 6,
        'cartella_input': 'input',      # <-- Cartella con file da elaborare
        'cartella_output': 'output',    # <-- Cartella per risultati
        'pattern': '*.xlsx'              # <-- Pattern file (es. 'supermercato_*.xlsx')
    }
    
    # Inizializza e avvia
    processor = BatchProcessor(config)
    processor.elabora_tutti()


if __name__ == "__main__":
    main()
