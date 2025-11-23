"""
Sistema di Riconciliazione Contabile - Dare/Avere
Questo script elabora un singolo file di movimenti contabili.
"""
from pathlib import Path
import warnings
import argparse
from core import RiconciliatoreContabile # Importa il core
import pandas as pd

warnings.filterwarnings('ignore')

def main(args):
    """Funzione principale che esegue la riconciliazione su un singolo file."""
    
    if not args.silent:
        print(f"⚙️  Avvio riconciliazione per: {args.input}")
    
    # Parametri algoritmo
    riconciliatore = RiconciliatoreContabile(
        tolleranza=args.tolleranza,
        giorni_finestra=args.giorni_finestra,
        max_combinazioni=args.max_combinazioni,
        soglia_residui=args.soglia_residui,               # <-- Corretto
        giorni_finestra_residui=args.giorni_finestra_residui # <-- Corretto
    )
    
    try:
        df = riconciliatore.carica_file(args.input)
        dare_df, avere_df = riconciliatore.separa_movimenti(df)
        dare_df, avere_df, df_abbinamenti = riconciliatore.riconcilia(dare_df, avere_df, verbose=not args.silent)
        
        # Esporta risultati
        with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
            # Pulisci e formatta il DataFrame degli abbinamenti prima di salvarlo
            if not df_abbinamenti.empty:
                df_abbinamenti['dare_indices'] = df_abbinamenti['dare_indices'].apply(lambda x: ', '.join(map(str, x)))
                df_abbinamenti['dare_date'] = df_abbinamenti['dare_date'].apply(lambda x: ', '.join([d.strftime('%d/%m/%y') for d in x]))
                df_abbinamenti['dare_importi'] = df_abbinamenti['dare_importi'].apply(lambda x: ', '.join([f'{i:.2f}' for i in x]))
                df_abbinamenti['avere_data'] = df_abbinamenti['avere_data'].dt.strftime('%d/%m/%y')

            df_abbinamenti.to_excel(writer, sheet_name='Abbinamenti', index=False)
            
            avere_non_riconc = avere_df[~avere_df['usato']][['indice_orig', 'Data', 'Avere']].copy()
            avere_non_riconc.columns = ['Indice', 'Data', 'Importo']
            avere_non_riconc['Data'] = avere_non_riconc['Data'].dt.strftime('%d/%m/%y')
            avere_non_riconc.to_excel(writer, sheet_name='AVERE non riconciliati', index=False)
            
            dare_non_util = dare_df[~dare_df['usato']][['indice_orig', 'Data', 'Dare']].copy()
            dare_non_util.columns = ['Indice', 'Data', 'Importo']
            dare_non_util['Data'] = dare_non_util['Data'].dt.strftime('%d/%m/%y')
            dare_non_util.to_excel(writer, sheet_name='DARE non utilizzati', index=False)
            
            stats = {
                'Metrica': ['Totale Incassi (DARE)', 'Totale Versamenti (AVERE)', 'Incassi (DARE) utilizzati', 'Versamenti (AVERE) riconciliati', 'Incassi (DARE) non utilizzati', 'Versamenti (AVERE) non riconciliati', '% Incassi (DARE) utilizzati', '% Versamenti (AVERE) riconciliati'],
                'Valore': [
                    len(dare_df), len(avere_df), 
                    int(dare_df['usato'].sum()), int(avere_df['usato'].sum()),
                    int((~dare_df['usato']).sum()), int((~avere_df['usato']).sum()),
                    f"{dare_df['usato'].sum()/len(dare_df)*100:.1f}%" if len(dare_df) > 0 else "0.0%",
                    f"{avere_df['usato'].sum()/len(avere_df)*100:.1f}%" if len(avere_df) > 0 else "0.0%"
                ]
            }
            pd.DataFrame(stats).to_excel(writer, sheet_name='Statistiche', index=False)
        
        if not args.silent:
            print(f"✓ Risultati esportati in: {args.output}")
        
    except FileNotFoundError:
        print(f"\n❌ ERRORE: File '{args.input}' non trovato!")
    except Exception as e:
        print(f"\n❌ ERRORE: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Riconciliazione Contabile per un singolo file.")
    parser.add_argument('-i', '--input', required=True, help="File di input (Excel o CSV).")
    parser.add_argument('-o', '--output', required=True, help="Nome del file di output Excel.")
    parser.add_argument('--tolleranza', type=float, default=0.01, help="Tolleranza di importo.")
    parser.add_argument('--giorni-finestra', type=int, default=30, help="Finestra temporale in giorni.")
    parser.add_argument('--max-combinazioni', type=int, default=6, help="Numero massimo di combinazioni.")
    parser.add_argument('--soglia-residui', type=float, default=100, help="Soglia per i DARE residui.")
    parser.add_argument('--giorni-finestra-residui', type=int, default=60, help="Finestra temporale per i residui.")
    parser.add_argument('--silent', action='store_true', help="Esegui senza output verboso (usato dal batch).")
    
    args = parser.parse_args()
    main(args)
