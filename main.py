"""
Sistema di Riconciliazione Contabile - Dare/Avere
Questo script elabora un singolo file di movimenti contabili.
"""
from pathlib import Path
import warnings
from datetime import datetime
import argparse
from core import RiconciliatoreContabile # Importa il core
import pandas as pd

warnings.filterwarnings('ignore')

def _generate_markdown_report(input_file_path, original_df, df_abbinamenti, final_dare_df, final_avere_df, config_params, output_excel_filename):
    """Genera il contenuto del report Markdown."""
    report_content = []

def run_reconciliation(input_file, config, output_file=None, silent=False):
    """
    Esegue la logica di riconciliazione per un dato file e configurazione.
    Restituisce un dizionario con le statistiche.
    """
    if not silent:
        print(f"⚙️  Avvio riconciliazione per: {input_file}")
    
    riconciliatore = RiconciliatoreContabile(
        tolleranza=config['tolleranza'],
        giorni_finestra=config['giorni_finestra'],
        max_combinazioni=config['max_combinazioni'],
        soglia_residui=config['soglia_residui'],
        giorni_finestra_residui=config['giorni_finestra_residui'],
        sorting_strategy=config['sorting_strategy'],
        search_direction=config['search_direction']
    )
    
    try:
        # Carica il file e conserva una copia dell'originale per il report Markdown
        loaded_df = riconciliatore.carica_file(input_file)
        original_df_for_report = loaded_df.copy() # Questa è la copia originale dei dati
        
        dare_df, avere_df = riconciliatore.separa_movimenti(loaded_df) # Lavora sulla copia che verrà modificata
        dare_df, avere_df, df_abbinamenti = riconciliatore.riconcilia(dare_df, avere_df, verbose=not silent)
        
        # Calcola i dataframe dei non utilizzati, necessari sia per l'export che per il return
        dare_non_util = dare_df[~dare_df['usato']].copy()
        avere_non_riconc = avere_df[~avere_df['usato']].copy()
        
        # Se viene fornito un file di output, salva i risultati
        if output_file:
            # Genera il report Markdown PRIMA di modificare df_abbinamenti per l'Excel
            markdown_report_filename = Path(output_file).parent / f"report_{Path(output_file).stem}.md" # Path(output_file).stem
            markdown_content = _generate_markdown_report(
                input_file, original_df_for_report, df_abbinamenti, dare_df, avere_df, config, Path(output_file).name
            )
            with open(markdown_report_filename, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            if not silent:
                print(f"✓ Report Markdown esportato in: {markdown_report_filename}")
                
            # Ora procedi con la scrittura del file Excel
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Funzione helper per formattare le liste in modo pulito
                def format_list_as_string(data_list, is_float=False):
                    # Aggiungi +2 agli indici per farli corrispondere alle righe di Excel
                    if not is_float:
                        data_list = [i + 2 for i in data_list]

                    if len(data_list) == 1:
                        return f"{data_list[0]:.2f}" if is_float else str(data_list[0])
                    else:
                        formatter = (lambda x: f"{x:.2f}") if is_float else str
                        return ', '.join(map(formatter, data_list))

                if not df_abbinamenti.empty:
                    df_abbinamenti['dare_indices'] = df_abbinamenti['dare_indices'].apply(format_list_as_string)
                    df_abbinamenti['avere_indices'] = df_abbinamenti['avere_indices'].apply(format_list_as_string)
                    df_abbinamenti['dare_importi'] = df_abbinamenti['dare_importi'].apply(lambda x: format_list_as_string(x, is_float=True))
                    df_abbinamenti['avere_importi'] = df_abbinamenti['avere_importi'].apply(lambda x: format_list_as_string(x, is_float=True))
                    
                    df_abbinamenti['dare_date'] = df_abbinamenti['dare_date'].apply(lambda x: x[0].strftime('%d/%m/%y') if isinstance(x, list) and len(x) == 1 else ', '.join([d.strftime('%d/%m/%y') for d in x]))
                    df_abbinamenti['avere_data'] = df_abbinamenti['avere_data'].dt.strftime('%d/%m/%y')

                df_abbinamenti.to_excel(writer, sheet_name='Abbinamenti', index=False)
                
                # Prepara e formatta i fogli dei non riconciliati
                dare_non_util['Data'] = dare_non_util['Data'].dt.strftime('%d/%m/%y')
                dare_non_util[['indice_orig', 'Data', 'Dare']].rename(columns={'indice_orig': 'Indice Riga', 'Dare': 'Importo'}).to_excel(writer, sheet_name='DARE non utilizzati', index=False)

                avere_non_riconc['Data'] = avere_non_riconc['Data'].dt.strftime('%d/%m/%y')
                avere_non_riconc[['indice_orig', 'Data', 'Avere']].rename(columns={'indice_orig': 'Indice Riga', 'Avere': 'Importo'}).to_excel(writer, sheet_name='AVERE non riconciliati', index=False)

                # --- Creazione del nuovo foglio Statistiche ---
                # Calcoli per Incassi (DARE)
                num_dare_tot = len(dare_df)
                num_dare_usati = int(dare_df['usato'].sum())
                perc_num_dare = (num_dare_usati / num_dare_tot * 100) if num_dare_tot > 0 else 0
                imp_dare_tot = dare_df['Dare'].sum()
                imp_dare_usati = dare_df[dare_df['usato']]['Dare'].sum()
                imp_dare_delta = dare_non_util['Dare'].sum()
                perc_imp_dare = (imp_dare_usati / imp_dare_tot * 100) if imp_dare_tot > 0 else 0

                # Calcoli per Versamenti (AVERE)
                num_avere_tot = len(avere_df)
                num_avere_usati = int(avere_df['usato'].sum())
                perc_num_avere = (num_avere_usati / num_avere_tot * 100) if num_avere_tot > 0 else 0
                imp_avere_tot = avere_df['Avere'].sum()
                imp_avere_usati = avere_df[avere_df['usato']]['Avere'].sum()
                imp_avere_delta = avere_non_riconc['Avere'].sum()
                perc_imp_avere = (imp_avere_usati / imp_avere_tot * 100) if imp_avere_tot > 0 else 0

                # Funzione di formattazione
                def format_eur(value):
                    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

                # Creazione DataFrame Incassi
                df_incassi = pd.DataFrame({
                    'TOT': [num_dare_tot, format_eur(imp_dare_tot)],
                    'USATI': [num_dare_usati, format_eur(imp_dare_usati)],
                    '% USATI': [f"{perc_num_dare:.2f}%", f"{perc_imp_dare:.2f}%"],
                    'Delta': [len(dare_non_util), format_eur(imp_dare_delta)]
                }, index=['Numero', 'Importo'])

                # Creazione DataFrame Versamenti
                df_versamenti = pd.DataFrame({
                    'TOT': [num_avere_tot, format_eur(imp_avere_tot)],
                    'USATI': [num_avere_usati, format_eur(imp_avere_usati)],
                    '% USATI': [f"{perc_num_avere:.2f}%", f"{perc_imp_avere:.2f}%"],
                    'Delta': [len(avere_non_riconc), format_eur(imp_avere_delta)]
                }, index=['Numero', 'Importo'])

                # Creazione DataFrame Confronto Finale
                df_confronto = pd.DataFrame({
                    'Delta Conteggio': [len(dare_non_util) - len(avere_non_riconc)],
                    'Delta Importo (€)': [format_eur(imp_dare_delta - imp_avere_delta)]
                }, index=['Incassi vs Versamenti'])

                # Scrittura dei DataFrame sul foglio 'Statistiche'. La prima di queste chiamate crea il foglio.
                df_incassi.to_excel(writer, sheet_name='Statistiche', startrow=2)
                df_versamenti.to_excel(writer, sheet_name='Statistiche', startrow=8)
                df_confronto.to_excel(writer, sheet_name='Statistiche', startrow=14)

                # Ora che il foglio esiste, possiamo accedervi per scrivere i titoli.
                sheet_stats = writer.sheets['Statistiche'] # Questa riga ora funziona correttamente.
                sheet_stats.cell(row=1, column=1, value="Riepilogo Incassi (DARE)")
                sheet_stats.cell(row=7, column=1, value="Riepilogo Versamenti (AVERE)")
                sheet_stats.cell(row=13, column=1, value="Confronto Sbilancio Finale")

            if not silent:
                print(f"✓ Risultati esportati in: {output_file}")
        
        # Restituisce le statistiche grezze per l'elaborazione batch/ottimizzatore
        return {
            'Totale Incassi (DARE)': len(dare_df), 'Incassi (DARE) utilizzati': int(dare_df['usato'].sum()),
            'Totale Versamenti (AVERE)': len(avere_df), 'Versamenti (AVERE) riconciliati': int(avere_df['usato'].sum()),
            '_raw_importo_dare_non_util': dare_non_util['Dare'].sum(),
            '_raw_importo_avere_non_riconc': avere_non_riconc['Avere'].sum()
        }

    except FileNotFoundError:
        print(f"\n❌ ERRORE: File '{input_file}' non trovato!")
        return None
    except Exception as e:
        print(f"\n❌ ERRORE: {str(e)}")
        return None

def _generate_markdown_report(input_file_path, original_df, df_abbinamenti, final_dare_df, final_avere_df, config_params, output_excel_filename):
    """Genera il contenuto del report Markdown."""
    report_content = []
    report_content.append(f"# 📊 Report di Riconciliazione - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_content.append(f"## 📄 File Analizzato: `{input_file_path}`\n")
    report_content.append(f"Questo report visualizza la sequenza degli abbinamenti trovati e lo stato finale dei movimenti.\n")

    report_content.append("## ⚙️ Configurazione Utilizzata\n")
    report_content.append("| Parametro | Valore |\n|---|---|\n")
    for key, value in config_params.items():
        # Filtra i parametri non rilevanti per la configurazione dell'algoritmo
        if key not in ['input', 'output', 'commento', 'pattern', 'silent']:
            report_content.append(f"| `{key}` | `{value}` |\n")
    report_content.append("\n")

    report_content.append("## 🔍 Sequenza degli Abbinamenti Trovati\n")
    report_content.append("Ogni sezione mostra un abbinamento e i movimenti coinvolti.\n")

    current_pass_name = ""
    # Funzione helper per creare una tabella Markdown ben formattata
    def create_md_table(headers, rows):
        if not rows:
            return ""
        
        # Calcola la larghezza massima per ogni colonna
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Crea la tabella
        header_line = "| " + " | ".join([h.ljust(col_widths[i]) for i, h in enumerate(headers)]) + " |"
        separator_line = "|-" + "-|-".join(["-" * w for w in col_widths]) + "-|"
        
        table_lines = [header_line, separator_line]
        for row in rows:
            table_lines.append("| " + " | ".join([str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)]) + " |")
        
        return "\n".join(table_lines) + "\n"

    for idx, match_row in df_abbinamenti.iterrows():
        if match_row['pass_name'] != current_pass_name:
            current_pass_name = match_row['pass_name']
            report_content.append(f"\n### {current_pass_name}\n")

        report_content.append(f"#### Match #{idx + 1} ({match_row['tipo_match']})\n")

        # Tabella DARE
        report_content.append(f"**🔴 Incassi (DARE) coinvolti:**\n")
        dare_rows = [[dare_idx + 2, original_df.loc[dare_idx]['Data'].strftime('%d/%m/%y'), f"{original_df.loc[dare_idx]['Dare']:.2f}"] for dare_idx in match_row['dare_indices']]
        report_content.append(create_md_table(['Riga (Excel)', 'Data', 'Importo'], dare_rows))

        # Tabella AVERE
        report_content.append(f"**🟢 Versamenti (AVERE) coinvolti:**\n")
        avere_rows = [[avere_idx + 2, original_df.loc[avere_idx]['Data'].strftime('%d/%m/%y'), f"{original_df.loc[avere_idx]['Avere']:.2f}"] for avere_idx in match_row['avere_indices']]
        report_content.append(create_md_table(['Riga (Excel)', 'Data', 'Importo'], avere_rows))

        report_content.append(f"**Dettagli:** Somma AVERE: {match_row['somma_avere']:.2f} €, Differenza: {match_row['differenza']:.2f} €\n")
        report_content.append("---\n")

    report_content.append("\n## 🏁 Stato Finale dei Movimenti\n")
    report_content.append("Movimenti non abbinati dopo tutte le passate.\n")

    dare_non_util = final_dare_df[~final_dare_df['usato']]
    if not dare_non_util.empty:
        report_content.append("\n### 🔴 Incassi (DARE) Non Utilizzati\n")
        dare_rows_final = [[row['indice_orig'] + 2, row['Data'].strftime('%d/%m/%y'), f"{row['Dare']:.2f}"] for _, row in dare_non_util.iterrows()]
        report_content.append(create_md_table(['Riga (Excel)', 'Data', 'Importo'], dare_rows_final))
    else:
        report_content.append("\n### ✅ Tutti gli Incassi (DARE) sono stati utilizzati!\n")

    avere_non_riconc = final_avere_df[~final_avere_df['usato']]
    if not avere_non_riconc.empty:
        report_content.append("\n### 🟢 Versamenti (AVERE) Non Riconciliati\n")
        avere_rows_final = [[row['indice_orig'] + 2, row['Data'].strftime('%d/%m/%y'), f"{row['Avere']:.2f}"] for _, row in avere_non_riconc.iterrows()]
        report_content.append(create_md_table(['Riga (Excel)', 'Data', 'Importo'], avere_rows_final))
    else:
        report_content.append("\n### ✅ Tutti i Versamenti (AVERE) sono stati riconciliati!\n")

    report_content.append(f"\nPer statistiche dettagliate, consulta il file Excel: `{output_excel_filename}`\n")
    report_content.append("---\n")
    report_content.append("Generato automaticamente dal sistema di riconciliazione.")

    return "".join(report_content)


def create_stats_dict(dare_df, avere_df, dare_non_util, avere_non_riconc):
    """Crea un dizionario di statistiche dai DataFrame elaborati."""
    importo_dare_non_util = dare_non_util['Dare'].sum()
    importo_avere_non_riconc = avere_non_riconc['Avere'].sum()
    delta_finale = importo_dare_non_util - importo_avere_non_riconc

    stats_data = {
        'Totale Incassi (DARE)': len(dare_df),
        'Totale Versamenti (AVERE)': len(avere_df),
        'Incassi (DARE) utilizzati': int(dare_df['usato'].sum()),
        'Versamenti (AVERE) riconciliati': int(avere_df['usato'].sum()),
        'Incassi (DARE) non utilizzati': len(dare_non_util),
        'Versamenti (AVERE) non riconciliati': len(avere_non_riconc),
        '% Incassi (DARE) utilizzati': f"{dare_df['usato'].sum()/len(dare_df)*100:.1f}%" if len(dare_df) > 0 else "0.0%",
        '% Versamenti (AVERE) riconciliati': f"{avere_df['usato'].sum()/len(avere_df)*100:.1f}%" if len(avere_df) > 0 else "0.0%",
        'Importo DARE non utilizzato': f"{importo_dare_non_util:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."),
        'Importo AVERE non riconciliato': f"{importo_avere_non_riconc:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."),
        'Delta finale (DARE - AVERE)': f"{delta_finale:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."),
        # Aggiungi i valori grezzi per l'elaborazione aggregata
        '_raw_importo_dare_non_util': importo_dare_non_util,
        '_raw_importo_avere_non_riconc': importo_avere_non_riconc
    }
    return stats_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Riconciliazione Contabile per un singolo file.")
    parser.add_argument('-i', '--input', required=True, help="File di input (Excel o CSV).")
    parser.add_argument('-o', '--output', required=True, help="Nome del file di output Excel.")
    parser.add_argument('--tolleranza', type=float, default=0.01, help="Tolleranza di importo.")
    parser.add_argument('--giorni-finestra', type=int, default=30, help="Finestra temporale in giorni.")
    parser.add_argument('--max-combinazioni', type=int, default=6, help="Numero massimo di combinazioni.")
    parser.add_argument('--soglia-residui', type=float, default=100, help="Soglia per i DARE residui.")
    parser.add_argument('--giorni-finestra-residui', type=int, default=60, help="Finestra temporale per i residui.")
    parser.add_argument('--sorting-strategy', type=str, default="date", help="Strategia di ordinamento iniziale: 'date' o 'amount'.")
    parser.add_argument('--search-direction', type=str, default="both", help="Direzione della ricerca temporale: 'both', 'future_only', 'past_only'.")
    parser.add_argument('--silent', action='store_true', help="Esegui senza output verboso (usato dal batch).")
    
    # Carica la configurazione di base da config.json
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {} # Usa i default di argparse se config.json non esiste

    # Aggiorna i default di argparse con i valori da config.json
    parser.set_defaults(**config)
    
    # Ora il parsing degli argomenti darà priorità a quelli passati da linea di comando,
    # altrimenti userà i valori da config.json, altrimenti i default originali.
    args = parser.parse_args()
    stats = run_reconciliation(args.input, vars(args), args.output, args.silent)

    if args.silent and stats:
        print(json.dumps(stats)) # Stampa le statistiche in JSON per il processo padre
def create_stats_dict(dare_df, avere_df, dare_non_util, avere_non_riconc):
    """Crea un dizionario di statistiche dai DataFrame elaborati."""
    importo_dare_non_util = dare_non_util['Dare'].sum()
    importo_avere_non_riconc = avere_non_riconc['Avere'].sum()
    delta_finale = importo_dare_non_util - importo_avere_non_riconc

    stats_data = {
        'Totale Incassi (DARE)': len(dare_df),
        'Totale Versamenti (AVERE)': len(avere_df),
        'Incassi (DARE) utilizzati': int(dare_df['usato'].sum()),
        'Versamenti (AVERE) riconciliati': int(avere_df['usato'].sum()),
        'Incassi (DARE) non utilizzati': len(dare_non_util),
        'Versamenti (AVERE) non riconciliati': len(avere_non_riconc),
        '% Incassi (DARE) utilizzati': f"{dare_df['usato'].sum()/len(dare_df)*100:.1f}%" if len(dare_df) > 0 else "0.0%",
        '% Versamenti (AVERE) riconciliati': f"{avere_df['usato'].sum()/len(avere_df)*100:.1f}%" if len(avere_df) > 0 else "0.0%",
        'Importo DARE non utilizzato': f"{importo_dare_non_util:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."),
        'Importo AVERE non riconciliato': f"{importo_avere_non_riconc:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."),
        'Delta finale (DARE - AVERE)': f"{delta_finale:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."),
        # Aggiungi i valori grezzi per l'elaborazione aggregata
        '_raw_importo_dare_non_util': importo_dare_non_util,
        '_raw_importo_avere_non_riconc': importo_avere_non_riconc
    }
    return stats_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Riconciliazione Contabile per un singolo file.")
    parser.add_argument('-i', '--input', required=True, help="File di input (Excel o CSV).")
    parser.add_argument('-o', '--output', required=True, help="Nome del file di output Excel.")
    parser.add_argument('--tolleranza', type=float, default=0.01, help="Tolleranza di importo.")
    parser.add_argument('--giorni-finestra', type=int, default=30, help="Finestra temporale in giorni.")
    parser.add_argument('--max-combinazioni', type=int, default=6, help="Numero massimo di combinazioni.")
    parser.add_argument('--soglia-residui', type=float, default=100, help="Soglia per i DARE residui.")
    parser.add_argument('--giorni-finestra-residui', type=int, default=60, help="Finestra temporale per i residui.")
    parser.add_argument('--sorting-strategy', type=str, default="date", help="Strategia di ordinamento iniziale: 'date' o 'amount'.")
    parser.add_argument('--search-direction', type=str, default="both", help="Direzione della ricerca temporale: 'both', 'future_only', 'past_only'.")
    parser.add_argument('--silent', action='store_true', help="Esegui senza output verboso (usato dal batch).")
    
    # Carica la configurazione di base da config.json
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {} # Usa i default di argparse se config.json non esiste

    # Aggiorna i default di argparse con i valori da config.json
    parser.set_defaults(**config)
    
    # Ora il parsing degli argomenti darà priorità a quelli passati da linea di comando,
    # altrimenti userà i valori da config.json, altrimenti i default originali.
    args = parser.parse_args()
    stats = run_reconciliation(args.input, vars(args), args.output, args.silent)

    if args.silent and stats:
        print(json.dumps(stats)) # Stampa le statistiche in JSON per il processo padre
