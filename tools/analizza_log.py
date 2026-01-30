import os
import re
from datetime import datetime
import argparse
import sys
import pandas as pd


def analizza_log(data_str=None, export_csv=False):
    """
    Analizza i file di log per una data specifica (o per oggi se non specificata).

    Args:
        data_str (str, optional): La data da analizzare nel formato 'AAAA-MM-GG'.
                                  Se None, viene utilizzata la data odierna.

        export_csv (bool, optional): Se True, esporta i risultati in un file CSV.
    Questo script è stato adattato alla struttura dei log generati da batch.py.
    """
    log_dir = 'log'
    # Cerca la cartella log nella directory corrente o in quella superiore (se spostato in tools/)
    if not os.path.isdir(log_dir):
        if os.path.isdir(os.path.join('..', 'log')):
            log_dir = os.path.join('..', 'log')
        else:
            print(f"Errore: La cartella '{log_dir}' non è stata trovata.")
            print("Assicurati di eseguire lo script dalla cartella principale del progetto 'riconcilia_casse'.")
            return

    if data_str:
        try:
            # Valida il formato della data fornita dall'utente
            datetime.strptime(data_str, '%Y-%m-%d')
            data_da_analizzare = data_str
        except ValueError:
            print(f"Errore: Formato data non valido '{data_str}'. Usa il formato AAAA-MM-GG.", file=sys.stderr)
            return
    else:
        data_da_analizzare = datetime.now().strftime('%Y-%m-%d')
    risultati = []

    # Regex estesa per catturare tutti i parametri di ottimizzazione.
    # Usa gruppi nominati (?P<nome>...) per rendere il codice più leggibile e robusto.
    # I parametri sono opzionali (.*?) per gestire log più vecchi che potrebbero non averli.
    block_regex = re.compile(
        r"File: (.*?)\n"  # Cattura il nome del file originale
        r".*?% Importo DARE utilizzato: (?P<perc_dare>\d+\.\d+)%\n"
        r".*?% Importo AVERE utilizzato: (?P<perc_avere>\d+\.\d+)%\n"
        r".*?Parametri Ottimali Usati:.*?\n"
        r".*?- giorni_finestra: (?P<giorni_finestra>\d+)\n"
        r".*?- max_combinazioni: (?P<max_combinazioni>\d+)\n"
        r".*?- giorni_finestra_residui: (?P<giorni_finestra_residui>\d+)\n"
        r".*?- soglia_residui: (?P<soglia_residui>[\d\.]+)\n"
        r".*?- sorting_strategy: (?P<sorting_strategy>\w+)\n"
        r".*?- search_direction: (?P<search_direction>\w+)\n"
        r".*?- tolleranza: (?P<tolleranza>[\d\.]+)",
        re.DOTALL  # Permette a '.' di includere anche i newline
    )

    # I file di log sono nominati con il timestamp, es: 2025-12-10_16-02-18_summary.log
    for filename in sorted(os.listdir(log_dir)):
        # Filtra i file per la data target
        if filename.endswith("_summary.log") and filename.startswith(data_da_analizzare):
            filepath = os.path.join(log_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Estrai l'orario dal nome del file
                    try:
                        timestamp_str = filename.split('_summary.log')[0]
                        orario = datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S').strftime('%H:%M:%S')
                    except ValueError:
                        orario = "N/D"
                    
                    # Trova tutti i blocchi di risultati nel contenuto del file di log
                    matches = block_regex.finditer(content)
                    
                    for i, match in enumerate(matches):
                        original_filename = match.group(1)
                        match_dict = match.groupdict()

                        if original_filename and match_dict.get('perc_dare') and match_dict.get('perc_avere'):
                            perc_dare = float(match_dict['perc_dare'])
                            perc_avere = float(match_dict['perc_avere'])
                            risultati.append({
                                'orario': orario,
                                'original_filename': original_filename.strip(),
                                'perc_dare': perc_dare,
                                'perc_avere': perc_avere,
                                'delta': perc_dare - perc_avere,
                                # Aggiungi tutti gli altri parametri, gestendo i tipi
                                'giorni_finestra': int(match_dict.get('giorni_finestra', 0)),
                                'max_combinazioni': int(match_dict.get('max_combinazioni', 0)),
                                'tolleranza': float(match_dict.get('tolleranza', 0.0)),
                                'giorni_finestra_residui': int(match_dict.get('giorni_finestra_residui', 0)),
                                'soglia_residui': float(match_dict.get('soglia_residui', 0.0)),
                                'sorting_strategy': match_dict.get('sorting_strategy', 'N/D'),
                                'search_direction': match_dict.get('search_direction', 'N/D'),
                                'file_log': filename,
                                'run_index': i + 1 # Per distinguere i risultati all'interno dello stesso log
                            })

            except Exception as e:
                print(f"Errore durante la lettura del file {filename}: {e}")

    if not risultati:
        print(f"Nessun risultato trovato per la data {data_da_analizzare}.")
        print(f"Lo script cerca file come '{data_da_analizzare}_*_summary.log' nella cartella '{log_dir}'.")
        return

    # Ordina i risultati: chiave primaria 'original_filename', chiave secondaria 'orario'
    risultati.sort(key=lambda x: (x['original_filename'], x['orario']))

    # --- Stampa a console ---
    _stampa_tabella(risultati, data_da_analizzare)

    # --- Esportazione CSV (se richiesta) ---
    if export_csv:
        _esporta_csv(risultati, data_da_analizzare)

def _stampa_tabella(risultati, data_da_analizzare):
    """Stampa i risultati formattati in una tabella a console."""
    # Stampa la tabella dei risultati
    print(f"--- Analisi Risultati Riconciliazione del {data_da_analizzare} ---")
    # Adattiamo la larghezza e le colonne per una migliore leggibilità
    header = f"{'File Originale':<25} | {'Orario':<10} | {'Finestra':<8} | {'Max Comb':<8} | {'Toller.':<7} | {'% DARE':<10} | {'% AVERE':<10} | {'Delta %':<10} | {'File di Log'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Raggruppa per nome file per una visualizzazione chiara
    last_filename = None
    for res in risultati:
        if res['original_filename'] != last_filename and last_filename is not None:
            print("-" * len(header))
        
        # Mostra il nome del file solo per la prima riga del suo gruppo
        filename_display = res['original_filename'] if res['original_filename'] != last_filename else ""
        
        print(f"{filename_display:<25} | {res['orario']:<10} | {res['giorni_finestra']:<8} | {res['max_combinazioni']:<8} | {res['tolleranza']:<7.2f} | {res['perc_dare']:>9.2f}% | {res['perc_avere']:>9.2f}% | {res['delta']:>+9.2f}% | {res['file_log']}")
        last_filename = res['original_filename']

    print("-" * len(header))

def _esporta_csv(risultati, data_da_analizzare):
    """Esporta i risultati in un file CSV."""
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f"analisi_log_{data_da_analizzare}.csv")
    
    df = pd.DataFrame(risultati)
    df.to_csv(csv_filename, index=False, sep=';', decimal=',')
    print(f"\n✅ Risultati esportati con successo in: {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analizza i log di riconciliazione per una data specifica.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--data', help="La data da analizzare nel formato 'AAAA-MM-GG'.\nSe omesso, analizza i log di oggi.")
    parser.add_argument('--csv', action='store_true', help="Esporta l'output in un file CSV nella cartella 'output'.")
    args = parser.parse_args()

    analizza_log(data_str=args.data, export_csv=args.csv)
