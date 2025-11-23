"""
Sistema di Riconciliazione Contabile

Script principale per l'elaborazione di uno o più file di movimenti contabili.

Logica:
1. Carica la configurazione da `config.json`.
2. Trova i file da elaborare nella cartella di input.
3. Per ogni file:
    a. Carica i dati (Excel o CSV).
    b. Applica il mapping delle colonne definito in `config.json`.
    c. Esegue l'algoritmo di riconciliazione tramite il modulo `core`.
    d. Salva i risultati in un file Excel dettagliato nella cartella di output.
4. Stampa un riepilogo finale.
"""

from pathlib import Path
from datetime import datetime
import json
import sys
from tqdm import tqdm
import pandas as pd
from core import RiconciliatoreContabile, Movimento

def carica_config():
    """Carica la configurazione da config.json o usa i valori di default."""
    config_file = Path('config.json')
    default_config = {
        "tolleranza": 0.01,
        "giorni_finestra": 10,
        "cartella_input": "input",
        "cartella_output": "output",
        "pattern": ["*.xlsx", "*.csv"],
        "colonne": {
            "data": "DATA",
            "versamenti": "VERSAMENTI",
            "incassi": "INCASSI"
        },
        "mapping_colonne": {
            "Data": "DATA",
            "Dare": "INCASSI",
            "Avere": "VERSAMENTI"
        },
        "algoritmo": {"max_combinazioni": 6},
        "residui": {
            "attiva": True,
            "soglia_importo": 100,
            "giorni_finestra": 90
        }
    }
    
    if not config_file.exists():
        print(f"⚠️  File '{config_file}' non trovato. Utilizzo configurazione di default.")
        return default_config
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            print(f"📄 Caricamento configurazione da '{config_file}'...")
            user_config = json.load(f)
            default_config.update(user_config)
            print("✓ Configurazione caricata con successo.")
            return default_config
    except json.JSONDecodeError as e:
        print(f"❌ ERRORE: Formato JSON non valido in '{config_file}': {e}", file=sys.stderr)
        sys.exit(1)

def carica_e_mappa_dati(file_path, config):
    """Carica i dati da un file e applica il mapping delle colonne."""
    try:
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        raise IOError(f"Impossibile leggere il file: {e}")

    # Inverte il mapping per trovare il nome della colonna sorgente
    # Esempio: {'DATA': 'Data', 'INCASSI': 'Dare', 'VERSAMENTI': 'Avere'}
    mapping_inverso = {v: k for k, v in config['mapping_colonne'].items()}
    
    colonne_interne = config['colonne']
    colonne_da_mappare = {
        'data': colonne_interne['data'],
        'incassi': colonne_interne['incassi'],
        'versamenti': colonne_interne['versamenti']
    }

    colonne_rinominate = {}
    for nome_logico, nome_interno in colonne_da_mappare.items():
        # Trova il nome della colonna sorgente (es. 'Dare') che mappa al nome interno (es. 'INCASSI')
        nome_sorgente = next((k for k, v in config['mapping_colonne'].items() if v == nome_interno and k in df.columns), None)
        
        if not nome_sorgente:
            raise ValueError(f"Nessuna colonna mappata a '{nome_interno}' trovata nel file. Controllare 'mapping_colonne' in config.json.")
        
        colonne_rinominate[nome_sorgente] = nome_interno

    df.rename(columns=colonne_rinominate, inplace=True)

    # Convalida che tutte le colonne necessarie siano presenti dopo la rinomina
    colonne_necessarie = set(colonne_interne.values())
    if not colonne_necessarie.issubset(df.columns):
        colonne_mancanti = colonne_necessarie - set(df.columns)
        raise ValueError(f"Colonne interne mancanti dopo il mapping: {colonne_mancanti}")

    # --- Validazione e pulizia dei dati ---
    righe_scartate_totali = 0

    col_data = colonne_interne['data']
    col_incassi = colonne_interne['incassi']
    col_versamenti = colonne_interne['versamenti']

    # 1. Validazione Date
    df_original_dates = df[col_data].copy()
    df[col_data] = pd.to_datetime(df[col_data], errors='coerce', dayfirst=True) # Aggiunto dayfirst=True per flessibilità
    invalid_date_rows = df[df[col_data].isna()]
    if not invalid_date_rows.empty:
        print(f"⚠️ ATTENZIONE: Trovate {len(invalid_date_rows)} righe con date non valide. Saranno ignorate.")
        for idx, row in invalid_date_rows.iterrows():
            print(f"    - Riga {idx} (Originale: '{df_original_dates.loc[idx]}'): Data non valida.")
        df.dropna(subset=[col_data], inplace=True)
        righe_scartate_totali += len(invalid_date_rows)

    # 2. Validazione Importi Incassi
    df_original_incassi = df[col_incassi].copy()
    df[col_incassi] = pd.to_numeric(df[col_incassi], errors='coerce')
    invalid_incassi_amount_rows = df[df[col_incassi].isna() & df_original_incassi.notna()]
    if not invalid_incassi_amount_rows.empty:
        print(f"⚠️ ATTENZIONE: Trovate {len(invalid_incassi_amount_rows)} righe con importi INCASSI non numerici. Saranno ignorate.")
        for idx, row in invalid_incassi_amount_rows.iterrows():
            print(f"    - Riga {idx} (Originale: '{df_original_incassi.loc[idx]}'): Importo INCASSI non numerico.")
        righe_scartate_totali += len(invalid_incassi_amount_rows)

    # 3. Validazione Importi Versamenti
    df_original_versamenti = df[col_versamenti].copy()
    df[col_versamenti] = pd.to_numeric(df[col_versamenti], errors='coerce')
    invalid_versamenti_amount_rows = df[df[col_versamenti].isna() & df_original_versamenti.notna()]
    if not invalid_versamenti_amount_rows.empty:
        print(f"⚠️ ATTENZIONE: Trovate {len(invalid_versamenti_amount_rows)} righe con importi VERSAMENTI non numerici. Saranno ignorate.")
        for idx, row in invalid_versamenti_amount_rows.iterrows():
            print(f"    - Riga {idx} (Originale: '{df_original_versamenti.loc[idx]}'): Importo VERSAMENTI non numerico.")
        righe_scartate_totali += len(invalid_versamenti_amount_rows)

    if righe_scartate_totali > 0:
        print(f"ℹ️ Totale righe scartate a causa di dati non validi: {righe_scartate_totali}")

    incassi = [Movimento(row.Index, row[col_data], row[col_incassi], 'incasso') for row in df[df[col_incassi].notna()].itertuples()]
    versamenti = [Movimento(row.Index, row[col_data], row[col_versamenti], 'versamento') for row in df[df[col_versamenti].notna()].itertuples()]
    
    return incassi, versamenti

def main():
    """Funzione principale"""
    config = carica_config()
    cartella_input = Path(config['cartella_input'])
    cartella_output = Path(config['cartella_output'])
    cartella_output.mkdir(exist_ok=True)

    patterns = config.get('pattern', ['*.xlsx', '*.csv'])
    files_da_elaborare = []
    for p in patterns:
        files_da_elaborare.extend(cartella_input.glob(p))
    
    if not files_da_elaborare:
        print(f"⚠️ Nessun file trovato in '{cartella_input}' con i pattern: {patterns}")
        return

    print(f"Trovati {len(files_da_elaborare)} file da elaborare.")

    for file_path in tqdm(files_da_elaborare, desc="Elaborazione file"):
        print(f"\n{'='*20} Elaborazione di: {file_path.name} {'='*20}")
        try:
            incassi, versamenti = carica_e_mappa_dati(file_path, config)
            print(f"✓ Caricati {len(incassi)} incassi e {len(versamenti)} versamenti validi.")

            riconciliatore = RiconciliatoreContabile(
                incassi, versamenti,
                tolleranza=config['tolleranza'],
                giorni_finestra=config['giorni_finestra'],
                max_combinazioni=config['algoritmo']['max_combinazioni'],
                residui_config=config.get('residui')
            )

            riconciliatore.esegui_riconciliazione()
            
            output_file = cartella_output / f"risultato_{file_path.stem}.xlsx"
            riconciliatore.salva_risultati(output_file)
            print(f"💾 Risultati salvati in: {output_file}")

        except (IOError, ValueError, FileNotFoundError) as e:
            print(f"❌ ERRORE durante l'elaborazione di {file_path.name}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
