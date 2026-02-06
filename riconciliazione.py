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
from core import ReconciliationEngine

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
            # Istanzia il riconciliatore con i parametri dalla configurazione
            engine = ReconciliationEngine(
                tolerance=config['tolleranza'],
                days_window=config['giorni_finestra'],
                max_combinations=config['algoritmo']['max_combinazioni'],
                residual_threshold=config['residui']['soglia_importo'],
                residual_days_window=config['residui']['giorni_finestra'],
                column_mapping=config.get('mapping_colonne'),
                algorithm=config.get('algorithm', 'subset_sum'),
                search_direction=config.get('search_direction', 'past_only')
            )

            output_file = cartella_output / f"risultato_{file_path.stem}.xlsx"
            
            # Esegui l'intero processo passando i percorsi dei file
            engine.run(str(file_path), str(output_file))

        except (IOError, ValueError, FileNotFoundError) as e:
            print(f"❌ ERRORE durante l'elaborazione di {file_path.name}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
