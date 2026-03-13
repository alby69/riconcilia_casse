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

def load_config(config_file: Path) -> dict:
    """Loads the configuration from a JSON file."""
    if not config_file.exists():
        print(f"❌ ERRORE: File di configurazione '{config_file}' non trovato.", file=sys.stderr)
        sys.exit(1)
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            print(f"📄 Caricamento configurazione da '{config_file}'...")
            config = json.load(f)
            print("✓ Configurazione caricata con successo.")
            return config
    except json.JSONDecodeError as e:
        print(f"❌ ERRORE: Formato JSON non valido in '{config_file}': {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """Funzione principale"""
    config = load_config(Path('config.json'))
    common_params = config.get('common', {})
    
    input_folder = Path(common_params.get('input_folder', 'input'))
    output_folder = Path(common_params.get('output_folder', 'output'))
    output_folder.mkdir(exist_ok=True)

    patterns = common_params.get('file_patterns', ['*.xlsx', '*.csv'])
    files_to_process = []
    for p in patterns:
        files_to_process.extend(input_folder.glob(p))
    
    if not files_to_process:
        print(f"⚠️ Nessun file trovato in '{input_folder}' con i pattern: {patterns}")
        return

    print(f"Trovati {len(files_to_process)} file da elaborare.")

    # Get the algorithm name and merge its specific params with common ones
    algorithm_name = common_params.get('algorithm', 'subset_sum')
    engine_params = common_params.copy()
    engine_params.update(config.get(algorithm_name, {}))

    # Prepare params for ReconciliationEngine, ensuring correct names and types
    final_params = {
        'tolerance': engine_params.get('tolerance', 0.01),
        'days_window': engine_params.get('days_window', 10),
        'max_combinations': engine_params.get('max_combinations', 6),
        'residual_threshold': engine_params.get('residual_threshold', 100),
        'residual_days_window': engine_params.get('residual_days_window', 90),
        'column_mapping': engine_params.get('column_mapping'),
        'algorithm': algorithm_name,
        'search_direction': engine_params.get('search_direction', 'past_only')
    }

    for file_path in tqdm(files_to_process, desc="Elaborazione file"):
        print(f"\n{'='*20} Elaborazione di: {file_path.name} {'='*20}")
        try:
            # Instantiate the reconciler with the correct parameters
            engine = ReconciliationEngine(**final_params)
            
            output_file = output_folder / f"risultato_{file_path.stem}.xlsx"
            
            # Run the entire process
            engine.run(str(file_path), str(output_file))

        except (IOError, ValueError, FileNotFoundError) as e:
            print(f"❌ ERRORE durante l'elaborazione di {file_path.name}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
