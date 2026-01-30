import os
import json
import argparse
import subprocess
import sys

from datetime import datetime
import time

try:
    # Add the directory of convert_to_feather.py to the Python path
    # This assumes batch.py and convert_to_feather.py are in the same directory
    sys.path.append(os.path.dirname(__file__))
    try:
        from convert_to_feather import convert_excel_to_feather
    except ImportError:
        # Fallback: cerca nella cartella tools se il file è stato spostato
        sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))
        from convert_to_feather import convert_excel_to_feather

except ImportError:
    print("Errore: impossibile importare 'convert_to_feather'. Assicurati che il file 'convert_to_feather.py' sia nella stessa directory.")
    sys.exit(1)

def process_single_file(filename, input_dir, output_dir, base_config):
    """
    Elabora un singolo file: converte in feather, esegue l'ottimizzatore e la riconciliazione.
    Restituisce le statistiche e il tempo di esecuzione.
    """
    start_time = time.time()
    file_path = os.path.join(input_dir, filename)
    file_base_name, file_ext = os.path.splitext(filename)

    current_output_folder = os.path.join(output_dir, file_base_name)
    os.makedirs(current_output_folder, exist_ok=True)

    print(f"\n--- Processing file: {filename} ---")

    # Step 1: Conversione a Feather (se necessario)
    processed_input_path = _handle_file_conversion(file_path, file_base_name, file_ext, current_output_folder)
    if not processed_input_path:
        return None, 0

    # Step 2: Preparazione della configurazione locale
    local_config_path = _prepare_local_config(processed_input_path, file_base_name, current_output_folder, base_config)
    if not local_config_path:
        return None, 0

    # Step 3: Esecuzione dell'ottimizzatore
    print(f"Avvio ottimizzazione parametri per '{filename}'...")
    optimizer_success = _run_optimizer(local_config_path, filename, sequential=getattr(process_single_file, 'sequential_optimizer', False))
    if not optimizer_success:
        return None, 0

    # Step 4: Esecuzione della riconciliazione finale
    print(f"Avvio riconciliazione finale per '{filename}'...")
    stats = _run_main_reconciliation(local_config_path, filename)
    
    if stats:
        stats['original_filename'] = filename

    end_time = time.time()
    execution_time = end_time - start_time

    if stats:
        print(f"Completato '{filename}' in {execution_time:.2f} secondi.")
    
    return stats, execution_time

def process_single_file_wrapper(args):
    return process_single_file(*args)

def run_batch(sequential_optimizer=False, save_log=False):
    input_dir = 'input/'
    output_dir = 'output/'
    base_config_path = 'config.json'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load base config
    try:
        with open(base_config_path, 'r', encoding='utf-8') as f:
            base_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Base config file '{base_config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{base_config_path}'. Check file format.")
        sys.exit(1)

    # List files in input directory
    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    if not input_files:
        print(f"No files found in '{input_dir}'. Please place your Excel or CSV files there.")
        return

    all_stats = []

    for filename in input_files:
        # Imposta l'attributo per passarlo alla funzione
        process_single_file.sequential_optimizer = sequential_optimizer
        file_stats, exec_time = process_single_file(filename, input_dir, output_dir, base_config)
        if file_stats:
            file_stats['execution_time_seconds'] = exec_time
            all_stats.append(file_stats)

    # --- Step 5: Genera, stampa e salva il riepilogo globale ---
    _generate_and_save_summary(all_stats, output_dir, save_log)

def _handle_file_conversion(file_path, file_base_name, file_ext, output_folder, sequential_optimizer=False):
    """Converte file Excel in Feather per ottimizzare le performance."""
    if file_ext.lower() in ['.xlsx', '.xls']:
        print(f"  - Conversione in formato Feather per performance...")
        feather_output_path = os.path.join(output_folder, f"{file_base_name}.feather")
        try:
            processed_path = convert_excel_to_feather(file_path, feather_path=feather_output_path, force_conversion=True)
            print(f"  - File convertito utilizzato: {processed_path}")
            return processed_path
        except Exception as e:
            print(f"Errore durante la conversione di '{file_base_name}{file_ext}' in Feather: {e}")
            return None
    elif file_ext.lower() in ['.feather', '.csv']:
        print(f"  - Rilevato file '{file_base_name}{file_ext}'. Nessuna conversione necessaria.")
        return file_path
    else:
        print(f"Attenzione: Tipo di file non supportato '{file_ext}' per '{file_base_name}{file_ext}'. Salto il file.")
        return None

def _prepare_local_config(input_path, file_base_name, output_folder, base_config):
    """Prepara e salva un file config.json locale per l'elaborazione corrente."""
    local_config_path = os.path.join(output_folder, 'config.json')

    # --- LOGICA EVOLUTIVA ---
    # Se esiste già una configurazione locale ottimizzata, usa quella come base.
    # Altrimenti, parti dalla configurazione di base.
    if os.path.exists(local_config_path):
        print(f"  - Trovata configurazione ottimizzata precedente. La uso come base per la nuova ottimizzazione.")
        with open(local_config_path, 'r', encoding='utf-8') as f:
            local_config = json.load(f)
    else:
        print(f"  - Nessuna configurazione precedente trovata. Parto dalla configurazione di base.")
        local_config = base_config.copy()

    # Aggiorna sempre i percorsi di input/output per la sessione corrente
    local_config['file_input'] = input_path
    reconciliation_output_filename = f"risultato_{file_base_name}.xlsx"
    local_config['file_output'] = os.path.join(output_folder, reconciliation_output_filename)

    try:
        with open(local_config_path, 'w', encoding='utf-8') as f:
            json.dump(local_config, f, indent=4)
        return local_config_path
    except IOError as e:
        print(f"Errore nel salvataggio della configurazione locale per '{file_base_name}': {e}")
        return None

def _run_optimizer(config_path, filename, sequential=False):
    """Esegue lo script optimizer.py."""
    try:
        command = ['python', '-u', 'optimizer.py', '--config', config_path, '--auto']
        if sequential:
            command.append('--sequential')
            print("   - Esecuzione ottimizzatore in modalità SEQUENZIALE.")

        process = subprocess.Popen(command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            print(f"\nErrore durante l'esecuzione dell'ottimizzatore per '{filename}'. Codice di uscita: {process.returncode}.")
            return False
        return True
    except FileNotFoundError:
        print("Errore: 'optimizer.py' non trovato. Assicurati che sia nella stessa directory di 'batch.py'.")
        return False
    except Exception as e:
        print(f"Errore imprevisto durante l'esecuzione di optimizer.py per '{filename}': {e}")
        return False

def _run_main_reconciliation(config_path, filename):
    """Esegue lo script main.py e recupera le statistiche in formato JSON."""
    try:
        main_result = subprocess.run(
            ['python', 'main.py', '--config', config_path, '--silent'],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        
        # Cerca la riga JSON nell'output
        stats_line = None
        for line in reversed(main_result.stdout.splitlines()):
            stripped_line = line.strip()
            if stripped_line.startswith('{') and stripped_line.endswith('}'):
                stats_line = stripped_line
                break
        
        if not stats_line:
            print(f"Attenzione: Nessuna statistica JSON trovata nell'output di main.py per '{filename}'.")
            print(f"Output ricevuto:\n{main_result.stdout}")
            return None

        try:
            file_stats = json.loads(stats_line)
            return file_stats
        except json.JSONDecodeError:
            print(f"Attenzione: Impossibile decodificare le statistiche JSON dall'output di main.py per '{filename}'.")
            print(f"Riga JSON rilevata: {stats_line}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione della riconciliazione principale per '{filename}': {e}")
        print(f"Stderr:\n{e.stderr}")
        return None
    except FileNotFoundError:
        print("Errore: 'main.py' non trovato. Assicurati che sia nella stessa directory di 'batch.py'.")
        return None
    except Exception as e:
        print(f"Errore imprevisto durante l'esecuzione di main.py per '{filename}': {e}")
        return None

def _generate_and_save_summary(all_stats, output_dir, save_log=False):
    """Genera il riepilogo, lo stampa a console e lo salva in un file di log."""
    summary_lines = []
    separator = "="*50

    summary_lines.append(separator)
    summary_lines.append("GLOBAL RECONCILIATION SUMMARY")
    summary_lines.append(separator)

    if all_stats:
        total_time = sum(s.get('execution_time_seconds', 0) for s in all_stats)
        for stats in all_stats:
            summary_lines.append(f"\nFile: {stats.get('original_filename', 'N/A')}")
            summary_lines.append(f"  Tempo di esecuzione: {stats.get('execution_time_seconds', 0):.2f} secondi")
            summary_lines.append(f"  Incassi (DARE): {stats.get('Incassi (DARE) utilizzati', 'N/A')} / {stats.get('Totale Incassi (DARE)', 'N/A')} ({stats.get('% Incassi (DARE) utilizzati', 'N/A')})")
            summary_lines.append(f"  Versamenti (AVERE): {stats.get('Versamenti (AVERE) riconciliati', 'N/A')} / {stats.get('Totale Versamenti (AVERE)', 'N/A')} ({stats.get('% Versamenti (AVERE) riconciliati', 'N/A')})")
            summary_lines.append(f"  % Importo DARE utilizzato: {stats.get('_raw_perc_dare_importo', 0):.2f}%")
            summary_lines.append(f"  % Importo AVERE utilizzato: {stats.get('_raw_perc_avere_importo', 0):.2f}%")
            summary_lines.append(f"  Sbilancio finale: {stats.get('Delta finale (DARE - AVERE)', 'N/A')}")
            summary_lines.append(f"  (Di cui Strutturale/Iniziale: {stats.get('Sbilancio Strutturale (Origine)', 'N/A')})")

            file_base_name, _ = os.path.splitext(stats.get('original_filename', ''))
            local_config_path = os.path.join(output_dir, file_base_name, 'config.json')
            try:
                with open(local_config_path, 'r') as f:
                    optimized_config = json.load(f)
                
                summary_lines.append("  Parametri Ottimali Usati:")
                summary_lines.append(f"    - giorni_finestra: {optimized_config.get('giorni_finestra')}")
                summary_lines.append(f"    - max_combinazioni: {optimized_config.get('max_combinazioni')}")
                summary_lines.append(f"    - giorni_finestra_residui: {optimized_config.get('giorni_finestra_residui')}")
                summary_lines.append(f"    - soglia_residui: {optimized_config.get('soglia_residui')}")
                summary_lines.append(f"    - sorting_strategy: {optimized_config.get('sorting_strategy')}")
                summary_lines.append(f"    - search_direction: {optimized_config.get('search_direction')}")
                summary_lines.append(f"    - tolleranza: {optimized_config.get('tolleranza')}")
            except (FileNotFoundError, json.JSONDecodeError):
                summary_lines.append("  Non è stato possibile leggere i parametri ottimali dal file di configurazione locale.")
        
        summary_lines.append("\n" + "-"*50)
        summary_lines.append(f"Tempo totale di esecuzione batch: {total_time:.2f} secondi")
    else:
        summary_lines.append("No files were successfully processed or no statistics were collected.")
    
    summary_lines.append(separator)
    
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text) # Stampa a console

    if save_log:
        # Salva su file di log solo se richiesto
        log_dir = 'log'
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = os.path.join(log_dir, f"{timestamp}_summary.log")
        with open(log_filename, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"\n📄 Riepilogo salvato in: {log_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegue il processo di riconciliazione in batch su una cartella di file.")
    parser.add_argument(
        '--sequential-optimizer',
        action='store_true',
        help="Forza l'esecuzione dell'ottimizzatore in modalità sequenziale (un processo alla volta) per evitare un uso eccessivo della CPU."
    )
    parser.add_argument(
        '--save-log',
        action='store_true',
        help="Salva il riepilogo dell'esecuzione in un file nella cartella log/."
    )
    args = parser.parse_args()
    run_batch(sequential_optimizer=args.sequential_optimizer, save_log=args.save_log)