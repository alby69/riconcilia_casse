"""
Batch Processor - Wrapper per l'elaborazione multipla di file.
Questo script orchestra l'elaborazione, chiamando `main.py` per ogni file.
"""

from pathlib import Path
from datetime import datetime
import json
import shutil
import sys
import os
from tqdm import tqdm
import pandas as pd
import subprocess
import concurrent.futures

def run_main_py_worker(file_path, config, output_dir):
    """Funzione eseguita da ogni worker per lanciare main.py come subprocess."""
    output_file = output_dir / f"risultato_{file_path.stem}.xlsx"
    command = [
        sys.executable, 'main.py',
        '--input', str(file_path), '--output', str(output_file),
        '--tolleranza', str(config['tolleranza']), '--giorni-finestra', str(config['giorni_finestra']),
        '--max-combinazioni', str(config['max_combinazioni']), '--soglia-residui', str(config['soglia_residui']),
        '--giorni-finestra-residui', str(config['giorni_finestra_residui']), '--silent'
        '--sorting-strategy', str(config['sorting_strategy']),
        '--search-direction', str(config['search_direction'])
    ]
    return subprocess.run(command, capture_output=True, text=True)

def run_main_py_sequentially(file_path, config, output_dir):
    """Esegue main.py in modo sequenziale per mostrare l'output dettagliato."""
    print(f"\n{'='*60}")
    print(f"📂 Elaborazione dettagliata di: {file_path.name}")
    print(f"{'='*60}")
    
    # Importa la funzione di riconciliazione refattorizzata da main.py
    from main import run_reconciliation

    output_file = output_dir / f"risultato_{file_path.stem}.xlsx"

    try:
        # Chiama la funzione direttamente con i parametri necessari
        stats = run_reconciliation(str(file_path), config, output_file, silent=False)
        if stats:
            return True, "", stats
        else:
            return False, "La riconciliazione non ha restituito statistiche.", None
    except Exception as e:
        return False, str(e), None

class BatchProcessor:
    """Orchestra l'elaborazione in batch chiamando main.py per ogni file."""
    
    def __init__(self, config=None):
        self.config = config if config is not None else {}
        self.cartella_input = Path(self.config['cartella_input'])
        self.cartella_output = Path(self.config['cartella_output'])
        
    def crea_cartelle(self):
        self.cartella_output.mkdir(exist_ok=True)
        log_dir = self.cartella_output / 'logs'
        if log_dir.exists():
            shutil.rmtree(log_dir)
        log_dir.mkdir(exist_ok=True)
        
    def trova_files(self):
        if not self.cartella_input.exists():
            raise FileNotFoundError(f"Cartella '{self.cartella_input}' non trovata!")
        
        patterns = self.config['pattern'] if isinstance(self.config['pattern'], list) else [self.config['pattern']]
        files = []
        for p in patterns:
            files.extend(self.cartella_input.glob(p))
        files = [f for f in files if not f.name.startswith('~')]
        return sorted(files)
    
    def elabora_tutti(self):
        print("""
╔════════════════════════════════════════════════════════════╗
║          BATCH PROCESSOR - ELABORAZIONE MULTIPLA           ║
╚════════════════════════════════════════════════════════════╝
        """)
        self.crea_cartelle()
        print(f"🔍 Ricerca file in: {self.cartella_input}")
        files = self.trova_files()
        
        if not files:
            print(f"⚠️  Nessun file trovato con i pattern specificati in: {self.cartella_input}")
            return
        print(f"✓ Trovati {len(files)} file da elaborare\n")
        
        print(f"⚙️ CONFIGURAZIONE:")
        for key, value in self.config.items():
            if key not in ['cartella_input', 'cartella_output', 'pattern', 'commento']:
                print(f"   - {key}: {value}")
        print(f"   - Output: {self.cartella_output}/")
        
        errori = []
        statistiche_globali = []
        
        # Esegui tutti i file tranne l'ultimo in parallelo (in background)
        if len(files) > 1:
            files_in_parallelo = files[:-1]
            print(f"\n🚀 Avvio elaborazione parallela di {len(files_in_parallelo)} file su {os.cpu_count()} core...")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = {executor.submit(run_main_py_worker, file_path, self.config, self.cartella_output): file_path for file_path in files_in_parallelo}
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(files_in_parallelo), desc="Avanzamento Batch"):
                    result = future.result()
                    if result.returncode != 0:
                        file_path = futures[future]
                        errori.append((file_path.name, result.stderr))
                    else:
                        # Decodifica le statistiche JSON dall'output del processo
                        try:
                            stats = json.loads(result.stdout)
                            statistiche_globali.append(stats)
                        except json.JSONDecodeError:
                            pass # Ignora se l'output non è JSON valido

        # Esegui l'ultimo file in modo sequenziale per mostrare il progresso dettagliato
        if files:
            ultimo_file = files[-1]
            success, error_msg, stats = run_main_py_sequentially(ultimo_file, self.config, self.cartella_output)
            if not success:
                errori.append((ultimo_file.name, error_msg))
            elif stats:
                statistiche_globali.append(stats)
        
        print("\n\n🎉 Processo Batch completato.")
        print(f"💾 I risultati sono stati salvati singolarmente nella cartella: {self.cartella_output}")
        
        # Stampa il riepilogo aggregato se sono state raccolte statistiche
        if statistiche_globali:
            self.stampa_riepilogo_globale(statistiche_globali)

        if errori:
            print("\n⚠️  Si sono verificati degli errori durante l'elaborazione:")
            for filename, stderr in errori:
                print(f"  - File: {filename}\n    Errore: {str(stderr)[:200]}...")

    def stampa_riepilogo_globale(self, stats_list):
        """Stampa una tabella con le statistiche aggregate di tutti i file."""
        total_dare = sum(s.get('Totale Incassi (DARE)', 0) for s in stats_list)
        total_avere = sum(s.get('Totale Versamenti (AVERE)', 0) for s in stats_list)
        total_dare_usati = sum(s.get('Incassi (DARE) utilizzati', 0) for s in stats_list)
        total_avere_usati = sum(s.get('Versamenti (AVERE) riconciliati', 0) for s in stats_list)
        total_importo_dare_non_util = sum(s.get('_raw_importo_dare_non_util', 0) for s in stats_list)
        total_importo_avere_non_riconc = sum(s.get('_raw_importo_avere_non_riconc', 0) for s in stats_list)
        
        perc_dare = (total_dare_usati / total_dare * 100) if total_dare > 0 else 0
        perc_avere = (total_avere_usati / total_avere * 100) if total_avere > 0 else 0
        total_delta = total_importo_dare_non_util - total_importo_avere_non_riconc
        
        formatted_delta = f"{total_delta:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")

        print("\n\n╔════════════════════════════════════════════════════════════╗")
        print("║              📊 RIEPILOGO GLOBALE AGGREGATO 📊             ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print(f"  - File elaborati con successo: {len(stats_list)}")
        print(f"  - Totale Incassi (DARE): {total_dare_usati} / {total_dare} ({perc_dare:.1f}%)")
        print(f"  - Totale Versamenti (AVERE): {total_avere_usati} / {total_avere} ({perc_avere:.1f}%)")
        print(f"  - Delta importo non agganciato: {formatted_delta}")
        print("="*60)

def carica_config():
    """Carica la configurazione da config.json o usa i valori di default."""
    config_file = 'config.json'
    
    # Configurazione di default
    default_config = {
        'tolleranza': 0.01,
        'giorni_finestra': 30,
        'max_combinazioni': 6,
        'giorni_finestra_residui': 90,
        'soglia_residui': 100,
        'cartella_input': 'input',
        'cartella_output': 'output',
        'pattern': ['*.xlsx', '*.csv'],
        'sorting_strategy': 'date', # Default: ordina per data
        'search_direction': 'both' # Default: cerca in entrambe le direzioni
    }
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            print(f"📄 Caricamento configurazione da '{config_file}'...")
            user_config = json.load(f)
            # I valori nel file JSON sovrascrivono quelli di default
            default_config.update(user_config)
            print("✓ Configurazione caricata con successo.")
        return default_config
    except FileNotFoundError:
        print(f"⚠️  File '{config_file}' non trovato. Utilizzo configurazione di default.")
        return default_config
    except json.JSONDecodeError as e:
        print(f"❌ ERRORE: Formato JSON non valido in '{config_file}': {e}")
        print("Il programma verrà terminato.")
        sys.exit(1) # Termina lo script in caso di errore JSON


def main():
    """Funzione principale"""
    config = carica_config()
    processor = BatchProcessor(config)
    processor.elabora_tutti()


if __name__ == "__main__":
    main()
