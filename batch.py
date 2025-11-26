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
import subprocess
import threading

def run_subprocess_interactive(command, description, timeout=None):
    """
    Esegue un sottoprocesso mostrando l'output in tempo reale.
    Utile per processi con barre di avanzamento (es. optimizer).
    Non cattura stdout/stderr, ma restituisce solo il codice di uscita.
    """
    print(f"   -> {description}...")
    class CompletedProcessMock:
        def __init__(self, returncode, stdout="", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr
    try:
        # Esegue il processo e attende il completamento. L'output viene stampato direttamente.
        result = subprocess.run(command, text=True, timeout=timeout)
        return CompletedProcessMock(result.returncode)
    except subprocess.TimeoutExpired:
        print(f"\n   ❌ ERRORE: Il processo ha superato il tempo massimo di {timeout} secondi ed è stato terminato.")
        return CompletedProcessMock(1, stderr=f"TimeoutExpired: Il processo ha superato i {timeout} secondi.")
    except Exception as e:
        return CompletedProcessMock(1, stderr=str(e))

def run_subprocess_capture(command, description, timeout=None):
    """Esegue un sottoprocesso catturando l'output. Utile per leggere risultati (es. main.py)."""
    print(f"   -> {description} (l'output verrà mostrato al termine)...")

    class CompletedProcessMock:
        def __init__(self, returncode, stdout, stderr):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if process.returncode != 0:
            print("--- ERRORE NEL SOTTOPROCESSO ---")
            print(process.stderr)
            print("---------------------------------")
        return CompletedProcessMock(process.returncode, process.stdout, process.stderr)
    except subprocess.TimeoutExpired:
        print(f"\n   ❌ ERRORE: Il processo ha superato il tempo massimo di {timeout} secondi ed è stato terminato.")
        return CompletedProcessMock(1, stdout="", stderr=f"TimeoutExpired: Il processo ha superato i {timeout} secondi.")
    except Exception as e:
        return CompletedProcessMock(1, stdout="", stderr=str(e))

class BatchProcessor:
    """Orchestra l'elaborazione in batch chiamando main.py per ogni file."""
    
    def __init__(self, config=None):
        self.config = config if config is not None else {}
        self.base_config_path = 'config.json'
        self.cartella_input = Path(self.config['cartella_input'])
        self.cartella_output = Path(self.config['cartella_output'])
        self.timeout_per_file = self.config.get('timeout_per_file_seconds')
        
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
        print(f"   - Strategia di ottimizzazione: Automatica per ogni file")
        print(f"   - Cartella di output principale: {self.cartella_output}/")
        if self.timeout_per_file:
            print(f"   - Tempo massimo per file: {self.timeout_per_file} secondi")
        
        errori = []
        successi = 0
        statistiche_globali = [] # Lista per raccogliere le statistiche di ogni file
        
        with tqdm(total=len(files), desc="Avanzamento Batch") as pbar:
            for i, file_path in enumerate(files):
                start_time_file = datetime.now() # Registra l'inizio dell'elaborazione del file
                pbar.set_description(f"Processing {file_path.name}")
                print(f"\n--- [File {i+1}/{len(files)}] Elaborazione di: {file_path.name} ---")

                # 1. Per ogni file in input creare una cartella in output con lo stesso nome
                file_output_dir = self.cartella_output / file_path.stem
                file_output_dir.mkdir(exist_ok=True)
                print(f"1. Creata cartella di lavoro: {file_output_dir}")

                # 2. Copiare il file config base nella cartella appena creata.
                local_config_path = file_output_dir / 'config.json'
                shutil.copy(self.base_config_path, local_config_path)
                
                # 2.1 AGGIORNAMENTO: Scrivi i percorsi corretti nel config locale
                output_excel_path = file_output_dir / f"risultato_{file_path.stem}.xlsx"
                with open(local_config_path, 'r+') as f:
                    config_data = json.load(f)
                    config_data['file_input'] = str(file_path.resolve())
                    config_data['file_output'] = str(output_excel_path.resolve())
                    f.seek(0)
                    json.dump(config_data, f, indent=2)
                    f.truncate()
                print(f"2. Creato e aggiornato config locale: {local_config_path}")

                # 3. Lanciare l'optimizer.py usando il file config appena sopra
                print("3. Avvio ottimizzazione parametri (modalità automatica)...")
                # Aggiungiamo un controllo per assicurarci che l'optimizer non fallisca silenziosamente
                # se il file di input non è specificato correttamente.
                optimizer_cmd = [sys.executable, 'optimizer.py', '--config', str(local_config_path), '--auto']
                result = run_subprocess_interactive(optimizer_cmd, "Esecuzione optimizer.py", timeout=self.timeout_per_file)
                if result.returncode != 0:
                    errori.append((file_path.name, f"Optimizer fallito: {result.stderr}"))
                    pbar.update(1)
                    continue
                print("   ✓ Ottimizzazione completata. I parametri migliori sono stati salvati.")
                
                # 4. Lanciare l'analisi usando il nuovo file config ottimizzato
                print("4. Avvio riconciliazione finale con parametri ottimizzati...")
                main_cmd = [sys.executable, 'main.py', '--config', str(local_config_path)]
                result = run_subprocess_capture(main_cmd, "Esecuzione main.py", timeout=self.timeout_per_file)
                if result.returncode != 0:
                    errori.append((file_path.name, f"Main fallito: {result.stderr}"))
                    pbar.update(1)
                    continue
                
                # Cattura e parsifica le statistiche JSON dall'output di main.py
                try:
                    end_time_file = datetime.now() # Registra la fine
                    processing_time = (end_time_file - start_time_file).total_seconds()
                    if result.stdout.strip(): # Assicurati che ci sia output prima di parsare
                        stats = json.loads(result.stdout)
                        stats['processing_time_seconds'] = processing_time # Aggiungi il tempo
                        stats['file_name'] = file_path.name # Aggiungi il nome del file per il riepilogo
                        statistiche_globali.append(stats)
                except json.JSONDecodeError:
                    errori.append((file_path.name, f"Errore nel parsing delle statistiche JSON da main.py: {result.stdout}"))

                print(f"   ✓ Riconciliazione completata. Risultati salvati in {file_output_dir}")
                successi += 1
                pbar.update(1)
        
        print("\n\n🎉 Processo Batch completato.")
        print(f"   - File elaborati con successo: {successi}/{len(files)}")
        print(f"💾 I risultati per ogni file sono stati salvati in cartelle dedicate dentro: {self.cartella_output.resolve()}")

        # Stampa il riepilogo aggregato se sono state raccolte statistiche
        if statistiche_globali:
            self.stampa_riepilogo_globale(statistiche_globali)
            
        if errori:
            print("\n⚠️  Si sono verificati degli errori durante l'elaborazione:")
            for filename, stderr in errori:
                print(f"  - File: {filename}\n    Errore: {str(stderr)[:200]}...")

    def stampa_riepilogo_globale(self, stats_list):
        """Stampa una tabella con le statistiche aggregate di tutti i file."""
        def format_eur(value):
            return f"{value:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")

        print("\n\n╔════════════════════════════════════════════════════════════╗")
        print("║              📊 RIEPILOGO GLOBALE AGGREGATO 📊             ║")
        print("╚════════════════════════════════════════════════════════════╝")

        # Header della tabella
        header = f"{'File':<30} | {'% DARE (€)':>10} | {'% AVERE (€)':>11} | {'Sbilancio':>18} | {'Tempo (s)':>10}"
        print(header)
        print("-" * len(header))

        # Righe della tabella
        for stats in stats_list:
            file_name = stats.get('file_name', 'N/D')
            perc_dare_importo = f"{stats.get('_raw_perc_dare_importo', 0):.1f}%"
            perc_avere_importo = f"{stats.get('_raw_perc_avere_importo', 0):.1f}%"
            sbilancio = stats.get('Delta finale (DARE - AVERE)', 'N/D')
            tempo = f"{stats.get('processing_time_seconds', 0):.1f}"
            
            row = (f"{file_name:<30} | "
                   f"{perc_dare_importo:>10} | "
                   f"{perc_avere_importo:>11} | "
                   f"{sbilancio:>18} | "
                   f"{tempo:>10}")
            print(row)

        print("-" * len(header))

def carica_config():
    """Carica la configurazione da config.json o usa i valori di default."""
    config_file = 'config.json'
    
    # Configurazione di default
    default_config = {
        'timeout_per_file_seconds': None, # Nessun timeout di default
        'tolleranza': 0.01,
        'giorni_finestra': 30,
        'max_combinazioni': 6,
        'giorni_finestra_residui': 90,
        'soglia_residui': 100,
        'cartella_input': 'input',
        'cartella_output': 'output',
        'file_input': None, # Aggiunto per il nuovo flusso
        'file_output': None, # Aggiunto per il nuovo flusso
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
