import cProfile
import pstats
import io
import json
import os
from optimizer import run_simulation, load_optimizer_config, generate_dynamic_ranges

def run_profiling():
    """
    Funzione wrapper che esegue la logica di ottimizzazione da profilare.
    Questo è più utile perché l'ottimizzazione è la parte più lenta del processo.
    """
    # --- CONFIGURA QUI ---
    # 1. Specifica il file di configurazione di base e il file di input
    base_config_path = 'config.json'
    input_file_path = 'output/alessano_311025/alessano_311025.feather' # Usa un file già convertito
    
    # 2. Carica le configurazioni
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    optimizer_settings, optimization_params = load_optimizer_config()

    # 3. Scegli la modalità di ottimizzazione da profilare
    #    'True' per una prima esecuzione (esplorazione ampia), 'False' per un affinamento.
    is_first_run = False

    if is_first_run:
        print(">>> Profilazione in modalità ESPLORAZIONE AMPIA.")
        ranges_to_use = optimization_params
        n_trials = optimizer_settings.get('n_trials_first_run', 20) # Riduci i trial per una profilazione più rapida
    else:
        print(">>> Profilazione in modalità AFFINAMENTO MIRATO.")
        range_percentage = optimizer_settings.get('range_percentage', 0.30)
        ranges_to_use = generate_dynamic_ranges(base_config, optimization_params, range_percentage)
        n_trials = optimizer_settings.get('n_trials_refinement', 10) # Riduci i trial

    # Esegui la simulazione (la funzione che vuoi misurare)
    print(f"Avvio profilazione di 'run_simulation' per il file: {input_file_path}...")
    # Forziamo la modalità sequenziale per una profilazione più pulita e leggibile
    run_simulation(base_config, ranges_to_use, input_file_path, n_trials, show_progress=False, sequential=True)
    print("Profilazione completata.")

if __name__ == "__main__":
    # 1. Crea un oggetto Profiler
    profiler = cProfile.Profile()

    # 2. Esegui la tua funzione sotto il controllo del profiler
    profiler.run('run_profiling()')

    # 3. Stampa le statistiche ordinate per il tempo cumulativo speso in ogni funzione
    print("\n--- Risultati Profilazione (ordinate per Tempo Totale 'tottime') ---")
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(20) # Mostra le 20 funzioni più lente