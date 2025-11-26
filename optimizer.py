"""
Optimizer - Script per la Ricerca Operativa dei Parametri di Riconciliazione.

Questo script esegue simulazioni multiple su un singolo file di input,
variando i parametri chiave per trovare la combinazione che massimizza
le percentuali di riconciliazione.
"""

import pandas as pd
import itertools
import argparse
from datetime import datetime
import time
from pathlib import Path
import json
import sys
import multiprocessing
from tqdm import tqdm # Importa tqdm
import optuna # Importa Optuna
def run_auto_optimization(config, config_path):
    """Esegue l'ottimizzazione automatica basata su range predefiniti."""
    print("🔬 Avvio ottimizzazione in modalità automatica...")
    params_to_test = AUTO_OPTIMIZATION_RANGES
    
    # Esegui la simulazione e trova i parametri migliori
    best_params = run_simulation(config, params_to_test)

def load_optimizer_config(config_path='config_optimizer.json'):
    """Carica la configurazione dell'ottimizzatore da un file JSON."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            print(f"📄 Caricamento configurazione optimizer da '{config_path}'...")
            optimizer_config = json.load(f)
            print("✓ Configurazione optimizer caricata con successo.")
            return optimizer_config
    except FileNotFoundError:
        print(f"⚠️  File '{config_path}' non trovato. Utilizzo configurazione di default.")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ ERRORE: Formato JSON non valido in '{config_path}': {e}")
        print("Il programma verrà terminato.")
        sys.exit(1)

def run_auto_optimization(config, config_path):
    """Esegue l'ottimizzazione automatica basata su range predefiniti."""
    print("🔬 Avvio ottimizzazione in modalità automatica...")
    
    # Carica la configurazione dell'ottimizzatore dal file
    optimizer_config = load_optimizer_config()
    
    # Esegui la simulazione e trova i parametri migliori
    # Passiamo direttamente la configurazione dell'optimizer a run_simulation
    best_params = run_simulation(config, optimizer_config)

    # Scrivi i parametri migliori nel file di configurazione
    update_config_file(config_path, best_params)

# Nuova funzione helper per l'esecuzione in parallelo
def _run_single_simulation_worker(args):
    """
    Funzione worker per eseguire una singola simulazione di riconciliazione.
    Progettata per essere utilizzata con multiprocessing.Pool.
    """
    run_config, params = args
    
    # Importa RiconciliatoreContabile all'interno della funzione worker
    # Questo è cruciale per evitare problemi di serializzazione (pickling)
    from core import RiconciliatoreContabile 

    riconciliatore_sim = RiconciliatoreContabile(
        tolleranza=run_config.get('tolleranza', 0.01),
        giorni_finestra=run_config.get('giorni_finestra', 30),
        max_combinazioni=run_config.get('max_combinazioni', 6),
        soglia_residui=run_config.get('soglia_residui', 100),
        giorni_finestra_residui=run_config.get('giorni_finestra_residui', 60),
        sorting_strategy=run_config.get('sorting_strategy', 'date'),
        search_direction=run_config.get('search_direction', 'both')
    )
    
    INPUT_FILE_DA_OTTIMIZZARE = run_config['file_input']
    start_time = time.time()
    # verbose=False per evitare output disordinato dai processi paralleli
    stats = riconciliatore_sim.run(INPUT_FILE_DA_OTTIMIZZARE, output_file=None, verbose=False) 
    end_time = time.time()
    execution_time = end_time - start_time

    if stats:
        perc_dare_str = stats.get('% Incassi (DARE) utilizzati', '0.0%')
        perc_avere_str = stats.get('% Versamenti (AVERE) riconciliati', '0.0%')

        perc_dare = float(str(perc_dare_str).replace('%', '')) if perc_dare_str else 0.0
        perc_avere = float(str(perc_avere_str).replace('%', '')) if perc_avere_str else 0.0

        # Restituisce tutte le informazioni necessarie al processo principale
        return {
            "params": params,
            "perc_dare": perc_dare,
            "perc_avere": perc_avere,
            "execution_time": execution_time,
            "full_stats": stats # Opzionalmente restituisce le statistiche complete per il logging
        }
    return None # Restituisce None se la simulazione è fallita o non ha prodotto statistiche

def run_simulation(base_config, optimizer_config_ranges):
    """Esegue l'ottimizzazione usando Optuna per trovare i parametri migliori."""

    def objective(trial):
        """Funzione obiettivo che Optuna cercherà di massimizzare."""
        # 1. Suggerisci i parametri per questo "trial"
        params = {}
        for param_name, details in optimizer_config_ranges.items():
            if details['type'] == 'numeric':
                # Usa suggest_int per i parametri interi
                params[param_name] = trial.suggest_int(param_name, details['min'], details['max'], step=details['step'])
            elif details['type'] == 'categorical':
                # Usa suggest_categorical per i parametri testuali
                params[param_name] = trial.suggest_categorical(param_name, details['values'])

        # 2. Esegui la simulazione con i parametri suggeriti
        run_config = base_config.copy()
        run_config.update(params)
        
        # Importa qui per essere compatibile con la parallelizzazione di Optuna
        from core import RiconciliatoreContabile

        # Filtra il dizionario di configurazione per passare solo i parametri
        # attesi dal costruttore di RiconciliatoreContabile.
        expected_params = [
            'tolleranza', 'giorni_finestra', 'max_combinazioni', 
            'soglia_residui', 'giorni_finestra_residui', 
            'sorting_strategy', 'search_direction'
        ]
        riconciliatore_config = {
            key: run_config[key] for key in expected_params if key in run_config
        }

        # Usa sempre il file di input completo per la massima accuratezza
        input_data_for_run = run_config['file_input']

        riconciliatore_sim = RiconciliatoreContabile(**riconciliatore_config)
        stats = riconciliatore_sim.run(input_data_for_run, output_file=None, verbose=False)

        # 3. Calcola e restituisci il punteggio da massimizzare
        if stats:
            perc_dare_str = stats.get('% Incassi (DARE) utilizzati', '0.0%')
            perc_avere_str = stats.get('% Versamenti (AVERE) riconciliati', '0.0%')
            perc_dare = float(str(perc_dare_str).replace('%', ''))
            perc_avere = float(str(perc_avere_str).replace('%', ''))
            
            # Vogliamo massimizzare la somma delle percentuali
            return perc_dare + perc_avere
        
        # Se la simulazione fallisce, restituisci un punteggio molto basso
        return 0.0

    # Disabilita il logging verboso di Optuna per mantenere l'output pulito
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Crea uno "studio" di ottimizzazione.
    # La direzione è "maximize" perché vogliamo il punteggio più alto.
    study = optuna.create_study(direction="maximize")

    # Avvia l'ottimizzazione. n_trials è il numero di simulazioni da eseguire.
    # 100 trial sono spesso sufficienti per trovare ottimi risultati.
    # n_jobs=-1 usa tutti i core della CPU per parallelizzare i trial.
    # Dopo le ottimizzazioni in core.py, ogni trial è molto più veloce.
    # Riduciamo il numero di trial per accelerare il processo batch,
    # mantenendo comunque una buona capacità di ricerca.
    n_trials = 30 
    print(f"🚀 Avvio ottimizzazione con Optuna per {n_trials} trial (in parallelo)...")

    # Aggiungi una barra di avanzamento con tqdm
    with tqdm(total=n_trials, desc="Ottimizzazione Trial") as pbar:
        # Definisci un callback per aggiornare la barra di avanzamento dopo ogni trial
        def callback(study, trial):
            pbar.update(1)

        study.optimize(objective, n_trials=n_trials, n_jobs=-1, callbacks=[callback])

    # Stampa il suggerimento finale
    best_params = study.best_params
    best_score = study.best_value

    print("\n\n╔════════════════════════════════════════════════════════════╗")
    print("║              🏆 RISULTATO OTTIMALE TROVATO 🏆              ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print("\nLa combinazione di parametri che ha prodotto i migliori risultati è:")
    for key, value in best_params.items():
        print(f"  - {key}: {value}")
    print(f"\nCon queste impostazioni, hai raggiunto:")
    # Nota: il punteggio esatto potrebbe non essere recuperabile facilmente, ma possiamo indicare il valore ottimizzato
    print(f"  - Punteggio ottimizzato (Somma % DARE + % AVERE): {best_score:.2f}")
    print("\nSuggerimento: Aggiorna il tuo file 'config.json' con questi valori per le elaborazioni future.")
    
    return best_params

def get_user_parameters():
    """Funzione interattiva per definire i parametri e i range della simulazione."""
    print("\n--- Configurazione Parametri di Ottimizzazione ---")
    # Definizione strutturata dei parametri disponibili per l'ottimizzazione
    available_params = {
        "1": {"name": "giorni_finestra", "type": "numeric", "prompt": "Finestra temporale standard (giorni)"},
        "2": {"name": "max_combinazioni", "type": "numeric", "prompt": "Numero massimo di combinazioni"},
        "3": {"name": "giorni_finestra_residui", "type": "numeric", "prompt": "Finestra temporale per i residui (giorni)"},
        "4": {"name": "soglia_residui", "type": "numeric", "prompt": "Soglia importo per analisi residui (€)"},
        "5": {"name": "tolleranza", "type": "numeric", "prompt": "Tolleranza di importo (es. 0.01)"},
        "6": {"name": "sorting_strategy", "type": "categorical", "values": ["date", "amount"], "prompt": "Strategia di ordinamento"},
        "7": {"name": "search_direction", "type": "categorical", "values": ["future_only", "past_only", "both"], "prompt": "Direzione della ricerca temporale"}
    }

    params_to_test = {}
    while True:
        print("\nScegli un parametro da variare (o premi Invio per iniziare la simulazione):")
        for key, value in available_params.items():
            print(f"  {key}) {value['prompt']}")

        choice = input("> ")
        if not choice:
            if not params_to_test:
                print("❌ Nessun parametro selezionato. Uscita.")
                exit()
            break

        if choice not in available_params:
            print("❌ Scelta non valida.")
            continue

        param_info = available_params.pop(choice) # Rimuovi per non sceglierlo di nuovo
        param_name = param_info['name']

        if param_info['type'] == 'numeric':
            try:
                value_type = float if param_name in ['tolleranza', 'soglia_residui'] else int
                min_val = value_type(input(f"  - Valore MIN per '{param_name}': "))
                max_val = value_type(input(f"  - Valore MAX per '{param_name}': "))
                step = value_type(input(f"  - Passo (step) per '{param_name}': "))
                
                # Genera la sequenza di valori
                values = []
                current = min_val
                while current <= max_val:
                    values.append(current)
                    current += step
                params_to_test[param_name] = values
                print(f"✓ Parametro '{param_name}' configurato per testare i valori: {values}")
            except (ValueError, TypeError):
                print("❌ Input non valido. Riprova.")
                available_params[choice] = param_info # Reinserisci per poterlo riselezionare
        
        elif param_info['type'] == 'categorical':
            possible_values = param_info['values']
            print(f"  Valori possibili per '{param_name}':")
            for i, val in enumerate(possible_values):
                print(f"    {i+1}) {val}")
            
            user_choice = input("  Scegli i valori da testare (es. '1,3' per il primo e il terzo, o 'tutti'): ")
            
            selected_values = []
            if user_choice.lower() == 'tutti':
                selected_values = possible_values
            else:
                try:
                    indices = [int(i.strip()) - 1 for i in user_choice.split(',')]
                    for idx in indices:
                        if 0 <= idx < len(possible_values):
                            selected_values.append(possible_values[idx])
                        else:
                            print(f"❌ Indice '{idx+1}' non valido.")
                    if not selected_values:
                        raise ValueError("Nessun valore valido selezionato.")
                except (ValueError, IndexError):
                    print("❌ Selezione non valida. Riprova.")
                    available_params[choice] = param_info # Reinserisci
                    continue
            
            params_to_test[param_name] = list(set(selected_values)) # Rimuovi duplicati
            print(f"✓ Parametro '{param_name}' configurato per testare i valori: {params_to_test[param_name]}")

    return params_to_test

def update_config_file(config_path, best_params):
    """Aggiorna il file di configurazione JSON con i parametri migliori trovati."""
    with open(config_path, 'r+') as f:
        config_data = json.load(f)
        config_data.update(best_params)
        f.seek(0) # Riavvolgi all'inizio del file
        json.dump(config_data, f, indent=2)
        f.truncate() # Rimuovi il contenuto rimanente se il nuovo file è più corto
    print(f"\n✅ File di configurazione '{config_path}' aggiornato con i parametri ottimali.")

def main():
    """Funzione principale che orchestra la simulazione."""
    parser = argparse.ArgumentParser(description="Ottimizzatore dei parametri di riconciliazione.")
    parser.add_argument('--config', required=True, help="Percorso del file di configurazione JSON da usare e aggiornare.")
    parser.add_argument('--auto', action='store_true', help="Esegui in modalità automatica non interattiva.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Errore: File di configurazione non trovato in '{config_path}'")
        sys.exit(1)

    # Carica la configurazione di base dal file specificato
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("╔════════════════════════════════════════════════════════════╗")
    print("║        🚀 AVVIO OTTIMIZZATORE PARAMETRI 🚀                 ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"\n🎯 File di configurazione in uso: {config_path.resolve()}")
    print(f"📄 File di input per l'analisi: {config.get('file_input')}")

    if args.auto:
        run_auto_optimization(config, config_path)
    else:
        # Modalità interattiva
        print("\n⚙️  Configurazione di base (da config.json):")
        for key, value in config.items():
            if key != "commento":
                print(f"   - {key}: {value}")
        
        params_to_test = get_user_parameters()
        best_params = run_simulation(config, params_to_test)
        update_config_file(config_path, best_params)

if __name__ == "__main__":
    main()