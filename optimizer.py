"""
Optimizer - Script per la Ricerca Operativa dei Parametri di Riconciliazione.

Questo script esegue simulazioni multiple su un singolo file di input,
variando i parametri chiave per trovare la combinazione che massimizza
le percentuali di riconciliazione.
"""

import pandas as pd
import itertools
from datetime import datetime
import time
from pathlib import Path
import json
from main import run_reconciliation  # Importa la logica di riconciliazione da main.py

# --- CONFIGURAZIONE DELLA SIMULAZIONE ---

# 1. Specifica il file di input su cui eseguire i test
INPUT_FILE_DA_OTTIMIZZARE = "input/sancesareo_311025.xlsx"

# 2. Definisci la cartella di output per i log
LOG_DIR = Path("output") / "logs"

# --- FINE CONFIGURAZIONE ---

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

def main():
    """Funzione principale che orchestra la simulazione."""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║        🚀 AVVIO OTTIMIZZATORE PARAMETRI 🚀             ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"\n🎯 File di input per l'analisi: {INPUT_FILE_DA_OTTIMIZZARE}")

    # Carica la configurazione di base da config.json
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Mostra la configurazione di base
    print("\n⚙️  Configurazione di base (da config.json):")
    for key, value in config.items():
        if key != "commento":
            print(f"   - {key}: {value}")

    # Ottieni i parametri e i range dall'utente
    params_to_test = get_user_parameters()

    # Genera tutte le combinazioni di parametri
    keys, values = zip(*params_to_test.items())
    parameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_runs = len(parameter_combinations)
    print(f"🔬 Verranno eseguite {total_runs} simulazioni.\n")

    log_results = []
    best_result = {"percentuale_dare": 0, "percentuale_avere": 0, "tempo_esecuzione_sec": float('inf')}

    for i, params in enumerate(parameter_combinations):
        run_config = config.copy()
        run_config.update(params)

        print(f"--- [Simulazione {i+1}/{total_runs}] ---")
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        print(f"Parametri: {param_str}")

        # Esegui la riconciliazione con la configurazione corrente
        start_time = time.time()
        stats = run_reconciliation(INPUT_FILE_DA_OTTIMIZZARE, run_config, silent=True)
        end_time = time.time()
        execution_time = end_time - start_time

        if stats:
            # Estrai le percentuali per il confronto
            perc_dare_str = stats.get('% Incassi (DARE) utilizzati', '0.0%')
            perc_avere_str = stats.get('% Versamenti (AVERE) riconciliati', '0.0%')
            
            perc_dare = float(perc_dare_str.replace('%', ''))
            perc_avere = float(perc_avere_str.replace('%', ''))

            print(f"📊 Risultati: % DARE = {perc_dare_str}, % AVERE = {perc_avere_str} (in {execution_time:.2f} sec)\n")

            # Aggiungi i risultati al log
            log_entry = params.copy()
            log_entry['tempo_esecuzione_sec'] = round(execution_time, 2)
            log_entry.update(stats)
            log_results.append(log_entry)

            # Controlla se questo è il risultato migliore finora (basato sulla somma delle percentuali)
            if (perc_dare + perc_avere) > (best_result.get("percentuale_dare", 0) + best_result.get("percentuale_avere", 0)):
                best_result = {
                    **params,
                    "percentuale_dare": perc_dare,
                    "percentuale_avere": perc_avere,
                    "tempo_esecuzione_sec": execution_time
                }

    # Salva il log completo in un file CSV
    if log_results:
        # Crea la cartella di log se non esiste
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Costruisci il nome e il percorso completo del file di log
        log_filename = f"optimization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        log_filepath = LOG_DIR / log_filename
        
        log_df = pd.DataFrame(log_results)
        log_df.to_csv(log_filepath, index=False, sep=';')
        print(f"\n💾 Log completo di tutte le simulazioni salvato in: {log_filepath.resolve()}")

    # Stampa il suggerimento finale
    print("\n\n╔════════════════════════════════════════════════════════════╗")
    print("║              🏆 RISULTATO OTTIMALE TROVATO 🏆              ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print("\nLa combinazione di parametri che ha prodotto i migliori risultati è:")
    for key, value in best_result.items():
        if key not in ["percentuale_dare", "percentuale_avere", "tempo_esecuzione_sec"]:
            print(f"  - {key}: {value}")
    print(f"\nCon queste impostazioni, hai raggiunto:")
    print(f"  - % Incassi (DARE) utilizzati: {best_result['percentuale_dare']:.1f}%")
    print(f"  - % Versamenti (AVERE) riconciliati: {best_result['percentuale_avere']:.1f}%")
    print(f"  - Tempo di esecuzione: {best_result['tempo_esecuzione_sec']:.2f} secondi")
    print("\nSuggerimento: Aggiorna il tuo file 'config.json' con questi valori per le elaborazioni future.")


if __name__ == "__main__":
    main()