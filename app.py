import io
import os
import json
import uuid
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template, flash, redirect, url_for, jsonify, send_from_directory, session
from core import RiconciliatoreContabile
from optimizer import run_simulation, update_config_file

# --- Configurazione dell'App Flask ---
app = Flask(__name__)
app.secret_key = 'supersecretkey_dev' # Cambiare in produzione

# --- Configurazione delle cartelle ---
LOG_FOLDER = 'log'
INPUT_FOLDER = 'input'
OUTPUT_FOLDER = os.path.join('output', 'processed_files')
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Assicura che le cartelle esistano all'avvio
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --- Pagina Principale ---
@app.route('/')
def index():
    """Mostra la pagina iniziale con il form di upload."""
    with open('config.json', 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    return render_template('index.html', config=config_data)


# --- Endpoint per la Configurazione ---
@app.route('/config', methods=['GET', 'POST'])
def handle_config():
    """Gestisce la lettura e la scrittura della configurazione."""
    if request.method == 'POST':
        try:
            new_config = request.get_json()
            # La funzione update_config_file di optimizer.py è più robusta
            # per gestire anche i parametri annidati.
            success = update_config_file('config.json', new_config)
            if not success:
                raise ValueError("La funzione update_config_file ha fallito.")
            
            return jsonify({'success': True, 'message': 'Configurazione salvata con successo.'})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Errore nel salvataggio della configurazione: {str(e)}'}), 500
    else: # GET
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return jsonify(config_data)
        except FileNotFoundError:
            return jsonify({'error': 'File di configurazione non trovato.'}), 404
        except Exception as e:
            return jsonify({'error': f'Errore nella lettura della configurazione: {str(e)}'}), 500


# --- Endpoint per l'Elaborazione (modificato per AJAX) ---
@app.route('/processa', methods=['POST'])
def processa_file():
    """
    Gestisce il caricamento del file, l'elaborazione, salva i risultati
    e restituisce un JSON con il link per il download e il log.
    Salva anche il file di input per l'ottimizzatore.
    """
    if 'file_input' not in request.files:
        return jsonify({'error': 'Nessun file selezionato.'}), 400

    file = request.files['file_input']

    if file.filename == '':
        return jsonify({'error': 'Nessun file selezionato.'}), 400

    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        return jsonify({'error': 'Formato file non supportato. Si prega di caricare un file Excel (.xlsx o .xls).'}), 400
    
    try:
        # --- 1. Salvataggio del file per l'ottimizzatore ---
        # Usiamo un nome fisso per semplicità, sovrascrivendo il file precedente.
        # In un'applicazione multi-utente, qui servirebbe un ID di sessione o utente.
        input_filepath = os.path.join(INPUT_FOLDER, 'last_uploaded_file.xlsx')
        file.save(input_filepath)
        session['last_uploaded_file'] = input_filepath
        
        # --- 2. Preparazione del DataFrame ---
        # Riapri il file appena salvato per l'elaborazione
        df_input = pd.read_excel(input_filepath)
        df_input['Data'] = pd.to_datetime(df_input['Data'], errors='coerce', dayfirst=True)
        df_input.dropna(subset=['Data'], inplace=True)
        
        # Conversione sicura a numerico, gestendo virgole e punti
        for col in ['Dare', 'Avere']:
            if df_input[col].dtype == 'object':
                df_input[col] = df_input[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
        
        df_input[['Dare', 'Avere']] = df_input[['Dare', 'Avere']].fillna(0)
        
        # Conversione in centesimi
        df_input['Dare'] = (df_input['Dare'] * 100).round().astype(int)
        df_input['Avere'] = (df_input['Avere'] * 100).round().astype(int)
        df_input['indice_orig'] = df_input.index

        # --- 3. Esecuzione della logica di riconciliazione ---
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        riconciliatore_config = {
            "tolleranza": int(config.get("tolleranza", 0.01) * 100),
            "giorni_finestra": config.get("giorni_finestra", 10),
            "max_combinazioni": config.get("max_combinazioni", 6),
            "soglia_residui": int(config.get("residui", {}).get("soglia_importo", 100) * 100),
            "giorni_finestra_residui": config.get("residui", {}).get("giorni_finestra", 90)
        }
        riconciliatore = RiconciliatoreContabile(**riconciliatore_config)
        stats = riconciliatore.run(df_input.copy(), output_file=None, verbose=False)

        # --- 4. Salvataggio del file di output con nome unico ---
        unique_id = uuid.uuid4()
        sanitized_filename = "".join(c for c in file.filename if c.isalnum() or c in ('.', '_')).rstrip()
        unique_output_filename = f"{unique_id}_{sanitized_filename}"
        output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], unique_output_filename)
        
        # Passiamo df_input originale (non modificato da run) per il report
        riconciliatore._crea_report_excel(output_filepath, df_input)

        # --- 5. Creazione e salvataggio del file di log ---
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f"{timestamp}_{sanitized_filename}_summary.log"
        log_filepath = os.path.join(LOG_FOLDER, log_filename)
        
        with open(log_filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        
        # --- 6. Preparazione dei nomi e salvataggio in sessione ---
        base_name, extension = os.path.splitext(sanitized_filename)
        pretty_download_filename = f"{base_name}_risultato{extension}"
        session['download_map'] = {pretty_download_filename: unique_output_filename}
        
        # --- 7. Restituzione del JSON per il frontend ---
        return jsonify({
            'log_content': json.dumps(stats, indent=4, ensure_ascii=False),
            'download_url': url_for('download_file', filename=pretty_download_filename),
            'version': '3.1' 
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Si è verificato un errore critico: {str(e)}"}), 500


# --- Endpoint per l'Ottimizzazione ---
@app.route('/optimize', methods=['POST'])
def optimize_params():
    """
    Esegue l'ottimizzazione dei parametri sul file caricato più di recente.
    """
    last_file = session.get('last_uploaded_file')
    if not last_file or not os.path.exists(last_file):
        return jsonify({'error': 'Nessun file valido su cui eseguire l\'ottimizzazione. Si prega di elaborare un file prima.'}), 400

    try:
        data = request.get_json()
        n_trials = data.get('n_trials', 50)

        # Carica la configurazione di base
        with open('config.json', 'r', encoding='utf-8') as f:
            base_config = json.load(f)

        # Carica il df di input
        loader = RiconciliatoreContabile()
        input_df = loader.carica_file(last_file) # carica_file gestisce la preparazione

        # Avvia la simulazione
        # show_progress=True stamperà la barra di avanzamento nella console del server
        results = run_simulation(
            base_config=base_config,
            file_input_df=input_df,
            n_trials=n_trials,
            show_progress=True,
            sequential=False # Usa tutti i core
        )

        return jsonify(results)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Si è verificato un errore durante l'ottimizzazione: {str(e)}"}), 500




# --- Nuovo Endpoint per il Download ---
@app.route('/download/<filename>')
def download_file(filename):
    """
    Serve il file di output usando una mappa salvata in sessione
    per trovare il file fisico con nome univoco.
    """
    download_map = session.get('download_map', {})
    actual_filename = download_map.get(filename)
    
    if not actual_filename:
        return "File non trovato o sessione scaduta.", 404
        
    return send_from_directory(
        app.config['OUTPUT_FOLDER'], 
        actual_filename, 
        as_attachment=True,
        download_name=filename  # FIX: Specifica il nome del file per il download
    )


# --- Avvio del Server ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
