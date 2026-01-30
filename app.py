import io
import os
import json
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_from_directory, session, url_for
import uuid
from core import RiconciliatoreContabile

# --- Configurazione dell'App Flask ---
app = Flask(__name__)
app.secret_key = 'supersecretkey_dev' # Cambiare in produzione

# --- Configurazione delle cartelle ---
LOG_FOLDER = 'log'
OUTPUT_FOLDER = 'output'
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
CONFIG_FILE_PATH = 'config.json'

# Assicura che le cartelle esistano all'avvio
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Funzioni Helper per la Configurazione ---
def load_config():
    """Carica la configurazione da config.json."""
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_config(new_config):
    """Salva la configurazione su config.json."""
    # Prima carica la configurazione esistente per non perdere chiavi non presenti nella UI
    current_config = load_config()
    # Aggiorna la configurazione con i nuovi valori
    current_config.update(new_config)
    # Salva il file completo
    with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(current_config, f, indent=2, ensure_ascii=False)

def robust_currency_parser(value):
    """Converte in modo robusto una stringa o un numero in un formato numerico standard per pd.to_numeric."""
    # Se è già un numero, è a posto.
    if isinstance(value, (int, float)):
        return value
    # Se non è una stringa, non possiamo farci nulla.
    if not isinstance(value, str):
        return None # Verrà convertito in NaN
    
    # Pulisci la stringa da spazi e simbolo euro
    cleaned_str = str(value).strip().replace('€', '').replace(' ', '')
    
    # Caso 1: Formato italiano completo (es. "1.234,56")
    # La presenza di entrambi i separatori è un forte indicatore.
    if '.' in cleaned_str and ',' in cleaned_str:
        return cleaned_str.replace('.', '').replace(',', '.')
    
    # Caso 2: Formato italiano con solo decimali (es. "1234,56")
    if ',' in cleaned_str:
        return cleaned_str.replace(',', '.')
        
    # Caso 3: Formato senza virgole (es. "1234" o "1234.56"). Lasciamo il punto.
    return cleaned_str

# --- Pagina Principale ---
@app.route('/')
def index():
    """Mostra la pagina iniziale con il form di upload."""
    return render_template('index.html')

# --- API per la Configurazione ---
@app.route('/api/config', methods=['GET'])
def get_config():
    """Restituisce la configurazione corrente in formato JSON."""
    try:
        return jsonify(load_config())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return jsonify({'error': f"Impossibile leggere config.json: {e}"}), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Aggiorna e salva la configurazione da un JSON inviato."""
    new_config_data = request.json
    save_config(new_config_data)
    return jsonify({'status': 'success', 'message': 'Configurazione salvata con successo.'})

# --- Endpoint per l'Elaborazione (modificato per AJAX) ---
@app.route('/processa', methods=['POST'])
def processa_file():
    """
    Gestisce il caricamento del file, l'elaborazione, salva i risultati
    e restituisce un JSON con il link per il download e il log.
    """
    if 'file_input' not in request.files:
        return jsonify({'error': 'Nessun file selezionato.'}), 400

    file = request.files['file_input']

    if file.filename == '':
        return jsonify({'error': 'Nessun file selezionato.'}), 400

    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        return jsonify({'error': 'Formato file non supportato. Si prega di caricare un file Excel (.xlsx o .xls).'}), 400
    
    try:
        # --- 1. Estrazione dei parametri di configurazione dal form ---
        # I valori inviati dal form sono stringhe, vanno convertiti al tipo corretto.
        # Estraiamo 'save_log' separatamente perché non va passato al costruttore di RiconciliatoreContabile
        save_log = request.form.get('save_log') == 'true'

        config_params = {
            'tolleranza': request.form.get('tolleranza', 0.01, type=float),
            'giorni_finestra': request.form.get('giorni_finestra', 7, type=int),
            'max_combinazioni': request.form.get('max_combinazioni', 10, type=int),
            'soglia_residui': request.form.get('soglia_residui', 100.0, type=float),
            'giorni_finestra_residui': request.form.get('giorni_finestra_residui', 30, type=int),
            'sorting_strategy': request.form.get('sorting_strategy', 'date', type=str),
            'search_direction': request.form.get('search_direction', 'past_only', type=str),
            'algorithm': request.form.get('algorithm', 'subset_sum', type=str),
            'ignore_tolerance': request.form.get('ignore_tolerance') == 'true'
        }

        # --- 2. Preparazione del DataFrame ---
        df_input = pd.read_excel(file.stream)

        # Applicazione del mapping colonne (External -> Internal) prima di accedere ai dati
        full_config = load_config()
        mapping_conf = full_config.get('mapping_colonne', {})
        # Inverte il mapping: {'Data': 'DATA'} -> {'DATA': 'Data'} per rinominare correttamente
        rename_mapping = {v: k for k, v in mapping_conf.items()}
        df_input.rename(columns=rename_mapping, inplace=True)

        df_input['Data'] = pd.to_datetime(df_input['Data'], errors='coerce', dayfirst=True)
        df_input.dropna(subset=['Data'], inplace=True)

        # --- NUOVA LOGICA DI PARSING ROBUSTA ---
        # Applica il parser a ogni cella, poi converte l'intera colonna.
        df_input['Dare'] = pd.to_numeric(df_input['Dare'].apply(robust_currency_parser), errors='coerce')
        df_input['Avere'] = pd.to_numeric(df_input['Avere'].apply(robust_currency_parser), errors='coerce')

        df_input[['Dare', 'Avere']] = df_input[['Dare', 'Avere']].fillna(0)
        df_input['Dare'] = (df_input['Dare'] * 100).round().astype(int)
        df_input['Avere'] = (df_input['Avere'] * 100).round().astype(int)
        df_input['indice_orig'] = df_input.index

        # --- 3. Esecuzione della logica di riconciliazione con i parametri dalla UI ---
        riconciliatore = RiconciliatoreContabile(**config_params)
        stats = riconciliatore.run(df_input, output_file=None, verbose=False)

        # --- 4. Salvataggio del file di output con nome unico ---
        unique_id = uuid.uuid4()
        sanitized_filename = "".join(c for c in file.filename if c.isalnum() or c in ('.', '_')).rstrip()
        unique_output_filename = f"{unique_id}_{sanitized_filename}"
        output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], unique_output_filename)
        
        riconciliatore._crea_report_excel(output_filepath, df_input)

        # --- 5. Creazione e salvataggio del file di log ---
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f"{timestamp}_{sanitized_filename}_summary.log"
        log_filepath = os.path.join(LOG_FOLDER, log_filename)
        
        formatted_stats = stats.copy()
        for key, value in stats.items():
            if '_raw_importo' in key:
                formatted_stats[key] = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " €"
            elif '_raw_perc' in key:
                formatted_stats[key] = f"{value:.2f} %".replace('.', ',')

        # Salva su disco solo se richiesto esplicitamente
        if save_log:
            with open(log_filepath, 'w', encoding='utf-8') as f:
                json.dump(formatted_stats, f, indent=4, ensure_ascii=False)
        
        # --- 6. Preparazione dei nomi e salvataggio in sessione ---
        base_name, extension = os.path.splitext(sanitized_filename)
        pretty_download_filename = f"{base_name}_risultato{extension}"
        
        # Salva la mappa {nome_bello: nome_unico} nella sessione utente
        session['download_map'] = {pretty_download_filename: unique_output_filename}
        
        # --- 7. Restituzione del JSON per il frontend ---
        return jsonify({
            'log_content': json.dumps(formatted_stats, indent=4, ensure_ascii=False),
            'download_url': url_for('download_file', filename=pretty_download_filename),
            'version': '3.0' # Versione incrementata per il nuovo fix
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Si è verificato un errore critico: {str(e)}"}), 500


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
