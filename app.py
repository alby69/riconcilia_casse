import io
import os
import json
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_from_directory, session, url_for
import uuid
from core import RiconciliatoreContabile

# --- Flask App Configuration ---
app = Flask(__name__)
app.secret_key = 'supersecretkey_dev' # Change in production

# --- Folder Configuration ---
LOG_FOLDER = 'log'
OUTPUT_FOLDER = 'output'
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
CONFIG_FILE_PATH = 'config.json'

# Ensure folders exist on startup
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Configuration Helper Functions ---
def load_config():
    """Loads the configuration from config.json."""
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_config(new_config):
    """Saves the configuration to config.json."""
    # First, load the existing configuration to avoid losing keys not present in the UI
    current_config = load_config()
    # Update the configuration with the new values
    current_config.update(new_config)
    # Save the complete file
    with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(current_config, f, indent=2, ensure_ascii=False)

def robust_currency_parser(value):
    """Robustly converts a string or number into a standard numeric format for pd.to_numeric."""
    # If it's already a number, it's fine.
    if isinstance(value, (int, float)):
        return value
    # If it's not a string, we can't do anything.
    if not isinstance(value, str):
        return None # Will be converted to NaN
    
    # Clean the string from spaces and euro symbol
    cleaned_str = str(value).strip().replace('€', '').replace(' ', '')
    
    # Case 1: Full Italian format (e.g., "1.234,56")
    # The presence of both separators is a strong indicator.
    if '.' in cleaned_str and ',' in cleaned_str:
        return cleaned_str.replace('.', '').replace(',', '.')
    
    # Case 2: Italian format with only decimals (e.g., "1234,56")
    if ',' in cleaned_str:
        return cleaned_str.replace(',', '.')
        
    # Case 3: Format without commas (e.g., "1234" or "1234.56"). We leave the dot.
    return cleaned_str

# --- Pagina Principale ---
@app.route('/')
def index():
    """Displays the initial page with the upload form."""
    return render_template('index.html')

# --- API per la Configurazione ---
@app.route('/api/config', methods=['GET'])
def get_config():
    """Returns the current configuration in JSON format."""
    try:
        return jsonify(load_config())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return jsonify({'error': f"Unable to read config.json: {e}"}), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Updates and saves the configuration from a submitted JSON."""
    new_config_data = request.json
    save_config(new_config_data)
    return jsonify({'status': 'success', 'message': 'Configuration saved successfully.'})

# --- Endpoint per l'Elaborazione (modificato per AJAX) ---
@app.route('/processa', methods=['POST'])
def processa_file():
    """
    Handles file upload, processing, saves the results,
    and returns a JSON with the download link and log.
    """
    if 'file_input' not in request.files:
        return jsonify({'error': 'No file selected.'}), 400

    file = request.files['file_input']

    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        return jsonify({'error': 'Unsupported file format. Please upload an Excel file (.xlsx or .xls).'}), 400
    
    try:
        # --- 1. Extract configuration parameters from the form ---
        # Values sent from the form are strings, they need to be converted to the correct type.
        # We extract 'save_log' separately because it's not passed to the RiconciliatoreContabile constructor
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

        # --- 2. DataFrame Preparation ---
        # Fix for Python 3.9 SpooledTemporaryFile error: read into BytesIO
        file.stream.seek(0)
        df_input = pd.read_excel(io.BytesIO(file.stream.read()))

        # Apply column mapping (External -> Internal) before accessing data
        full_config = load_config()
        mapping_conf = full_config.get('mapping_colonne', {})
        # Invert the mapping: {'Data': 'DATE'} -> {'DATE': 'Data'} to rename correctly
        rename_mapping = {v: k for k, v in mapping_conf.items()}
        df_input.rename(columns=rename_mapping, inplace=True)

        df_input['Data'] = pd.to_datetime(df_input['Data'], errors='coerce', dayfirst=True)
        df_input.dropna(subset=['Data'], inplace=True)

        # --- NUOVA LOGICA DI PARSING ROBUSTA ---
        # Apply the parser to each cell, then convert the entire column.
        df_input['Dare'] = pd.to_numeric(df_input['Dare'].apply(robust_currency_parser), errors='coerce')
        df_input['Avere'] = pd.to_numeric(df_input['Avere'].apply(robust_currency_parser), errors='coerce')

        df_input[['Dare', 'Avere']] = df_input[['Dare', 'Avere']].fillna(0)
        df_input['Dare'] = (df_input['Dare'] * 100).round().astype(int)
        df_input['Avere'] = (df_input['Avere'] * 100).round().astype(int)
        df_input['indice_orig'] = df_input.index

        # --- 3. Execute reconciliation logic with parameters from the UI ---
        riconciliatore = RiconciliatoreContabile(**config_params)
        stats = riconciliatore.run(df_input, output_file=None, verbose=False)

        # --- 4. Save the output file with a unique name ---
        unique_id = uuid.uuid4()
        sanitized_filename = "".join(c for c in file.filename if c.isalnum() or c in ('.', '_')).rstrip()
        unique_output_filename = f"{unique_id}_{sanitized_filename}"
        output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], unique_output_filename)
        
        riconciliatore._crea_report_excel(output_filepath, df_input)

        # --- 5. Create and save the log file ---
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f"{timestamp}_{sanitized_filename}_summary.log"
        log_filepath = os.path.join(LOG_FOLDER, log_filename)
        
        formatted_stats = stats.copy()
        for key, value in stats.items():
            if '_raw_importo' in key:
                formatted_stats[key] = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " €"
            elif '_raw_perc' in key:
                formatted_stats[key] = f"{value:.2f} %".replace('.', ',')

        # Save to disk only if explicitly requested
        if save_log:
            with open(log_filepath, 'w', encoding='utf-8') as f:
                json.dump(formatted_stats, f, indent=4, ensure_ascii=False)
        
        # --- 6. Prepare names and save in session ---
        base_name, extension = os.path.splitext(sanitized_filename)
        pretty_download_filename = f"{base_name}_result{extension}"
        
        # Save the map {pretty_name: unique_name} in the user session
        session['download_map'] = {pretty_download_filename: unique_output_filename}
        
        # --- 7. Return JSON for the frontend ---
        return jsonify({
            'log_content': json.dumps(formatted_stats, indent=4, ensure_ascii=False),
            'download_url': url_for('download_file', filename=pretty_download_filename),
            'version': '3.0' # Versione incrementata per il nuovo fix
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"A critical error occurred: {str(e)}"}), 500


# --- Nuovo Endpoint per il Download ---
@app.route('/download/<filename>')
def download_file(filename):
    """
    Serves the output file using a map saved in the session
    to find the physical file with a unique name.
    """
    download_map = session.get('download_map', {})
    actual_filename = download_map.get(filename)
    
    if not actual_filename:
        return "File not found or session expired.", 404
        
    return send_from_directory(
        app.config['OUTPUT_FOLDER'], 
        actual_filename, 
        as_attachment=True,
        download_name=filename  # FIX: Specify the filename for the download
    )


# --- Avvio del Server ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
