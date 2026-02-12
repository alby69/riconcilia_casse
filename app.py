import io
import os
import json
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_from_directory, session, url_for
import uuid
from core import ReconciliationEngine
from optimizer import find_best_parameters

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

# --- Helper Functions ---
def load_config():
    """Loads the configuration from config.json."""
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def robust_currency_parser(value):
    """Robustly converts a string or number into a standard numeric format."""
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return None
    cleaned_str = str(value).strip().replace('€', '').replace(' ', '')
    if '.' in cleaned_str and ',' in cleaned_str:
        return cleaned_str.replace('.', '').replace(',', '.')
    if ',' in cleaned_str:
        return cleaned_str.replace(',', '.')
    return cleaned_str

def prepare_dataframe(file_stream):
    """Reads an Excel file from a stream and prepares the DataFrame for the engine."""
    df = pd.read_excel(io.BytesIO(file_stream.read()))
    df.columns = df.columns.str.strip()

    config = load_config()
    mapping_conf = config.get('common', {}).get('column_mapping', {})
    
    # Apply renaming based on config { "Nome Colonna File": "Nome Interno" }
    df.rename(columns=mapping_conf, inplace=True)

    required_columns = ['Date', 'Debit', 'Credit']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti dopo il mapping: {', '.join(missing)}. Colonne trovate: {df.columns.tolist()}")

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df.dropna(subset=['Date'], inplace=True)
    
    # Parse currency and convert to cents in the final columns
    df['Debit'] = pd.to_numeric(df['Debit'].apply(robust_currency_parser), errors='coerce')
    df['Credit'] = pd.to_numeric(df['Credit'].apply(robust_currency_parser), errors='coerce')
    df[['Debit', 'Credit']] = df[['Debit', 'Credit']].fillna(0)
    
    # The engine expects integer cents
    df['Debit'] = (df['Debit'] * 100).round().astype(int)
    df['Credit'] = (df['Credit'] * 100).round().astype(int)
    
    df['orig_index'] = df.index
    return df

# --- Routes ---
@app.route('/')
def index():
    """Displays the main page, passing the full configuration to the template."""
    config = load_config()
    return render_template('index.html', config=config)

@app.route('/optimize', methods=['POST'])
def optimize_parameters():
    """Analyzes the uploaded file and returns optimal parameters."""
    if 'file_input' not in request.files:
        return jsonify({'error': 'Nessun file selezionato.'}), 400
    file = request.files['file_input']
    if file.filename == '':
        return jsonify({'error': 'Nessun file selezionato.'}), 400

    try:
        file.stream.seek(0)
        df = prepare_dataframe(file.stream)
        config = load_config()
        # Usa 'common' come fallback se 'reconciliation_defaults' non esiste
        base_config = config.get('reconciliation_defaults', config.get('common', {}))
        optimizer_config = config.get('optimizer', {})
        
        # Run optimization
        # Pass both base parameters and optimizer-specific configurations
        # sequential=False abilita il multiprocessing (configurato in modo sicuro in optimizer.py)
        best_params = find_best_parameters(df, base_config, optimizer_config, sequential=False)
        
        return jsonify(best_params)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Errore critico durante l'ottimizzazione: {str(e)}"}), 500

@app.route('/processa', methods=['POST'])
def processa_file():
    """Handles the main file processing with user-provided or optimized parameters."""
    if 'file_input' not in request.files:
        return jsonify({'error': 'Nessun file selezionato.'}), 400
    file = request.files['file_input']
    if file.filename == '':
        return jsonify({'error': 'Nessun file selezionato.'}), 400

    try:
        form_data = request.form.to_dict()
        engine_params = {
            'tolerance': float(form_data.get('tolerance', 0.01)),
            'days_window': int(form_data.get('days_window', 7)),
            'max_combinations': int(form_data.get('max_combinations', 10)),
            'residual_threshold': float(form_data.get('residual_threshold', 100.0)),
            'residual_days_window': int(form_data.get('residual_days_window', 30)),
            'search_direction': form_data.get('search_direction', 'past_only'),
            'algorithm': form_data.get('algorithm', 'auto'),
            'ignore_tolerance': form_data.get('ignore_tolerance') == 'true'
        }
        
        file.stream.seek(0)
        df_input = prepare_dataframe(file.stream)

        # The engine receives the dataframe with amounts already in cents
        engine = ReconciliationEngine(**engine_params)
        stats = engine.run(df_input, verbose=False)

        unique_id = uuid.uuid4()
        sanitized_filename = "".join(c for c in file.filename if c.isalnum() or c in ('.', '_')).rstrip()
        unique_output_filename = f"{unique_id}_{sanitized_filename}"
        output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], unique_output_filename)
        
        engine.create_excel_report(output_filepath, df_input)

        base_name, _ = os.path.splitext(sanitized_filename)
        pretty_download_filename = f"{base_name}_result.xlsx"
        session['download_map'] = {pretty_download_filename: unique_output_filename}
        
        return jsonify({
            'log_content': json.dumps(stats, indent=4, ensure_ascii=False),
            'download_url': url_for('download_file', filename=pretty_download_filename)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Errore critico durante l'elaborazione: {str(e)}"}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Handles secure downloading of the generated report file."""
    download_map = session.get('download_map', {})
    actual_filename = download_map.get(filename)
    
    if not actual_filename:
        return "File non trovato o sessione scaduta.", 404
        
    return send_from_directory(
        app.config['OUTPUT_FOLDER'], 
        actual_filename, 
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
