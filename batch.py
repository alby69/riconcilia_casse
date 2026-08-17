"""Batch processing script for accounting reconciliation.

This script automates the reconciliation of multiple files. It is designed
to be run from the command line to process an entire directory of accounting
files based on a central configuration.

The script's workflow is:
1.  **Load Configuration**: Reads settings from the main `config.json` file,
    including the input/output folder paths and file patterns (e.g., `*.xlsx`).
2.  **File Discovery**: Scans the input directory for all files matching the
    configured patterns.
3.  **Batch Iteration**: Loops through each discovered file, showing overall
    progress with a `tqdm` progress bar.
4.  **Processing**: For each file, it initializes the `ReconciliationEngine`
    based on the settings in `config.json` and runs the full reconciliation
    process.
5.  **Output Generation**: Saves the detailed Excel report for each processed
    file to the output directory, with a name derived from the original
    input file (e.g., `result_my-data.xlsx`).
"""

from pathlib import Path
import json
import sys
from tqdm import tqdm
from core import ReconciliationEngine

def load_config():
    """Loads configuration from config.json."""
    config_file = Path('config.json')
    if not config_file.exists():
        print(f"⚠️  '{config_file}' not found.")
        return {}
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            print(f"📄 Loading configuration from '{config_file}'...")
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON format in '{config_file}': {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main function"""
    config = load_config()
    common_params = config.get('common', {})
    input_folder = Path(common_params.get('input_folder', 'input'))
    output_folder = Path(common_params.get('output_folder', 'output'))
    output_folder.mkdir(exist_ok=True)

    patterns = common_params.get('file_patterns', ['*.xlsx', '*.csv'])
    files_to_process = []
    for p in patterns:
        files_to_process.extend(input_folder.glob(p))
    
    if not files_to_process:
        print(f"⚠️ No files found in '{input_folder}' with patterns: {patterns}")
        return

    print(f"Found {len(files_to_process)} files to process.")

    algorithm_name = common_params.get('algorithm', 'progressive_balance')
    engine_params = common_params.copy()
    if algorithm_name in config:
        engine_params.update(config[algorithm_name])

    for file_path in tqdm(files_to_process, desc="Processing files"):
        print(f"\n{'='*20} Processing: {file_path.name} {'='*20}")
        try:
            engine = ReconciliationEngine(
                tolerance=engine_params.get('tolerance', 50.0),
                days_window=engine_params.get('days_window', 5),
                max_combinations=engine_params.get('max_combinations', 10),
                residual_threshold=engine_params.get('residual_threshold', 50.0),
                residual_days_window=engine_params.get('residual_days_window', 5),
                sorting_strategy=engine_params.get('sorting_strategy', 'date'),
                search_direction=engine_params.get('search_direction', 'past_only'),
                column_mapping=engine_params.get('column_mapping'),
                algorithm=algorithm_name,
                store_id_column=engine_params.get('store_id_column'),
                valuta_date_column=engine_params.get('valuta_date_column')
            )

            output_file = output_folder / f"result_{file_path.stem}.xlsx"
            
            engine.run(str(file_path), str(output_file))

        except (IOError, ValueError, FileNotFoundError) as e:
            print(f"❌ ERROR while processing {file_path.name}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
