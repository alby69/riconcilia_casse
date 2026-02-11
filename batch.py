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
from core import ReconciliationEngine  # Assuming core.py is refactored

def load_config():
    """Loads configuration from config.json or uses default values."""
    config_file = Path('config.json')
    default_config = {
        "tolerance": 0.01,
        "days_window": 10,
        "input_folder": "input",
        "output_folder": "output",
        "file_patterns": ["*.xlsx", "*.csv"],
        "column_mapping": {
            "Date": "date",
            "Debit": "debit",
            "Credit": "credit"
        },
        "algorithm": {
            "name": "subset_sum",
            "max_combinations": 6
        },
        "residuals": {
            "enabled": True,
            "amount_threshold": 100,
            "days_window": 90
        }
    }
    
    if not config_file.exists():
        print(f"⚠️  '{config_file}' not found. Using default configuration.")
        return default_config
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            print(f"📄 Loading configuration from '{config_file}'...")
            user_config = json.load(f)
            default_config.update(user_config)
            print("✓ Configuration loaded successfully.")
            return default_config
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON format in '{config_file}': {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main function"""
    config = load_config()
    input_folder = Path(config['input_folder'])
    output_folder = Path(config['output_folder'])
    output_folder.mkdir(exist_ok=True)

    patterns = config.get('file_patterns', ['*.xlsx', '*.csv'])
    files_to_process = []
    for p in patterns:
        files_to_process.extend(input_folder.glob(p))
    
    if not files_to_process:
        print(f"⚠️ No files found in '{input_folder}' with patterns: {patterns}")
        return

    print(f"Found {len(files_to_process)} files to process.")

    for file_path in tqdm(files_to_process, desc="Processing files"):
        print(f"\n{'='*20} Processing: {file_path.name} {'='*20}")
        try:
            engine = ReconciliationEngine(
                tolerance=config['tolerance'],
                days_window=config['days_window'],
                max_combinations=config['algorithm']['max_combinations'],
                residual_threshold=config['residuals']['amount_threshold'],
                residual_days_window=config['residuals']['days_window'],
                column_mapping=config.get('column_mapping')
            )

            output_file = output_folder / f"result_{file_path.stem}.xlsx"
            
            engine.run(str(file_path), str(output_file))

        except (IOError, ValueError, FileNotFoundError) as e:
            print(f"❌ ERROR while processing {file_path.name}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
