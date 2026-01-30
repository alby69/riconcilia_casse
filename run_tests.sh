#!/bin/bash

# Script per eseguire automaticamente i test unitari
# Esegue la discovery dei test nella cartella 'tests'

echo "=========================================="
echo "   Esecuzione Test Unitari Riconciliazione"
echo "=========================================="

# Esegue i test scoprendo i file che iniziano con test_ nella cartella tests
# -s tests: start directory
# -t .: top level directory (per permettere gli import corretti dalla root)
python3 -m unittest discover -s tests -t . -p "test_*.py"