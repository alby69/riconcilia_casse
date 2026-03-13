#!/bin/bash

# Script to automatically run unit tests
# Performs test discovery in the 'tests' folder

echo "=========================================="
echo "   Running Reconciliation Unit Tests"
echo "=========================================="

# Runs tests by discovering files starting with test_ in the tests folder
# -s tests: start directory
# -t .: top level directory (per permettere gli import corretti dalla root)
python3 -m unittest discover -s tests -t . -p "test_*.py"