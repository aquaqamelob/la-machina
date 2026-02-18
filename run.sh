#!/bin/bash

# 1. Get the folder where this script is saved
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# 2. Check if the 'venv' folder exists
if [ ! -d "venv" ]; then
    echo "------------------------------------------------"
    echo "First run detected (or moved). Setting up..."
    echo "------------------------------------------------"
    
    # Create the virtual environment locally in this folder
    python3 -m venv venv
    
    # Activate it
    source venv/bin/activate
    
    # Upgrade pip just in case
    pip install --upgrade pip
    
    # Install the requirements inside this folder only
    pip install -r requirements.txt
    
    echo "------------------------------------------------"
    echo "Installation complete."
    echo "------------------------------------------------"
else
    # Just activate if it already exists
    source venv/bin/activate
fi

# 3. Run the Python script
# Passes any arguments (like video names) to the python script
echo "Starting Analysis..."
python main.py "$@"

# 4. Deactivate when done
deactivate
