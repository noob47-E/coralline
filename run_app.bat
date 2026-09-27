@echo off
cd /d "%~dp0"
python -c "import numpy, scipy, PIL" 2>nul || (
    echo Installing required packages...
    python -m pip install -r requirements.txt
)
start "" pythonw app.py
