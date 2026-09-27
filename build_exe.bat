@echo off
rem Builds dist\CoralPatternGenerator.exe - a single file that runs without Python installed.
cd /d "%~dp0"
python -m pip install -r requirements.txt pyinstaller
python -m PyInstaller --noconfirm --onefile --windowed --name CoralPatternGenerator ^
    --add-data "presets;presets" app.py
echo.
echo Done. The program is in the "dist" folder.
pause
