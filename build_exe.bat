@echo off
rem Builds dist\CoralPatternGenerator.exe - a single file that runs without Python installed.
cd /d "%~dp0"
python -m pip install -r requirements.txt pyinstaller
python -m PyInstaller --noconfirm --onefile --windowed --name CoralPatternGenerator ^
    --add-data "presets;presets" app.py
rem GPL: the license must travel with the program
copy /Y LICENSE dist\LICENSE.txt >nul
echo.
echo Done. The program is in the "dist" folder, together with LICENSE.txt.
echo Source code: https://github.com/noob47-E/coralline
pause
