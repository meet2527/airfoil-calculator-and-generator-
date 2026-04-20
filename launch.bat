@echo off
echo Starting RC Airfoil Configuration System...
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python run_app.py
pause
