@echo off
setlocal

rem --- Demo senza browser: finestra OpenCV con webcam + YOLO ---
rem Premi Q nella finestra per chiudere.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Ambiente non ancora configurato.
    echo         Esegui prima start.bat per installare le dipendenze.
    pause
    exit /b 1
)

call ".venv\Scripts\python.exe" examples\detect_cli.py %*
pause
endlocal
