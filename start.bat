@echo off
rem Real-Time AI Object Detection: one-click launcher
rem First run creates a virtual env and installs dependencies (~500 MB).
rem Next runs just launch the dashboard.

cd /d "%~dp0"

echo.
echo === Real-Time AI Object Detection ===
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python non trovato nel PATH.
    echo         Installa Python 3.10+ da https://www.python.org/downloads/
    echo         e riprova ^(spunta "Add python.exe to PATH" durante l'installazione^).
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creo l'ambiente virtuale .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Creazione venv fallita.
        pause
        exit /b 1
    )

    echo [2/3] Installo le dipendenze ^(puo' richiedere qualche minuto^)...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Installazione dipendenze fallita.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Ambiente virtuale gia' presente.
    echo [2/3] Dipendenze gia' installate.
)

echo [3/3] Avvio del server...
echo.
echo   Apri il browser su: http://127.0.0.1:5000
echo   Premi CTRL+C in questa finestra per fermare il server.
echo.

start "" /b cmd /c "timeout /t 5 /nobreak >nul && start http://127.0.0.1:5000"

".venv\Scripts\python.exe" run.py %*

echo.
echo Server fermato.
pause
