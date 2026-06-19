@echo off
REM Start the Agentic AI Explainer server.
REM Double-click this file, or run `run.bat` from a terminal.
REM cd to this script's own folder so it works regardless of where it's launched from.
cd /d "%~dp0"

echo Starting Agentic AI Explainer on http://localhost:8000  (Ctrl+C to stop)
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

REM Keep the window open after the server stops so any error is readable.
pause
