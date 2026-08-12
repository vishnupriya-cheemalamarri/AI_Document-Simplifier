@echo off
REM ============================================================
REM AI Document Simplifier - start backend + frontend
REM Prodapt Hackathon - Group 17
REM
REM Starts the FastAPI backend, waits 30 seconds for it to finish
REM loading models, then starts the Streamlit frontend. Each runs
REM in its own console window so you can watch its live output.
REM
REM Run from anywhere - this script cd's to its own folder first.
REM ============================================================

cd /d "%~dp0"

echo ============================================================
echo  AI Document Simplifier - startup
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo         Set it up first:
    echo           python -m venv .venv
    echo           .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if not exist ".env" (
    echo [WARNING] .env not found - copy .env.example to .env and set OPENAI_API_KEY
    echo           before asking questions ^(uploads will still fail without it^).
    echo.
)

echo [1/3] Starting backend  ^(FastAPI / uvicorn^) on http://localhost:8001 ...
start "AI Doc Simplifier - Backend" cmd /k ""%~dp0.venv\Scripts\python.exe" -m uvicorn backend.main:app --reload --port 8001"

echo       Backend is loading in its own window ^(first boot also loads the
echo       embedding model, which takes a bit^). Waiting 30 seconds before
echo       starting the frontend...
timeout /t 30 /nobreak

echo.
echo [2/3] Starting frontend ^(Streamlit^) on http://localhost:8501 ...
REM --server.headless true is required here: without it, Streamlit's first-ever
REM run in a real interactive console (which this spawned window is) blocks on
REM a "send usage statistics?" prompt that never gets answered, and the
REM frontend silently never comes up.
start "AI Doc Simplifier - Frontend" cmd /k ""%~dp0.venv\Scripts\python.exe" -m streamlit run frontend\app.py --server.headless true"

echo.
echo [3/3] Both servers are launching in their own windows:
echo         - Backend  window: "AI Doc Simplifier - Backend"   -^> http://localhost:8001
echo         - Frontend window: "AI Doc Simplifier - Frontend"  -^> http://localhost:8501
echo.
echo       Watch those windows for live logs / errors / progress.
echo       Close a window (or Ctrl+C inside it) to stop that server.
echo ============================================================
