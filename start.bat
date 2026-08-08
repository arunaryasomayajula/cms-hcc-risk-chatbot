@echo off
setlocal
echo ============================================================
echo  CMS HCC Risk Scoring Chatbot
echo ============================================================
echo.

:: Anaconda ships two OpenMP runtimes; without this torch aborts on import with
:: "OMP: Error #15 ... libiomp5md.dll already initialized".
set KMP_DUPLICATE_LIB_OK=TRUE

:: An API key is only needed for the 'claude' provider. A fully local setup
:: (AirLLM / Ollama / vLLM) runs without one, so warn rather than refuse.
if "%ANTHROPIC_API_KEY%"=="" (
    echo [WARN] ANTHROPIC_API_KEY is not set - the 'claude' provider will be unavailable.
    echo        Set it with:  set ANTHROPIC_API_KEY=sk-ant-...
    echo        Or pick a local provider ^(Ollama / AirLLM^) in the sidebar dropdowns.
    echo.
)

:: Install dependencies
echo [1/2] Installing Python dependencies...
pip install -r backend\requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] pip install failed. Make sure Python 3.10+ is installed.
    pause
    exit /b 1
)

:: Optional: launch the AirLLM sidecar alongside the app  (start.bat --airllm)
if /i "%1"=="--airllm" (
    echo.
    echo [+] Installing AirLLM dependencies...
    pip install -r backend\requirements-local.txt --quiet
    echo [+] Starting AirLLM sidecar on port 8002 in a new window...
    :: No space before ^&^& or the env var value picks up a trailing space.
    start "AirLLM sidecar" cmd /k "cd backend&& set KMP_DUPLICATE_LIB_OK=TRUE&& python airllm_server.py"
    echo     First load downloads and re-splits the model - watch that window,
    echo     or poll http://localhost:8002/health for progress.
)

echo.
echo [2/2] Starting server at http://localhost:8000
echo.
echo NOTE: On first run, the ICD-10 index will be built (~2-5 minutes).
echo       The status badge in the UI will turn green when ready.
echo.
echo Press Ctrl+C to stop.
echo.

cd backend
python app.py
