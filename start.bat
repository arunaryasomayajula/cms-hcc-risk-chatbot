@echo off
echo ============================================================
echo  CMS HCC Risk Scoring Chatbot
echo ============================================================
echo.

:: Check for ANTHROPIC_API_KEY
if "%ANTHROPIC_API_KEY%"=="" (
    echo [ERROR] ANTHROPIC_API_KEY environment variable is not set.
    echo.
    echo Set it with:
    echo   set ANTHROPIC_API_KEY=sk-ant-...
    echo.
    pause
    exit /b 1
)

:: Install dependencies
echo [1/2] Installing Python dependencies...
pip install -r backend\requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] pip install failed. Make sure Python 3.10+ is installed.
    pause
    exit /b 1
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
