@echo off
setlocal

:: ============================================================================
::
::   ◈ PARALLAX STUDIO v1.2 - LAUNCHER
::   Your Photos. Alive.
::
:: ============================================================================

title ◈ Parallax Studio v1.2

:: Get directory where this script is located
set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

cls
echo.
echo  ╔════════════════════════════════════════════════════════════╗
echo  ║                                                            ║
echo  ║           ◈  PARALLAX STUDIO v1.2  ◈                       ║
echo  ║                                                            ║
echo  ║               Your Photos. Alive.                          ║
echo  ║                                                            ║
echo  ╚════════════════════════════════════════════════════════════╝
echo.

:: ============================================================================
:: Check Conda
:: ============================================================================

call conda --version >nul 2>&1
if %errorLevel% neq 0 (
    echo  ╔════════════════════════════════════════════════════════════╗
    echo  ║  ERROR: Conda is not installed                             ║
    echo  ║                                                            ║
    echo  ║  Please run install.bat first.                             ║
    echo  ╚════════════════════════════════════════════════════════════╝
    echo.
    pause
    exit /b 1
)

:: ============================================================================
:: Activate Environment
:: ============================================================================

echo  Activating environment...
call conda activate parallax_studio

if %errorLevel% neq 0 (
    echo.
    echo  ╔════════════════════════════════════════════════════════════╗
    echo  ║  ERROR: Environment 'parallax_studio' not found            ║
    echo  ║                                                            ║
    echo  ║  Please run install.bat first.                             ║
    echo  ╚════════════════════════════════════════════════════════════╝
    echo.
    pause
    exit /b 1
)

:: ============================================================================
:: Check for App File
:: ============================================================================

if not exist "%APP_DIR%parallax_studio.py" (
    echo.
    echo  ╔════════════════════════════════════════════════════════════╗
    echo  ║  ERROR: parallax_studio.py not found                       ║
    echo  ║                                                            ║
    echo  ║  Make sure this file is in the same folder as run.bat:    ║
    echo  ║  %APP_DIR%
    echo  ╚════════════════════════════════════════════════════════════╝
    echo.
    pause
    exit /b 1
)

:: ============================================================================
:: Check GPU
:: ============================================================================

echo  Checking GPU...
python -c "import torch; gpu=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if gpu else 'None'; print(f'  GPU: {name}' if gpu else '  WARNING: No GPU detected!')"
echo.

:: ============================================================================
:: Launch Application
:: ============================================================================

echo  ────────────────────────────────────────────────────────────
echo.
echo   Starting Parallax Studio...
echo.
echo   The app will open in your browser at:
echo.
echo       http://localhost:8501
echo.
echo   To STOP: Close this window or press Ctrl+C
echo.
echo  ────────────────────────────────────────────────────────────
echo.

:: Run Streamlit with optimized settings
streamlit run parallax_studio.py ^
    --server.maxUploadSize=500 ^
    --server.headless=false ^
    --browser.gatherUsageStats=false ^
    --theme.base=dark

:: If we reach here, the app has stopped
echo.
echo  ────────────────────────────────────────────────────────────
echo   Parallax Studio has stopped.
echo  ────────────────────────────────────────────────────────────
echo.
pause
exit /b 0
