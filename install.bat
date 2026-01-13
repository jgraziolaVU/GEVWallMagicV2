@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
::
::   ◈ PARALLAX STUDIO v1.2 - INSTALLER
::   Your Photos. Alive.
::
:: ============================================================================

title Parallax Studio v1.2 - Installation

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
echo  This installer will set up everything you need:
echo.
echo    [1] Python environment (Miniconda)
echo    [2] PyTorch with CUDA (GPU acceleration)
echo    [3] Apple SHARP (3D depth extraction)
echo    [4] Qwen-Image-Edit (AI style transfer)
echo    [5] FFmpeg (video encoding)
echo.
echo  ────────────────────────────────────────────────────────────
echo   Requirements:
echo     • NVIDIA GPU with 24GB+ VRAM (RTX 4090/5090)
echo     • 64GB+ system RAM recommended
echo     • 60GB free disk space
echo     • Internet connection
echo  ────────────────────────────────────────────────────────────
echo.
echo  Estimated time: 20-40 minutes (first install)
echo.

pause
cls

:: ============================================================================
:: Configuration
:: ============================================================================

set "APP_DIR=%USERPROFILE%\ParallaxStudio"
set "ENV_NAME=parallax_studio"

:: ============================================================================
:: STEP 1: Check GPU
:: ============================================================================

echo.
echo  [Step 1/8] Checking for NVIDIA GPU...
echo.

nvidia-smi >nul 2>&1
if %errorLevel% neq 0 (
    echo  ╔════════════════════════════════════════════════════════════╗
    echo  ║  WARNING: NVIDIA GPU not detected!                         ║
    echo  ║                                                            ║
    echo  ║  Parallax Studio requires an RTX 4090/5090 or similar.     ║
    echo  ║  Please install NVIDIA drivers first.                      ║
    echo  ╚════════════════════════════════════════════════════════════╝
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader') do (
    echo  ✓ Found GPU: %%i
)
echo.

:: ============================================================================
:: STEP 2: Check/Install Git
:: ============================================================================

echo.
echo  [Step 2/8] Checking for Git...
echo.

where git >nul 2>&1
if %errorLevel% neq 0 (
    echo  Git not found. Downloading installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/Git-2.43.0-64-bit.exe' -OutFile '%TEMP%\git_installer.exe'"
    
    echo  Please install Git with default options...
    start /wait "" "%TEMP%\git_installer.exe"
    del "%TEMP%\git_installer.exe"
    
    echo.
    echo  ════════════════════════════════════════════════════════════
    echo   Git installed. Please RESTART this installer to continue.
    echo  ════════════════════════════════════════════════════════════
    echo.
    pause
    exit /b 0
)

echo  ✓ Git is installed
echo.

:: ============================================================================
:: STEP 3: Check/Install Miniconda
:: ============================================================================

echo.
echo  [Step 3/8] Checking for Conda...
echo.

where conda >nul 2>&1
if %errorLevel% neq 0 (
    echo  Conda not found. Downloading Miniconda...
    powershell -Command "Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe' -OutFile '%TEMP%\miniconda.exe'"
    
    echo.
    echo  Installing Miniconda...
    echo  IMPORTANT: Select "Add to PATH" when prompted!
    echo.
    start /wait "" "%TEMP%\miniconda.exe"
    del "%TEMP%\miniconda.exe"
    
    echo.
    echo  ════════════════════════════════════════════════════════════
    echo   Miniconda installed. Please RESTART this installer.
    echo  ════════════════════════════════════════════════════════════
    echo.
    pause
    exit /b 0
)

echo  ✓ Conda is installed
echo.

:: ============================================================================
:: STEP 4: Create/Update Environment
:: ============================================================================

echo.
echo  [Step 4/8] Setting up Python environment...
echo.

call conda env list | findstr /C:"%ENV_NAME%" >nul
if %errorLevel% equ 0 (
    echo  Environment '%ENV_NAME%' exists. Activating...
) else (
    echo  Creating new environment...
    call conda create -n %ENV_NAME% python=3.11 -y
)

call conda activate %ENV_NAME%

if %errorLevel% neq 0 (
    echo  ERROR: Failed to activate environment
    pause
    exit /b 1
)

echo  ✓ Environment ready (Python 3.11)
echo.

:: ============================================================================
:: STEP 5: Install PyTorch
:: ============================================================================

echo.
echo  [Step 5/8] Installing PyTorch with CUDA...
echo  (This may take 5-10 minutes)
echo.

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}'); print(f'  ✓ CUDA available: {torch.cuda.is_available()}')"
echo.

:: ============================================================================
:: STEP 6: Install SHARP
:: ============================================================================

echo.
echo  [Step 6/8] Installing Apple SHARP...
echo.

if not exist "%APP_DIR%" mkdir "%APP_DIR%"
cd /d "%APP_DIR%"

if not exist "%APP_DIR%\ml-sharp" (
    echo  Cloning SHARP repository...
    git clone https://github.com/apple/ml-sharp.git
) else (
    echo  Updating SHARP...
    cd ml-sharp
    git pull
    cd ..
)

cd ml-sharp
pip install -r requirements.txt
pip install -e .
cd "%APP_DIR%"

echo  ✓ SHARP installed
echo.

:: ============================================================================
:: STEP 7: Install Qwen & Dependencies
:: ============================================================================

echo.
echo  [Step 7/8] Installing Qwen-Image-Edit and dependencies...
echo  (This may take 10-15 minutes)
echo.

pip install git+https://github.com/huggingface/diffusers
pip install streamlit>=1.31.0
pip install huggingface-hub>=0.20.0
pip install Pillow>=10.0.0
pip install plyfile
pip install gsplat
pip install tqdm

echo.
echo  Downloading SHARP model checkpoint (~500MB)...
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='apple/Sharp', filename='sharp_2572gikvuh.pt', local_dir='%USERPROFILE%\.cache\sharp')"

echo.
echo  ✓ All Python packages installed
echo.

:: ============================================================================
:: STEP 8: Install FFmpeg
:: ============================================================================

echo.
echo  [Step 8/8] Setting up FFmpeg...
echo.

where ffmpeg >nul 2>&1
if %errorLevel% neq 0 (
    echo  Installing FFmpeg via conda...
    call conda install -c conda-forge ffmpeg -y
) else (
    echo  ✓ FFmpeg already installed
)

echo.

:: ============================================================================
:: Setup Application Files
:: ============================================================================

echo.
echo  Setting up application...
echo.

:: Copy app file if present
if exist "%~dp0parallax_studio.py" (
    copy "%~dp0parallax_studio.py" "%APP_DIR%\parallax_studio.py" >nul
    echo  ✓ Application file installed
)

:: Create run.bat
(
echo @echo off
echo title ◈ Parallax Studio v1.2
echo call conda activate %ENV_NAME%
echo cd /d "%APP_DIR%"
echo cls
echo echo.
echo echo  ╔════════════════════════════════════════════════════════════╗
echo echo  ║                                                            ║
echo echo  ║           ◈  PARALLAX STUDIO v1.2  ◈                       ║
echo echo  ║                                                            ║
echo echo  ║               Your Photos. Alive.                          ║
echo echo  ║                                                            ║
echo echo  ╚════════════════════════════════════════════════════════════╝
echo echo.
echo echo  Starting application...
echo echo.
echo echo  The app will open in your browser at:
echo echo  http://localhost:8501
echo echo.
echo echo  To stop: Close this window or press Ctrl+C
echo echo.
echo streamlit run parallax_studio.py --server.maxUploadSize=500 --browser.gatherUsageStats=false
echo pause
) > "%APP_DIR%\run.bat"

:: Create desktop shortcut
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%USERPROFILE%\Desktop\Parallax Studio.lnk'); $s.TargetPath = '%APP_DIR%\run.bat'; $s.WorkingDirectory = '%APP_DIR%'; $s.Description = 'Your Photos. Alive.'; $s.Save()"

echo  ✓ Desktop shortcut created
echo.

:: ============================================================================
:: Complete
:: ============================================================================

cls
echo.
echo  ╔════════════════════════════════════════════════════════════╗
echo  ║                                                            ║
echo  ║        ◈  INSTALLATION COMPLETE!  ◈                        ║
echo  ║                                                            ║
echo  ╚════════════════════════════════════════════════════════════╝
echo.
echo  Parallax Studio v1.2 is installed at:
echo  %APP_DIR%
echo.
echo  ────────────────────────────────────────────────────────────
echo.
echo  TO RUN THE APP:
echo.
echo    • Double-click "Parallax Studio" on your Desktop
echo.
echo    • Or run: %APP_DIR%\run.bat
echo.
echo  ────────────────────────────────────────────────────────────
echo.
echo  FIRST RUN NOTE:
echo    The Qwen model (~40GB) downloads on first use of
echo    the enhancement feature. This is normal!
echo.
echo  ────────────────────────────────────────────────────────────
echo.

set /p LAUNCH="  Launch Parallax Studio now? (y/n): "
if /i "%LAUNCH%"=="y" (
    start "" "%APP_DIR%\run.bat"
)

echo.
echo  Enjoy making your photos come alive!
echo.
pause
exit /b 0
