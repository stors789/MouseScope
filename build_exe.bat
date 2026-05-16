@echo off
setlocal

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

set "PYTHON_EXE=E:\miniconda\envs\mouse\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    echo.
    echo Please update build_exe.bat to point to the correct conda env.
    pause
    exit /b 1
)

echo [INFO] Building MouseScope with:
echo        %PYTHON_EXE%
echo.

echo [INFO] Installing/updating runtime dependencies...
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -m PyInstaller --clean --noconfirm MouseScope.spec
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

echo.
echo [OK] Build finished.
echo [OK] Run this file instead of anything under build\:
echo      %PROJECT_DIR%dist\MouseScope\MouseScope.exe
echo.
echo [OK] Distribution folder:
echo      %PROJECT_DIR%dist\MouseScope
echo.

start "" explorer "%PROJECT_DIR%dist\MouseScope"
pause
