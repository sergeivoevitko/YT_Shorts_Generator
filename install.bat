@echo off
setlocal
cd /d "%~dp0"
title ClipForge installer
echo.
echo  === ClipForge installer ===
echo.

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY ( where python >nul 2>nul && set "PY=python" )
if not defined PY (
  echo Python was not found. Installing Python 3.12 with winget...
  winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
  echo.
  echo Python was installed. Close this window and run install.bat again.
  pause
  exit /b 1
)

echo [1/4] Creating the Python environment...
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv
if errorlevel 1 goto :fail

echo [2/4] Installing Python packages (a few minutes the first time)...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  echo NVIDIA graphics card found. Installing GPU libraries for faster speech recognition...
  ".venv\Scripts\python.exe" -m pip install -r requirements-gpu.txt
  if errorlevel 1 echo Could not install the GPU libraries. The app will still work on the processor.
)

echo Updating the Microsoft Visual C++ runtime (needed by speech recognition)...
winget install -e --id Microsoft.VCRedist.2015+.x64 --accept-source-agreements --accept-package-agreements >nul 2>nul
winget upgrade -e --id Microsoft.VCRedist.2015+.x64 --accept-source-agreements --accept-package-agreements >nul 2>nul

echo [3/4] Checking FFmpeg (video engine)...
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo FFmpeg not found. Installing with winget...
  winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
) else ( echo FFmpeg is already installed. )

echo [4/4] Checking Deno (needed for YouTube downloads)...
where deno >nul 2>nul
if errorlevel 1 (
  where node >nul 2>nul
  if errorlevel 1 (
    echo Deno not found. Installing with winget...
    winget install -e --id DenoLand.Deno --accept-source-agreements --accept-package-agreements
  ) else ( echo Node.js found, it will be used instead. )
) else ( echo Deno is already installed. )

echo.
echo  Done. Double-click start.bat to launch ClipForge.
echo  If FFmpeg or Deno was just installed, close this window first so Windows picks them up.
echo.
pause
exit /b 0

:fail
echo.
echo  Installation failed. Scroll up to see the error.
pause
exit /b 1
