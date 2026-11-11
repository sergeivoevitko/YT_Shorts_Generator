@echo off
cd /d "%~dp0"
title ClipForge updater
echo Updating ClipForge and its packages...
where git >nul 2>nul && git pull
".venv\Scripts\python.exe" -m pip install --upgrade -r requirements.txt
where nvidia-smi >nul 2>nul && ".venv\Scripts\python.exe" -m pip install --upgrade -r requirements-gpu.txt
echo.
echo Done. YouTube changes often, so run this whenever downloads stop working.
pause
