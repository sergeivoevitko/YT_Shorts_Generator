@echo off
cd /d "%~dp0"
title ClipForge
if not exist ".venv\Scripts\python.exe" (
  echo ClipForge is not installed yet. Run install.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m app.main
pause
