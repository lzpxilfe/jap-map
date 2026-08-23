@echo off
setlocal
cd /d "%~dp0"
git pull --ff-only
if errorlevel 1 (
  echo.
  echo Update failed. Do not start reviewing until the Git conflict or local change is resolved.
)
pause
