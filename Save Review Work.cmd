@echo off
setlocal
cd /d "%~dp0"
git add -- data\derived\annotation_package\contour_annotations.gpkg
git diff --cached --quiet
if not errorlevel 1 (
  echo No review-label changes to save.
  pause
  exit /b 0
)
git commit -m "data: update contour review labels"
if errorlevel 1 (
  echo Commit failed. Review the message above before retrying.
  pause
  exit /b 1
)
git push origin HEAD
if errorlevel 1 (
  echo Push failed. Your commit is still safe on this computer.
  pause
  exit /b 1
)
echo Review labels were committed and pushed.
pause
