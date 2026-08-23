@echo off
setlocal
set "REVIEW_PROJECT=%~dp0data\derived\annotation_package\annotation_project.qgz"
if not exist "%REVIEW_PROJECT%" (
  echo Review project not found: "%REVIEW_PROJECT%"
  pause
  exit /b 1
)
start "" "%REVIEW_PROJECT%"
