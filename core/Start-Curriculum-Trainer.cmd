@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Core.ps1" -Curriculum
if errorlevel 1 pause
