@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Core.ps1" -Matches
if errorlevel 1 pause
