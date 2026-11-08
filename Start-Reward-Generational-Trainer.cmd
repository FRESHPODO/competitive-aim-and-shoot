@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0core\Start-Core.ps1" -RewardGenerational
if errorlevel 1 pause
