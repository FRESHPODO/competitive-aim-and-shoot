@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Unity-Simulator.ps1"
if errorlevel 1 pause
