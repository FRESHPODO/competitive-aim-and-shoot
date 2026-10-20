@echo off
cd /d "%~dp0"
set "TCL_LIBRARY=%~dp0.venv\tcl\tcl8.6"
set "TK_LIBRARY=%~dp0.venv\tcl\tk8.6"
"%~dp0.venv\Scripts\python.exe" -m src.scripts.gui_simulator --model-name ijhcs_default --ckpt latest
if errorlevel 1 pause
