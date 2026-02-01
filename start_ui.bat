
@echo off
REM MASt3R Enhanced UI - Windows Start Script
REM ==================================================
REM This script will activate the 'mast3r' Conda environment
REM and then launch the Gradio user interface.
REM ==================================================

echo Activating Conda environment 'mast3r' and starting the UI...
echo.

REM Activate the conda environment and run the python script.
CALL conda.bat activate mast3r && python app.py

echo.
echo The program has exited.
echo Press any key to close this window...
pause > nul
