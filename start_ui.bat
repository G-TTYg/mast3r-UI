
@echo off
REM MASt3R Enhanced UI - Windows Start Script
REM ==================================================
REM Instructions:
REM 1. Ensure you have installed all dependencies as per the INSTRUCTIONS.md guide.
REM 2. Place this script in the root directory of the project.
REM 3. Double-click this script to run the UI.
REM ==================================================

echo Starting MASt3R Enhanced UI...
echo.
echo All configurations, including model and device selection, are now available within the UI.
echo.

REM Launch app.py without any command-line arguments
python app.py

echo.
echo The program has exited.
echo Press any key to close this window...
pause > nul
