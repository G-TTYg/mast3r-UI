
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
echo Defaulting to CPU mode for maximum compatibility.
echo If you have a compatible NVIDIA GPU and want to use it,
echo please edit this script and remove the "--device cpu" argument.
echo.

REM Set the default model name
set MODEL_NAME=MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric

REM Launch app.py. Defaulting to CPU for broader compatibility.
python app.py --model_name %MODEL_NAME% --device cpu

echo.
echo The program has exited.
echo Press any key to close this window...
pause > nul
