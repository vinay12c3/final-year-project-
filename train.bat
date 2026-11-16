@echo off
rem Trains both models from the command line (same as the buttons on the "Record and train" page).
setlocal
cd /d "%~dp0"
call tools\find_conda.bat
call conda activate signvision || (echo Run setup.bat first & pause & exit /b 1)
cd backend
echo === Letters and numbers model ===
python -m scripts.train_static
echo.
echo === Words model ===
python -m scripts.train_dynamic
echo.
echo Results and figures are in backend\reports
pause
