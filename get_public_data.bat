@echo off
rem Downloads the public ISL alphabet dataset (~290 MB) and converts it to hand landmarks.
rem Only needed if you want to RE-TRAIN the letters model together with the public data.
setlocal
cd /d "%~dp0"
call tools\find_conda.bat
call conda activate signvision || (echo Run setup.bat first & pause & exit /b 1)
cd backend
python -m scripts.download_isl_images || goto :fail
python -m scripts.prepare_images --src data\raw\isl_images --name isl_public --per-class 0 || goto :fail
echo [OK] Public data ready. Train from the "Record and train" page or run train.bat
pause
exit /b 0
:fail
echo [X] Stopped because of the error above.
pause
exit /b 1
