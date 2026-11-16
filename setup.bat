@echo off
rem SignVision one-time setup for Windows. Double-click it, or run it in "Anaconda Prompt".
setlocal
cd /d "%~dp0"
call tools\find_conda.bat
where conda >nul 2>nul
if errorlevel 1 (
  echo.
  echo [X] Conda was not found. Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
  echo     then open "Anaconda Prompt" from the Start menu and run setup.bat again.
  pause
  exit /b 1
)

echo.
echo === Step 1 of 5: Python environment "signvision" (first time takes 5-15 minutes) ===
call conda env list | findstr /b /c:"signvision " >nul
if errorlevel 1 (
  call conda env create -f environment.yml
) else (
  call conda env update -n signvision -f environment.yml --prune
)
if errorlevel 1 goto :fail
call conda activate signvision
if errorlevel 1 goto :fail

echo.
echo === Step 2 of 5: Web app packages ===
pushd frontend
call npm install --no-audit --no-fund
if errorlevel 1 (popd & goto :fail)

echo.
echo === Step 3 of 5: Building the web app ===
call npm run build
if errorlevel 1 (popd & goto :fail)
popd

echo.
echo === Step 4 of 5: Speech model for Conversation (about 145 MB, one time) ===
pushd backend
python -m scripts.download_speech_model
if errorlevel 1 (popd & goto :fail)

echo.
echo === Step 5 of 5: Sharing tool for internet conversation rooms (about 40 MB) ===
python -m scripts.download_tunnel
if errorlevel 1 echo [!] Could not download it now; the app will try again when you press "Invite Person B".
popd

echo.
echo [OK] Setup finished. Double-click run.bat to start SignVision.
pause
exit /b 0

:fail
echo.
echo [X] Setup stopped because of the error above. Check your internet connection and run setup.bat again.
pause
exit /b 1
