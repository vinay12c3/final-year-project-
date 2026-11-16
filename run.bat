@echo off
rem Starts SignVision at http://localhost:8000  (close this window to stop it)
setlocal
cd /d "%~dp0"
call tools\find_conda.bat
call conda activate signvision
if errorlevel 1 (
  echo [X] The "signvision" environment is missing. Run setup.bat first.
  pause
  exit /b 1
)
if not exist frontend\dist\index.html (
  echo Building the web app...
  pushd frontend & call npm run build & popd
)
echo.
echo  SignVision is starting. Your browser will open http://localhost:8000
echo  Keep this window open while you use the app. Close it to stop.
echo.
start "" cmd /c "timeout /t 5 >nul & start http://localhost:8000"
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
