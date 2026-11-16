@echo off
rem Developer mode: backend auto-reloads on Python changes, frontend hot-reloads at http://localhost:5173
setlocal
cd /d "%~dp0"
call tools\find_conda.bat
call conda activate signvision || (echo Run setup.bat first & pause & exit /b 1)
start "SignVision backend" cmd /k "cd /d %~dp0backend && python -m uvicorn app.main:app --reload --port 8000"
start "SignVision frontend" cmd /k "cd /d %~dp0frontend && npm run dev"
start "" cmd /c "timeout /t 6 >nul & start http://localhost:5173"
