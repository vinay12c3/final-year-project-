@echo off
rem Makes "conda" available even when the script is double-clicked from Explorer.
where conda >nul 2>nul
if not errorlevel 1 goto :eof
for %%D in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3" "C:\miniconda3" "C:\anaconda3") do (
  if exist "%%~D\condabin\conda.bat" set "PATH=%%~D\condabin;%PATH%"
)
