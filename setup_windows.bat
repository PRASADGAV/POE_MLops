@echo off
setlocal
cd /d "%~dp0"
echo [1/3] Creating virtual environment...
if not exist .venv py -3.11 -m venv .venv
if not exist .venv\Scripts\python.exe (
  echo Could not create .venv with Python 3.11.
  echo Install Python 3.11 or edit this file to use your installed Python version.
  pause
  exit /b 1
)
echo [2/3] Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto error
echo [3/3] Installing project requirements...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto error
echo.
echo Setup completed. Now double-click run_windows.bat
pause
exit /b 0
:error
echo.
echo Setup failed. Check the internet connection and package installation messages.
pause
exit /b 1
