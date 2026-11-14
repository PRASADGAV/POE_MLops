@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Virtual environment not found. Run setup_windows.bat first.
  pause
  exit /b 1
)
echo Running Pytest...
.venv\Scripts\python.exe -m pytest -q
if errorlevel 1 (
  echo Tests failed. Fix the errors before running the pipeline.
  pause
  exit /b 1
)
echo.
echo Running the full MLOps pipeline...
.venv\Scripts\python.exe main.py
if errorlevel 1 goto error
echo.
echo Completed. Open outputs\evidently_carshare_data_drift.html in a browser.
echo Run start_mlflow_ui.bat in a second terminal to inspect MLflow.
pause
exit /b 0
:error
echo Pipeline failed. Review the error message above.
pause
exit /b 1
