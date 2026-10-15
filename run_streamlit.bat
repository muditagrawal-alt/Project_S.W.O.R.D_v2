@echo off
REM ====================================================================
REM S.W.O.R.D. Phase 4 - Launch Streamlit Action Recognition App
REM ====================================================================

set PYTHON_EXE=C:\Users\nidhi\AppData\Local\Programs\Python\Python311\python.exe

cd /d "%~dp0"
echo ===============================================================
echo   Launching S.W.O.R.D. Phase 4 Streamlit Surveillance App...
echo ===============================================================
echo.

"%PYTHON_EXE%" -m streamlit run app_streamlit.py
pause
