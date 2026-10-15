@echo off
REM ====================================================================
REM S.W.O.R.D. Phase 4 - Easy Video Action Recognition Runner
REM Drag and drop a video file onto this .bat file or pass it as argument
REM ====================================================================

set PYTHON_EXE=C:\Users\nidhi\AppData\Local\Programs\Python\Python311\python.exe

if "%~1"=="" (
    echo.
    echo ===============================================================
    echo   S.W.O.R.D. Action Recognition Runner
    echo ===============================================================
    echo.
    echo Please provide a video file!
    echo Usage:
    echo   run_sword_action.bat "C:\path\to\your_video.mp4"
    echo.
    echo Or DRAG AND DROP your video file directly onto this icon.
    echo.
    set /p VIDEO_PATH="Enter the full path to your video file: "
) else (
    set VIDEO_PATH=%~1
)

cd /d "%~dp0"
"%PYTHON_EXE%" run_pipeline.py --video "%VIDEO_PATH%"
pause
