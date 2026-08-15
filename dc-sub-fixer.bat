@echo off
setlocal

rem Launches the dc-sub-fixer tuning window using the project's own venv, so it
rem works from a desktop shortcut without activating anything first.
rem
rem You can also drop an RGB clip and its depth map straight onto this file and
rem it will open them; any dc-sub-fixer option works here too.

set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo.
    echo   Could not find the virtual environment at:
    echo     %PY%
    echo.
    echo   Create it and install the dependencies first - see requirements.txt
    echo   for the exact commands:
    echo.
    echo     python -m venv .venv
    echo     .venv\Scripts\python -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

cd /d "%ROOT%"

rem Run with a console attached rather than pythonw. A GUI failure would
rem otherwise be silent, and a silent exit is exactly what made an earlier
rem crash impossible to diagnose. -X faulthandler makes a hard crash print a
rem stack rather than just closing the window.
"%PY%" -X faulthandler -m dcsubfixer --gui %*
set "CODE=%ERRORLEVEL%"

if not "%CODE%"=="0" (
    echo.
    echo   dc-sub-fixer exited with code %CODE%.
    echo   The error above is the useful part - please keep it.
    echo.
    pause
)

exit /b %CODE%
