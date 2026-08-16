@echo off
rem Double-clickable wrapper around install.ps1, which does the real work.
rem Bypasses the execution policy for this one invocation only; it does not
rem change any machine setting.

setlocal
set "ROOT=%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%install.ps1" %*
set "CODE=%ERRORLEVEL%"

echo.
if not "%CODE%"=="0" (
    echo   Install script exited with code %CODE%.
)
pause
exit /b %CODE%
