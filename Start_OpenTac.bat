@echo off
rem OpenTac CAD on this computer (Windows): double-click this file.
rem The first run downloads what the app needs (a few minutes, once). Later runs start in seconds.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 if exist "%USERPROFILE%\.local\bin\uv.exe" set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
    echo.
    echo   OpenTac needs uv ^(a small Python installer, https://docs.astral.sh/uv/^).
    set /p answer="  Install it now? [y/N] "
    if /i not "!answer!"=="y" (
        echo   Install uv, then run this file again.
        pause
        exit /b 1
    )
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;!PATH!"
)

uv run opentac cad
pause
