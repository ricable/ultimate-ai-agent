@echo off
REM SPARC IDE - Windows Installer Creation Batch Wrapper
REM This batch file runs the PowerShell script to create the Windows installer

echo ===== SPARC IDE Windows Installer Creation =====
echo.

REM Check for administrative privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script requires administrative privileges.
    echo Please run this script as Administrator.
    echo.
    pause
    exit /b 1
)

echo Running PowerShell script with administrative privileges...
echo.

REM Run the PowerShell script with execution policy bypass
powershell.exe -ExecutionPolicy Bypass -File "%~dp0create-windows-installer.ps1"

if %errorLevel% neq 0 (
    echo.
    echo ERROR: PowerShell script execution failed with error code %errorLevel%.
    echo Please check the log file for details.
    echo.
    pause
    exit /b %errorLevel%
)

echo.
echo ===== Windows Installer Creation Completed =====
echo.
echo The Windows installer is available in the package\windows directory.
echo.

pause