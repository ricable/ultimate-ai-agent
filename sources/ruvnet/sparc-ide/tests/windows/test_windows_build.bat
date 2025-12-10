@echo off
REM SPARC IDE - Windows Build Test Script
REM This script tests the Windows build of SPARC IDE

setlocal enabledelayedexpansion

REM Configuration
set "PACKAGE_DIR=%~dp0..\..\package\windows"
set "LOG_FILE=%~dp0..\..\test-reports\windows_test_%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log"
set "LOG_FILE=%LOG_FILE: =0%"

REM Create log directory
if not exist "%~dp0..\..\test-reports" mkdir "%~dp0..\..\test-reports"

REM Start logging
echo ===== SPARC IDE Windows Build Test ===== > "%LOG_FILE%"
echo Date: %date% %time% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

echo ===== SPARC IDE Windows Build Test =====
echo Date: %date% %time%
echo.

REM Check if package directory exists
echo Checking package directory...
echo Checking package directory... >> "%LOG_FILE%"
if not exist "%PACKAGE_DIR%" (
    echo ERROR: Package directory not found at %PACKAGE_DIR%
    echo ERROR: Package directory not found at %PACKAGE_DIR% >> "%LOG_FILE%"
    exit /b 1
)

REM Check for installer
echo Checking for installer...
echo Checking for installer... >> "%LOG_FILE%"
set "INSTALLER_FOUND=0"
for %%F in ("%PACKAGE_DIR%\*.exe") do (
    set "INSTALLER=%%F"
    set "INSTALLER_NAME=%%~nxF"
    set "INSTALLER_FOUND=1"
)

if "%INSTALLER_FOUND%"=="0" (
    echo ERROR: Installer not found in %PACKAGE_DIR%
    echo ERROR: Installer not found in %PACKAGE_DIR% >> "%LOG_FILE%"
    exit /b 1
)

echo Found installer: %INSTALLER_NAME%
echo Found installer: %INSTALLER_NAME% >> "%LOG_FILE%"

REM Verify installer size
echo Verifying installer size...
echo Verifying installer size... >> "%LOG_FILE%"
for %%A in ("%INSTALLER%") do set "SIZE=%%~zA"
set /a "SIZE_MB=%SIZE% / 1048576"
if %SIZE_MB% LSS 50 (
    echo ERROR: Installer size is too small (%SIZE_MB% MB^). Expected at least 50 MB.
    echo ERROR: Installer size is too small (%SIZE_MB% MB^). Expected at least 50 MB. >> "%LOG_FILE%"
    exit /b 1
)
echo Installer size: %SIZE_MB% MB
echo Installer size: %SIZE_MB% MB >> "%LOG_FILE%"

REM Verify checksums
echo Verifying checksums...
echo Verifying checksums... >> "%LOG_FILE%"
set "CHECKSUM_FILE=%PACKAGE_DIR%\checksums.sha256"
if not exist "%CHECKSUM_FILE%" (
    echo ERROR: Checksums file not found at %CHECKSUM_FILE%
    echo ERROR: Checksums file not found at %CHECKSUM_FILE% >> "%LOG_FILE%"
    exit /b 1
)

REM Extract expected checksum for the installer
set "EXPECTED_CHECKSUM="
for /f "tokens=1,2 delims= " %%a in (%CHECKSUM_FILE%) do (
    if "%%b"=="%INSTALLER_NAME%" set "EXPECTED_CHECKSUM=%%a"
)

if "%EXPECTED_CHECKSUM%"=="" (
    echo ERROR: Checksum not found for %INSTALLER_NAME% in %CHECKSUM_FILE%
    echo ERROR: Checksum not found for %INSTALLER_NAME% in %CHECKSUM_FILE% >> "%LOG_FILE%"
    exit /b 1
)

REM Calculate actual checksum (requires certutil)
echo Computing actual checksum...
echo Computing actual checksum... >> "%LOG_FILE%"
for /f "skip=1 tokens=* delims=" %%a in ('certutil -hashfile "%INSTALLER%" SHA256') do (
    if not defined ACTUAL_CHECKSUM set "ACTUAL_CHECKSUM=%%a"
)
set "ACTUAL_CHECKSUM=%ACTUAL_CHECKSUM: =%"

if /i not "%EXPECTED_CHECKSUM%"=="%ACTUAL_CHECKSUM%" (
    echo ERROR: Checksum verification failed.
    echo Expected: %EXPECTED_CHECKSUM%
    echo Actual: %ACTUAL_CHECKSUM%
    echo ERROR: Checksum verification failed. >> "%LOG_FILE%"
    echo Expected: %EXPECTED_CHECKSUM% >> "%LOG_FILE%"
    echo Actual: %ACTUAL_CHECKSUM% >> "%LOG_FILE%"
    exit /b 1
)
echo Checksum verification passed.
echo Checksum verification passed. >> "%LOG_FILE%"

REM Test installer extraction (silent mode)
echo Testing installer extraction (silent mode^)...
echo Testing installer extraction (silent mode^)... >> "%LOG_FILE%"
set "TEMP_DIR=%TEMP%\sparc-ide-test-%RANDOM%"
mkdir "%TEMP_DIR%"

echo Running installer in silent mode...
echo Running installer in silent mode... >> "%LOG_FILE%"
start /wait "" "%INSTALLER%" /VERYSILENT /SUPPRESSMSGBOXES /DIR="%TEMP_DIR%"

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Installer extraction failed with exit code %ERRORLEVEL%
    echo ERROR: Installer extraction failed with exit code %ERRORLEVEL% >> "%LOG_FILE%"
    exit /b 1
)
echo Installer extraction passed.
echo Installer extraction passed. >> "%LOG_FILE%"

REM Verify extracted files
echo Verifying extracted files...
echo Verifying extracted files... >> "%LOG_FILE%"
set "REQUIRED_FILES=sparc-ide.exe resources\app\node_modules.asar resources\app\out\main.js"
for %%F in (%REQUIRED_FILES%) do (
    if not exist "%TEMP_DIR%\%%F" (
        echo ERROR: Required file not found: %%F
        echo ERROR: Required file not found: %%F >> "%LOG_FILE%"
        exit /b 1
    )
)
echo All required files found.
echo All required files found. >> "%LOG_FILE%"

REM Verify Roo Code extension integration
echo Verifying Roo Code extension integration...
echo Verifying Roo Code extension integration... >> "%LOG_FILE%"
if not exist "%TEMP_DIR%\resources\app\extensions\roo-code" (
    echo WARNING: Roo Code extension directory not found.
    echo WARNING: Roo Code extension directory not found. >> "%LOG_FILE%"
) else (
    echo Roo Code extension integration verified.
    echo Roo Code extension integration verified. >> "%LOG_FILE%"
)

REM Verify registry settings (if installed)
echo Verifying registry settings...
echo Verifying registry settings... >> "%LOG_FILE%"
reg query "HKLM\Software\SPARC\IDE" /v "InstallLocation" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Registry settings verified.
    echo Registry settings verified. >> "%LOG_FILE%"
) else (
    echo WARNING: Registry settings not found. This is expected for portable installations.
    echo WARNING: Registry settings not found. This is expected for portable installations. >> "%LOG_FILE%"
)

REM Clean up
echo Cleaning up...
echo Cleaning up... >> "%LOG_FILE%"
rd /s /q "%TEMP_DIR%"

echo.
echo ===== All tests passed! =====
echo The Windows build of SPARC IDE has been verified successfully.
echo.
echo Test log saved to: %LOG_FILE%

echo. >> "%LOG_FILE%"
echo ===== All tests passed! ===== >> "%LOG_FILE%"
echo The Windows build of SPARC IDE has been verified successfully. >> "%LOG_FILE%"

endlocal
exit /b 0