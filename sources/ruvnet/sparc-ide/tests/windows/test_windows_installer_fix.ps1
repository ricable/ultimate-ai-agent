# SPARC IDE - Windows Installer Fix Test Script
# This script tests the fix for the Windows installer issue

# Stop on errors
$ErrorActionPreference = "Stop"

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = (Get-Item "$ScriptDir\..\..\").FullName
$PackageDir = Join-Path $RootDir "package\windows"
$BuildScript = Join-Path $RootDir "scripts\build-real-windows-installer.sh"
$MockScript = Join-Path $RootDir "scripts\mock-windows-installer.sh"
$BrandingScript = Join-Path $RootDir "scripts\prepare-windows-branding.sh"
$TestReportDir = Join-Path $RootDir "test-reports"
$TestReportFile = Join-Path $TestReportDir "windows_installer_fix_test_$(Get-Date -Format 'yyyyMMdd_HHmmss').html"

# Create test report directory
New-Item -ItemType Directory -Force -Path $TestReportDir | Out-Null

# Function to write colored output
function Write-ColoredOutput {
    param (
        [Parameter(Mandatory=$true)]
        [string]$Message,
        
        [Parameter(Mandatory=$true)]
        [string]$Type
    )
    
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    switch ($Type) {
        "INFO" { Write-Host "[$Timestamp] [INFO] $Message" -ForegroundColor Cyan }
        "SUCCESS" { Write-Host "[$Timestamp] [SUCCESS] $Message" -ForegroundColor Green }
        "ERROR" { Write-Host "[$Timestamp] [ERROR] $Message" -ForegroundColor Red }
        "WARNING" { Write-Host "[$Timestamp] [WARNING] $Message" -ForegroundColor Yellow }
        "HEADER" { 
            $HeaderLine = "===== $Message ====="
            Write-Host "[$Timestamp] $HeaderLine" -ForegroundColor Yellow 
        }
    }
}

# Function to append to HTML report
function Add-ToReport {
    param (
        [Parameter(Mandatory=$true)]
        [string]$Content,
        
        [Parameter(Mandatory=$false)]
        [string]$Type = "info"
    )
    
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $HtmlClass = $Type.ToLower()
    
    $HtmlContent = "<div class='$HtmlClass'><span class='timestamp'>[$Timestamp]</span> $Content</div>`n"
    Add-Content -Path $TestReportFile -Value $HtmlContent
}

# Initialize HTML report
$HtmlHeader = @"
<!DOCTYPE html>
<html>
<head>
    <title>SPARC IDE Windows Installer Fix Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .timestamp { color: #888; }
        .header { background-color: #f8f8f8; padding: 10px; margin: 10px 0; font-weight: bold; }
        .info { margin: 5px 0; }
        .success { color: green; margin: 5px 0; }
        .error { color: red; margin: 5px 0; }
        .warning { color: orange; margin: 5px 0; }
        pre { background-color: #f0f0f0; padding: 10px; border-radius: 5px; overflow-x: auto; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
    </style>
</head>
<body>
    <h1>SPARC IDE Windows Installer Fix Test Report</h1>
    <div class='header'>Test Date: $(Get-Date)</div>
    <div class='header'>System Information: $(Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture | Format-List | Out-String)</div>
"@

Set-Content -Path $TestReportFile -Value $HtmlHeader

# Function to test installer file
function Test-InstallerFile {
    param (
        [Parameter(Mandatory=$true)]
        [string]$InstallerPath
    )
    
    Write-ColoredOutput "Testing installer file: $InstallerPath" "INFO"
    Add-ToReport "Testing installer file: $InstallerPath"
    
    # Check if file exists
    if (-not (Test-Path $InstallerPath)) {
        Write-ColoredOutput "ERROR: Installer file not found at $InstallerPath" "ERROR"
        Add-ToReport "ERROR: Installer file not found at $InstallerPath" "error"
        return $false
    }
    
    # Get file info
    $FileInfo = Get-Item $InstallerPath
    $SizeMB = [math]::Round($FileInfo.Length / 1MB, 2)
    Write-ColoredOutput "Installer size: $SizeMB MB" "INFO"
    Add-ToReport "Installer size: $SizeMB MB"
    
    # Check if it's a real executable
    $Bytes = [System.IO.File]::ReadAllBytes($InstallerPath)
    if ($Bytes[0] -ne 0x4D -or $Bytes[1] -ne 0x5A) { # "MZ" signature for Windows executables
        Write-ColoredOutput "ERROR: Not a valid Windows executable (missing MZ signature)" "ERROR"
        Add-ToReport "ERROR: Not a valid Windows executable (missing MZ signature)" "error"
        
        # Read first 100 bytes as text to see if it's a text file
        $Content = Get-Content -Path $InstallerPath -TotalCount 1
        Write-ColoredOutput "File content (first line): $Content" "INFO"
        Add-ToReport "File content (first line): $Content" "info"
        
        if ($Content -like "*mock*") {
            Write-ColoredOutput "This appears to be a mock installer, not a real executable." "ERROR"
            Add-ToReport "This appears to be a mock installer, not a real executable." "error"
        }
        
        return $false
    }
    
    # Determine architecture
    $PEOffset = [BitConverter]::ToInt32($Bytes, 60)
    $MachineType = [BitConverter]::ToUInt16($Bytes, $PEOffset + 4)
    
    $ArchMap = @{
        0x014C = "x86 (32-bit)"
        0x8664 = "x64 (64-bit)"
        0x0200 = "IA64 (Itanium)"
        0x01C4 = "ARM"
        0xAA64 = "ARM64"
    }
    
    if ($ArchMap.ContainsKey($MachineType)) {
        $DetectedArch = $ArchMap[$MachineType]
        Write-ColoredOutput "Detected architecture: $DetectedArch" "SUCCESS"
        Add-ToReport "Detected architecture: $DetectedArch" "success"
    } else {
        Write-ColoredOutput "Unknown architecture type: 0x$($MachineType.ToString('X4'))" "WARNING"
        Add-ToReport "Unknown architecture type: 0x$($MachineType.ToString('X4'))" "warning"
    }
    
    # Check VS_VERSION_INFO resource for version information
    Write-ColoredOutput "Checking version information..." "INFO"
    Add-ToReport "Checking version information..."
    
    try {
        $VersionInfo = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($InstallerPath)
        Write-ColoredOutput "Product Version: $($VersionInfo.ProductVersion)" "INFO"
        Write-ColoredOutput "File Version: $($VersionInfo.FileVersion)" "INFO"
        Write-ColoredOutput "Product Name: $($VersionInfo.ProductName)" "INFO"
        
        Add-ToReport "<pre>Product Version: $($VersionInfo.ProductVersion)
File Version: $($VersionInfo.FileVersion)
Product Name: $($VersionInfo.ProductName)
Company Name: $($VersionInfo.CompanyName)
Description: $($VersionInfo.FileDescription)</pre>"
        
        if ($VersionInfo.ProductName -like "*SPARC*IDE*" -or $VersionInfo.FileDescription -like "*SPARC*IDE*") {
            Write-ColoredOutput "Installer appears to be for SPARC IDE" "SUCCESS"
            Add-ToReport "Installer appears to be for SPARC IDE" "success"
        } else {
            Write-ColoredOutput "WARNING: Installer may not be for SPARC IDE" "WARNING"
            Add-ToReport "WARNING: Installer may not be for SPARC IDE" "warning"
        }
    } catch {
        Write-ColoredOutput "Unable to read version information: $_" "WARNING"
        Add-ToReport "Unable to read version information: $_" "warning"
    }
    
    Write-ColoredOutput "File is a valid Windows executable." "SUCCESS"
    Add-ToReport "File is a valid Windows executable." "success"
    return $true
}

# Function to compare mock vs. real installer
function Compare-Installers {
    # Backup the current installer if it exists
    $CurrentInstallerPath = Get-ChildItem -Path $PackageDir -Filter "*.exe" | Select-Object -First 1 -ExpandProperty FullName
    $BackupPath = $null
    
    if ($CurrentInstallerPath) {
        $BackupPath = "$CurrentInstallerPath.backup"
        Write-ColoredOutput "Backing up current installer to $BackupPath" "INFO"
        Add-ToReport "Backing up current installer to $BackupPath"
        Copy-Item -Path $CurrentInstallerPath -Destination $BackupPath -Force
    }
    
    try {
        # Create a mock installer
        Write-ColoredOutput "Creating mock installer for comparison..." "HEADER"
        Add-ToReport "Creating mock installer for comparison..." "header"
        
        if (Test-Path $MockScript) {
            if (Get-Command bash -ErrorAction SilentlyContinue) {
                & bash $MockScript
                
                $MockInstallerPath = Get-ChildItem -Path $PackageDir -Filter "*.exe" | Select-Object -First 1 -ExpandProperty FullName
                Write-ColoredOutput "Mock installer created at: $MockInstallerPath" "INFO"
                Add-ToReport "Mock installer created at: $MockInstallerPath"
                
                Write-ColoredOutput "Testing mock installer..." "INFO"
                Add-ToReport "Testing mock installer..."
                $MockResult = Test-InstallerFile -InstallerPath $MockInstallerPath
                
                # Create real installer
                Write-ColoredOutput "Creating real installer for comparison..." "HEADER"
                Add-ToReport "Creating real installer for comparison..." "header"
                
                if (Test-Path $BuildScript) {
                    & bash $BuildScript
                    
                    $RealInstallerPath = Get-ChildItem -Path $PackageDir -Filter "*.exe" | Select-Object -First 1 -ExpandProperty FullName
                    Write-ColoredOutput "Real installer created at: $RealInstallerPath" "INFO"
                    Add-ToReport "Real installer created at: $RealInstallerPath"
                    
                    Write-ColoredOutput "Testing real installer..." "INFO"
                    Add-ToReport "Testing real installer..."
                    $RealResult = Test-InstallerFile -InstallerPath $RealInstallerPath
                    
                    # Compare results
                    Write-ColoredOutput "Comparing installers..." "HEADER"
                    Add-ToReport "Comparing installers..." "header"
                    
                    Add-ToReport "<table>
<tr><th>Test</th><th>Mock Installer</th><th>Real Installer</th></tr>
<tr><td>Valid Windows Executable</td><td>$($MockResult ? 'Yes ✓' : 'No ✗')</td><td>$($RealResult ? 'Yes ✓' : 'No ✗')</td></tr>
<tr><td>Size</td><td>$(if (Test-Path $MockInstallerPath) { [math]::Round((Get-Item $MockInstallerPath).Length / 1MB, 2) } else { 'N/A' }) MB</td><td>$(if (Test-Path $RealInstallerPath) { [math]::Round((Get-Item $RealInstallerPath).Length / 1MB, 2) } else { 'N/A' }) MB</td></tr>
</table>"
                    
                    if ($MockResult -eq $false -and $RealResult -eq $true) {
                        Write-ColoredOutput "Fix appears successful. Mock installer was invalid, real installer is valid." "SUCCESS"
                        Add-ToReport "Fix appears successful. Mock installer was invalid, real installer is valid." "success"
                    } elseif ($MockResult -eq $true -and $RealResult -eq $true) {
                        Write-ColoredOutput "Both installers appear valid. Further testing needed." "WARNING"
                        Add-ToReport "Both installers appear valid. Further testing needed." "warning"
                    } elseif ($MockResult -eq $false -and $RealResult -eq $false) {
                        Write-ColoredOutput "Fix unsuccessful. Both installers are invalid." "ERROR"
                        Add-ToReport "Fix unsuccessful. Both installers are invalid." "error"
                    } else {
                        Write-ColoredOutput "Unexpected result. Mock installer is valid but real installer is invalid." "ERROR"
                        Add-ToReport "Unexpected result. Mock installer is valid but real installer is invalid." "error"
                    }
                } else {
                    Write-ColoredOutput "ERROR: Build script not found at $BuildScript" "ERROR"
                    Add-ToReport "ERROR: Build script not found at $BuildScript" "error"
                }
            } else {
                Write-ColoredOutput "ERROR: bash not found. Cannot run mock installer script." "ERROR"
                Add-ToReport "ERROR: bash not found. Cannot run mock installer script." "error"
            }
        } else {
            Write-ColoredOutput "ERROR: Mock script not found at $MockScript" "ERROR"
            Add-ToReport "ERROR: Mock script not found at $MockScript" "error"
        }
    } finally {
        # Restore the original installer if a backup was made
        if ($BackupPath -and (Test-Path $BackupPath)) {
            Write-ColoredOutput "Restoring original installer from backup" "INFO"
            Add-ToReport "Restoring original installer from backup"
            Copy-Item -Path $BackupPath -Destination $CurrentInstallerPath -Force
            Remove-Item -Path $BackupPath -Force
        }
    }
}

# Function to test branding assets
function Test-BrandingAssets {
    Write-ColoredOutput "Testing Windows branding assets..." "HEADER"
    Add-ToReport "Testing Windows branding assets..." "header"
    
    $BrandingDir = Join-Path $RootDir "branding\windows"
    
    if (-not (Test-Path $BrandingDir)) {
        Write-ColoredOutput "ERROR: Windows branding directory not found at $BrandingDir" "ERROR"
        Add-ToReport "ERROR: Windows branding directory not found at $BrandingDir" "error"
        return
    }
    
    # Required branding files
    $RequiredFiles = @(
        "sparc-ide.ico",
        "sparc-ide-installer.ico",
        "sparc-ide-installer-banner.bmp",
        "sparc-ide-installer-dialog.bmp"
    )
    
    $MissingFiles = @()
    
    foreach ($File in $RequiredFiles) {
        $FilePath = Join-Path $BrandingDir $File
        if (-not (Test-Path $FilePath)) {
            $MissingFiles += $File
            Write-ColoredOutput "Missing branding file: $File" "ERROR"
            Add-ToReport "Missing branding file: $File" "error"
        } else {
            Write-ColoredOutput "Found branding file: $File" "SUCCESS"
            Add-ToReport "Found branding file: $File" "success"
            
            # Check file type
            $FileBytes = [System.IO.File]::ReadAllBytes($FilePath)
            $FileType = ""
            
            if ($File -like "*.ico" -and $FileBytes[0] -eq 0x00 -and $FileBytes[1] -eq 0x00 -and $FileBytes[2] -eq 0x01 -and $FileBytes[3] -eq 0x00) {
                $FileType = "Valid ICO file"
                Write-ColoredOutput "$File is a valid ICO file" "SUCCESS"
                Add-ToReport "$File is a valid ICO file" "success"
            } elseif ($File -like "*.bmp" -and $FileBytes[0] -eq 0x42 -and $FileBytes[1] -eq 0x4D) {
                $FileType = "Valid BMP file"
                Write-ColoredOutput "$File is a valid BMP file" "SUCCESS"
                Add-ToReport "$File is a valid BMP file" "success"
            } else {
                $FileType = "Unknown/Invalid"
                Write-ColoredOutput "WARNING: $File may not be a valid format" "WARNING"
                Add-ToReport "WARNING: $File may not be a valid format" "warning"
            }
        }
    }
    
    if ($MissingFiles.Count -gt 0) {
        Write-ColoredOutput "Running branding preparation script..." "INFO"
        Add-ToReport "Running branding preparation script..." "info"
        
        if (Test-Path $BrandingScript) {
            if (Get-Command bash -ErrorAction SilentlyContinue) {
                & bash $BrandingScript
                
                # Check if files were created
                $StillMissing = @()
                foreach ($File in $MissingFiles) {
                    $FilePath = Join-Path $BrandingDir $File
                    if (-not (Test-Path $FilePath)) {
                        $StillMissing += $File
                    } else {
                        Write-ColoredOutput "Successfully created: $File" "SUCCESS"
                        Add-ToReport "Successfully created: $File" "success"
                    }
                }
                
                if ($StillMissing.Count -eq 0) {
                    Write-ColoredOutput "All branding assets created successfully" "SUCCESS"
                    Add-ToReport "All branding assets created successfully" "success"
                } else {
                    Write-ColoredOutput "Some branding assets still missing: $($StillMissing -join ', ')" "ERROR"
                    Add-ToReport "Some branding assets still missing: $($StillMissing -join ', ')" "error"
                }
            } else {
                Write-ColoredOutput "ERROR: bash not found. Cannot run branding preparation script." "ERROR"
                Add-ToReport "ERROR: bash not found. Cannot run branding preparation script." "error"
            }
        } else {
            Write-ColoredOutput "ERROR: Branding preparation script not found at $BrandingScript" "ERROR"
            Add-ToReport "ERROR: Branding preparation script not found at $BrandingScript" "error"
        }
    } else {
        Write-ColoredOutput "All required branding assets exist" "SUCCESS"
        Add-ToReport "All required branding assets exist" "success"
    }
}

# Test documentation
function Test-Documentation {
    Write-ColoredOutput "Testing documentation..." "HEADER"
    Add-ToReport "Testing documentation..." "header"
    
    $DocFile = Join-Path $RootDir "docs\windows-installer-fix.md"
    
    if (-not (Test-Path $DocFile)) {
        Write-ColoredOutput "ERROR: Documentation file not found at $DocFile" "ERROR"
        Add-ToReport "ERROR: Documentation file not found at $DocFile" "error"
    } else {
        Write-ColoredOutput "Documentation file found at $DocFile" "SUCCESS"
        Add-ToReport "Documentation file found at $DocFile" "success"
        
        $DocContent = Get-Content -Path $DocFile -Raw
        
        # Check for key sections
        $ExpectedSections = @(
            "Problem Diagnosis",
            "Solution Implemented",
            "How to Build",
            "Compatibility",
            "Verification"
        )
        
        foreach ($Section in $ExpectedSections) {
            if ($DocContent -match $Section) {
                Write-ColoredOutput "Documentation includes section: $Section" "SUCCESS"
                Add-ToReport "Documentation includes section: $Section" "success"
            } else {
                Write-ColoredOutput "Documentation missing section: $Section" "WARNING"
                Add-ToReport "Documentation missing section: $Section" "warning"
            }
        }
    }
}

# Run the tests
function Run-Tests {
    Write-ColoredOutput "Starting Windows Installer Fix Tests" "HEADER"
    Add-ToReport "Starting Windows Installer Fix Tests" "header"
    
    # Test branding assets
    Test-BrandingAssets
    
    # Test documentation
    Test-Documentation
    
    # Test current installer
    $CurrentInstallerPath = Get-ChildItem -Path $PackageDir -Filter "*.exe" | Select-Object -First 1 -ExpandProperty FullName
    if ($CurrentInstallerPath) {
        Write-ColoredOutput "Testing current installer: $CurrentInstallerPath" "HEADER"
        Add-ToReport "Testing current installer: $CurrentInstallerPath" "header"
        Test-InstallerFile -InstallerPath $CurrentInstallerPath
    } else {
        Write-ColoredOutput "No current installer found in $PackageDir" "WARNING"
        Add-ToReport "No current installer found in $PackageDir" "warning"
    }
    
    # Compare mock vs. real installer
    Compare-Installers
    
    Write-ColoredOutput "Windows Installer Fix Tests Completed" "HEADER"
    Add-ToReport "Windows Installer Fix Tests Completed" "header"
}

# Finalize the HTML report
function Finalize-Report {
    Add-Content -Path $TestReportFile -Value @"
</body>
</html>
"@

    Write-ColoredOutput "Test report saved to: $TestReportFile" "SUCCESS"
    
    # Open the report
    try {
        Invoke-Item $TestReportFile
    } catch {
        Write-ColoredOutput "Could not open test report: $_" "WARNING"
    }
}

# Main function
function Main {
    try {
        Run-Tests
    } catch {
        Write-ColoredOutput "Error running tests: $_" "ERROR"
        Add-ToReport "Error running tests: $_" "error"
        Add-ToReport "<pre>$($_.ScriptStackTrace)</pre>" "error"
    } finally {
        Finalize-Report
    }
}

# Run the main function
Main