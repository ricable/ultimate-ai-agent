# SPARC IDE - Windows Branding Application Script
# This script applies Windows-specific branding to the SPARC IDE build

# Stop on errors
$ErrorActionPreference = "Stop"

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = (Get-Item "$ScriptDir\..\..\").FullName
$VscodiumDir = Join-Path $RootDir "vscodium"
$WindowsBrandingDir = Join-Path $RootDir "branding\windows"
$BrandingConfigFile = Join-Path $ScriptDir "branding-config.json"
$InstallerConfigFile = Join-Path $ScriptDir "installer-config.nsh"

# Function to write colored output
function Write-ColoredOutput {
    param (
        [Parameter(Mandatory=$true)]
        [string]$Message,
        
        [Parameter(Mandatory=$true)]
        [string]$Type
    )
    
    switch ($Type) {
        "INFO" { 
            Write-Host "[INFO] $Message" -ForegroundColor Cyan 
        }
        "SUCCESS" { 
            Write-Host "[SUCCESS] $Message" -ForegroundColor Green 
        }
        "ERROR" { 
            Write-Host "[ERROR] $Message" -ForegroundColor Red 
        }
        "HEADER" { 
            Write-Host "===== $Message =====" -ForegroundColor Yellow 
        }
    }
}

# Function to verify file exists
function Verify-FileExists {
    param (
        [Parameter(Mandatory=$true)]
        [string]$FilePath,
        
        [Parameter(Mandatory=$true)]
        [string]$Description
    )
    
    if (-not (Test-Path $FilePath)) {
        Write-ColoredOutput "Required $Description file not found: $FilePath" "ERROR"
        exit 1
    }
    
    return $true
}

# Function to verify directory exists
function Verify-DirectoryExists {
    param (
        [Parameter(Mandatory=$true)]
        [string]$DirPath,
        
        [Parameter(Mandatory=$true)]
        [string]$Description
    )
    
    if (-not (Test-Path $DirPath -PathType Container)) {
        Write-ColoredOutput "Required $Description directory not found: $DirPath" "ERROR"
        exit 1
    }
    
    return $true
}

# Function to create directory if it doesn't exist
function Create-DirectoryIfNotExists {
    param (
        [Parameter(Mandatory=$true)]
        [string]$DirPath
    )
    
    if (-not (Test-Path $DirPath -PathType Container)) {
        New-Item -ItemType Directory -Path $DirPath -Force | Out-Null
        Write-ColoredOutput "Created directory: $DirPath" "INFO"
    }
}

# Main function
function Main {
    Write-ColoredOutput "SPARC IDE Windows Branding Application" "HEADER"
    
    # Verify required directories and files
    Verify-DirectoryExists $VscodiumDir "VSCodium"
    Verify-DirectoryExists $WindowsBrandingDir "Windows branding"
    Verify-FileExists $BrandingConfigFile "branding configuration"
    Verify-FileExists $InstallerConfigFile "installer configuration"
    
    # Load branding configuration
    Write-ColoredOutput "Loading branding configuration..." "INFO"
    $BrandingConfig = Get-Content $BrandingConfigFile -Raw | ConvertFrom-Json
    
    # Create Windows build directory in VSCodium
    $WindowsBuildDir = Join-Path $VscodiumDir "build\win32-x64"
    Create-DirectoryIfNotExists $WindowsBuildDir
    
    # Copy Windows branding assets
    Write-ColoredOutput "Copying Windows branding assets..." "INFO"
    Copy-Item -Path "$WindowsBrandingDir\*" -Destination $WindowsBuildDir -Recurse -Force
    
    # Copy installer configuration
    Write-ColoredOutput "Copying installer configuration..." "INFO"
    $NsisDir = Join-Path $WindowsBuildDir "nsis"
    Create-DirectoryIfNotExists $NsisDir
    Copy-Item -Path $InstallerConfigFile -Destination $NsisDir -Force
    
    # Update product.json with Windows-specific settings
    Write-ColoredOutput "Updating product.json with Windows-specific settings..." "INFO"
    $ProductJsonPath = Join-Path $VscodiumDir "product.json"
    
    if (Test-Path $ProductJsonPath) {
        $ProductJson = Get-Content $ProductJsonPath -Raw | ConvertFrom-Json
        
        # Update Windows-specific properties
        $ProductJson.nameShort = $BrandingConfig.productName
        $ProductJson.nameLong = $BrandingConfig.productName + ": AI-Powered Development Environment"
        $ProductJson.applicationName = $BrandingConfig.applicationName
        $ProductJson.dataFolderName = $BrandingConfig.dataFolderName
        $ProductJson.win32MutexName = $BrandingConfig.applicationName.Replace("-", "")
        $ProductJson.win32DirName = $BrandingConfig.productName
        $ProductJson.win32NameVersion = $BrandingConfig.productName
        $ProductJson.win32RegValueName = $BrandingConfig.productName
        $ProductJson.win32AppUserModelId = "SPARC.IDE"
        $ProductJson.win32ShellNameShort = $BrandingConfig.productName
        
        # Save updated product.json
        $ProductJson | ConvertTo-Json -Depth 10 | Set-Content $ProductJsonPath -Encoding UTF8
        Write-ColoredOutput "Updated product.json with Windows-specific settings" "SUCCESS"
    } else {
        Write-ColoredOutput "product.json not found at $ProductJsonPath" "ERROR"
        exit 1
    }
    
    # Create custom NSIS installer script
    Write-ColoredOutput "Creating custom NSIS installer script..." "INFO"
    $NsisTemplatePath = Join-Path $NsisDir "VSCodeSetup.nsi.template"
    
    # We'll create this file during the build process if it doesn't exist yet
    if (-not (Test-Path $NsisTemplatePath)) {
        Write-ColoredOutput "NSIS template file not found. It will be created during the build process." "INFO"
    } else {
        # Backup original template
        Copy-Item -Path $NsisTemplatePath -Destination "$NsisTemplatePath.bak" -Force
        
        # Modify NSIS template
        $NsisTemplate = Get-Content $NsisTemplatePath -Raw
        
        # Replace product name
        $NsisTemplate = $NsisTemplate -replace "VSCodium", $BrandingConfig.productName
        
        # Add include for our custom configuration
        $NsisTemplate = $NsisTemplate -replace "!include `".\installer.nsh`"", "!include `".\installer.nsh`"`n!include `".\installer-config.nsh`""
        
        # Add file associations
        $FileAssociationsCode = ""
        foreach ($assoc in $BrandingConfig.fileAssociations) {
            $FileAssociationsCode += "  !insertmacro RegisterExtension `"$($assoc.ext)`" `"$($assoc.description)`" 0`n"
        }
        
        # Add the file associations to the install section
        $NsisTemplate = $NsisTemplate -replace "; Create shortcuts", "; Register file associations`n$FileAssociationsCode`n  ; Create shortcuts"
        
        # Save modified template
        Set-Content -Path $NsisTemplatePath -Value $NsisTemplate -Encoding UTF8
        Write-ColoredOutput "Modified NSIS template with custom settings" "SUCCESS"
    }
    
    # Create registry entries file
    Write-ColoredOutput "Creating registry entries file..." "INFO"
    $RegEntriesPath = Join-Path $NsisDir "registry-entries.nsh"
    
    $RegEntries = @"
; SPARC IDE - Windows Registry Entries

; Add registry entries for SPARC IDE
!macro AddRegistryEntries
  ; Application information
  WriteRegStr HKLM "Software\SPARC\IDE" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "Software\SPARC\IDE" "Version" "$\{PRODUCT_VERSION\}"
  WriteRegStr HKLM "Software\SPARC\IDE" "DataDirectory" "$LOCALAPPDATA\SPARCIde"
  
  ; Add to App Paths (allows Windows to find the executable)
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\App Paths\sparc-ide.exe" "" "$INSTDIR\sparc-ide.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\App Paths\sparc-ide.exe" "Path" "$INSTDIR"
  
  ; Add uninstall information
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\$\{PRODUCT_NAME\}" "DisplayName" "$\{PRODUCT_NAME\}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\$\{PRODUCT_NAME\}" "DisplayIcon" "$INSTDIR\sparc-ide.exe,0"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\$\{PRODUCT_NAME\}" "DisplayVersion" "$\{PRODUCT_VERSION\}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\$\{PRODUCT_NAME\}" "Publisher" "$\{PRODUCT_PUBLISHER\}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\$\{PRODUCT_NAME\}" "UninstallString" "$INSTDIR\uninstall.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\$\{PRODUCT_NAME\}" "QuietUninstallString" "$INSTDIR\uninstall.exe /SILENT"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\$\{PRODUCT_NAME\}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\$\{PRODUCT_NAME\}" "HelpLink" "$\{PRODUCT_HELP_LINK\}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\$\{PRODUCT_NAME\}" "URLInfoAbout" "$\{PRODUCT_ABOUT_LINK\}"
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\$\{PRODUCT_NAME\}" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\$\{PRODUCT_NAME\}" "NoRepair" 1
!macroend

; Remove registry entries for SPARC IDE
!macro RemoveRegistryEntries
  ; Remove application information
  DeleteRegKey HKLM "Software\SPARC\IDE"
  
  ; Remove from App Paths
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\App Paths\sparc-ide.exe"
  
  ; Remove uninstall information
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\$\{PRODUCT_NAME\}"
!macroend
"@
    
    Set-Content -Path $RegEntriesPath -Value $RegEntries -Encoding UTF8
    Write-ColoredOutput "Created registry entries file" "SUCCESS"
    
    Write-ColoredOutput "Windows branding applied successfully" "HEADER"
}

# Run main function
Main