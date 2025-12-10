; SPARC IDE - Windows Installer Configuration
; This file contains custom configuration for the NSIS installer

; Product information
!define PRODUCT_NAME "SPARC IDE"
!define PRODUCT_PUBLISHER "SPARC IDE Team"
!define PRODUCT_WEB_SITE "https://github.com/sparc-ide/sparc-ide"
!define PRODUCT_HELP_LINK "https://github.com/sparc-ide/sparc-ide/issues"
!define PRODUCT_ABOUT_LINK "https://github.com/sparc-ide/sparc-ide"

; Registry keys
!define PRODUCT_UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
!define PRODUCT_UNINST_ROOT_KEY "HKLM"
!define PRODUCT_SETTINGS_KEY "Software\SPARC\IDE"

; Custom installer appearance
!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"

; Custom welcome/finish page
!define MUI_WELCOMEFINISHPAGE_BITMAP "sparc-ide-installer-dialog.bmp"
!define MUI_UNWELCOMEFINISHPAGE_BITMAP "sparc-ide-installer-dialog.bmp"

; Custom header
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_BITMAP "sparc-ide-installer-banner.bmp"
!define MUI_HEADERIMAGE_RIGHT

; Custom finish page
!define MUI_FINISHPAGE_RUN "$INSTDIR\sparc-ide.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Launch SPARC IDE"
!define MUI_FINISHPAGE_SHOWREADME "$INSTDIR\resources\app\docs\index.html"
!define MUI_FINISHPAGE_SHOWREADME_TEXT "View Documentation"
!define MUI_FINISHPAGE_LINK "Visit SPARC IDE Website"
!define MUI_FINISHPAGE_LINK_LOCATION "${PRODUCT_WEB_SITE}"

; Custom installer strings
!define MUI_TEXT_WELCOME_INFO_TITLE "Welcome to the SPARC IDE Setup Wizard"
!define MUI_TEXT_WELCOME_INFO_TEXT "This wizard will guide you through the installation of SPARC IDE, an AI-powered development environment.$\r$\n$\r$\nIt is recommended that you close all other applications before starting Setup. This will make it possible to update relevant system files without having to reboot your computer.$\r$\n$\r$\n$_CLICK"

; Custom file associations
!macro RegisterExtension extension description iconIndex
  WriteRegStr HKCR "${extension}" "" "SPARC.IDE.${extension}"
  WriteRegStr HKCR "SPARC.IDE.${extension}" "" "${description}"
  WriteRegStr HKCR "SPARC.IDE.${extension}\DefaultIcon" "" "$INSTDIR\sparc-ide.exe,${iconIndex}"
  WriteRegStr HKCR "SPARC.IDE.${extension}\shell\open\command" "" '"$INSTDIR\sparc-ide.exe" "%1"'
!macroend

!macro UnRegisterExtension extension
  DeleteRegKey HKCR "SPARC.IDE.${extension}"
  DeleteRegKey HKCR "${extension}"
!macroend

; Custom installation directory
!define CUSTOM_APPDATA_DIR "$LOCALAPPDATA\SPARCIde"

; Custom start menu folder
!define CUSTOM_STARTMENU_DIR "$SMPROGRAMS\SPARC IDE"

; Custom uninstaller
!macro customUninstall
  ; Remove file associations
  !insertmacro UnRegisterExtension ".sparc"
  !insertmacro UnRegisterExtension ".md"
  !insertmacro UnRegisterExtension ".json"
  
  ; Remove registry keys
  DeleteRegKey ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}"
  DeleteRegKey HKLM "${PRODUCT_SETTINGS_KEY}"
  
  ; Remove Start Menu shortcuts
  RMDir /r "${CUSTOM_STARTMENU_DIR}"
  
  ; Remove application data
  RMDir /r "${CUSTOM_APPDATA_DIR}"
!macroend

; Custom installation
!macro customInstall
  ; Create application data directory
  CreateDirectory "${CUSTOM_APPDATA_DIR}"
  
  ; Register file associations
  !insertmacro RegisterExtension ".sparc" "SPARC IDE Project File" 0
  !insertmacro RegisterExtension ".md" "Markdown Document" 1
  !insertmacro RegisterExtension ".json" "JSON File" 2
  
  ; Create Start Menu shortcuts
  CreateDirectory "${CUSTOM_STARTMENU_DIR}"
  CreateShortCut "${CUSTOM_STARTMENU_DIR}\SPARC IDE.lnk" "$INSTDIR\sparc-ide.exe"
  CreateShortCut "${CUSTOM_STARTMENU_DIR}\Documentation.lnk" "$INSTDIR\resources\app\docs\index.html"
  CreateShortCut "${CUSTOM_STARTMENU_DIR}\Uninstall.lnk" "$INSTDIR\uninstall.exe"
  
  ; Add additional registry settings
  WriteRegStr HKLM "${PRODUCT_SETTINGS_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "${PRODUCT_SETTINGS_KEY}" "Version" "${PRODUCT_VERSION}"
  WriteRegStr HKLM "${PRODUCT_SETTINGS_KEY}" "DataDirectory" "${CUSTOM_APPDATA_DIR}"
  
  ; Add to PATH (optional)
  ; EnVar::SetHKLM
  ; EnVar::AddValue "PATH" "$INSTDIR"
!macroend