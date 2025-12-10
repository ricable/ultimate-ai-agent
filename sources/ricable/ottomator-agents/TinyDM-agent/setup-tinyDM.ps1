# TinyDM Setup and Startup Script
# Save this as setup-tinyDM.ps1

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Please run this script as Administrator" -ForegroundColor Red
    exit
}

# Configuration
$projectName = "TinyDM"
$projectPath = "C:\$projectName"
$pythonVersion = "3.11"
$requirementsFile = "$projectPath\requirements.txt"
$envFile = "$projectPath\.env"
$mainFile = "$projectPath\main.py"

# Create project directory
Write-Host "Creating project directory..." -ForegroundColor Green
New-Item -ItemType Directory -Force -Path $projectPath | Out-Null

# Check if Python is installed
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "Python not found. Installing Python $pythonVersion..." -ForegroundColor Yellow
    # Download Python installer
    $pythonUrl = "https://www.python.org/ftp/python/$pythonVersion/python-$pythonVersion-amd64.exe"
    $installer = "$env:TEMP\python-installer.exe"
    Invoke-WebRequest -Uri $pythonUrl -OutFile $installer
    
    # Install Python
    Start-Process -FilePath $installer -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1" -Wait
    Remove-Item $installer
    
    # Refresh environment variables
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

# Create virtual environment
Write-Host "Setting up virtual environment..." -ForegroundColor Green
python -m venv "$projectPath\venv"

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& "$projectPath\venv\Scripts\Activate.ps1"

# Create requirements.txt
Write-Host "Creating requirements.txt..." -ForegroundColor Green
@"
fastapi==0.104.1
uvicorn==0.24.0
python-dotenv==1.0.0
supabase==2.0.3
httpx==0.25.1
python-multipart==0.0.6
google-cloud-aiplatform==1.36.4
"@ | Out-File -FilePath $requirementsFile -Encoding utf8

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Green
pip install -r $requirementsFile

# Create .env template
Write-Host "Creating .env template..." -ForegroundColor Green
@"
# Supabase Configuration
SUPABASE_URL=your-project.supabase.co
SUPABASE_KEY=your-anon-key

# Google AI Configuration
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
GOOGLE_PROJECT_ID=your-project-id

# Server Configuration
HOST=localhost
PORT=8001
BEARER_TOKEN=your-bearer-token

# Open5e API Configuration
OPEN5E_API_URL=https://api.open5e.com/v1
"@ | Out-File -FilePath $envFile -Encoding utf8

# Create main.py
Write-Host "Creating main.py..." -ForegroundColor Green
@"
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import httpx
from supabase import create_client, Client
from typing import Optional
import json
import uuid

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="TinyDM")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

@app.post("/api/dm-assistant")
async def handle_request(request: dict, authorization: str = Header(None)):
    # Validate bearer token
    if authorization != f"Bearer {os.getenv('BEARER_TOKEN')}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        # Extract request information
        query = request.get("query")
        session_id = request.get("session_id")
        user_id = request.get("user_id")
        
        if not all([query, session_id, user_id]):
            raise HTTPException(status_code=400, detail="Missing required fields")

        # Store user message
        await store_message(session_id, "human", query)
        
        # Generate response (placeholder)
        response = "Greetings! I am TinyDM, ready to assist with your D&D session!"
        
        # Store assistant message
        await store_message(session_id, "ai", response)
        
        return {
            "success": True,
            "content": response,
            "message": {
                "type": "ai",
                "content": response
            }
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

async def store_message(session_id: str, msg_type: str, content: str):
    message = {
        "type": msg_type,
        "content": content
    }
    
    supabase.table("messages").insert({
        "session_id": session_id,
        "message": message
    }).execute()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "localhost"),
        port=int(os.getenv("PORT", 8001)),
        reload=True
    )
"@ | Out-File -FilePath $mainFile -Encoding utf8

# Create startup script
$startupScript = @"
# Save this as start-tinyDM.ps1
Set-Location "$projectPath"
& "$projectPath\venv\Scripts\Activate.ps1"
python main.py
"@
$startupScript | Out-File -FilePath "$projectPath\start-tinyDM.ps1" -Encoding utf8

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Edit $envFile with your configuration details" -ForegroundColor Yellow
Write-Host "2. Run start-tinyDM.ps1 to start the server" -ForegroundColor Yellow
Write-Host "3. Configure Agent Zero with:" -ForegroundColor Yellow
Write-Host "   - Supabase Project URL: your-project.supabase.co" -ForegroundColor Yellow
Write-Host "   - Supabase Anon Key: your-anon-key" -ForegroundColor Yellow
Write-Host "   - Agent Endpoint: http://localhost:8001/api/dm-assistant" -ForegroundColor Yellow
Write-Host "   - Bearer Token: your-bearer-token" -ForegroundColor Yellow

# Create desktop shortcut for startup
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\Start TinyDM.lnk")
$Shortcut.TargetPath = "powershell.exe"
$Shortcut.Arguments = "-NoExit -File `"$projectPath\start-tinyDM.ps1`""
$Shortcut.WorkingDirectory = $projectPath
$Shortcut.Save()