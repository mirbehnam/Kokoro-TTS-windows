# Setup script for Windows
Write-Host "Setting up Kokoro TTS Local..."

# Check if Python is installed
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Python is not installed. Please install Python 3.8 or higher."
    exit 1
}

# Create and activate virtual environment
Write-Host "Creating virtual environment..."
python -m venv venv
.\venv\Scripts\Activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies..."
pip install -r requirements.txt

Write-Host "Setup complete! You can now run: python tts_demo.py" 