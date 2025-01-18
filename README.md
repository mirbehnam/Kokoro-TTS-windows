# Kokoro TTS Local

A local implementation of the Kokoro Text-to-Speech model, featuring dynamic module loading, automatic dependency management, and a web interface.

## Current Status

✅ **WORKING - READY TO USE** ✅

The project has been updated with:
- Automatic espeak-ng installation and configuration
- Dynamic module loading from Hugging Face
- Improved error handling and debugging
- Interactive CLI interface
- Cross-platform setup scripts
- Web interface with Gradio

## Features

- Local text-to-speech synthesis using the Kokoro model
- Automatic espeak-ng setup using espeakng-loader
- Multiple voice support with easy voice selection
- Phoneme output support and visualization
- Interactive CLI for custom text input
- Voice listing functionality
- Dynamic module loading from Hugging Face
- Comprehensive error handling and logging
- Cross-platform support (Windows, Linux, macOS)
- **NEW: Web Interface Features**
  - Modern, user-friendly UI
  - Real-time generation progress
  - Multiple output formats (WAV, MP3, AAC)
  - Network sharing capabilities
  - Audio playback and download
  - Voice selection dropdown
  - Detailed process logging

## Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)
- Internet connection (for initial model download)
- FFmpeg (required for MP3/AAC conversion):
  - Windows: Automatically installed with pydub
  - Linux: `sudo apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`

### Windows-Specific Requirements
For optimal performance on Windows, you should either:
1. Enable Developer Mode:
   - Open Windows Settings
   - Navigate to System > Developer settings
   - Turn on Developer Mode
   
OR

2. Run Python as Administrator:
   - Right-click your terminal (PowerShell/Command Prompt)
   - Select "Run as administrator"
   - Run the commands from there

This is needed for proper symlink support in the Hugging Face cache system.
If you skip this, the system will still work but may use more disk space.

## Dependencies

```txt
torch
phonemizer-fork
transformers
scipy
munch
soundfile
huggingface-hub
espeakng-loader
gradio>=4.0.0
pydub  # For audio format conversion
```

## Setup

### Windows
```powershell
# Clone the repository
git clone https://github.com/PierrunoYT/Kokoro-TTS-Local.git
cd Kokoro-TTS-Local

# Run the setup script
.\setup.ps1

# Download initial model and voices
python tts_demo.py --list-voices
```

### Linux/macOS
```bash
# Clone the repository
git clone https://github.com/PierrunoYT/Kokoro-TTS-Local.git
cd Kokoro-TTS-Local

# Run the setup script
chmod +x setup.sh
./setup.sh

# Install FFmpeg (if needed)
# Linux:
sudo apt-get install ffmpeg
# macOS:
brew install ffmpeg

# Download initial model and voices
python tts_demo.py --list-voices
```

### Manual Setup
If you prefer to set up manually:

1. Create a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Install system dependencies:
```bash
# Windows
# FFmpeg is automatically installed with pydub

# Linux
sudo apt-get update
sudo apt-get install espeak-ng ffmpeg

# macOS
brew install espeak ffmpeg
```

4. Download initial model and voices:
```bash
# This will download the model and voices from Hugging Face
python tts_demo.py --list-voices
```

## Usage

### Web Interface
```bash
# Start the web interface
python gradio_interface.py
```

After running the command:
1. Open your web browser and visit: http://localhost:7860
2. The interface will also create a public share link (optional)
3. You can now:
   - Input text to synthesize
   - Select from available voices
   - Choose output format (WAV/MP3/AAC)
   - Monitor generation progress
   - Play or download generated audio

Note: If port 7860 is already in use, Gradio will automatically try the next available port (7861, 7862, etc.).
Check the terminal output for the correct URL.

### Command Line Interface
```bash
python tts_demo.py
```

The script will:
1. Download necessary model files from Hugging Face
2. Set up espeak-ng automatically using espeakng-loader
3. Import required modules dynamically
4. Test the phonemizer functionality
5. Generate speech from your text with phoneme visualization
6. Save the output as 'output.wav' (22050Hz sample rate)

## Project Structure

- `models.py`: Core model loading and speech generation functionality
  - Model building and initialization with dynamic imports
  - Voice loading and management from Hugging Face
  - Speech generation with phoneme output
  - Voice listing functionality
  - Automatic espeak-ng configuration
  - Error handling and logging
- `tts_demo.py`: Demo script showing basic usage
  - Command-line interface with argparse
  - Interactive text input mode
  - Voice selection and listing
  - Error handling and user feedback
- `gradio_interface.py`: Web interface implementation
  - Modern, responsive UI
  - Real-time progress monitoring
  - Multiple output formats
  - Network sharing capabilities
- `setup.ps1`: Windows PowerShell setup script
  - Environment creation
  - Dependency installation
  - Automatic configuration
- `setup.sh`: Linux/macOS bash setup script
  - Environment creation
  - Dependency installation
  - Automatic configuration
- `requirements.txt`: Project dependencies

## Model Information

The project uses the Kokoro-82M model from Hugging Face:
- Repository: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- Model file: `kokoro-v0_19.pth`
- Voice files: Located in the `voices/` directory
- Available voices:
  - African Female: `af_bella`, `af_nicole`, `af_sarah`, `af_sky`
  - African Male: `am_adam`, `am_michael`
  - British Female: `bf_emma`, `bf_isabella`
  - British Male: `bm_george`, `bm_lewis`
- Automatically downloads required files from Hugging Face

## Technical Details

- Sample rate: 22050Hz
- Input: Text in any language (English recommended)
- Output: WAV/MP3/AAC audio file
- Dependencies are automatically managed
- Modules are dynamically loaded from Hugging Face
- Error handling includes stack traces for debugging
- Cross-platform compatibility through setup scripts

## Contributing

Feel free to contribute by:
1. Opening issues for bugs or feature requests
2. Submitting pull requests with improvements
3. Helping with documentation
4. Testing different voices and reporting issues
5. Suggesting new features or optimizations
6. Testing on different platforms and reporting results

## License

This project is licensed under the Apache 2.0 License. 