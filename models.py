import torch
import os
import sys
import platform
import glob
import warnings
from huggingface_hub import hf_hub_download, list_repo_files
import espeakng_loader
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

# Filter out specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

__all__ = ['list_available_voices', 'build_model', 'load_voice', 'generate_speech', 'load_and_validate_voice']

def get_voices_path():
    """Get the path where voice files are stored."""
    # Store voices in a 'voices' directory in the project root
    return str(Path(__file__).parent / "voices")

def load_and_validate_voice(voice_name: str, device: str) -> torch.Tensor:
    """Load and validate the requested voice.
    
    Args:
        voice_name: Name of the voice to load
        device: Device to load the voice on ('cuda' or 'cpu')
        
    Returns:
        Loaded voice tensor
        
    Raises:
        ValueError: If the requested voice doesn't exist
    """
    available_voices = list_available_voices()
    if voice_name not in available_voices:
        raise ValueError(f"Voice '{voice_name}' not found. Available voices: {', '.join(available_voices)}")
    return load_voice(voice_name, device)

def list_available_voices():
    """List all available voices from the official voicepacks."""
    voices_path = get_voices_path()
    try:
        # Create voices directory if it doesn't exist
        os.makedirs(voices_path, exist_ok=True)
        
        # Download voices if they don't exist
        if not any(f.endswith('.pt') for f in os.listdir(voices_path)):
            print("Downloading voice files...")
            repo_id = "hexgrad/Kokoro-82M"
            repo_files = list_repo_files(repo_id)
            voice_files = [f for f in repo_files if f.startswith('voices/') and f.endswith('.pt')]
            
            for voice_file in voice_files:
                try:
                    voice_name = os.path.splitext(os.path.basename(voice_file))[0]
                    target_path = os.path.join(voices_path, f"{voice_name}.pt")
                    if not os.path.exists(target_path):
                        print(f"Downloading voice: {voice_name}")
                        hf_hub_download(
                            repo_id=repo_id,
                            filename=voice_file,
                            local_dir=str(Path(voices_path).parent),
                            local_dir_use_symlinks=False
                        )
                except Exception as e:
                    print(f"Error downloading voice {voice_file}: {e}")
        
        # List available voice files
        voices = [os.path.splitext(f)[0] for f in os.listdir(voices_path) if f.endswith('.pt')]
        return sorted(voices)
    except Exception as e:
        print(f"Error accessing voices: {e}")
        return ["af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_michael", 
                "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]

def get_platform_paths():
    """Get platform-specific paths for espeak-ng"""
    system = platform.system().lower()
    
    # First check for local espeak directory in project folder
    local_espeak_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "espeak")
    local_lib_path = os.path.join(local_espeak_dir, "libespeak-ng.dll")
    local_data_path = os.path.join(local_espeak_dir, "espeak-ng-data")
    
    # If local files exist, use them
    if os.path.exists(local_lib_path) and os.path.exists(local_data_path):
        return local_lib_path, local_data_path
    
    # Fall back to default system paths if local files don't exist
    if system == "windows":
        lib_path = os.path.join(os.getenv("ProgramFiles"), "eSpeak NG", "libespeak-ng.dll")
        data_path = os.path.join(os.getenv("ProgramFiles"), "eSpeak NG", "espeak-ng-data")
    
    elif system == "darwin":  # macOS
        lib_path = "/opt/homebrew/lib/libespeak-ng.dylib"
        brew_data = "/opt/homebrew/share/espeak-ng-data"
        sys_lib = "/usr/local/lib/libespeak-ng.dylib"
        sys_data = "/usr/local/share/espeak-ng-data"
        lib_path = lib_path if os.path.exists(lib_path) else sys_lib
        data_path = brew_data if os.path.exists(brew_data) else sys_data
    
    else:  # Linux
        data_path = "/usr/lib/x86_64-linux-gnu/espeak-ng-data"
        lib_paths = [
            "/lib/x86_64-linux-gnu/libespeak-ng.so.1",
            "/usr/lib/x86_64-linux-gnu/libespeak-ng.so",
            "/usr/lib/libespeak-ng.so",
            "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
            "/usr/lib/aarch64-linux-gnu/libespeak-ng.so",
            "/usr/lib64/libespeak-ng.so"
        ]
        
        lib_path = None
        for path in lib_paths:
            if os.path.exists(path):
                lib_path = path
                break
        
        if lib_path is None:
            lib_path = lib_paths[0]  # Default for error message
    
    return lib_path, data_path
def setup_espeak():
    """Set up espeak library paths for phonemizer."""
    try:
        lib_path, data_path = get_platform_paths()
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"espeak-ng library not found at {lib_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"espeak-ng data not found at {data_path}")
            
        EspeakWrapper.set_library(lib_path)
        EspeakWrapper.data_path = data_path
        
        # Configure phonemizer for UTF-8
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = lib_path
        os.environ["PHONEMIZER_ESPEAK_PATH"] = data_path
        os.environ["PYTHONIOENCODING"] = "utf-8"
        
        print("espeak-ng library paths set up successfully")
        
    except Exception as e:
        print(f"Error setting up espeak: {e}")
        print("\nPlease ensure espeak-ng is installed:")
        print("- Windows: Download from https://github.com/espeak-ng/espeak-ng/releases")
        print("- macOS: brew install espeak-ng")
        print("- Linux: sudo apt install espeak-ng")
        raise e

def import_module_from_path(module_name, module_path):
    """Import a module from file path."""
    try:
        spec = spec_from_file_location(module_name, module_path)
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error importing module {module_name}: {e}")
        raise e

def build_model(model_file, device='cpu'):
    """Build the Kokoro model following official implementation."""
    try:
        setup_espeak()
        
        repo_id = "hexgrad/Kokoro-82M"
      #  Download all required files
        resources_dir = "resources"   # Your preferred local directory

        model_path = hf_hub_download(repo_id=repo_id, filename="kokoro-v0_19.pth", cache_dir=resources_dir )
        kokoro_py = hf_hub_download(repo_id=repo_id, filename="kokoro.py", cache_dir=resources_dir )
        models_py = hf_hub_download(repo_id=repo_id, filename="models.py", cache_dir=resources_dir )
        istftnet_py = hf_hub_download(repo_id=repo_id, filename="istftnet.py", cache_dir=resources_dir )
        plbert_py = hf_hub_download(repo_id=repo_id, filename="plbert.py", cache_dir=resources_dir)
        config_json = hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=resources_dir )
        
        # Import required modules
        print("Importing plbert module...")
        plbert_module = import_module_from_path("plbert", plbert_py)
        print("Importing istftnet module...")
        istftnet_module = import_module_from_path("istftnet", istftnet_py)
        print("Importing models module...")
        models_module = import_module_from_path("models", models_py)
        print("Importing kokoro module...")
        kokoro_module = import_module_from_path("kokoro", kokoro_py)
        
        # Test phonemizer
        from phonemizer import phonemize
        test_phonemes = phonemize("Hello")
        print(f"Phonemizer test successful: 'Hello' -> {test_phonemes}")
        
        # Build model
        print("Building model...")
        model = models_module.build_model(model_path, device)
        print(f"Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        print(f"Error building model: {e}")
        import traceback
        traceback.print_exc()
        raise e

def load_voice(voice_name='af_bella', device='cpu'):
    """Load a voice from the local voices directory."""
    try:
        voices_path = get_voices_path()
        voice_path = os.path.join(voices_path, f"{voice_name}.pt")
        
        # Download voice if it doesn't exist locally
        if not os.path.exists(voice_path):
            print(f"Downloading voice: {voice_name}")
            repo_id = "hexgrad/Kokoro-82M"
            hf_hub_download(
                repo_id=repo_id,
                filename=f"voices/{voice_name}.pt",
                local_dir=str(Path(voices_path).parent),
                local_dir_use_symlinks=False
            )
        
        voice = torch.load(voice_path, weights_only=True).to(device)
        print(f"Loaded voice: {voice_name}")
        return voice
    except Exception as e:
        print(f"Error loading voice: {e}")
        raise e

def generate_speech(model, text, voice=None, lang='a', device='cpu',speed=1):
    """Generate speech using the Kokoro model."""
    try:
        repo_id = "hexgrad/Kokoro-82M"
        kokoro_py = hf_hub_download(repo_id=repo_id, filename="kokoro.py")
        kokoro_module = import_module_from_path("kokoro", kokoro_py)
        
        # Generate speech
        audio, phonemes = kokoro_module.generate(model, text, voice, lang=lang,speed=speed)
        
        # Handle phonemes encoding
        if phonemes:
            try:
                # Debug info
                print(f"Debug - Original phonemes type: {type(phonemes)}")
                print(f"Debug - Original phonemes: {repr(phonemes)}")
                
                # Convert to string if it's bytes
                if isinstance(phonemes, bytes):
                    phonemes = phonemes.decode('utf-8', errors='replace')
                # If it's a string, ensure it's valid UTF-8
                elif isinstance(phonemes, str):
                    # Replace problematic characters with their ASCII approximations
                    replacements = {
                        'É™': 'ə',
                        'ÊŠ': 'ʊ',
                        'Ê': 'ʃ',
                        'æ': 'ae'
                    }
                    for old, new in replacements.items():
                        phonemes = phonemes.replace(old, new)
                
                print(f"Debug - Processed phonemes: {repr(phonemes)}")
            except Exception as e:
                print(f"Debug - Encoding error: {str(e)}")
                # Last resort: strip to ASCII
                phonemes = ''.join(c for c in str(phonemes) if ord(c) < 128)
        
        return audio, phonemes
    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback
        traceback.print_exc()
        return None, None