import torch
from typing import Optional, Tuple, List
from models import build_model, load_voice, generate_speech, list_available_voices
import argparse
from tqdm.auto import tqdm
import soundfile as sf
from pathlib import Path

# Constants
SAMPLE_RATE = 22050
DEFAULT_MODEL_PATH = 'kokoro-v0_19.pth'
DEFAULT_OUTPUT_FILE = 'output.wav'
DEFAULT_LANGUAGE = 'a'  # TODO: Document why this is 'a' or make configurable
DEFAULT_TEXT = "Hello, welcome to this text-to-speech test."

# Configure tqdm for better Windows console support
tqdm.monitor_interval = 0  # Disable monitor thread to prevent encoding issues

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

def main() -> None:
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Kokoro TTS Demo')
        parser.add_argument('--text', type=str, help='Text to synthesize (optional)')
        parser.add_argument('--voice', type=str, default='af_bella', help='Voice to use (default: af_bella)')
        parser.add_argument('--list-voices', action='store_true', help='List all available voices')
        parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help=f'Path to model file (default: {DEFAULT_MODEL_PATH})')
        parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_FILE, help=f'Output WAV file (default: {DEFAULT_OUTPUT_FILE})')
        parser.add_argument('--lang', type=str, default=DEFAULT_LANGUAGE, help=f'Language code (default: {DEFAULT_LANGUAGE})')
        args = parser.parse_args()

        if args.list_voices:
            voices = list_available_voices()
            print("\nAvailable voices:")
            for voice in voices:
                print(f"- {voice}")
            return

        # Set up device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Build model and load voice with progress indication
        print("\nLoading model...")
        with tqdm(total=1, desc="Building model") as pbar:
            model = build_model(args.model, device)
            pbar.update(1)
            
        print("\nLoading voice...")
        with tqdm(total=1, desc="Loading voice") as pbar:
            try:
                voice = load_and_validate_voice(args.voice, device)
                pbar.update(1)
            except ValueError as e:
                print(f"Error: {e}")
                return
        
        # Get text input
        if args.text:
            text = args.text
        else:
            print("\nEnter the text you want to convert to speech (or press Enter for default text):")
            text = input("> ").strip()
            if not text:
                text = DEFAULT_TEXT
        
        print(f"\nGenerating speech for: '{text}'")
        with tqdm(total=1, desc="Generating speech") as pbar:
            audio, phonemes = generate_speech(model, text, voice, lang=args.lang, device=device)
            pbar.update(1)
        
        if audio is not None:
            try:
                if phonemes:
                    try:
                        print(f"Generated phonemes: {phonemes}")
                    except UnicodeEncodeError:
                        print("Generated phonemes: [Unicode display error - phonemes were generated but cannot be displayed]")
                output_path = Path(args.output)
                sf.write(output_path, audio, SAMPLE_RATE)
                print(f"\nAudio saved to {output_path.absolute()}")
            except Exception as e:
                print(f"Error saving output: {e}")
                print("Audio generation was successful, but saving failed.")
        else:
            print("Error: Failed to generate audio")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if 'voice' in locals():
            del voice
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 