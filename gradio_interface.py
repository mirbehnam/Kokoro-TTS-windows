import os

# Set HF_HOME to a folder named "my_model" inside your project directory
project_dir = os.path.dirname(os.path.abspath(__file__))  # Path to the script directory
hf_home = os.path.join(project_dir, "my_model")  # Your custom HF cache directory
os.environ["HF_HOME"] = hf_home

# Ensure the directory exists
os.makedirs(hf_home, exist_ok=True)
print(f"Hugging Face cache directory is set to: {hf_home}")

import gradio as gr
import os
import sys
import platform
from datetime import datetime
import shutil
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment
import torch
from models import (
    list_available_voices, build_model, load_voice,
    generate_speech, load_and_validate_voice
)

# Global configuration
CONFIG_FILE = "tts_config.json"  # Stores user preferences and paths
DEFAULT_OUTPUT_DIR = "outputs"    # Directory for generated audio files
SAMPLE_RATE = 22050

# Initialize model globally
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = None

def get_available_voices():
    """Get list of available voice models."""
    try:
        voices = list_available_voices()
        print("Available voices:", voices)
        return voices
    except Exception as e:
        print(f"Error retrieving voices: {e}")
        return []

def convert_audio(input_path: str, output_path: str, format: str):
    """Convert audio to specified format using pydub."""
    try:
        audio = AudioSegment.from_wav(input_path)
        if format == "mp3":
            audio.export(output_path, format="mp3", bitrate="192k")
        elif format == "aac":
            audio.export(output_path, format="aac", bitrate="192k")
        else:  # wav
            shutil.copy2(input_path, output_path)
        return True
    except Exception as e:
        print(f"Error converting audio: {e}")
        return False

def generate_tts_with_logs(voice_name, text, format, speed):
    """Generate TTS audio with real-time logging and format conversion."""
    global model

    if not text.strip():
        return "❌ Error: Text required", None

    logs_text = ""
    try:
        # Initialize model if not done yet
        if model is None:
            logs_text += "Loading model...\n"
            model = build_model("kokoro-v0_19.pth", device)

        # Load voice
        logs_text += f"Loading voice: {voice_name}\n"
        yield logs_text, None
        voice = load_and_validate_voice(voice_name, device)

        # Generate speech
        logs_text += f"Generating speech for: '{text}'\n"
        yield logs_text, None
        audio, phonemes = generate_speech(model, text, voice, lang='a', device=device, speed=speed)

        if audio is not None and phonemes:
            try:
                logs_text += f"Generated phonemes: {phonemes}\n"
            except UnicodeEncodeError:
                logs_text += "Generated phonemes: [Unicode display error]\n"

            # Save temporary WAV file
            temp_wav = "output.wav"
            sf.write(temp_wav, audio, SAMPLE_RATE)

            # Convert to desired format
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output_{timestamp}.{format}"
            os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
            output_path = Path(DEFAULT_OUTPUT_DIR) / filename

            if convert_audio(temp_wav, str(output_path), format):
                logs_text += f"✅ Saved: {output_path}\n"
                os.remove(temp_wav)
                yield logs_text, str(output_path)
            else:
                logs_text += "❌ Audio conversion failed\n"
                yield logs_text, None
        else:
            logs_text += "❌ Failed to generate audio\n"
            yield logs_text, None

    except Exception as e:
        logs_text += f"❌ Error: {str(e)}\n"
        yield logs_text, None

def create_interface(server_name="0.0.0.0", server_port=7860):
    """Create and configure Gradio interface with network sharing capabilities.
    
    Creates a web interface with:
    - Text input area
    - Voice model selection
    - Audio format selection (WAV/MP3/AAC)
    - Real-time progress logging
    - Audio playback and download
    - Example inputs for testing
    
    Args:
        server_name (str): Server address for network sharing (default: "0.0.0.0" for all interfaces)
        server_port (int): Port number to serve on (default: 7860)
    
    Returns:
        gr.Blocks: Configured Gradio interface ready for launching
    """
    theme = gr.themes.Base(
        primary_hue="zinc",
        secondary_hue="slate",
        neutral_hue="zinc",
        font=gr.themes.GoogleFont("Inter")
    )

    with gr.Blocks(theme=theme) as demo:
        gr.HTML(
            """
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 style="font-size: 2.5em; margin-bottom: 0.5rem;">🎙️ Kokoro-TTS Local Generator</h1>
                
          <!-- YouTube-like Channel Section -->
<div style="display: flex; justify-content: center; align-items: center; background-color: #fafafa; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); margin-bottom: 2rem; width: 100%; max-width: 700px; margin: 0 auto;">
    <!-- Channel Profile Image and Info Section -->
    <div style="display: flex; align-items: center; gap: 1.5rem; width: 100%;">
        <!-- Channel Profile Image -->
        <img src="data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAKAAAACgCAYAAACLz2ctAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAxRpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDkuMS1jMDAyIDc5LmYzNTRlZmM3MCwgMjAyMy8xMS8wOS0xMjowNTo1MyAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6NTQxOTk2NEVENUUwMTFFRkFDQ0E4ODRCOTQ5Qzc3NDgiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6NTQxOTk2NERENUUwMTFFRkFDQ0E4ODRCOTQ5Qzc3NDgiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIDIwMjQgV2luZG93cyI+IDx4bXBNTTpEZXJpdmVkRnJvbSBzdFJlZjppbnN0YW5jZUlEPSJGNTNDRTMxRjZEMjlEQTAwOTM4MzcyNTdERUUxM0YxQSIgc3RSZWY6ZG9jdW1lbnRJRD0iRjUzQ0UzMUY2RDI5REEwMDkzODM3MjU3REVFMTNGMUEiLz4gPC9yZGY6RGVzY3JpcHRpb24+IDwvcmRmOlJERj4gPC94OnhtcG1ldGE+IDw/eHBhY2tldCBlbmQ9InIiPz7PDZcOAABScUlEQVR42ux9B4AdZbX/b/rcun2TTSeEhJKA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" alt="Channel Logo" style="width: 60px; height: 60px; border-radius: 50%; border: 2px solid #ccc; object-fit: cover;">
        
        <!-- Channel Info -->
        <div style="text-align: left; flex-grow: 1;">
            <h3 style="margin: 0; font-size: 1.5rem; font-weight: 600; color: #333;">HowToIn1minute</h3>
            <p style="font-size: 1rem; color: #555; margin-top: 0.5rem;">Follow my YouTube page to get the best of AI. 😉</p>
        </div>
        
       <!-- Subscribe Button -->
<a href="https://www.youtube.com/@HowToIn1Minute" target="_blank" 
   style="background-color: #FF0000; color: white; padding: 10px 20px; font-size: 1.1rem; 
          text-decoration: none; border-radius: 4px; font-weight: 600; display: inline-flex; 
          align-items: center; gap: 12px; box-shadow: 0 4px 8px rgba(255, 0, 0, 0.2); 
          transition: background-color 0.3s, box-shadow 0.3s;">


    
    Subscribe
</a>

    </div>
</div>


                <!-- Instructions and Introduction Section -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; background: rgba(0,0,0,0.05); padding: 1.5rem; border-radius: 8px; margin-top: 1rem;">
                    <div style="text-align: left;">
                        <h3>✨ Instructions</h3>
                        <p>1. Type or paste your text into the input box</p>
                        <p>2. Choose a voice from the dropdown menu</p>
                        <p>3. Click Generate and wait for processing</p>
                        <p>4. Play or download your generated audio</p>
                    </div>
                    <div style="text-align: left; border-left: 1px solid rgba(255,255,255,0.1); padding-left: 2rem;">
                        <h3>Introduction</h3>
                        <p>By Youtube channel <a href="https://www.youtube.com/@HowToIn1Minute">HowToIn1Minute</a></p>
                        <p>A local text-to-speech system using the Kokoro-82M model for natural-sounding voice synthesis.</p>
                        <p>Based on <a href="https://github.com/PierrunoYT/Kokoro-TTS-Local">Kokoro-TTS-Local</a> by <a href="https://github.com/PierrunoYT">PierrunoYT</a></p>
                        <p>Model: <a href="https://huggingface.co/hexgrad/Kokoro-82M">Kokoro-82M</a> by <a href="https://huggingface.co/hexgrad">hexgrad</a></p>
                    </div>
                </div>
            </div>
            """
        )
        
        text_input = gr.Textbox(
            label="✍️ Text to Synthesize",
            placeholder="Enter text here...",
            lines=3
        )
        
        generate_button = gr.Button("🔊 Generate", variant="primary")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    voice = gr.Dropdown(
                        choices=get_available_voices(),
                        label="🗣️ Select Voice",
                        value="af_bella"
                    )
                    format = gr.Radio(
                        choices=["wav", "mp3", "aac"],
                        label="🎵 Output Format",
                        value="wav"
                    )
                    speed = gr.Slider(
                        minimum=0.5, maximum=2.0, step=0.1, value=1.0,
                        label="🎚️ Voice Speed"
                    )
            
            with gr.Column(scale=2):
                audio_output = gr.Audio(
                    label="🎧 Output",
                    type="filepath"
                )
        
        logs_output = gr.Textbox(
            label="📋 Process Log",
            lines=8,
            interactive=False
        )
        
        generate_button.click(
            fn=generate_tts_with_logs,
            inputs=[voice, text_input, format, speed],
            outputs=[logs_output, audio_output]
        )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="localhost",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False,             # Enable Gradio sharing link
        inbrowser=True,          # Automatically open in browser
        show_error=True
    )
