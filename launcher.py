import os
import sys

def main():
    # Get the absolute path to the project directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Add the project root to Python path
    sys.path.insert(0, project_root)
    
    # Set Hugging Face cache directory
    os.environ['HF_HOME'] = os.path.join(project_root, 'my_model')
    
    try:
        print("Importing gradio_interface...")
        import gradio_interface
        
        print("Creating Gradio interface...")
        # Create the interface
        interface = gradio_interface.create_interface()
        
        print("Launching Gradio interface...")
        # Launch the interface
        interface.launch(
        server_name="127.0.0.9",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False,             # Enable Gradio sharing link
        inbrowser=True,          # Automatically open in browser
        show_error=True
    )

    except Exception as e:
        print("\nError occurred:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()