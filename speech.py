# speech.py
import os
import whisper
import tempfile
from fastapi import UploadFile

# Load the Whisper model once (outside the function to avoid reloading)
model = whisper.load_model("base")

def speech_to_text(file: UploadFile) -> str:
    """
    Convert an uploaded audio file into text using Whisper Tiny.
    """
    # Read the file bytes from the UploadFile
    audio_bytes = file.file.read()
    
    # Determine the file extension from the original filename (default to .wav)
    suffix = ".wav"
    if file.filename and "." in file.filename:
        suffix = os.path.splitext(file.filename)[1]  # includes the dot
    
    # Create a temporary file; on Windows, we disable automatic deletion so that we can close it
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.close()  # Close the file so ffmpeg can access it

        # Load the Whisper Tiny model. (Consider caching this model on app startup in production.)
        model = whisper.load_model("tiny")

        # Transcribe the audio file.
        result = model.transcribe(tmp.name, language="en")
        return result.get("text", "")
    finally:
        # Delete the temporary file
        try:
            os.unlink(tmp.name)
        except Exception as e:
            # Log the exception if needed; we don't want to fail the request if cleanup fails.
            print(f"Failed to delete temporary file {tmp.name}: {e}")
