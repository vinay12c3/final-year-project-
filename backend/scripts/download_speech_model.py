"""Downloads the offline speech-to-text model (Whisper base.en, ~145 MB) once.

    python -m scripts.download_speech_model
"""
from signvision import speech

if __name__ == "__main__":
    if speech.is_downloaded():
        print(f"Speech model already downloaded: {speech.MODEL_DIR}")
    else:
        print("Downloading speech model (about 145 MB, one time only)...")
        print(f"Saved to {speech.download()}")
