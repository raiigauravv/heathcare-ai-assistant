# src/ingestion.py - Hugging Face Spaces compatible
import os
from PIL import Image
import librosa
import numpy as np
from typing import Union, Tuple
import logging

logger = logging.getLogger(__name__)

def ingest_text(text_input: Union[str, object]) -> str:
    """
    Handle text input - can be a string or file object
    Compatible with Gradio text inputs
    """
    if isinstance(text_input, str):
        return text_input.strip()
    elif hasattr(text_input, 'read'):
        # Handle file-like objects
        content = text_input.read()
        if isinstance(content, bytes):
            return content.decode('utf-8').strip()
        return str(content).strip()
    else:
        return str(text_input).strip()

def ingest_image(image_input: Union[str, Image.Image, np.ndarray]) -> Union[Image.Image, None]:
    """
    Handle image input from various sources
    Compatible with Gradio image inputs
    Supports: JPG, JPEG, PNG, BMP, TIFF, WEBP
    """
    try:
        if isinstance(image_input, str):
            # File path
            if os.path.exists(image_input):
                # Check file extension
                file_ext = os.path.splitext(image_input)[1].lower()
                supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
                
                if file_ext not in supported_formats:
                    logger.error(f"Unsupported image format: {file_ext}. Supported formats: {', '.join(supported_formats)}")
                    return None
                
                # Try to open and convert to RGB
                img = Image.open(image_input)
                return img.convert('RGB')
            else:
                logger.error(f"Image file not found: {image_input}")
                return None
                
        elif isinstance(image_input, Image.Image):
            # PIL Image object
            return image_input.convert('RGB')
            
        elif isinstance(image_input, np.ndarray):
            # Numpy array from Gradio
            return Image.fromarray(image_input).convert('RGB')
            
        else:
            logger.error(f"Unsupported image input type: {type(image_input)}")
            return None
            
    except Exception as e:
        logger.error(f"Error ingesting image: {e}")
        return None

def ingest_audio(audio_input: Union[str, Tuple[int, np.ndarray]]) -> Union[Tuple[np.ndarray, int], None]:
    """
    Handle audio input from various sources
    Compatible with Gradio audio inputs
    
    Returns:
        Tuple of (waveform, sample_rate) or None if error
    """
    try:
        if isinstance(audio_input, str):
            # File path
            if os.path.exists(audio_input):
                waveform, sample_rate = librosa.load(audio_input, sr=None, mono=True)
                return waveform, sample_rate
            else:
                logger.error(f"Audio file not found: {audio_input}")
                return None
                
        elif isinstance(audio_input, tuple) and len(audio_input) == 2:
            # Gradio audio format: (sample_rate, numpy_array)
            sample_rate, waveform = audio_input
            
            # Convert to float and normalize if needed
            if waveform.dtype == np.int16:
                waveform = waveform.astype(np.float32) / 32768.0
            elif waveform.dtype == np.int32:
                waveform = waveform.astype(np.float32) / 2147483648.0
                
            # Ensure mono audio
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)
                
            return waveform, sample_rate
            
        else:
            logger.error(f"Unsupported audio input type: {type(audio_input)}")
            return None
            
    except Exception as e:
        logger.error(f"Error ingesting audio: {e}")
        return None

# Utility functions for Hugging Face Spaces
def validate_inputs(text: str, image=None, audio=None) -> bool:
    """
    Validate inputs before processing
    """
    if not text or len(text.strip()) < 5:
        return False
    return True

def get_file_info(file_path: str) -> dict:
    """
    Get file information for debugging
    """
    if not file_path or not os.path.exists(file_path):
        return {"exists": False, "path": file_path}
    
    try:
        stat = os.stat(file_path)
        return {
            "exists": True,
            "path": file_path,
            "size": stat.st_size,
            "extension": os.path.splitext(file_path)[1].lower()
        }
    except Exception as e:
        return {"exists": False, "path": file_path, "error": str(e)}
