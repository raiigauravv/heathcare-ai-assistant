# src/preprocess.py
import torchvision.transforms as T

def preprocess_text(text):
    """Preprocess text input"""
    return text.lower().strip()

def preprocess_image(image):
    """Preprocess PIL image"""
    transform = T.Compose([
        T.Resize((224, 224)), 
        T.ToTensor()
    ])
    return transform(image)

def preprocess_audio(audio_tuple):
    """Preprocess audio waveform"""
    waveform, rate = audio_tuple
    return waveform.mean(0)  # downmix to mono
