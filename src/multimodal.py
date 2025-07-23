# src/multimodal.py - Advanced Multimodal Healthcare AI Model
import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, 
    CLIPModel, CLIPProcessor,
    WhisperModel, WhisperProcessor
)
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedMultimodalHealthcareModel(nn.Module):
    """
    Enterprise-grade multimodal model for healthcare AI
    Supports text, image, and audio inputs with SOTA models
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        self.config = config or {
            "text_model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            "vision_model": "openai/clip-vit-base-patch32", 
            "audio_model": "openai/whisper-base",
            "hidden_dim": 768,
            "num_classes": 10,  # Adjust based on your healthcare categories
            "dropout": 0.1
        }
        
        self._initialize_models()
        self._initialize_fusion_layers()
        
    def _initialize_models(self):
        """Initialize pre-trained models for each modality"""
        try:
            # Text: Biomedical BERT for healthcare domain
            self.text_model = AutoModel.from_pretrained(self.config["text_model"])
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.config["text_model"])
            
            # Vision: CLIP for medical image understanding
            self.vision_model = CLIPModel.from_pretrained(self.config["vision_model"])
            self.vision_processor = CLIPProcessor.from_pretrained(self.config["vision_model"])
            
            # Audio: Whisper for speech/audio analysis
            self.audio_model = WhisperModel.from_pretrained(self.config["audio_model"])
            self.audio_processor = WhisperProcessor.from_pretrained(self.config["audio_model"])
            
            logger.info("Successfully loaded all pre-trained models")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Fallback to simpler models
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        """Fallback to simpler models if SOTA models fail to load"""
        self.text_model = AutoModel.from_pretrained('bert-base-uncased')
        self.text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Simple CNN for images
        self.vision_model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, self.config["hidden_dim"])
        )
        
        # Simple LSTM for audio
        self.audio_model = nn.LSTM(
            input_size=80,  # Mel spectrogram features
            hidden_size=self.config["hidden_dim"],
            batch_first=True
        )
        
        logger.warning("Using fallback models due to loading issues")
    
    def _initialize_fusion_layers(self):
        """Initialize cross-modal fusion layers"""
        hidden_dim = self.config["hidden_dim"]
        
        # Attention-based fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=self.config["dropout"]
        )
        
        # Feature projection layers
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.vision_proj = nn.Linear(hidden_dim, hidden_dim)
        self.audio_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.config["dropout"]),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config["dropout"]),
            nn.Linear(hidden_dim, self.config["num_classes"])
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, 1),
            nn.Sigmoid()
        )
    
    def encode_text(self, text_input):
        """Encode text using biomedical BERT"""
        try:
            if hasattr(self, 'text_tokenizer'):
                tokens = self.text_tokenizer(
                    text_input, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                outputs = self.text_model(**tokens)
                return outputs.last_hidden_state.mean(dim=1)  # Global average pooling
            else:
                # Fallback encoding
                return self.text_model(text_input).last_hidden_state.mean(dim=1)
        except Exception as e:
            logger.error(f"Text encoding error: {e}")
            return torch.zeros(1, self.config["hidden_dim"])
    
    def encode_vision(self, image_input):
        """Encode images using CLIP or fallback CNN"""
        try:
            if hasattr(self.vision_model, 'get_image_features'):
                # CLIP encoding
                if hasattr(self, 'vision_processor'):
                    processed = self.vision_processor(images=image_input, return_tensors="pt")
                    features = self.vision_model.get_image_features(**processed)
                else:
                    features = self.vision_model.get_image_features(image_input)
                return features
            else:
                # Fallback CNN encoding
                return self.vision_model(image_input)
        except Exception as e:
            logger.error(f"Vision encoding error: {e}")
            return torch.zeros(1, self.config["hidden_dim"])
    
    def encode_audio(self, audio_input):
        """Encode audio using Whisper or fallback LSTM"""
        try:
            if hasattr(self.audio_model, 'encoder'):
                # Whisper encoding
                if hasattr(self, 'audio_processor'):
                    processed = self.audio_processor(audio_input, return_tensors="pt")
                    features = self.audio_model.encoder(**processed)
                    return features.last_hidden_state.mean(dim=1)
                else:
                    return self.audio_model.encoder(audio_input).last_hidden_state.mean(dim=1)
            else:
                # Fallback LSTM encoding
                output, (hidden, _) = self.audio_model(audio_input)
                return hidden[-1]  # Use last hidden state
        except Exception as e:
            logger.error(f"Audio encoding error: {e}")
            return torch.zeros(1, self.config["hidden_dim"])
    
    def forward(self, text_input, image_input, audio_input):
        """
        Forward pass through the multimodal model
        
        Args:
            text_input: Text data (string or tokens)
            image_input: Image tensor [B, C, H, W]
            audio_input: Audio tensor [B, T, F] or raw waveform
            
        Returns:
            Dict containing predictions and confidence scores
        """
        # Encode each modality
        text_features = self.encode_text(text_input)
        vision_features = self.encode_vision(image_input) 
        audio_features = self.encode_audio(audio_input)
        
        # Project features to common space
        text_proj = self.text_proj(text_features)
        vision_proj = self.vision_proj(vision_features)
        audio_proj = self.audio_proj(audio_features)
        
        # Cross-modal attention fusion
        combined_features = torch.stack([text_proj, vision_proj, audio_proj], dim=0)
        attended_features, _ = self.cross_attention(
            combined_features, combined_features, combined_features
        )
        
        # Flatten for classification
        fused_features = attended_features.flatten(start_dim=1)
        
        # Generate predictions
        logits = self.classifier(fused_features)
        confidence = self.confidence_head(fused_features)
        
        return {
            "logits": logits,
            "predictions": torch.softmax(logits, dim=-1),
            "confidence": confidence,
            "features": {
                "text": text_features,
                "vision": vision_features, 
                "audio": audio_features,
                "fused": fused_features
            }
        }

class SimpleMultimodalModel(nn.Module):
    """Legacy simple model for backwards compatibility"""
    def __init__(self):
        super().__init__()
        self.text_model = AutoModel.from_pretrained('bert-base-uncased')

    def forward(self, text_tokens, image_tensor, audio_tensor):
        text_out = self.text_model(**text_tokens).last_hidden_state.mean(1)
        img_out = image_tensor.mean([1,2,3]) if image_tensor.dim() > 2 else image_tensor
        audio_out = audio_tensor.mean(1) if audio_tensor.dim() > 1 else audio_tensor
        all_feat = torch.cat([text_out, img_out.unsqueeze(1), audio_out.unsqueeze(1)], dim=1)
        return all_feat

# Factory function for model selection
def create_model(model_type: str = "advanced", config: Optional[Dict] = None):
    """
    Factory function to create appropriate model
    
    Args:
        model_type: "advanced" or "simple"
        config: Model configuration dictionary
        
    Returns:
        Instantiated model
    """
    if model_type == "advanced":
        return AdvancedMultimodalHealthcareModel(config)
    else:
        return SimpleMultimodalModel()
