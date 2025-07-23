# src/inference_service.py
from fastapi import FastAPI, File, UploadFile, Form
from ingestion import ingest_text, ingest_image, ingest_audio
from preprocess import preprocess_text, preprocess_image, preprocess_audio

app = FastAPI(title="Healthcare AI Assistant", description="Multimodal healthcare AI prediction API")

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {"message": "Healthcare AI Assistant API is running", "status": "healthy"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is operational"}

@app.post("/predict/")
async def predict(
    text: str = Form(...),
    image: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    # Ingestion - handle text as string, others as file objects
    text_data = ingest_text(text)
    img = ingest_image(image.file)
    aud_wf = ingest_audio(audio.file)
    
    # Preprocess
    text_p = preprocess_text(text_data)
    img_p = preprocess_image(img)
    aud_p = preprocess_audio(aud_wf)
    
    # Here you would pass into your model
    # result = model.forward(...)
    # For demo, return obfuscated "diagnosis"
    return {"diagnosis": "Healthy", "confidence": 0.95}
