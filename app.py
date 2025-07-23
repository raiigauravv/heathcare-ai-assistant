"""
Healthcare AI Assistant - Hugging Face Spaces Demo
Multimodal healthcare AI with OpenAI integration
Updated: July 23, 2025 - Improved styling and visibility
"""

import gradio as gr
import os
from typing import Optional, Tuple
import logging
from datetime import datetime

# Import our modules
from src.ingestion import ingest_text, ingest_image, ingest_audio
from src.preprocess import preprocess_text
from src.openai_integration import OpenAIHealthcareAssistant

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI assistant
healthcare_ai = OpenAIHealthcareAssistant()

def multimodal_predict(
    text_symptoms: str,
    medical_image: Optional[str] = None,
    audio_file: Optional[str] = None,
    patient_name: str = "",
    patient_age: int = 30,
    patient_gender: str = "Not specified"
) -> Tuple[str, str, float]:
    """
    Main prediction function for multimodal healthcare analysis
    
    Args:
        text_symptoms: Patient's symptom description
        medical_image: Uploaded medical image (optional)
        audio_file: Uploaded audio recording (optional)
        patient_name: Patient's name for personalization
        patient_age: Patient's age
        patient_gender: Patient's gender
        
    Returns:
        Tuple of (analysis, recommendations, confidence_score)
    """
    try:
        logger.info("Starting multimodal healthcare prediction")
        
        # Initialize variables
        final_symptoms_text = ""
        audio_analysis = ""
        image_analysis = ""
        
        # Process audio first if provided (to get transcription)
        if audio_file:
            try:
                # Get audio transcription using Whisper
                audio_analysis = healthcare_ai.analyze_audio_symptoms(audio_file, text_symptoms or "Audio symptoms")
                
                # Extract transcription from audio analysis if available
                if "Transcript:" in audio_analysis:
                    # Try to extract the transcript text
                    transcript_part = audio_analysis.split("Transcript:")[1].split("\n")[0].strip()
                    final_symptoms_text = transcript_part if transcript_part else text_symptoms or ""
                
                logger.info("Audio processed successfully")
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                audio_analysis = "⚠️ Audio analysis unavailable"
        
        # Use text symptoms if provided, otherwise use transcription from audio
        if text_symptoms and len(text_symptoms.strip()) >= 10:
            final_symptoms_text = text_symptoms
        elif not final_symptoms_text and audio_file:
            # Audio-only mode: create a basic prompt
            final_symptoms_text = "Patient provided audio description of symptoms"
        
        # Validate we have some form of symptom description
        if not final_symptoms_text or (not audio_file and len(final_symptoms_text.strip()) < 10):
            return "❌ Error: Please provide detailed symptoms in text (at least 10 characters) OR upload an audio recording describing your symptoms", "", 0.0
        
        # Process text input
        processed_text = ingest_text(final_symptoms_text)
        cleaned_text = preprocess_text(processed_text)
        
        # Process image if provided
        if medical_image:
            try:
                img = ingest_image(medical_image)
                image_analysis = healthcare_ai.analyze_medical_image(img, final_symptoms_text)
                logger.info("Medical image processed successfully")
            except Exception as e:
                logger.error(f"Image processing error: {e}")
                image_analysis = "⚠️ Image analysis unavailable"
        
        # Generate comprehensive analysis using OpenAI
        analysis_result = healthcare_ai.comprehensive_health_analysis(
            symptoms=cleaned_text,
            image_analysis=image_analysis,
            audio_analysis=audio_analysis,
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender
        )
        
        # Extract results
        main_analysis = analysis_result.get("analysis", "Analysis unavailable")
        recommendations = analysis_result.get("recommendations", "No recommendations available")
        confidence = analysis_result.get("confidence", 0.7)
        
        logger.info("Prediction completed successfully")
        
        return main_analysis, recommendations, confidence
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return f"❌ Error during analysis: {str(e)}", "Please try again or consult a healthcare professional", 0.0

def create_demo_interface():
    """Create and configure the Gradio interface"""
    
    # Custom CSS for healthcare theme
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .warning {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: #000000;
        font-weight: 500;
    }
    """
    
    with gr.Blocks(css=css, title="Healthcare AI Assistant") as interface:
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🏥 Healthcare AI Assistant</h1>
            <p>Advanced multimodal AI for healthcare analysis and recommendations</p>
        </div>
        """)
        
        # Disclaimer
        gr.HTML("""
        <div class="warning">
            ⚠️ <strong>Medical Disclaimer:</strong> This is a demonstration tool for educational purposes only. 
            It does not provide medical advice and should not be used as a substitute for professional medical consultation, 
            diagnosis, or treatment. Always seek advice from qualified healthcare professionals.
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📝 Input Information")
                
                # Text input
                text_symptoms = gr.Textbox(
                    label="Describe Symptoms",
                    placeholder="Please describe your symptoms in detail (e.g., 'I have been experiencing chest pain and shortness of breath for the past 2 days...')",
                    lines=4,
                    max_lines=8
                )
                
                # Personal information
                patient_name = gr.Textbox(
                    label="Name (Optional)",
                    placeholder="Your first name for personalized interaction",
                    max_lines=1
                )
                
                with gr.Row():
                    patient_age = gr.Slider(
                        minimum=0,
                        maximum=120,
                        value=30,
                        step=1,
                        label="Age"
                    )
                    patient_gender = gr.Dropdown(
                        choices=["Male", "Female", "Other", "Not specified"],
                        value="Not specified",
                        label="Gender"
                    )
                
                # File uploads
                medical_image = gr.Image(
                    label="Medical Image (Optional)",
                    type="filepath",
                    height=200
                )
                
                audio_file = gr.Audio(
                    label="Audio Recording (Optional)",
                    type="filepath"
                )
                
                # Submit button
                submit_btn = gr.Button(
                    "🔍 Analyze Health Data", 
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("## 📊 Analysis Results")
                
                # Output displays
                analysis_output = gr.Textbox(
                    label="🔬 Health Analysis",
                    lines=8,
                    max_lines=12,
                    interactive=False
                )
                
                recommendations_output = gr.Textbox(
                    label="💡 Recommendations",
                    lines=6,
                    max_lines=10,
                    interactive=False
                )
                
                confidence_output = gr.Slider(
                    label="🎯 Confidence Score",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    interactive=False,
                    show_label=True
                )
                
                # Status indicator
                gr.HTML("""
                <div style="margin-top: 20px; padding: 15px; background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <strong>🤖 Powered by:</strong> OpenAI GPT-4, CLIP, and Whisper models<br>
                    <strong>⚡ Status:</strong> <span style="color: #28a745; font-weight: bold;">Online and Ready</span>
                </div>
                """)
        
        # Example inputs
        gr.Markdown("## 💡 Example Inputs")
        
        examples = [
            [
                "I have been experiencing persistent headaches for the past week, along with nausea and sensitivity to light. The pain is usually on one side of my head and gets worse with physical activity.",
                None,
                None,
                28,
                "Female"
            ],
            [
                "I've had a persistent cough for 3 weeks with yellow-green phlegm, fever, and difficulty breathing. I'm also feeling very tired.",
                None,
                None,
                45,
                "Male"
            ],
            [
                "I noticed a small, dark mole on my arm that has changed color and size over the past month. It's also slightly raised and sometimes itches.",
                None,
                None,
                35,
                "Other"
            ]
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[text_symptoms, medical_image, audio_file, patient_name, patient_age, patient_gender],
            label="Click on an example to try it out"
        )
        
        # Connect the interface
        submit_btn.click(
            fn=multimodal_predict,
            inputs=[text_symptoms, medical_image, audio_file, patient_name, patient_age, patient_gender],
            outputs=[analysis_output, recommendations_output, confidence_output],
            show_progress=True
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <p><strong>Healthcare AI Assistant</strong> - Demonstrating multimodal AI in healthcare</p>
            <p>Built with 💝 using OpenAI, Gradio, and Hugging Face Spaces</p>
            <p style="font-size: 0.8em; color: #666;">
                Remember: This is for demonstration purposes only. Always consult healthcare professionals for medical advice.
            </p>
        </div>
        """)
    
    return interface

# Launch the interface
if __name__ == "__main__":
    demo = create_demo_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=True
    )
