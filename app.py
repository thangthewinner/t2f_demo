"""Text-to-Face Generation Demo - Gradio Web UI."""

import gradio as gr
import torch
from pathlib import Path
from PIL import Image
import time

from src.face_detector import FaceDescriptionDetector
from src.text_formatter import TextFormatter
from src.language_detector import LanguageDetector
from src.translator import VietnameseTranslator
from src.model_inference import T2FInference


# ============================================================================
# GLOBAL INITIALIZATION
# ============================================================================

print("Initializing T2F Demo...")

# Checkpoint path
CHECKPOINT_PATH = Path(__file__).parent / "models" / "checkpoint_epoch0500.pt"

# Check if checkpoint exists
if not CHECKPOINT_PATH.exists():
    print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
    CHECKPOINT_PATH = None

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Initialize components
detector = FaceDescriptionDetector()
formatter = TextFormatter()
language_detector = LanguageDetector()
translator = VietnameseTranslator()
model = None

# Load sample texts
SAMPLE_TEXTS_PATH = Path(__file__).parent / "data" / "sample_texts.txt"
with open(SAMPLE_TEXTS_PATH, 'r', encoding='utf-8') as f:
    SAMPLE_TEXTS = [line.strip() for line in f if line.strip()]

# Model evaluation metrics (from checkpoint epoch 500)
MODEL_FSD = 1.0595  # Fréchet Inception Distance
MODEL_FSS = 42.25   # Face Similarity Score


def get_model():
    """Lazy load model on first use."""
    global model
    if model is None and CHECKPOINT_PATH is not None:
        print("Loading T2F model...")
        model = T2FInference(str(CHECKPOINT_PATH), device=DEVICE)
        print("✓ Model loaded!")
    return model


# ============================================================================
# GENERATION FUNCTION
# ============================================================================

def generate_face(user_input: str, use_text_rewrite: bool = True):
    """Generate face from text description."""
    
    if not user_input or not user_input.strip():
        return None, "⚠️ Please enter a face description", ""
    
    if CHECKPOINT_PATH is None or not CHECKPOINT_PATH.exists():
        return None, "❌ Model checkpoint not found", ""
    
    start_time = time.time()
    
    # Translation info display
    translation_info = ""
    text_to_validate = user_input
    
    # Detect language and translate if needed (BEFORE validation)
    # Auto-translate is always enabled
    try:
        detected_language = language_detector.get_language(user_input)
        
        # Check if language is supported
        if detected_language == "other":
            error_msg = "⚠️ Language not supported!\n\nPlease enter your description in:\n• English 🇬🇧\n• Vietnamese 🇻🇳"
            return None, error_msg, ""
        
        if detected_language == "vi":
            translation_info = f"🌏 Language: Tiếng Việt\n"
            try:
                translated_text = translator.translate(user_input)
                text_to_validate = translated_text
                translation_info += f"🔄 Translated to: {translated_text}\n"
            except Exception as e:
                translation_info += f"⚠️ Translation failed: {str(e)}\n"
        else:  # "en"
            translation_info = f"🌏 Language: English\n"
    except Exception as e:
        # If language detection fails, continue with original text
        text_to_validate = user_input
    
    # Validate input (using translated text if Vietnamese)
    is_face, detection_msg = detector.is_face_description(text_to_validate)
    if not is_face:
        suggestions = detector.get_suggestions(text_to_validate)
        suggestion_text = "\n\n💡 Try adding:\n" + "\n".join(f"  • {s}" for s in suggestions)
        return None, detection_msg + suggestion_text, ""
    
    # Format text with GROQ API if enabled
    if use_text_rewrite:
        try:
            # Use the already translated text if available
            formatted_text, lang_info = formatter.format_text(user_input, auto_translate=True)
            
            # Update translation info if not already set
            if not translation_info:
                if lang_info['original_language'] == 'vi':
                    translation_info = f"🌏 Language: {lang_info['language_name']}\n"
                    if lang_info['was_translated']:
                        translation_info += f"🔄 Translated to: {lang_info['translated_text']}\n"
                    elif 'translation_error' in lang_info:
                        translation_info += f"⚠️ Translation failed, using original text\n"
                else:
                    translation_info = f"🌏 Language: {lang_info['language_name']}\n"
                
        except Exception as e:
            formatted_text = text_to_validate
            detection_msg += f"\n⚠️ GROQ API unavailable, using original text"
    else:
        formatted_text = text_to_validate
    
    # Remove pipe | and format for display
    formatted_display = formatted_text.replace("|", "\n")
    
    # Generate image
    try:
        model_instance = get_model()
        if model_instance is None:
            return None, "❌ Failed to load model", formatted_display
        
        generated_image = model_instance.generate(formatted_text)
        generation_time = time.time() - start_time
        
        status = f"✅ Generated in {generation_time:.2f}s | Device: {DEVICE}"
        
        return generated_image, status, formatted_display
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}", formatted_display


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_demo(theme_mode="soft"):
    """Create simplified Gradio interface."""
    
    # Available themes
    themes = {
        "soft": gr.themes.Soft(),
        "default": gr.themes.Default(),
        "glass": gr.themes.Glass(),
        "monochrome": gr.themes.Monochrome(),
    }
    
    theme = themes.get(theme_mode, gr.themes.Soft())
    
    # Custom CSS for formatted text styling
    custom_css = """
    .formatted-text textarea {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: #e8f5e9 !important;
        color: #2e7d32 !important;
        border: 1px solid #81c784 !important;
        border-radius: 8px !important;
        padding: 12px !important;
        line-height: 1.6 !important;
    }
    .dark .formatted-text textarea {
        background-color: #1b5e20 !important;
        color: #a5d6a7 !important;
        border: 1px solid #388e3c !important;
    }
    .theme-toggle {
        position: absolute;
        top: 16px;
        right: 16px;
        z-index: 1000;
    }
    """
    
    with gr.Blocks(title="Text-to-Face Demo", theme=theme, css=custom_css) as demo:
        
        # Header with theme toggle
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("""
                # 🎨 Text-to-Face Generation
                Generate realistic 1024×1024 face images from text descriptions.
                
                **Supports Vietnamese 🇻🇳 and English 🇬🇧 input** - Automatic translation included!
                """)
            with gr.Column(scale=1, elem_classes="theme-toggle"):
                theme_btn = gr.Button(
                    "🌓 Theme",
                    size="sm",
                    elem_id="theme-toggle-btn"
                )
        
        gr.Markdown("---")
        
        with gr.Row():
            # Left: Input
            with gr.Column(scale=1):
                user_input = gr.Textbox(
                    label="Face Description (Vietnamese or English)",
                    placeholder="e.g., Young woman with wavy blonde hair and blue eyes\nhoặc: Một cô gái trẻ với mái tóc dài màu vàng",
                    lines=4,
                )
                
                sample_dropdown = gr.Dropdown(
                    label="Sample Texts",
                    choices=SAMPLE_TEXTS,
                    value=None,
                )
                
                formatted_text_box = gr.Textbox(
                    label="Formatted Text (Model Input)",
                    lines=4,
                    interactive=False,
                    elem_classes="formatted-text",
                )
                
                with gr.Row():
                    use_rewrite = gr.Checkbox(
                        label="AI Text Rewrite (based on training text)",
                        value=True,
                        info="Use GROQ API to format text"
                    )
                
                generate_btn = gr.Button("🎨 Generate Face", variant="primary", size="lg")
            
            # Right: Output
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Generated Face",
                    type="pil",
                    height=512,
                    format="png",
                )
                
                status_box = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False,
                )
        
        # Footer with metrics explanation
        with gr.Accordion("ℹ️ About Metrics", open=False):
            gr.Markdown("""
            **FSD (Fréchet Inception Distance):** Measures similarity between generated and real images.
            - Lower is better (closer to real data distribution)
            - Our model: **1.0595**
            
            **FSS (Face Similarity Score):** Measures text-to-face alignment accuracy.
            - Higher is better (more accurate feature matching)
            - Our model: **42.25**
            """)
        
        gr.Markdown("""
        ---
        **💡 Tip:** Best results with structured descriptions (hair color/style, eyes, nose, lips, age, accessories).
        """)
        
        # Event handlers
        sample_dropdown.change(
            fn=lambda x: x,
            inputs=[sample_dropdown],
            outputs=[user_input],
        )
        
        generate_btn.click(
            fn=generate_face,
            inputs=[user_input, use_rewrite],
            outputs=[output_image, status_box, formatted_text_box],
        )
        
        # Theme toggle functionality
        theme_btn.click(
            None,
            None,
            None,
            js="""() => {
                document.body.classList.toggle('dark');
                return [];
            }"""
        )
    
    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if CHECKPOINT_PATH is None or not CHECKPOINT_PATH.exists():
        print("\n" + "="*70)
        print("⚠️  WARNING: Checkpoint file not found!")
        print("="*70)
        print(f"Expected: {Path(__file__).parent / 'models' / 'checkpoint_epoch0500.pt'}")
        print("\nPlease download checkpoint and place in models/ folder.")
        print("="*70 + "\n")
    
    # Theme options: "soft" (default), "default", "glass", "monochrome"
    # Change theme by editing the parameter below
    THEME = "soft"  # Options: soft, default, glass, monochrome
    
    demo = create_demo(theme_mode=THEME)
    
    print("\n" + "="*70)
    print("🚀 Launching Text-to-Face Demo")
    print(f"Theme: {THEME.capitalize()}")
    print("="*70)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
