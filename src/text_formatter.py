"""Format natural language text to training data format using GROQ API."""

import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from .language_detector import LanguageDetector
from .translator import VietnameseTranslator


class TextFormatter:
    """Convert natural language to training caption format using GROQ API."""
    
    def __init__(self):
        # Load environment variables
        env_path = Path(__file__).parent / '.env'
        if not env_path.exists():
            # Try parent directory
            env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(env_path)
        
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        self.client = Groq(api_key=api_key)
        
        # Initialize language detection and translation
        self.language_detector = LanguageDetector()
        self.translator = VietnameseTranslator()
        
        # Load features (data/ is at project root, not in src/)
        features_path = Path(__file__).parent.parent / 'data' / 'features.txt'
        with open(features_path, 'r', encoding='utf-8') as f:
            self.features_text = f.read()
    
    def format_text(self, user_input: str, auto_translate: bool = True) -> tuple[str, dict]:
        """
        Convert natural language to training caption format.
        Supports Vietnamese auto-translation.
        
        Args:
            user_input: Natural language face description (Vietnamese or English)
            auto_translate: Whether to auto-translate Vietnamese to English
            
        Returns:
            Tuple of (formatted_text, info_dict)
            - formatted_text: Formatted caption in training data style
            - info_dict: Dict with language detection and translation info
        """
        # Detect language
        lang_info = self.language_detector.detect_with_info(user_input)
        
        info = {
            "original_language": lang_info["language"],
            "language_name": lang_info["display_name"],
            "was_translated": False,
            "original_text": user_input,
            "translated_text": None,
        }
        
        text_to_format = user_input
        
        # Translate if Vietnamese and auto_translate is enabled
        if lang_info["is_vietnamese"] and auto_translate:
            try:
                translated = self.translator.translate(user_input)
                text_to_format = translated
                info["was_translated"] = True
                info["translated_text"] = translated
            except Exception as e:
                # If translation fails, use original text
                info["translation_error"] = str(e)
        
        # Format the text (Vietnamese original or English translated)
        prompt = self._build_prompt(text_to_format)
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a text formatter that converts natural language face descriptions into structured captions for a face generation model. Follow the exact format and vocabulary from the examples."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500,
            )
            
            formatted_text = response.choices[0].message.content.strip()
            return formatted_text, info
            
        except Exception as e:
            return f"Error formatting text: {str(e)}", info
    
    def _build_prompt(self, user_input: str) -> str:
        """Build prompt for GROQ API."""
        prompt = f"""Convert the following natural language face description into a structured caption format.

**IMPORTANT RULES:**
1. Start with "The woman..." or "The man..." based on gender
2. Follow these patterns:
   - "The woman/man has [feature]"
   - "She/He has [feature]"
   - "The [age] [appearance] woman/man..."
   - "She's/He's wearing [accessory]"
3. ONLY use features from this vocabulary list:
{self.features_text}

4. When writing features, do NOT use underscores (e.g., write "5 o'clock shadow" not "5_o'clock_shadow")
5. Common sentence structures:
   - "The woman has high cheekbones."
   - "She has wavy hair which is brown in colour."
   - "She has big lips and pointy nose with arched eyebrows and a slightly open mouth."
   - "The smiling, young attractive woman has heavy makeup."
   - "She's wearing earrings and lipstick."

**EXAMPLE CONVERSIONS:**

Input: "A young woman with long blonde hair and blue eyes, smiling"
Output: The woman has straight hair which is blond in colour. She has blue eyes. The smiling, young attractive woman has heavy makeup.

Input: "Man with beard and short brown hair"
Output: The man has straight hair which is brown in colour. He sports a goatee. The man looks young.

Input: "Attractive young woman with wavy black hair, wearing earrings and lipstick"
Output: The woman has wavy hair which is black in colour. The young attractive woman has heavy makeup. She's wearing earrings and lipstick.

**NOW CONVERT THIS:**
Input: {user_input}
Output:"""
        
        return prompt


# Test function
if __name__ == "__main__":
    formatter = TextFormatter()
    
    test_inputs = [
        "A young woman with long blonde hair and blue eyes, smiling",
        "Man with 5 o'clock shadow and brown hair",
        "Attractive woman with high cheekbones and heavy makeup",
        "Một cô gái trẻ với mái tóc dài màu vàng",
        "Người đàn ông có râu và tóc ngắn màu nâu",
    ]
    
    print("Testing Text Formatter with Auto-Translation\n" + "="*60)
    for text in test_inputs:
        print(f"\nInput: {text}")
        formatted, info = formatter.format_text(text, auto_translate=True)
        print(f"Language: {info['language_name']}")
        if info['was_translated']:
            print(f"Translated: {info['translated_text']}")
        print(f"Formatted: {formatted}")
