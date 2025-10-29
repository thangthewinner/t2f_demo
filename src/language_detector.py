"""Detect if input text is Vietnamese or English using GROQ API."""

import os
from pathlib import Path
from typing import Tuple
from groq import Groq
from dotenv import load_dotenv


class LanguageDetector:
    """Detect Vietnamese vs English text using GROQ API."""
    
    # Vietnamese-specific characters (for fallback)
    VIETNAMESE_CHARS = set('ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ')
    
    def __init__(self):
        # Load environment variables
        env_path = Path(__file__).parent / '.env'
        if not env_path.exists():
            # Try parent directory
            env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(env_path)
        
        api_key = os.getenv('GROQ_API_KEY')
        if api_key:
            self.client = Groq(api_key=api_key)
            self.use_api = True
        else:
            self.client = None
            self.use_api = False
            print("⚠️ GROQ_API_KEY not found, using fallback heuristic method")
    
    def is_vietnamese(self, text: str) -> Tuple[bool, float]:
        """
        Detect if text is Vietnamese.
        
        Args:
            text: Input text to detect
            
        Returns:
            Tuple of (is_vietnamese, confidence_score)
            - is_vietnamese: True if Vietnamese detected
            - confidence_score: 0.0 to 1.0 confidence level
        """
        if not text or not text.strip():
            return False, 0.0
        
        # Try API method first
        if self.use_api:
            try:
                return self._detect_with_api(text)
            except Exception as e:
                # Fallback to heuristic if API fails
                print(f"⚠️ API detection failed: {e}, using fallback")
                return self._detect_with_heuristic(text)
        else:
            # Use heuristic if no API key
            return self._detect_with_heuristic(text)
    
    def _detect_with_api(self, text: str) -> Tuple[bool, float]:
        """Detect language using GROQ API."""
        prompt = f"""Detect if the following text is Vietnamese or English. Reply with ONLY ONE WORD: "vietnamese" or "english".

Text: {text}

Language:"""
        
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a language detector. Reply with ONLY ONE WORD: 'vietnamese' or 'english'."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=10,
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Parse result
        if "vietnamese" in result or "việt" in result:
            return True, 0.95
        elif "english" in result:
            return False, 0.95
        else:
            # If unclear, fallback to heuristic
            return self._detect_with_heuristic(text)
    
    def _detect_with_heuristic(self, text: str) -> Tuple[bool, float]:
        """Fallback heuristic method for language detection."""
        text_lower = text.lower()
        
        # Count Vietnamese characters
        vietnamese_char_count = sum(1 for char in text_lower if char in self.VIETNAMESE_CHARS)
        
        # Count total letters (exclude spaces, punctuation)
        total_letters = sum(1 for char in text_lower if char.isalpha())
        
        if total_letters == 0:
            return False, 0.0
        
        # Calculate percentage of Vietnamese characters
        vietnamese_percentage = vietnamese_char_count / total_letters
        
        # Detection logic:
        # - If >= 5% Vietnamese chars → definitely Vietnamese
        # - If >= 2% Vietnamese chars → probably Vietnamese
        if vietnamese_percentage >= 0.05:
            confidence = min(1.0, vietnamese_percentage * 2)  # Scale confidence
            return True, confidence
        elif vietnamese_percentage >= 0.02:
            return True, 0.7
        
        # Additional check: Vietnamese common words
        vietnamese_words = [
            'một', 'cô', 'gái', 'người', 'với', 'tóc', 'mắt', 
            'màu', 'đẹp', 'trẻ', 'nam', 'nữ', 'khuôn', 'mặt'
        ]
        
        word_matches = sum(1 for word in vietnamese_words if word in text_lower)
        if word_matches >= 2:
            return True, 0.8
        
        return False, 0.0
    
    def get_language(self, text: str) -> str:
        """
        Get language name.
        
        Returns:
            "vi" for Vietnamese, "en" for English, "other" for other languages
        """
        if not text or not text.strip():
            return "other"
        
        # Try API method first
        if self.use_api:
            try:
                return self._detect_language_with_api(text)
            except Exception as e:
                # Fallback to heuristic if API fails
                is_viet, confidence = self.is_vietnamese(text)
                return "vi" if is_viet else "en"
        else:
            # Use heuristic if no API key
            is_viet, confidence = self.is_vietnamese(text)
            return "vi" if is_viet else "en"
    
    def _detect_language_with_api(self, text: str) -> str:
        """Detect specific language using GROQ API."""
        prompt = f"""Detect the language of the following text. Reply with ONLY ONE of these words: "vietnamese", "english", or "other".

Text: {text}

Language:"""
        
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a language detector. Reply with ONLY ONE WORD: 'vietnamese', 'english', or 'other' (for any other language)."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=10,
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Parse result
        if "vietnamese" in result or "việt" in result:
            return "vi"
        elif "english" in result:
            return "en"
        else:
            return "other"
    
    def detect_with_info(self, text: str) -> dict:
        """
        Detect language with detailed information.
        
        Returns:
            Dict with language, confidence, and display name
        """
        is_viet, confidence = self.is_vietnamese(text)
        
        return {
            "language": "vi" if is_viet else "en",
            "is_vietnamese": is_viet,
            "confidence": confidence,
            "display_name": "Tiếng Việt" if is_viet else "English"
        }


# Test function
if __name__ == "__main__":
    detector = LanguageDetector()
    
    test_cases = [
        "Một cô gái trẻ với mái tóc dài màu vàng",
        "A young woman with long blonde hair",
        "Người đàn ông có râu và tóc ngắn",
        "Man with beard and short hair",
        "Cô gái xinh đẹp với đôi mắt xanh",
    ]
    
    print("Language Detection Tests\n" + "="*60)
    for text in test_cases:
        result = detector.detect_with_info(text)
        print(f"\nText: {text}")
        print(f"Language: {result['display_name']} ({result['language']})")
        print(f"Confidence: {result['confidence']:.2%}")
