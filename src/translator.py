"""Translate Vietnamese to English using GROQ API."""

import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv


class VietnameseTranslator:
    """Translate Vietnamese face descriptions to English using GROQ API."""
    
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
    
    def translate(self, vietnamese_text: str) -> str:
        """
        Translate Vietnamese to English.
        
        Args:
            vietnamese_text: Vietnamese face description
            
        Returns:
            English translation
        """
        prompt = self._build_translation_prompt(vietnamese_text)
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional Vietnamese-to-English translator specializing in face and appearance descriptions. Translate naturally and accurately, maintaining all descriptive details."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=500,
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            # Remove any prefix like "Translation:" if the model adds it
            if ":" in translated_text and len(translated_text.split(":")[0]) < 20:
                translated_text = translated_text.split(":", 1)[1].strip()
            
            return translated_text
            
        except Exception as e:
            raise Exception(f"Translation failed: {str(e)}")
    
    def _build_translation_prompt(self, vietnamese_text: str) -> str:
        """Build prompt for translation."""
        prompt = f"""Translate this Vietnamese face description to English. Keep it natural and descriptive.

**RULES:**
1. Translate accurately, maintaining all details
2. Keep it natural and fluent
3. Focus on physical appearance descriptions
4. Output ONLY the English translation, no explanations

**EXAMPLES:**

Vietnamese: Một cô gái trẻ với mái tóc dài màu vàng và đôi mắt xanh
English: A young woman with long blonde hair and blue eyes

Vietnamese: Người đàn ông có râu và tóc ngắn màu nâu
English: A man with a beard and short brown hair

Vietnamese: Cô gái xinh đẹp với khuôn mặt trái xoan và đôi môi đầy đặn
English: An attractive woman with an oval face and full lips

Vietnamese: Người phụ nữ trung niên đang cười, đeo hoa tai
English: A middle-aged woman smiling, wearing earrings

**NOW TRANSLATE:**
Vietnamese: {vietnamese_text}
English:"""
        
        return prompt


# Test function
if __name__ == "__main__":
    translator = VietnameseTranslator()
    
    test_inputs = [
        "Một cô gái trẻ với mái tóc dài màu vàng",
        "Người đàn ông có râu và tóc ngắn",
        "Cô gái xinh đẹp với đôi mắt xanh và má hồng",
        "Người phụ nữ trung niên đang cười, đeo hoa tai và son môi",
    ]
    
    print("Vietnamese to English Translation Tests\n" + "="*60)
    for text in test_inputs:
        print(f"\nVietnamese: {text}")
        try:
            translated = translator.translate(text)
            print(f"English: {translated}")
        except Exception as e:
            print(f"Error: {e}")
