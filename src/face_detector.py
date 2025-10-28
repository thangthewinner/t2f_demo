"""Detect if text is a face description."""

from .vocabulary import VOCABULARY, ALL_KEYWORDS


class FaceDescriptionDetector:
    """Detect if input text describes a face/person."""
    
    def __init__(self):
        self.face_keywords = ['face', 'woman', 'man', 'person', 'girl', 'boy', 'lady', 'guy']
        self.feature_keywords = ALL_KEYWORDS
        
    def is_face_description(self, text: str) -> tuple[bool, str]:
        """
        Check if text is a face description.
        
        Returns:
            tuple: (is_face_description, reason)
        """
        text_lower = text.lower()
        
        # Check 1: Contains face/person keywords
        has_face_keyword = any(keyword in text_lower for keyword in self.face_keywords)
        
        # Check 2: Contains facial feature keywords
        matching_features = [kw for kw in self.feature_keywords if kw.lower() in text_lower]
        
        # Decision logic
        if has_face_keyword and len(matching_features) >= 1:
            return True, f"✓ Face description detected (found keywords: {', '.join(matching_features[:5])})"
        elif len(matching_features) >= 2:
            # Even without explicit face/person keyword, if has multiple features
            return True, f"✓ Face description detected (found features: {', '.join(matching_features[:5])})"
        elif has_face_keyword and len(matching_features) == 0:
            return False, "✗ Contains person keyword but no facial features. Please add details like hair, eyes, nose, etc."
        else:
            return False, "✗ Not a face description. Please describe a person's face with features like hair, eyes, appearance, etc."
    
    def get_suggestions(self, text: str) -> list[str]:
        """Get suggestions for improving the description."""
        text_lower = text.lower()
        suggestions = []
        
        # Check what's missing
        has_gender = any(g in text_lower for g in VOCABULARY['gender'])
        has_hair = any(h in text_lower for h in VOCABULARY['hair_color'] + VOCABULARY['hair_style'])
        has_face_features = any(f in text_lower for f in VOCABULARY['face_features'])
        has_appearance = any(a in text_lower for a in VOCABULARY['appearance'])
        
        if not has_gender:
            suggestions.append("Add gender (woman/man)")
        if not has_hair:
            suggestions.append("Add hair description (color, style)")
        if not has_face_features:
            suggestions.append("Add facial features (nose, lips, eyes, etc.)")
        if not has_appearance:
            suggestions.append("Add appearance (attractive, smiling, etc.)")
        
        return suggestions
