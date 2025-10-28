"""Vocabulary extracted from training captions."""

VOCABULARY = {
    "gender": ['woman', 'man'],
    "age": ['young'],
    "hair_color": ['black', 'brown', 'blond', 'gray'],
    "hair_style": ['wavy', 'straight', 'bangs', 'receding hairline', 'bald'],
    "face_features": ['high cheekbones', 'oval face', 'chubby', 'double chin', 'double chined', 'chubby face'],
    "facial_hair": ["5 o'clock shadow", 'goatee', 'sideburns', 'mustache'],
    "nose": ['pointy nose', 'big nose', 'big pointy nose'],
    "lips": ['big lips'],
    "eyes": ['narrow eyes'],
    "eyebrows": ['arched eyebrows', 'bushy eyebrows'],
    "mouth": ['slightly open mouth'],
    "appearance": ['attractive', 'smiling', 'heavy makeup', 'wearing lipstick', 'looks young', 'rosy cheeks', 'pale skin'],
    "accessories": ['lipstick', 'earrings', 'necklace', 'necktie', 'glasses', 'hat', 'a hat'],
}

# All valid keywords for face description detection
ALL_KEYWORDS = []
for keywords in VOCABULARY.values():
    ALL_KEYWORDS.extend(keywords)
