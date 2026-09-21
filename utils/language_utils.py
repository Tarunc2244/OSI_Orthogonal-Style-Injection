# utils/language_utils.py
import re

LANGUAGE_CONFIGS = {
    # Indic languages
    'bengali': {'name': 'Bengali', 'unicode_range': r'[\u0980-\u09FF\u09E6-\u09EF]', 'column_suffix': 'bengali_dialogue'},
    'hindi':   {'name': 'Hindi',   'unicode_range': r'[\u0900-\u097F\u0966-\u096F]', 'column_suffix': 'hindi_dialogue'},
    'tamil':   {'name': 'Tamil',   'unicode_range': r'[\u0B80-\u0BFF\u0BE6-\u0BEF]', 'column_suffix': 'tamil_dialogue'},
    'telugu':  {'name': 'Telugu',  'unicode_range': r'[\u0C00-\u0C7F\u0C66-\u0C6F]', 'column_suffix': 'telugu_dialogue'},
    'kannada': {'name': 'Kannada', 'unicode_range': r'[\u0C80-\u0CFF\u0CE6-\u0CEF]', 'column_suffix': 'kannada_dialogue'},
    # East Asian
    'chinese': {'name': 'Chinese', 'unicode_range': r'[\u4E00-\u9FFF\u3000-\u303F]', 'column_suffix': 'chinese_dialogue'},
    'japanese':{'name': 'Japanese','unicode_range': r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', 'column_suffix': 'japanese_dialogue'},
    # European languages (handled by a special regex, not the unicode_range)
    'italian': {'name': 'Italian', 'unicode_range': None, 'column_suffix': 'italian_dialogue'},
    'spanish': {'name': 'Spanish', 'unicode_range': None, 'column_suffix': 'spanish_dialogue'},
    'german':  {'name': 'German',  'unicode_range': None, 'column_suffix': 'german_dialogue'},
    'french':  {'name': 'French',  'unicode_range': None, 'column_suffix': 'french_dialogue'},
}

def post_prune_generated_text(text: str, language: str = 'bengali') -> str:
    """Clean generated text, removing prompt leftovers and non‑target characters."""
    if not text:
        return "N/A"
    
    lang_config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS['bengali'])
    
    # 1. Remove common assistant preambles and metadata
    preamble_patterns = [
        r'Cutting Knowledge Date:.*?(?=\n|$)',
        r'Today Date:.*?(?=\n|$)',
        r'You are a helpful translation assistant\..*?(?=\n|$)',
        r'^Sure[,\s]+(?:here\s+is|I\s+can\s+help\s+with\s+that)[:\s]*',
        r'^Of\s+course[,\s]+',
        r'^Here\s+is\s+the\s+translation[:\s]*',
        r'^The\s+translation\s+is[:\s]*',
        r'^Translation[:\s]*',
        r'^\"|\"$',  # remove surrounding quotes
        r'^“|”$',    # smart quotes
    ]
    for pat in preamble_patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # 2. Remove leftover prompt tokens
    patterns = [
        r'Translate.*?(?=\n|$)',
        r'English.*?(?=\n|$)',
        r'Hindi.*?(?=\n|$)',
        r'Bengali.*?(?=\n|$)',
        r'Tamil.*?(?=\n|$)',
        r'Telugu.*?(?=\n|$)',
        r'Kannada.*?(?=\n|$)',
        r'Spanish.*?(?=\n|$)',
        r'Italian.*?(?=\n|$)',
        r'German.*?(?=\n|$)',
        r'French.*?(?=\n|$)',
        r'Chinese.*?(?=\n|$)',
        r'Japanese.*?(?=\n|$)',
        r'Character.*?(?=\n|$)',
        r'Dialogue:.*?(?=\n|$)',
        r'<\|.*?\|>',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # 3. Remove repeated characters
    text = re.sub(r'(.)\1{3,}', r'\1', text)
    
    # 4. Keep only target script characters and allowed punctuation
    if language in ['spanish', 'italian', 'german', 'french']:
        # For European languages, keep Latin letters, accents, and common punctuation
        allowed = r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ¡¿äöüßÄÖÜœŒçÇàèéêëîïôùûüÿ\-\.\,\!\?\;\:\s]'
        text = re.sub(r'[^' + allowed[1:-1] + ']', '', text)
    else:
        # For Indic/East Asian, use the unicode range from config
        unicode_range = lang_config['unicode_range']
        if unicode_range:
            chars = unicode_range[1:-1]
            allowed = f'[{chars}\\s।,!?.]'
            text = re.sub(f'[^{chars}\\s।,!?.]', '', text)
        else:
            text = re.sub(r'[^\x20-\x7E]', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text if text else "N/A"