"""
Text Normalization for Air Traffic Control (ATC) ASR.
Handles NATO alphabet expansion, callsign formatting, and airline code standardization.
"""
import re
from whisper.normalizers import EnglishTextNormalizer

# Initialize Whisper normalizer once
_whisper_normalizer = EnglishTextNormalizer()

def filterAndNormalize(text: str) -> str:
    """
    Main entry point for ATC text normalization.
    Applies a pipeline of cleaning, expansion, and standardization.
    """
    if not text:
        return ""
        
    text = removeCharSet(text, '[', ']')
    text = removeCharSet(text, '<', '>')
    text = removeNonAlphaNum(text)
    text = separateNumbersAndText(text)
    text = aerospaceTransform(text)
    text = removeSpokenSeparators(text)
    
    # Apply Whisper's standard English normalization
    text = _whisper_normalizer(text)
    # Run twice to handle edge cases like "zero five" -> "05" -> "5"
    text = _whisper_normalizer(text)
    
    text = splitNumbersIntoDigits(text)
    text = splitGreetings(text)
    text = text.lower()
    text = standard_words(text)
    
    return text.strip()

# ---------------------------------------------------------------------------
# Core Transformation Logic
# ---------------------------------------------------------------------------

def aerospaceTransform(text: str) -> str:
    """Expands NATO alphabet, maps terminology, and fixes airline codes."""
    wrds = text.split()
    for i, word in enumerate(wrds):
        word_upper = word.upper()
        word_lower = word.lower()
        
        # NATO Alphabet
        if word_upper in NATO_ALPHABET_MAPPING:
            wrds[i] = NATO_ALPHABET_MAPPING[word_upper]
        elif word_lower in NATO_SIMILARITIES:
            wrds[i] = NATO_SIMILARITIES[word_lower]
            
        # Terminology
        elif word_upper in TERMINOLOGY_MAPPING:
            wrds[i] = TERMINOLOGY_MAPPING[word_upper]
        elif word_lower in TEXT_SIMILARITIES:
            wrds[i] = TEXT_SIMILARITIES[word_lower]
            
        # Airlines
        elif word_upper in AIRLINES_IATA_CODES:
            wrds[i] = AIRLINES_IATA_CODES[word_upper]
        elif word_upper in AIRLINES_ICAO_CODES:
            wrds[i] = AIRLINES_ICAO_CODES[word_upper]
            
    return ' '.join(wrds)

def removePunctuation(text: str) -> str:
    return ''.join(' ' if c in '!@#$%^&*~-+=_\\|;:,.?' else c for c in text)

def separateNumbersAndText(text: str) -> str:
    # Insert space between numbers and text
    return ' '.join(re.split(r'(\d+)', text))

def splitNumbersIntoDigits(text: str) -> str:
    """Converts '123' into '1 2 3'."""
    wrds = text.split()
    for i, word in enumerate(wrds):
        if word.isnumeric():
            # "123" -> "1 2 3"
            wrds[i] = ' '.join(list(word))
    return ' '.join(wrds)

def removeSpokenSeparators(text: str) -> str:
    wrds = text.split()
    return ' '.join([w for w in wrds if w.lower() not in ['decimal', 'comma', 'point']])

def splitGreetings(text: str) -> str:
    wrds = text.split()
    for i, word in enumerate(wrds):
        if word.lower() == 'goodbye':
            wrds[i] = 'good bye'
    return ' '.join(wrds)

def removeCharSet(text: str, c1: str, c2: str) -> str:
    """Removes text between delimiters, e.g. [noise]."""
    while c1 in text and c2 in text:
        start = text.find(c1)
        end = text.rfind(c2)
        if end > start:
            text = text[:start] + text[end+1:]
        else:
            break
    return text

def removeNonAlphaNum(text: str) -> str:
    # Keep spaces, remove other non-alphanumeric chars
    return ''.join(c for c in text if c.isalnum() or c == ' ')

def standard_words(text: str) -> str:
    text = text.lower()
    replacements = {
        'lineup': 'line up',
        'centre': 'center',
        'k l m': 'klm',
        'niner': 'nine',
        'x-ray': 'xray'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

# ---------------------------------------------------------------------------
# Constants & Mappings
# ---------------------------------------------------------------------------

NATO_ALPHABET_MAPPING = {
    'A': 'alpha', 'B': 'bravo', 'C': 'charlie', 'D': 'delta', 'E': 'echo',
    'F': 'foxtrot', 'G': 'golf', 'H': 'hotel', 'I': 'india', 'J': 'juliett',
    'K': 'kilo', 'L': 'lima', 'M': 'mike', 'N': 'november', 'O': 'oscar',
    'P': 'papa', 'Q': 'quebec', 'R': 'romeo', 'S': 'sierra', 'T': 'tango',
    'U': 'uniform', 'V': 'victor', 'W': 'whiskey', 'X': 'xray', 'Y': 'yankee', 'Z': 'zulu',
    '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
    '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten',
    '0': 'zero', '00': 'hundred', '000': 'thousand',
    '.': 'decimal', ',': 'comma', '-': 'dash'
}

NATO_SIMILARITIES = {'alfa': 'alpha', 'oskar': 'oscar', 'ekko': 'echo', 'gulf': 'golf'}
TERMINOLOGY_MAPPING = {'FL': 'flight level'}
TEXT_SIMILARITIES = {'descent': 'descend'}

AIRLINES_IATA_CODES = {'BA': 'british airways', 'KL': 'klm', 'LH': 'lufthansa', 'EW': 'eurowings'}
AIRLINES_ICAO_CODES = {'BAW': 'british airways', 'DLH': 'lufthansa', 'KLM': 'klm', 'EWG': 'eurowings'}

# Used for prompt engineering in inference
CONTEXT_PROMPT_EXTENSION = [
    # Waypoints
    'ABNED', 'ABSAM', 'ADIKU', 'ADOMI', 'ADUNU', 'AGASO', 'AGISI', 'AGISU', 
    'AGOGO', 'AKOXA', 'AKZOM', 'ALFEN', 'ALINA', 'AMADA', 'AMEGA', 'AMGOD',
    # ... (Truncated for brevity in code, but full list from original script is assumed here)
    'ZANDVOORT', 'SCHIPHOL', 'EINDHOVEN', 'ROTTERDAM', 'MAASTRICHT'
]