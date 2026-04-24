"""
Rule-based punctuation and capitalization restoration.

For production, replace with a dedicated model:
  pip install deepmultilingualpunctuation
  from deepmultilingualpunctuation import PunctuationModel
  model = PunctuationModel()
  text  = model.restore_punctuation(text)
"""
import re


def capitalize_sentences(text: str) -> str:
    """Capitalize the first letter after sentence-ending punctuation."""
    text = text.strip()
    if not text:
        return text
    # Capitalize the very first character
    text = text[0].upper() + text[1:]
    # Capitalize after . ! ?
    text = re.sub(r'([.!?]\s+)([a-z])',
                  lambda m: m.group(1) + m.group(2).upper(), text)
    return text


def ensure_terminal_punctuation(text: str) -> str:
    """Add a period if the text doesn't end with punctuation."""
    text = text.strip()
    if text and text[-1] not in '.!?,;:':
        text += '.'
    return text


def remove_filler_words(text: str,
                         fillers: tuple = ('uh', 'um', 'er', 'ah', 'like')) -> str:
    """Strip common filler words (useful for clean transcripts)."""
    pattern = r'\b(' + '|'.join(fillers) + r')\b[,]?\s*'
    return re.sub(pattern, '', text, flags=re.IGNORECASE).strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def postprocess(text: str,
                capitalize: bool = True,
                terminal_punct: bool = True,
                strip_fillers: bool = False) -> str:
    text = normalize_whitespace(text)
    if strip_fillers:
        text = remove_filler_words(text)
    if capitalize:
        text = capitalize_sentences(text)
    if terminal_punct:
        text = ensure_terminal_punctuation(text)
    return text
