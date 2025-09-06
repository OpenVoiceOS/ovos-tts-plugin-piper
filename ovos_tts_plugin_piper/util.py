import datetime
import logging
import re
import string
from datetime import date

from ovos_date_parser import nice_time, nice_date
from ovos_number_parser import pronounce_number, is_fractional, pronounce_fraction
from ovos_number_parser.util import is_numeric
from unicode_rbnf import RbnfEngine, FormatPurpose

LOG = logging.getLogger("normalize")

# A dictionary of common contractions and their expanded forms.
# This list is very comprehensive for English.
CONTRACTIONS = {
    "en": {
        "I'd": "I would",
        "I'll": "I will",
        "I'm": "I am",
        "I've": "I have",
        "ain't": "is not",
        "aren't": "are not",
        "can't": "can not",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "gonna": "going to",
        "gotta": "got to",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "how'd": "how did",
        "how'll": "how will",
        "how's": "how is",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "might've": "might have",
        "mightn't": "might not",
        "must've": "must have",
        "mustn't": "must not",
        "needn't": "need not",
        "oughtn't": "ought not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "somebody's": "somebody is",
        "someone'd": "someone would",
        "someone'll": "someone will",
        "someone's": "someone is",
        "that'd": "that would",
        "that'll": "that will",
        "that's": "that is",
        "there'd": "there would",
        "there're": "there are",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'd": "what did",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "whats": "what is",
        "when'd": "when did",
        "when's": "when is",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'd": "who would",
        "who'd've": "who would have",
        "who'll": "who will",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "why'd": "why did",
        "why're": "why are",
        "why's": "why is",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'ain't": "you are not",
        "y'aint": "you are not",
        "y'all": "you all",
        "ya'll": "you all",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
        "I'm'a": "I am going to",
        "I'm'o": "I am going to",
        "I'll've": "I will have",
        "I'd've": "I would have",
        "Whatcha": "What are you",
        "amn't": "am not",
        "'cause": "because",
        "can't've": "cannot have",
        "couldn't've": "could not have",
        "daren't": "dare not",
        "daresn't": "dare not",
        "dasn't": "dare not",
        "everyone's": "everyone is",
        "gimme": "give me",
        "gon't": "go not",
        "hadn't've": "had not have",
        "he've": "he would have",
        "he'll've": "he will have",
        "he'd've": "he would have",
        "here's": "here is",
        "how're": "how are",
        "how'd'y": "how do you do",
        "howd'y": "how do you do",
        "howdy": "how do you do",
        "'tis": "it is",
        "'twas": "it was",
        "it'll've": "it will have",
        "it'd've": "it would have",
        "kinda": "kind of",
        "let's": "let us",
        "ma'am": "madam",
        "may've": "may have",
        "mayn't": "may not",
        "mightn't've": "might not have",
        "mustn't've": "must not have",
        "needn't've": "need not have",
        "ol'": "old",
        "oughtn't've": "ought not have",
        "sha'n't": "shall not",
        "shan't": "shall not",
        "shalln't": "shall not",
        "shan't've": "shall not have",
        "she'd've": "she would have",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "something's": "something is",
        "that're": "that are",
        "that'd've": "that would have",
        "there'll": "there will",
        "there'd've": "there would have",
        "these're": "these are",
        "they'll've": "they will have",
        "they'd've": "they would have",
        "this's": "this is",
        "this'll": "this will",
        "this'd": "this would",
        "those're": "those are",
        "to've": "to have",
        "wanna": "want to",
        "we'll've": "we will have",
        "we'd've": "we would have",
        "what'll've": "what will have",
        "when've": "when have",
        "where're": "where are",
        "which's": "which is",
        "who'll've": "who will have",
        "why've": "why have",
        "will've": "will have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "you'll've": "you will have"
    }
}

# Dictionaries for titles, units, and their full word equivalents.
TITLES = {
    "en": {
        "Dr.": "Doctor",
        "Mr.": "Mister",
        "Prof.": "Professor"
    },
    "ca": {
        "Dr.": "Doctor",
        "Sr.": "Senyor",
        "Sra.": "Senyora",
        "Prof.": "Professor"
    },
    "es": {
        "Dr.": "Doctor",
        "Sr.": "Señor",
        "Sra.": "Señora",
        "Prof.": "Profesor",
        "D.": "Don",
        "Dña.": "Doña"
    },
    "pt": {
        "Dr.": "Doutor",
        "Sr.": "Senhor",
        "Sra.": "Senhora",
        "Prof.": "Professor",
        "Drª.": "Doutora",
        "Eng.": "Engenheiro",
        "D.": "Dom",
        "Dª": "Dona"
    },
    "gl": {
        "Dr.": "Doutor",
        "Sr.": "Señor",
        "Sra.": "Señora",
        "Prof.": "Profesor",
        "Srta.": "Señorita"
    },
    "fr": {
        "Dr.": "Docteur",
        "M.": "Monsieur",
        "Mme": "Madame",
        "Mlle": "Mademoiselle",
        "Prof.": "Professeur",
        "Pr.": "Professeur"
    },
    "it": {
        "Dr.": "Dottore",
        "Sig.": "Signore",
        "Sig.ra": "Signora",
        "Prof.": "Professore",
        "Dott.ssa": "Dottoressa",
        "Sig.na": "Signorina"
    },
    "nl": {
        "Dr.": "Dokter",
        "Dhr.": "De Heer",
        "Mevr.": "Mevrouw",
        "Prof.": "Professor",
        "Drs.": "Dokterandus",
        "Ing.": "Ingenieur"
    },
    "de": {
        "Dr.": "Doktor",
        "Prof.": "Professor"
    }
}

UNITS = {
    "en": {
        "€": "euros",
        "%": "per cent",
        "°C": "degrees celsius",
        "°F": "degrees fahrenheit",
        "°K": "degrees kelvin",
        "°": "degrees",
        "$": "dollars",
        "£": "pounds",
        "km": "kilometers",
        "m": "meters",
        "cm": "centimeters",
        "mm": "millimeters",
        "ft": "feet",
        "in": "inches",
        "yd": "yards",
        "mi": "miles",
        "kg": "kilograms",
        "g": "grams",
        "lb": "pounds",
        "oz": "ounces",
        "L": "liters",
        "mL": "milliliters",
        "gal": "gallons",
        "qt": "quarts",
        "pt": "pints",
        "hr": "hours",
        "min": "minutes",
        "s": "seconds"
    },
    "pt": {
        "€": "euros",
        "%": "por cento",
        "°C": "graus celsius",
        "°F": "graus fahrenheit",
        "°K": "graus kelvin",
        "°": "graus",
        "$": "dólares",
        "£": "libras",
        "km": "quilômetros",
        "m": "metros",
        "cm": "centímetros",
        "mm": "milímetros",
        "kg": "quilogramas",
        "g": "gramas",
        "L": "litros",
        "mL": "mililitros",
        "h": "horas",
        "min": "minutos",
        "s": "segundos"
    },
    "es": {
        "€": "euros",
        "%": "por ciento",
        "°C": "grados celsius",
        "°F": "grados fahrenheit",
        "°K": "grados kelvin",
        "°": "grados",
        "$": "dólares",
        "£": "libras",
        "km": "kilómetros",
        "m": "metros",
        "cm": "centímetros",
        "kg": "kilogramos",
        "g": "gramos",
        "L": "litros",
        "mL": "millilitros"
    },
    "fr": {
        "€": "euros",
        "%": "pour cent",
        "°C": "degrés celsius",
        "°F": "degrés fahrenheit",
        "°K": "degrés kelvin",
        "°": "degrés",
        "$": "dollars",
        "£": "livres",
        "km": "kilomètres",
        "m": "mètres",
        "cm": "centimètres",
        "kg": "kilogrammes",
        "g": "grammes",
        "L": "litres",
        "mL": "millilitres"
    },
    "de": {
        "€": "Euro",
        "%": "Prozent",
        "°C": "Grad Celsius",
        "°F": "Grad Fahrenheit",
        "°K": "Grad Kelvin",
        "°": "Grad",
        "$": "Dollar",
        "£": "Pfund",
        "km": "Kilometer",
        "m": "Meter",
        "cm": "Zentimeter",
        "kg": "Kilogramm",
        "g": "Gramm",
        "L": "Liter",
        "mL": "Milliliter"
    }
}


def _get_number_separators(full_lang: str) -> tuple[str, str]:
    """
    Determines decimal and thousands separators based on language.
    Defaults to '.' decimal and ',' thousands for most languages.
    Special cases:
    - 'pt', 'es', 'fr', 'de': ',' decimal and '.' thousands.
    """
    lang_code = full_lang.split("-")[0]
    decimal_separator = '.'
    thousands_separator = ','
    if lang_code in ["pt", "es", "fr", "de"]:
        decimal_separator = ','
        thousands_separator = '.'
    return decimal_separator, thousands_separator


def _normalize_number_word(word: str, full_lang: str, rbnf_engine) -> str:
    """
    Helper function to normalize a single word that is a number, handling
    decimal and thousands separators based on locale.
    """
    cleaned_word = word.rstrip(string.punctuation)

    # Handle fractions like '3/3'
    if is_fraction(cleaned_word):
        try:
            return pronounce_fraction(cleaned_word, full_lang) + word[len(cleaned_word):]
        except Exception as e:
            LOG.error(f"ovos-number-parser failed to pronounce fraction: {word} - ({e})")
            return word

    # Handle numbers with locale-specific separators
    decimal_separator, thousands_separator = _get_number_separators(full_lang)
    temp_cleaned_word = cleaned_word

    # Check if the word contains a thousands separator followed by digits and a decimal separator
    # This is a specific check for formats like '123.456,78'
    has_thousands_and_decimal = (
            thousands_separator in temp_cleaned_word and
            decimal_separator in temp_cleaned_word and
            temp_cleaned_word.index(thousands_separator) < temp_cleaned_word.index(decimal_separator)
    )

    if has_thousands_and_decimal:
        temp_cleaned_word = temp_cleaned_word.replace(thousands_separator, "")
        temp_cleaned_word = temp_cleaned_word.replace(decimal_separator, ".")
    elif decimal_separator in temp_cleaned_word and is_numeric(temp_cleaned_word.replace(decimal_separator, ".", 1)):
        # Handle cases like '1,2' -> '1.2'
        temp_cleaned_word = temp_cleaned_word.replace(decimal_separator, ".")
    elif thousands_separator in temp_cleaned_word and is_numeric(temp_cleaned_word.replace(thousands_separator, "", 1)):
        # Handle cases like '1.234' -> '1234'
        temp_cleaned_word = temp_cleaned_word.replace(thousands_separator, "")

    # Check if the word is a valid number after processing
    if is_numeric(temp_cleaned_word):
        try:
            num = float(temp_cleaned_word) if "." in temp_cleaned_word else int(temp_cleaned_word)
            return pronounce_number(num, lang=full_lang) + word[len(cleaned_word):]
        except Exception as e:
            LOG.error(f"ovos-number-parser failed to pronounce number: {word} - ({e})")
            return word

    elif rbnf_engine and cleaned_word.isdigit():
        try:
            pronounced_number = rbnf_engine.format_number(cleaned_word, FormatPurpose.CARDINAL).text
            return pronounced_number + word[len(cleaned_word):]
        except Exception as e:
            LOG.error(f"unicode-rbnf failed to pronounce number: {word} - ({e})")
            return word

    return word


# --- Date and Time Pronunciation ---
def pronounce_date(date_obj: date, full_lang: str) -> str:
    """
    Pronounces a date object using ovos-date-parser.
    """
    return nice_date(date_obj, full_lang)


def pronounce_time(time_string: str, full_lang: str) -> str:
    """
    Pronounces a time string using ovos-date-parser.
    Handles military time like "15h01" and converts it to a
    datetime.time object before passing it to nice_time.
    """
    try:
        hours, mins = time_string.split("h")
        time_obj = datetime.time(int(hours), int(mins))
        # Use nice_time from ovos-date-parser
        return nice_time(time_obj, full_lang, speech=True, use_24hour=True, use_ampm=False)
    except Exception as e:
        LOG.warning(f"Failed to parse time string '{time_string}': {e}")
        return time_string.replace("h", " ")


def _normalize_dates_and_times(text: str, full_lang: str, date_format: str = "DMY") -> str:
    """
    Helper function to normalize dates and times using regular expressions.
    This prepares the strings for pronunciation.
    """
    lang_code = full_lang.split("-")[0]
    # Pre-process with regex to handle English am/pm times
    if lang_code == "en":
        text = re.sub(r"(?i)(\d+)(am|pm)", r"\1 \2", text)
        # Handle the pronunciation for TTS
        text = text.replace("am", "A M").replace("pm", "P M")

    # Normalize times like "15h01" to words
    time_pattern = re.compile(r"(\d{1,2})h(\d{2})", re.IGNORECASE)

    def replace_time(match):
        time_str = match.group(0)
        return pronounce_time(time_str, full_lang)

    text = time_pattern.sub(replace_time, text)

    # Find dates like "DD/MM/YYYY" or "YYYY/MM/DD"
    date_pattern = re.compile(r"(\d{1,4})[/-](\d{1,2})[/-](\d{1,4})")

    match = date_pattern.search(text)

    if match:
        # Get the three parts of the date string
        part1_str, part2_str, part3_str = match.groups()
        p1, p2, p3 = int(part1_str), int(part2_str), int(part3_str)

        # Initialize month, day, and year
        month, day, year = None, None, None

        # Determine year first based on length (4 digits)
        if len(part1_str) == 4:
            year, rest_parts = p1, [p2, p3]
        elif len(part3_str) == 4:
            year, rest_parts = p3, [p1, p2]
        else:
            # If no 4-digit year, it's ambiguous, assume a 2-digit year.
            # We'll assume the last part is the year based on common patterns.
            year = p3
            # Expand 2-digit year to 4-digit year
            if year < 100:
                # Assume years 00-29 are 2000-2029, 30-99 are 1930-1999
                year = 2000 + year if year < 30 else 1900 + year
            rest_parts = [p1, p2]

        # From the remaining parts, try to determine day and month
        if day is None and any(p > 12 and len(str(p)) == 2 for p in rest_parts):
            # If a two-digit number is > 12, it's a day
            day_candidate = next((p for p in rest_parts if p > 12), None)
            if day_candidate:
                day = day_candidate
                rest_parts.remove(day_candidate)
                month = rest_parts[0]

        # Fallback to date_format if day/month are still ambiguous
        if day is None or month is None:
            if date_format.lower() == "mdy":
                month, day = rest_parts[0], rest_parts[1]
            else:  # default to DD/MM/YY
                day, month = rest_parts[0], rest_parts[1]

        try:
            date_obj = date(year, month, day)
            pronounced_date_str = pronounce_date(date_obj, full_lang)
            text = text.replace(match.group(0), pronounced_date_str)
        except (ValueError, IndexError) as e:
            LOG.warning(f"Could not parse date from '{match.group(0)}': {e}")

    return text


def _normalize_word_hyphen_digit(text: str) -> str:
    """
    Helper function to normalize words attached to digits with a hyphen,
    such as 'sub-23' -> 'sub 23'.
    """
    # Regex to find a word (\w+) followed by a hyphen and a digit (\d+)
    pattern = re.compile(r"(\w+)-(\d+)")
    text = pattern.sub(r"\1 \2", text)
    return text


def _normalize_units(text: str, full_lang: str) -> str:
    """
    Helper function to normalize units attached to numbers.
    This function handles symbolic and alphanumeric units separately
    to avoid issues with word boundaries.
    """
    text = text.replace("º", "°") # these characters look the same... but...
    lang_code = full_lang.split("-")[0]
    if lang_code in UNITS:
        # Determine number separators for the language
        decimal_separator, thousands_separator = _get_number_separators(full_lang)

        # Separate units into symbolic and alphanumeric
        symbolic_units = {k: v for k, v in UNITS[lang_code].items() if not k.isalnum()}
        alphanumeric_units = {k: v for k, v in UNITS[lang_code].items() if k.isalnum()}

        # Create regex pattern for symbolic units and replace them first
        sorted_symbolic = sorted(symbolic_units.keys(), key=len, reverse=True)
        symbolic_pattern_str = "|".join(re.escape(unit) for unit in sorted_symbolic)
        if symbolic_pattern_str:
            # Pattern to match numbers with optional thousands and decimal separators
            number_pattern_str = rf"(\d+[{re.escape(thousands_separator)}]?\d*[{re.escape(decimal_separator)}]?\d*)"
            symbolic_pattern = re.compile(number_pattern_str + r"\s*(" + symbolic_pattern_str + r")", re.IGNORECASE)

            def replace_symbolic(match):
                number = match.group(1)
                # Remove thousands separator and replace decimal separator for parsing
                if thousands_separator in number and decimal_separator in number:
                    number = number.replace(thousands_separator, "").replace(decimal_separator, ".")
                elif decimal_separator != "." and decimal_separator in number:
                    number = number.replace(decimal_separator, ".")
                unit_symbol = match.group(2)
                unit_word = symbolic_units[unit_symbol]
                try:
                    return f"{pronounce_number(float(number) if '.' in number else int(number), full_lang)} {unit_word}"
                except Exception as e:
                    LOG.error(f"Failed to pronounce number with unit: {number}{unit_symbol} - ({e})")
                    return match.group(0)
            text = symbolic_pattern.sub(replace_symbolic, text)

        # Create regex pattern for alphanumeric units and replace them next
        sorted_alphanumeric = sorted(alphanumeric_units.keys(), key=len, reverse=True)
        alphanumeric_pattern_str = "|".join(re.escape(unit) for unit in sorted_alphanumeric)
        if alphanumeric_pattern_str:
            number_pattern_str = rf"(\d+[{re.escape(thousands_separator)}]?\d*[{re.escape(decimal_separator)}]?\d*)"
            alphanumeric_pattern = re.compile(number_pattern_str + r"\s*(" + alphanumeric_pattern_str + r")\b",
                                              re.IGNORECASE)

            def replace_alphanumeric(match):
                number = match.group(1)
                # Remove thousands separator and replace decimal separator for parsing
                if thousands_separator in number and decimal_separator in number:
                    number = number.replace(thousands_separator, "").replace(decimal_separator, ".")
                elif decimal_separator != "." and decimal_separator in number:
                    number = number.replace(decimal_separator, ".")
                unit_symbol = match.group(2)
                unit_word = alphanumeric_units[unit_symbol]
                return f"{pronounce_number(float(number) if '.' in number else int(number), full_lang)} {unit_word}"

            text = alphanumeric_pattern.sub(replace_alphanumeric, text)
    return text


def _normalize_word(word: str, full_lang: str, rbnf_engine) -> str:
    """
    Helper function to normalize a single word.
    """
    lang_code = full_lang.split("-")[0]

    if word in CONTRACTIONS.get(lang_code, {}):
        return CONTRACTIONS[lang_code][word]

    if word in TITLES.get(lang_code, {}):
        return TITLES[lang_code][word]

    # Delegate number parsing to the new helper function
    normalized_number = _normalize_number_word(word, full_lang, rbnf_engine)
    if normalized_number != word:
        return normalized_number

    return word


def is_fraction(word: str) -> bool:
    """Checks if a word is a fraction like '3/3'."""
    if "/" in word:
        parts = word.split("/")
        if len(parts) == 2:
            n1, n2 = parts
            return n1.isdigit() and n2.isdigit()
    return False


def normalize(text: str, lang: str) -> str:
    """
    Normalizes a text string by expanding contractions, titles, and pronouncing
    numbers, units, and fractions.
    """
    full_lang = lang
    lang_code = full_lang.split("-")[0]
    dialog = text

    # Step 1: Handle dates and times with ovos-date-parser
    date_format = "MDY" if full_lang.lower() == "en-us" else "DMY"
    dialog = _normalize_dates_and_times(dialog, full_lang, date_format)

    # Step 2: Normalize words with hyphens and digits
    dialog = _normalize_word_hyphen_digit(dialog)

    # Step 3: Expand units attached to numbers
    dialog = _normalize_units(dialog, full_lang)

    # Step 4: Normalize word-by-word
    words = dialog.split()
    rbnf_engine = None
    try:
        rbnf_engine = RbnfEngine.for_language(lang_code)
    except (ValueError, KeyError) as e:
        LOG.debug(f"RBNF engine not available for language '{lang_code}': {e}")

    normalized_words = [_normalize_word(word, full_lang, rbnf_engine) for word in words]
    dialog = " ".join(normalized_words)

    return dialog


if __name__ == "__main__":
    # --- Example usage for demonstration purposes ---

    # General normalization examples
    print("General English example: " + normalize('I\'m Dr. Prof. 3/3 0.5% of 12345€, 5ft, and 10kg', 'en'))
    print(f"Word Salad Portuguese (Dr. Prof. 3/3 0,5% de 12345€, 5m, e 10kg): {normalize('Dr. Prof. 3/3 0,5% de 12345€, 5m, e 10kg', 'pt')}")
    print(f"Word Salad Portuguese (Dr. Prof. 3/3 0.5% de 12345€, 5m, e 10kg): {normalize('Dr. Prof. 3/3 0.5% de 12345€, 5m, e 10kg', 'pt')}")

    # Portuguese examples with comma decimal separator
    print("\n--- Portuguese Decimal Separator Examples ---")
    print(
        f"Original: 'A coima aplicada é de 1,2 milhões de euros.' Normalized: '{normalize('A coima aplicada é de 1,2 milhões de euros.', 'pt')}'")
    print(
        f"Original: 'Agora, tem 1,88 metros e muito para contar.' Normalized: '{normalize('Agora, tem 1,88 metros e muito para contar.', 'pt')}'")
    print(
        f"Original: 'Ainda temos 1,7 milhões de pobres!' Normalized: '{normalize('Ainda temos 1,7 milhões de pobres!', 'pt')}'")
    print(f"Original: 'O lucro foi de 123.456,78€.' Normalized: '{normalize('O lucro foi de 123.456,78€.', 'pt')}'")
    print(f"Normalized: '{normalize('O lucro foi de 123.456,78€.', 'pt-PT')}'")

    # English dates and times
    print("\n--- English Date & Time Examples ---")
    print(f"English date (MDY format): {normalize('The date is 08/03/2025', 'en-US')}")
    print(f"English ambiguous date (MDY assumed): {normalize('The report is due 15/05/2025', 'en-US')}")
    print(f"English date with dashes: {normalize('The event is on 11-04-2025', 'en-US')}")
    print(f"English AM/PM time: {normalize('The meeting is at 10am', 'en-US')}")
    print(f"English military time: {normalize('The party is at 19h30', 'en-US')}")
    print(f"English month name: {normalize('The report is due 15 May 2025', 'en-US')}")

    # Portuguese dates and times
    print("\n--- Portuguese Date & Time Examples ---")
    print(f"Portuguese date (A data é 03/08/2025): {normalize('A data é 03/08/2025', 'pt')}")
    print(f"Portuguese ambiguous date (O relatório é para 15/05/2025): {normalize('O relatório é para 15/05/2025', 'pt')}")
    print(f"Portuguese date with dashes (O evento é no dia 25-10-2024): {normalize('O evento é no dia 25-10-2024', 'pt')}")
    print(f"Portuguese military time (O encontro é às 14h30): {normalize('O encontro é às 14h30', 'pt')}")

    # Other examples
    print(f"\n--- Other Examples ---")
    print(f"English fraction: {normalize('The fraction is 1/2', 'en')}")
    print(f"English plural fraction: {normalize('There are 3/4 of a cup', 'en')}")
    print(f"Spanish example with units: {normalize('The temperature is 25ºC', 'es')}")
    print(f"Portuguese with punctuation: {normalize('12345€, 5m e 10kg', 'pt')}")
    print(
        f"Portuguese word-digit: {normalize('Esta temporada leva oito jogos ao serviço da equipa sub-23 leonina.', 'pt')}")
