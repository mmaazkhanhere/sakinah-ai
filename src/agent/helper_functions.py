import re
from src.agent.schema import QuranAyah, Hadith

def parse_quran_ayah(data: str) -> QuranAyah:
    lines = data.split('\n')
    metadata = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            metadata[key] = value
        elif not line.strip():
            break
    
    return QuranAyah(
        surah_no=int(metadata.get('surah_no', 0)),
        surah_name=metadata.get('surah_name_en', ''),
        ayah_no_surah=int(metadata.get('ayah_no_surah', 0)),
        ayah_eng=metadata.get('ayah_en', ''),
        ayah_arabic=metadata.get('ayah_ar', '')
    )

def parse_hadith(hadith_str: str):
    """Parse a raw hadith string into structured Hadith format
    focusing on Sahih Bukhari and Sahih Muslim"""
    # Filter for authentic sources
    if "SAHIH BUKHARI" not in hadith_str and "SAHIH MUSLIM" not in hadith_str:
        return None
    
    # Extract hadith number
    number_match = re.search(r'Number (\d+):', hadith_str)
    hadith_number = int(number_match.group(1)) if number_match else 0
    
    # Extract narrator
    narrator_match = re.search(r'Narrated (.*?)\n', hadith_str)
    narrator = narrator_match.group(1).strip() if narrator_match else ""
    
    # Extract book reference
    book_ref_match = re.search(r'(SAHIH (BUKHARI|MUSLIM).*?)$', hadith_str, re.MULTILINE)
    book_reference = book_ref_match.group(0).strip() if book_ref_match else ""
    
    # Extract hadith text
    content_start = 0
    if narrator_match:
        content_start = narrator_match.end()
    content_end = book_ref_match.start() if book_ref_match else len(hadith_str)
    
    hadith_text = hadith_str[content_start:content_end].strip()
    
    # Clean common prefixes
    hadith_text = re.sub(r'^Narrated .*?\n', '', hadith_text)
    hadith_text = re.sub(r'^that ', '', hadith_text, flags=re.IGNORECASE)
    
    return Hadith(
        hadith=hadith_text,
        hadith_number=hadith_number,
        narrator=narrator,
        book_reference=book_reference
    )