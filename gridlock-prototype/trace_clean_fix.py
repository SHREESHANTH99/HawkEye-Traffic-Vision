import re

DIGIT_TO_LETTER = {"0": "O", "1": "I", "8": "B", "5": "S", "6": "G"}
LETTER_TO_DIGIT = {"O": "0", "I": "1", "B": "8", "S": "5", "G": "6", "Z": "2"}

def trace_clean_plate(raw):
    cleaned = re.sub(r"[^A-Z0-9]", "", raw.upper())
    corrected = list(cleaned)
    print(f"cleaned (regex): {cleaned}")
    for i, ch in enumerate(corrected):
        before = ch
        if i < 2:
            if ch in DIGIT_TO_LETTER:
                corrected[i] = DIGIT_TO_LETTER[ch]
        elif i < 4:
            if ch in LETTER_TO_DIGIT:
                corrected[i] = LETTER_TO_DIGIT[ch]
        elif i >= max(6, len(cleaned) - 4):
            if ch in LETTER_TO_DIGIT:
                corrected[i] = LETTER_TO_DIGIT[ch]
        after = corrected[i]
        print(f"Index {i}: {before} -> {after} (Condition: i >= max(6, len-4): {i >= max(6, len(cleaned) - 4)})")
        
    print(f"Final: {''.join(corrected)}")

trace_clean_plate("MH 12AB 122")
