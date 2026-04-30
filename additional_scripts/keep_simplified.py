from opencc import OpenCC

# -----------------------------
# CONFIG
# -----------------------------
INPUT_TXT = "top_chars.txt"
OUTPUT_TXT = "top_chars_simplified.txt"

# OpenCC converter (traditional → simplified)
cc_t2s = OpenCC('t2s')

def is_simplified(char: str) -> bool:
    """
    Heuristic:
    If converting to simplified does NOT change the character,
    we treat it as simplified or neutral.
    """
    converted = cc_t2s.convert(char)
    return char == converted

# -----------------------------
# Load characters
# -----------------------------
with open(INPUT_TXT, "r", encoding="utf-8") as f:
    chars = [line.strip() for line in f if line.strip()]

# -----------------------------
# Deduplicate
# -----------------------------
seen = set()
unique_chars = []

for c in chars:
    if c not in seen:
        seen.add(c)
        unique_chars.append(c)

# -----------------------------
# Filter simplified-only
# -----------------------------
filtered = [c for c in unique_chars if is_simplified(c)]

# -----------------------------
# Save result
# -----------------------------
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    for c in filtered:
        f.write(c + "\n")

print(f"Original: {len(unique_chars)} chars")
print(f"Simplified-only: {len(filtered)} chars")
print(f"Saved to: {OUTPUT_TXT}")