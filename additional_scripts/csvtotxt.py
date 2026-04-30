import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
INPUT_CSV = "hanzi.csv"
OUTPUT_TXT = "top_chars.txt"

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(INPUT_CSV)

# Ensure expected columns exist
assert "char" in df.columns, "CSV must contain 'char' column"
assert "rank" in df.columns, "CSV must contain 'rank' column"

# -----------------------------
# Clean data
# -----------------------------
# Remove duplicates (keep first occurrence)
df = df.drop_duplicates(subset=["char"])

# Sort by rank (important for "top N")
df = df.sort_values(by="rank")

# -----------------------------
# Extract characters
# -----------------------------
chars = df["char"].astype(str).tolist()

# -----------------------------
# Save to txt (one char per line)
# -----------------------------
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    for c in chars:
        c = c.strip()
        if c:
            f.write(c + "\n")

print(f"Saved {len(chars)} unique characters to {OUTPUT_TXT}")