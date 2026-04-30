import os
import shutil

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = r"D:\uni\nir\code\img_samples\casia\Test"  # root folder with character folders
CHARS_FILE = "top_chars_simplified.txt"

DRY_RUN = False  # set False to actually delete

# -----------------------------
# Load allowed characters
# -----------------------------
with open(CHARS_FILE, "r", encoding="utf-8") as f:
    allowed_chars = set(line.strip() for line in f if line.strip())

print(f"Allowed characters loaded: {len(allowed_chars)}")

# -----------------------------
# Process folders
# -----------------------------
kept = 0
deleted = 0

for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)

    # skip files
    if not os.path.isdir(folder_path):
        continue

    # decision
    if folder in allowed_chars:
        kept += 1
        continue

    deleted += 1

    if DRY_RUN:
        print(f"[DRY RUN] Would delete: {folder_path}")
    else:
        shutil.rmtree(folder_path)
        print(f"Deleted: {folder_path}")

# -----------------------------
# Summary
# -----------------------------
print("\n--- SUMMARY ---")
print(f"Kept: {kept}")
print(f"Deleted: {deleted}")
print(f"Mode: {'DRY RUN' if DRY_RUN else 'REAL DELETE'}")