import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ─────────────────────────────
# PATH
# ─────────────────────────────

DATASET_DIR = r"D:\uni\nir\code\img_samples\casia\Test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ─────────────────────────────
# PRETRAINED MODEL
# ─────────────────────────────

model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
)

# CASIA has many classes → replace head
num_classes = len(os.listdir(DATASET_DIR))

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

MODEL_PATH = "efficientnet_casia.pth"

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Loaded fine-tuned CASIA model")
else:
    print("WARNING: using ImageNet pretrained only (no CASIA fine-tuning)")

model.to(DEVICE)
model.eval()

# ─────────────────────────────
# TRANSFORM (IMPORTANT FIX)
# ─────────────────────────────

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ─────────────────────────────
# LABEL MAPPING (CRITICAL FIX)
# ─────────────────────────────

classes = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

char2idx = {c: i for i, c in enumerate(classes)}

# ─────────────────────────────
# INFERENCE
# ─────────────────────────────

@torch.no_grad()
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    logits = model(x)
    pred = torch.argmax(logits, dim=1).item()

    return pred

# ─────────────────────────────
# EVALUATION
# ─────────────────────────────

correct = 0
total = 0

for cls in classes:
    folder = os.path.join(DATASET_DIR, cls)

    for img in os.listdir(folder):
        if os.path.splitext(img)[1].lower() not in IMG_EXTS:
            continue

        path = os.path.join(folder, img)

        pred = predict(path)
        true = char2idx[cls]

        if pred == true:
            correct += 1

        total += 1

acc = correct / total

print("\n======================")
print(f"ACCURACY: {acc*100:.2f}%")
print("======================")