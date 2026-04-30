"""
PaddleOCR 2.7 — CASIA dataset accuracy test
============================================
Same model as SingleOCRWorker in ui_paddle.py:
    PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    reader.ocr(image_path, cls=True)

Dataset layout:
    D:/uni/nir/code/img_samples/casia/Test/
        <character>/          ← folder name IS the ground-truth character
            *.png (20 images)

Two accuracy metrics (both computed as macro-average over classes):
  1. Image accuracy  — fraction of images correctly identified per class, then averaged
  2. Class accuracy  — 1.0 if ≥1 image in the class is correct, 0.0 otherwise, then averaged

Output:
  - Console progress
  - test_results.log  (next to this script)
"""

import os
import sys
import datetime
import logging

# ── Suppress PaddlePaddle C++ spam ──
os.environ["GLOG_minloglevel"] = "3"
logging.getLogger("ppocr").setLevel(logging.ERROR)

DATASET_DIR = r"D:\uni\nir\code\img_samples\casia\Test"
LOG_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results.log")
IMG_EXTS    = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}


# ── Init PaddleOCR (same as ui_paddle.py SingleOCRWorker) ──────────────────
print("Загрузка PaddleOCR 2.7…")
from paddleocr import PaddleOCR
reader = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
print("Модель загружена.\n")


def predict_char(image_path: str) -> tuple[str, float]:
    """
    Run PaddleOCR on a single image.
    Returns (predicted_text, confidence).
    predicted_text is filtered to CJK characters only.
    """
    try:
        results = reader.ocr(image_path, cls=True)
        if not results or not results[0]:
            return "", 0.0

        best     = max(results[0], key=lambda r: r[1][1])
        raw_text = best[1][0].strip()
        conf     = float(best[1][1])

        text = "".join(ch for ch in raw_text if "\u4e00" <= ch <= "\u9fff")
        return (text, conf) if text else ("", 0.0)
    except Exception as e:
        print(f"  [ERROR] {image_path}: {e}")
        return "", 0.0


# ── Collect classes ─────────────────────────────────────────────────────────
classes = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

if not classes:
    print(f"Папки не найдены в {DATASET_DIR}")
    sys.exit(1)

print(f"Найдено классов: {len(classes)}")
print(f"Лог: {LOG_PATH}\n")

# ── Run evaluation ──────────────────────────────────────────────────────────
class_records = []   # list of dicts, one per class

for cls_idx, cls_name in enumerate(classes):
    cls_dir = os.path.join(DATASET_DIR, cls_name)
    images  = sorted([
        f for f in os.listdir(cls_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ])

    if not images:
        print(f"[{cls_idx+1}/{len(classes)}] {cls_name} — нет изображений, пропуск")
        continue

    correct_images = 0
    image_details  = []

    for img_name in images:
        img_path = os.path.join(cls_dir, img_name)
        pred, conf = predict_char(img_path)

        # Match: predicted text must contain the ground-truth character
        is_correct = (cls_name in pred) if pred else False
        if is_correct:
            correct_images += 1

        image_details.append({
            "file":    img_name,
            "pred":    pred if pred else "(пусто)",
            "conf":    conf,
            "correct": is_correct,
        })

    n_images        = len(images)
    img_accuracy    = correct_images / n_images          # fraction correct in this class
    class_correct   = 1 if correct_images > 0 else 0    # 1 if any image correct

    class_records.append({
        "class":         cls_name,
        "n_images":      n_images,
        "correct_imgs":  correct_images,
        "img_accuracy":  img_accuracy,
        "class_correct": class_correct,
        "details":       image_details,
    })

    print(
        f"[{cls_idx+1:>4}/{len(classes)}] {cls_name}  "
        f"img_acc={img_accuracy*100:5.1f}%  "
        f"({correct_images}/{n_images})  "
        f"class={'✓' if class_correct else '✗'}"
    )

# ── Compute macro averages ───────────────────────────────────────────────────
if not class_records:
    print("Нет данных для подсчёта.")
    sys.exit(1)

macro_img_acc   = sum(r["img_accuracy"]  for r in class_records) / len(class_records)
macro_class_acc = sum(r["class_correct"] for r in class_records) / len(class_records)

print("\n" + "="*60)
print(f"  Macro image accuracy  (avg per-class img acc):  {macro_img_acc*100:.2f}%")
print(f"  Macro class accuracy  (≥1 correct per class):  {macro_class_acc*100:.2f}%")
print("="*60)

# ── Write log file ───────────────────────────────────────────────────────────
with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("PaddleOCR 2.7 — CASIA Test Accuracy Report\n")
    f.write(f"Date: {datetime.datetime.now()}\n")
    f.write(f"Dataset: {DATASET_DIR}\n")
    f.write(f"Total classes: {len(class_records)}\n")
    f.write("\n")
    f.write(f"MACRO IMAGE ACCURACY  (avg per-class image accuracy): {macro_img_acc*100:.2f}%\n")
    f.write(f"MACRO CLASS ACCURACY  (>=1 correct image per class):  {macro_class_acc*100:.2f}%\n")
    f.write("\n")
    f.write("="*80 + "\n")
    f.write("PER-CLASS BREAKDOWN\n")
    f.write("="*80 + "\n\n")

    for rec in class_records:
        cls       = rec["class"]
        n         = rec["n_images"]
        corr      = rec["correct_imgs"]
        img_acc   = rec["img_accuracy"] * 100
        cls_ok    = "YES" if rec["class_correct"] else "NO"

        f.write(f"Class: {cls}\n")
        f.write(f"  Image accuracy : {img_acc:5.1f}%  ({corr}/{n} correct)\n")
        f.write(f"  Class correct  : {cls_ok}\n")
        f.write(f"  {'File':<30} {'Predicted':<15} {'Conf':>6}  {'OK'}\n")
        f.write(f"  {'-'*60}\n")
        for d in rec["details"]:
            ok_mark = "✓" if d["correct"] else "✗"
            f.write(
                f"  {d['file']:<30} {d['pred']:<15} {d['conf']:>6.3f}  {ok_mark}\n"
            )
        f.write("\n")

print(f"\nЛог сохранён: {LOG_PATH}")