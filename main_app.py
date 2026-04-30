"""
Chinese Character Recognition App
==================================
Models used:
  - PaddleOCR 2.7  (ch)         — detection + recognition for printed & handwritten Chinese
  - pypinyin                    — pinyin / tone lookup (offline)
  - torchvision EfficientNet-B0 — deep feature extraction for handwriting comparison (1280-d)

Install dependencies:
    pip install paddlepaddle==2.6.2 paddleocr==2.7.0.3 pypinyin torch torchvision pillow numpy opencv-python PyQt6
"""

import sys
import os
import math
import unicodedata
import traceback

# ── Подавляем C++ логи PaddlePaddle ──
os.environ["GLOG_minloglevel"] = "3"

import numpy as np
import cv2
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget,
    QTextEdit, QScrollArea, QGridLayout,
    QListWidget, QFrame, QProgressBar, QSizePolicy,
    QListWidgetItem,
)
from PyQt6.QtGui import QPixmap, QFont, QImage, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMutex

# ─────────────────────────────────────────────
#  ERROR LOGGING
# ─────────────────────────────────────────────

_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_error.log")

def _write_log(text: str):
    """Append error text to ocr_error.log next to the script."""
    try:
        import datetime
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"{datetime.datetime.now()}\n")
            f.write(text)
            f.write("\n")
    except Exception:
        pass



def char_meaning(ch: str) -> str:
    """Return a short English gloss for a CJK character."""
    #if ch in MINI_DICT:
    #    return MINI_DICT[ch]
    # Try hanzipy if installed
    try:
        from hanzipy.decomposer import HanziDecomposer
        decomposer = HanziDecomposer()
        result = decomposer.decompose(ch, 1)
        if result and result.get("meaning"):
            return result["meaning"]
    except Exception:
        pass
    # Unicode name fallback
    try:
        name = unicodedata.name(ch, "")
        if name.startswith("CJK UNIFIED IDEOGRAPH"):
            return "(definition will be added later)"
        return name
    except Exception:
        return "(unknown)"


# ─────────────────────────────────────────────
#  BACKGROUND WORKERS
# ─────────────────────────────────────────────

class ModelLoader(QThread):
    """Loads PaddleOCR and EfficientNet-B0 in a background thread."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object)   # ocr_reader, feature_extractor
    error = pyqtSignal(str)

    def run(self):
        try:
            self.progress.emit("Загрузка PaddleOCR")
            import logging
            logging.getLogger("ppocr").setLevel(logging.ERROR)
            from paddleocr import PaddleOCR
            reader = PaddleOCR(
                use_angle_cls=True,
                lang="ch",
                show_log=False,
            )

            self.progress.emit("Загрузка EfficientNet-B0")
            import torch
            import torchvision.models as models
            import torchvision.transforms as T

            efficientnet = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )
            efficientnet.classifier = torch.nn.Identity()
            efficientnet.eval()

            transform = T.Compose([
                T.Resize((224, 224)),
                T.Grayscale(num_output_channels=3),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            ])

            self.progress.emit("Модели готовы")
            self.finished.emit(reader, (efficientnet, transform))

        except Exception as e:
            err = traceback.format_exc()
            _write_log(err)
            print(err)
            self.error.emit(err)


class OCRWorker(QThread):
    """Runs PaddleOCR on an image path."""
    finished = pyqtSignal(list)   # list of (char, x_center, y_center, crop_pil)
    error = pyqtSignal(str)

    def __init__(self, reader, image_path, parent=None):
        super().__init__(parent)
        self.reader = reader
        self.image_path = image_path

    def run(self):
        try:
            # PaddleOCR 2.x: ocr() → results[0] = list of [bbox, (text, conf)]
            # bbox = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            results = self.reader.ocr(self.image_path, cls=True)
            img_pil = Image.open(self.image_path).convert("RGB")
            img_arr = np.array(img_pil)

            characters = []
            if not results or not results[0]:
                self.finished.emit(characters)
                return

            for line in results[0]:
                bbox, (text, conf) = line
                if not text.strip():
                    continue

                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))

                region_w = max(x_max - x_min, 1)
                n        = len(text)
                char_w   = region_w / n if n > 0 else region_w

                for i, ch in enumerate(text):
                    if not ('\u4e00' <= ch <= '\u9fff'):
                        continue
                    cx_start = max(0, int(x_min + i * char_w))
                    cx_end   = min(img_arr.shape[1], int(x_min + (i + 1) * char_w))
                    cy_start = max(0, y_min)
                    cy_end   = min(img_arr.shape[0], y_max)

                    crop     = img_pil.crop((cx_start, cy_start, cx_end, cy_end))
                    x_center = (cx_start + cx_end) // 2
                    y_center = (cy_start + cy_end) // 2
                    characters.append((ch, x_center, y_center, crop))

            self.finished.emit(characters)
        except Exception as e:
            tb = traceback.format_exc()
            _write_log(tb)
            self.error.emit(tb)


class SingleOCRWorker(QThread):
    """Runs PaddleOCR on a single-character image."""
    finished = pyqtSignal(str, float)   # (detected_text, confidence)
    error = pyqtSignal(str)

    def __init__(self, reader, image_path, parent=None):
        super().__init__(parent)
        self.reader = reader
        self.image_path = image_path

    def run(self):
        try:
            # PaddleOCR 2.x: ocr() → results[0] = list of [bbox, (text, conf)]
            results = self.reader.ocr(self.image_path, cls=True)
            if not results or not results[0]:
                self.finished.emit("", 0.0)
                return

            # Берём строку с наибольшей уверенностью
            best      = max(results[0], key=lambda r: r[1][1])
            raw_text  = best[1][0].strip()
            conf      = float(best[1][1])

            text = "".join(ch for ch in raw_text if '\u4e00' <= ch <= '\u9fff')
            if not text:
                self.finished.emit("", 0.0)
            else:
                self.finished.emit(text, conf)
        except Exception as e:
            tb = traceback.format_exc()
            _write_log(tb)
            self.error.emit(tb)


class HandwritingWorker(QThread):
    """
    Computes pairwise handwriting similarity using EfficientNet-B0 features.
    Also adds stroke-level features from OpenCV for higher accuracy.
    """
    finished = pyqtSignal(float, str) 
    error = pyqtSignal(str)

    def __init__(self, feature_extractor, image_paths, parent=None):
        super().__init__(parent)
        self.efficientnet, self.transform = feature_extractor
        self.image_paths = image_paths

    # ---- helpers ----

    @staticmethod
    def _stroke_features(path: str) -> np.ndarray:
        """
        Extract simple stroke-style features:
          - mean stroke width (via skeleton)
          - stroke slant angle (via Hough lines)
          - ink density
        Returns a 3-d feature vector.
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"OpenCV не смог прочитать изображение: {path}")
        #if img is None:
        #   return np.zeros(3)

        # Binarise
        _, bw = cv2.threshold(img, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Ink density
        density = bw.mean() / 255.0

        # Stroke width via skeleton → distance transform
        dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
        mean_stroke_w = float(dist[dist > 0].mean()) if dist[dist > 0].size else 0.0

        # Slant via HoughLinesP on Canny edges
        edges = cv2.Canny(img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=20, minLineLength=15, maxLineGap=5)
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                angles.append(angle)
        slant = float(np.mean(angles)) if angles else 0.0

        return np.array([density, mean_stroke_w / 50.0, slant / 90.0])

    def _deep_features(self, path: str) -> np.ndarray:
        import torch
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = self.efficientnet(tensor)
        return feat.squeeze().numpy()

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def run(self):
        try:
            paths = self.image_paths
            n = len(paths)

            # Extract features for each image
            deep_feats   = [self._deep_features(p)   for p in paths]
            stroke_feats = [self._stroke_features(p) for p in paths]

            # Pairwise comparisons
            deep_sims, stroke_sims = [], []
            for i in range(n):
                for j in range(i + 1, n):
                    deep_sims.append(self._cosine(deep_feats[i], deep_feats[j]))
                    # stroke: use 1 - L2 distance (normalised)
                    s_dist = np.linalg.norm(stroke_feats[i] - stroke_feats[j])
                    stroke_sims.append(max(0.0, 1.0 - s_dist))

            deep_mean   = float(np.mean(deep_sims))
            stroke_mean = float(np.mean(stroke_sims))

            # Weighted combination (deep features are more reliable)
            combined = 0.75 * deep_mean + 0.25 * stroke_mean

            # Map cosine similarity [−1,1] → probability [0,1]
            # Empirically: same writer ≈ cos > 0.85, different ≈ cos < 0.65
            # Use a calibrated sigmoid
            probability = 1.0 / (1.0 + math.exp(-12.0 * (combined - 0.78)))
            probability = max(0.0, min(1.0, probability))

            detail = (
                f"Глубокие признаки (EfficientNet-B0): {deep_mean:.3f}\n"
                f"Признаки штриха (OpenCV):            {stroke_mean:.3f}\n"
                f"Взвешенная схожесть:                 {combined:.3f}\n"
                f"Вероятность одного автора:           {probability * 100:.1f}%"
            )
            self.finished.emit(probability, detail)

        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chinese Character Recognition")
        self.setMinimumSize(1280, 780)
        self.setStyleSheet(self.global_style())

        # ML models (filled after ModelLoader finishes)
        self._ocr_reader        = None
        self._feature_extractor = None

        # Active workers (keep refs to avoid GC)
        self._active_worker = None

        root = QVBoxLayout()

        # ── Status bar ──
        self._status_label = QLabel("Загрузка моделей")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet("color: #ffd740; font-size: 13px; padding: 4px;")
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)    # indeterminate
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setTextVisible(False)
        root.addWidget(self._status_label)
        root.addWidget(self._progress_bar)

        # ── Tabs ──
        self.tabs = QTabWidget()
        self.tabs.setEnabled(False)   # disabled until models load
        self.tabs.addTab(self._build_single_tab(), "Один символ")
        self.tabs.addTab(self._build_text_tab(),   "Фото текста")
        self.tabs.addTab(self._build_hw_tab(),     "Сравнение почерка(wip)")
        root.addWidget(self.tabs)

        self.setLayout(root)

        # Start model loading
        self._loader = ModelLoader()
        self._loader.progress.connect(self._on_load_progress)
        self._loader.finished.connect(self._on_models_ready)
        self._loader.error.connect(self._on_load_error)
        self._loader.start()

    # ──────────────────────────────────────────
    # STYLE
    # ──────────────────────────────────────────

    def global_style(self):
        return """
        QWidget {
            background-color: #1e1e2f;
            color: #e0e0e0;
            font-size: 15px;
        }
        QTabBar::tab {
            background: #2c2c3e;
            padding: 10px 26px;
            border-radius: 8px;
            margin: 4px;
            color: #aaa;
        }
        QTabBar::tab:selected { background: #3d5afe; color: white; }
        QPushButton {
            background-color: #3d5afe;
            border-radius: 8px;
            padding: 9px 14px;
            color: white;
        }
        QPushButton:hover  { background-color: #536dfe; }
        QPushButton:disabled { background-color: #444; color: #888; }
        QTextEdit, QListWidget {
            background-color: #252536;
            border: 1px solid #3a3a55;
            border-radius: 10px;
            padding: 6px;
        }
        QScrollArea { border: none; background: transparent; }
        QProgressBar {
            background: #2c2c3e;
            border-radius: 3px;
        }
        QProgressBar::chunk { background: #3d5afe; border-radius: 3px; }
        """

    def _card(self):
        f = QFrame()
        f.setStyleSheet(
            "QFrame { background-color: #252536; border-radius: 14px; padding: 16px; }"
        )
        return f

    # ──────────────────────────────────────────
    # MODEL LOADING CALLBACKS
    # ──────────────────────────────────────────

    def _on_load_progress(self, msg: str):
        self._status_label.setText(f"{msg}")

    def _on_models_ready(self, reader, feature_extractor):
        self._ocr_reader        = reader
        self._feature_extractor = feature_extractor
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(1)
        self._status_label.setText("Модели загружены")
        self._status_label.setStyleSheet("color: #69f0ae; font-size: 13px; padding: 4px;")
        self.tabs.setEnabled(True)

    def _on_load_error(self, msg: str):
        self._progress_bar.setRange(0, 1)
        self._status_label.setText(f"Ошибка загрузки: {msg}")
        self._status_label.setStyleSheet("color: #ff5252; font-size: 13px; padding: 4px;")

    # ──────────────────────────────────────────
    # TAB 1 — SINGLE CHARACTER
    # ──────────────────────────────────────────

    def _build_single_tab(self):
        tab = QWidget()
        outer = QHBoxLayout()

        card = self._card()
        lay  = QVBoxLayout()

        btn = QPushButton("Загрузить фото символа")
        btn.clicked.connect(self._load_single_image)

        self._single_img_lbl = QLabel("Загрузите изображение символа")
        self._single_img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._single_img_lbl.setMinimumHeight(220)
        self._single_img_lbl.setStyleSheet(
            "border: 2px dashed #3d5afe; border-radius: 12px; color: #888;"
        )

        self._single_status = QLabel("")
        self._single_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._single_status.setStyleSheet("color: #ffd740; font-size: 13px;")

        self._single_char_big = QLabel("")
        self._single_char_big.setFont(QFont("Microsoft YaHei", 100))
        self._single_char_big.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._single_char_info = QTextEdit()
        self._single_char_info.setReadOnly(True)
        self._single_char_info.setPlaceholderText("Здесь появится информация о символе")

        lay.addWidget(btn)
        lay.addWidget(self._single_img_lbl)
        lay.addWidget(self._single_status)
        lay.addWidget(self._single_char_big)
        lay.addWidget(self._single_char_info)
        card.setLayout(lay)
        outer.addWidget(card)
        tab.setLayout(outer)
        return tab

    def _load_single_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return

        px = QPixmap(path)
        self._single_img_lbl.setPixmap(
            px.scaled(280, 280, Qt.AspectRatioMode.KeepAspectRatio,
                      Qt.TransformationMode.SmoothTransformation)
        )
        self._single_char_big.setText("")
        self._single_char_info.setText("")
        self._single_status.setText("Распознавание")

        worker = SingleOCRWorker(self._ocr_reader, path)
        worker.finished.connect(self._on_single_ocr_done)
        worker.error.connect(lambda e: self._single_status.setText(f"❌ {e}"))
        self._active_worker = worker
        worker.start()

    def _on_single_ocr_done(self, text: str, conf: float):
        if not text:
            self._single_status.setText("Символ не распознан")
            return

        # Show first recognised character prominently
        #first_char = text[0]
        #self._single_char_big.setText(first_char)

        from pypinyin import pinyin, Style
        py_list = pinyin(text, style=Style.TONE)
        py_str  = " ".join(item[0] for item in py_list)

        # Collect meanings for each unique CJK char in text
        meanings = []
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff":
                m = char_meaning(ch)
                meanings.append(f"  {ch}  →  {m}")

        meaning_block = "\n".join(meanings) if meanings else "(не CJK символ)"

        self._single_char_info.setText(
            f"Распознанный текст: {text}\n"
            f"Пиньинь:  {py_str}\n"
            f"Уверенность модели: {conf * 100:.1f}%\n\n"
            f"Значения:\n{meaning_block}"
        )
        self._single_status.setText("Готово")
        self._single_status.setStyleSheet("color: #69f0ae; font-size: 13px;")

    # ──────────────────────────────────────────
    # TAB 2 — TEXT PAGE
    # ──────────────────────────────────────────

    def _build_text_tab(self):
        tab    = QWidget()
        outer  = QHBoxLayout()
        left_l = QVBoxLayout()

        btn = QPushButton("Загрузить фото текста")
        btn.clicked.connect(self._load_text_image)

        self._text_img_lbl = QLabel("Загрузите изображение текста")
        self._text_img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._text_img_lbl.setMinimumHeight(180)
        self._text_img_lbl.setStyleSheet(
            "border: 2px dashed #3d5afe; border-radius: 12px; color: #888;"
        )

        self._text_status = QLabel("")
        self._text_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._text_status.setStyleSheet("color: #ffd740; font-size: 13px;")

        self._chars_area   = QScrollArea()
        self._chars_widget = QWidget()
        self._chars_layout = QGridLayout()
        self._chars_layout.setSpacing(6)
        self._chars_widget.setLayout(self._chars_layout)
        self._chars_area.setWidget(self._chars_widget)
        self._chars_area.setWidgetResizable(True)

        left_l.addWidget(btn)
        left_l.addWidget(self._text_img_lbl)
        left_l.addWidget(self._text_status)
        left_l.addWidget(self._chars_area)

        # Info panel
        info_card = self._card()
        info_l    = QVBoxLayout()

        self._sel_char_lbl = QLabel("—")
        self._sel_char_lbl.setFont(QFont("Microsoft YaHei", 72))
        self._sel_char_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sel_char_lbl.setMinimumHeight(0)

        self._char_info_te = QTextEdit()
        self._char_info_te.setReadOnly(True)
        self._char_info_te.setPlaceholderText("Нажмите на символ, чтобы увидеть информацию")

        # Thumbnail of the cropped character
        self._crop_lbl = QLabel()
        self._crop_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._crop_lbl.setFixedHeight(0)

        info_l.addWidget(self._crop_lbl)
        #info_l.addWidget(self._sel_char_lbl)
        info_l.addWidget(self._char_info_te)
        info_card.setLayout(info_l)

        outer.addLayout(left_l, 2)
        outer.addWidget(info_card, 1)
        tab.setLayout(outer)
        return tab

    def _load_text_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return

        px = QPixmap(path)
        self._text_img_lbl.setPixmap(
            px.scaled(400, 240, Qt.AspectRatioMode.KeepAspectRatio,
                      Qt.TransformationMode.SmoothTransformation)
        )
        self._text_status.setText("PaddleOCR сегментирует символы")

        # Clear old buttons
        for i in reversed(range(self._chars_layout.count())):
            w = self._chars_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        worker = OCRWorker(self._ocr_reader, path)
        worker.finished.connect(self._on_text_ocr_done)
        worker.error.connect(lambda e: self._text_status.setText(f"❌ {e}"))
        self._active_worker = worker
        worker.start()

    def _on_text_ocr_done(self, characters):
        """characters: list of (char, x_center, y_center, crop_pil)"""
        if not characters:
            self._text_status.setText("Текст не найден")
            return

        # Sort left-to-right, top-to-bottom (reading order)
        characters.sort(key=lambda c: (c[2] // 40, c[1]))

        cols = 10
        for idx, (ch, xc, yc, crop) in enumerate(characters):
            btn = QPushButton(ch)
            btn.setFixedSize(64, 64)
            btn.setFont(QFont("Microsoft YaHei", 20))
            btn.setToolTip(f"({xc}, {yc})")
            btn.clicked.connect(
                lambda _, c=ch, cr=crop: self._show_char_info(c, cr)
            )
            self._chars_layout.addWidget(btn, idx // cols, idx % cols)

        self._text_status.setText(
            f"Найдено символов: {len(characters)}"
        )
        self._text_status.setStyleSheet("color: #69f0ae; font-size: 13px;")

    def _show_char_info(self, ch: str, crop: Image.Image):
        self._sel_char_lbl.setText(ch)

        # Show crop thumbnail
        try:
            qimg = self._pil_to_qimage(crop)
            px   = QPixmap.fromImage(qimg).scaled(
                80, 80,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self._crop_lbl.setPixmap(px)
        except Exception:
            self._crop_lbl.clear()

        from pypinyin import pinyin, Style
        py_list = pinyin(ch, style=Style.TONE)
        py_str  = " ".join(item[0] for item in py_list)
        meaning = char_meaning(ch)

        self._char_info_te.setText(
            f"Символ:   {ch}\n"
            f"Пиньинь:  {py_str}\n"
            f"Значение: {meaning}"
        )

    # ──────────────────────────────────────────
    # TAB 3 — HANDWRITING COMPARISON
    # ──────────────────────────────────────────

    def _build_hw_tab(self):
        tab    = QWidget()
        outer  = QHBoxLayout()

        # Left: file list + controls
        left_c = self._card()
        left_l = QVBoxLayout()

        add_btn = QPushButton("Добавить изображения")
        add_btn.clicked.connect(self._hw_add_images)

        clr_btn = QPushButton("Очистить список")
        clr_btn.clicked.connect(self._hw_clear)
        clr_btn.setStyleSheet(
            "QPushButton { background: #c62828; } QPushButton:hover { background: #e53935; }"
        )

        self._hw_list = QListWidget()
        self._hw_list.setMinimumHeight(200)

        cmp_btn = QPushButton("Сравнить почерк")
        cmp_btn.clicked.connect(self._hw_compare)

        self._hw_status = QLabel("")
        self._hw_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hw_status.setStyleSheet("color: #ffd740; font-size: 13px;")

        left_l.addWidget(add_btn)
        left_l.addWidget(clr_btn)
        left_l.addWidget(self._hw_list)
        left_l.addWidget(cmp_btn)
        left_l.addWidget(self._hw_status)
        left_c.setLayout(left_l)

        # Right: result panel
        right_c = self._card()
        right_l = QVBoxLayout()

        self._hw_verdict = QLabel("Добавьте минимум\n2 изображения")
        self._hw_verdict.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        self._hw_verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hw_verdict.setWordWrap(True)

        self._hw_prob_bar = QProgressBar()
        self._hw_prob_bar.setRange(0, 100)
        self._hw_prob_bar.setValue(0)
        self._hw_prob_bar.setFixedHeight(22)
        self._hw_prob_bar.setFormat("%v%")

        self._hw_detail = QTextEdit()
        self._hw_detail.setReadOnly(True)
        self._hw_detail.setPlaceholderText("Детали сравнения появятся здесь…")

        right_l.addWidget(self._hw_verdict)
        right_l.addSpacing(10)
        right_l.addWidget(QLabel("Вероятность одного автора:"))
        right_l.addWidget(self._hw_prob_bar)
        right_l.addSpacing(10)
        right_l.addWidget(QLabel("Подробности:"))
        right_l.addWidget(self._hw_detail)
        right_c.setLayout(right_l)

        outer.addWidget(left_c, 1)
        outer.addWidget(right_c, 1)
        tab.setLayout(outer)
        return tab

    def _hw_add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Выберите изображения", "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        for p in paths:
            self._hw_list.addItem(p)

    def _hw_clear(self):
        self._hw_list.clear()
        self._hw_verdict.setText("Добавьте минимум\n2 изображения")
        self._hw_detail.clear()
        self._hw_prob_bar.setValue(0)
        self._hw_status.setText("")

    def _hw_compare(self):
        n = self._hw_list.count()
        if n < 2:
            self._hw_status.setText("Нужно минимум 2 изображения")
            return

        paths = [self._hw_list.item(i).text() for i in range(n)]
        self._hw_status.setText("Анализ почерка")
        self._hw_verdict.setText("…")

        worker = HandwritingWorker(self._feature_extractor, paths)
        worker.finished.connect(self._on_hw_done)
        worker.error.connect(lambda e: self._hw_status.setText(f"❌ {e}"))
        self._active_worker = worker
        worker.start()

    def _on_hw_done(self, prob: float, detail: str):
        pct = int(prob * 100)
        self._hw_prob_bar.setValue(pct)

        if prob >= 0.75:
            verdict = f"Вероятно один автор\n({pct}%)"
            color   = "#69f0ae"
        elif prob >= 0.45:
            verdict = f"Неоднозначно\n({pct}%)"
            color   = "#ffd740"
        else:
            verdict = f"Вероятно разные авторы\n({pct}%)"
            color   = "#ff5252"

        self._hw_verdict.setText(verdict)
        self._hw_verdict.setStyleSheet(f"color: {color};")
        self._hw_detail.setText(detail)
        self._hw_status.setText("Анализ завершён")
        self._hw_status.setStyleSheet("color: #69f0ae; font-size: 13px;")

    # ──────────────────────────────────────────
    # UTILS
    # ──────────────────────────────────────────

    @staticmethod
    def _pil_to_qimage(pil_img: Image.Image) -> QImage:
        pil_img = pil_img.convert("RGBA")
        data    = pil_img.tobytes("raw", "RGBA")
        return QImage(data, pil_img.width, pil_img.height,
                      QImage.Format.Format_RGBA8888)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())