"""
Optional shelf-tag OCR (SAMGRAHA proposal mockup: product name/price read
off the shelf-lip price rail, shown as "OCR 0.89" next to each slot).

This is opt-in and degrades gracefully: it requires `pip install
pytesseract` *and* a local Tesseract OCR binary on PATH (not a pure-Python
dependency, so it isn't in requirements.txt by default). If either is
missing, `LabelReader.available` is False and `read()` always returns None
-- callers should treat OCR purely as a "nice to have" demo overlay, never
as something the core void/fill-ratio alerting logic depends on.

Not claimed as spec-accurate: the spec doesn't actually call for OCR (SKU
identity comes from the planogram slot mapping, not by reading the tag) --
this exists purely to reproduce the pitch deck's demo visual.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2

logger = logging.getLogger("ocr")

try:
    import pytesseract
    _PYTESSERACT_IMPORTED = True
except ImportError:
    _PYTESSERACT_IMPORTED = False


@dataclass
class LabelReading:
    text: str
    confidence: float  # 0.0-1.0


class LabelReader:
    def __init__(self):
        self.available = _PYTESSERACT_IMPORTED
        if not self.available:
            logger.info(
                "OCR disabled: `pip install pytesseract` and install the Tesseract "
                "binary to enable shelf-tag text overlays."
            )

    def read(self, frame, rect: Tuple[int, int, int, int]) -> Optional[LabelReading]:
        if not self.available:
            return None
        x, y, w, h = rect
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
        except Exception as exc:  # pytesseract.TesseractNotFoundError, etc.
            logger.warning("OCR read failed (%s); disabling further OCR calls", exc)
            self.available = False
            return None

        words, confidences = [], []
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            text = text.strip()
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                continue
            if text and conf >= 0:
                words.append(text)
                confidences.append(conf)

        if not words:
            return None
        return LabelReading(text=" ".join(words), confidence=(sum(confidences) / len(confidences)) / 100.0)
