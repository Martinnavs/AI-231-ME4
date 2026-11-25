"""OCR engine wrapper: RapidOCR + Tesseract fallback, pre-checks, dedupe, chunking."""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from voice_agent.rag.images import ImageRef

OCR_MODELS_DIR = Path(os.environ.get("VOICE_AGENT_OCR_MODELS", "assets/ocr"))
OCR_CHUNK_CHARS = 1200
OCR_CHUNK_SPLIT = 900
OCR_LINE_CONF_MIN = 0.60
OCR_LINE_ASCII_MIN = 0.50
OCR_TIMEOUT_S = 30
MIN_SIDE_PX = 64
MIN_AREA_PX2 = 8192
MIN_NBYTES = 2048
MAX_ASPECT = 12
NARROW_MIN = 40
SOLID_WINDOW = 32
SOLID_STDDEV = 5.0


@dataclass(frozen=True)
class OcrResult:
    """Text extracted from one image."""

    rel_key: str
    locator: str
    kind: str
    text: str
    confidence: float

    @property
    def chars(self) -> int:
        return len(re.sub(r"\s", "", self.text))


def _normalize_image(raw: bytes, *, max_long: int = 960) -> bytes:
    """Composite alpha, EXIF transpose, RGB, cap longest side."""
    from PIL import Image, ImageOps  # type: ignore[import-untyped]

    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    img = ImageOps.exif_transpose(img) or img
    if img.mode != "RGB":
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_long:
        scale = max_long / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def should_skip(ref: ImageRef, raw: bytes) -> bool:
    """Cheap pre-checks: skip tiny, solid, extreme aspect-ratio images."""
    w, h = ref.w, ref.h
    if min(w, h) < MIN_SIDE_PX:
        return True
    if w * h < MIN_AREA_PX2:
        return True
    if len(raw) < MIN_NBYTES:
        return True
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > MAX_ASPECT and min(w, h) < NARROW_MIN:
        return True
    # Near-solid check
    try:
        from PIL import Image  # type: ignore[import-untyped]

        img = Image.open(io.BytesIO(raw)).convert("L")
        img = img.resize((SOLID_WINDOW, SOLID_WINDOW))
        pixels = list(img.getdata())
        mean = sum(pixels) / len(pixels)
        variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)
        if variance**0.5 < SOLID_STDDEV:
            return True
    except Exception:
        pass
    return False


def extract_image_bytes(ref: ImageRef, raw_root: Path) -> bytes:
    """Read image bytes from source (zip, PDF, or file)."""
    if ref.kind == "pptx_media":
        parts = ref.locator.split(" / ", 1)
        zip_name = parts[1] if len(parts) == 2 else ref.locator
        zip_path = raw_root / ref.rel_key
        import zipfile  # noqa: PLC0415

        zf = zipfile.ZipFile(zip_path)
        try:
            return zf.read(zip_name)
        finally:
            zf.close()

    if ref.kind in ("pdf_image", "pdf_scan_page"):
        pdf_path = raw_root / ref.rel_key
        import pypdfium2 as pdfium  # type: ignore[import-untyped]  # noqa: PLC0415

        doc = pdfium.PdfDocument(pdf_path)
        try:
            page_num = 0
            m = re.search(r"page\s+(\d+)", ref.locator)
            if m:
                page_num = int(m.group(1)) - 1
            max_side = 1280 if ref.kind == "pdf_scan_page" else 960
            scale = 150 / 72 if ref.kind == "pdf_scan_page" else 1.0
            page = doc[page_num]
            try:
                pw = page.get_width()
                ph = page.get_height()
                longest = max(pw, ph)
                if longest * scale > max_side:
                    scale = max_side / longest
                bitmap = page.render(scale=scale)
            finally:
                page.close()
            pil_img = bitmap.to_pil()
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            return pil_img.tobytes()  # type: ignore[no-any-return]
        finally:
            doc.close()

    if ref.kind == "image_file":
        return (raw_root / ref.rel_key).read_bytes()

    raise OSError(f"cannot extract image for {ref.kind}: {ref.rel_key}")


def run_rapidocr(image_bytes: bytes) -> tuple[str, float] | None:
    """Run RapidOCR on image bytes. Returns (text, confidence) or None."""
    if not _build_rapidocr_engine():
        return None

    engine = _ENGINE
    if engine is None:
        return None
    try:
        result = engine(image_bytes)  # type: ignore[operator]
    except Exception:
        return None

    if result is None:
        return "", 0.0

    try:
        lines, confidences = _extract_lines_and_confs(result)
    except Exception:
        return None
    if not lines:
        return "", 0.0
    conf = sum(confidences) / len(confidences) if confidences else 0.0
    text = "\n".join(line[0] for line in lines)
    return text, conf


def _extract_lines_and_confs(result: object) -> tuple[list[tuple[str, float]], list[float]]:
    """Extract sorted text lines and confidences from RapidOCR result."""
    lines: list[tuple[str, float, float, float]] = []
    confidences: list[float] = []
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        boxes = result[0] if result[0] is not None else []
        recs = result[1] if len(result) > 1 and result[1] is not None else []
        for i, box in enumerate(boxes):
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            xs = [b[0] for b in box]
            ys = [b[1] for b in box]
            cx, cy = sum(xs) / 4, sum(ys) / 4
            text = ""
            conf = -1.0
            if i < len(recs):
                r = recs[i]
                if isinstance(r, (list, tuple)) and len(r) >= 1:
                    text = str(r[0])
                if isinstance(r, (list, tuple)) and len(r) >= 2:
                    try:
                        conf = float(r[1])
                    except (ValueError, TypeError):
                        conf = -1.0
            lines.append((text, conf, cx, cy))
            if conf >= 0:
                confidences.append(conf)
    if not lines:
        return [], []
    lines.sort(key=lambda item: (round(item[3] / 20) * 20, item[2]))
    return [(item[0], item[1]) for item in lines], confidences


_ENGINE: object | None = None


def _build_rapidocr_engine() -> bool:
    """Lazily build the RapidOCR engine. Returns True on success."""
    global _ENGINE  # noqa: PLW0603
    if _ENGINE is not None:
        return True
    try:
        from rapidocr import RapidOCR  # type: ignore[import-not-found]

        params: dict[str, object] = {
            "Det.engine_type": "onnxruntime",
            "Rec.engine_type": "onnxruntime",
            "Global.use_cls": False,
            "Det.limit_side_len": 960,
            "EngineConfig.onnxruntime.intra_op_num_threads": 1,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.enable_cpu_mem_arena": False,
        }
        det = OCR_MODELS_DIR / "ch_PP-OCRv4_det_infer.onnx"
        rec = OCR_MODELS_DIR / "en_PP-OCRv4_mobile_rec.onnx"
        keys = OCR_MODELS_DIR / "en_dict.txt"
        if det.exists():
            params["Det.model_path"] = str(det)
        if rec.exists():
            params["Rec.model_path"] = str(rec)
        if keys.exists():
            params["Rec.rec_keys_path"] = str(keys)
        params["Global.font_path"] = ""
        _ENGINE = RapidOCR(params=params)
        return True
    except Exception:
        return False


def engine_available() -> bool:
    """True when RapidOCR (with its models) can be built or the tesseract binary exists."""
    return _build_rapidocr_engine() or shutil.which("tesseract") is not None


def run_tesseract(image_bytes: bytes) -> tuple[str, float] | None:
    """Run Tesseract CLI on PNG bytes. Returns (text, confidence) or None."""
    if not shutil.which("tesseract"):
        return None
    try:
        proc = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", "eng", "--oem", "1", "--psm", "3"],
            input=image_bytes,
            capture_output=True,
            timeout=OCR_TIMEOUT_S,
        )
        text = proc.stdout.decode("utf-8", errors="replace").strip()
        # Tesseract doesn't give confidence via CLI easily; estimate from text quality
        conf = 0.9 if text else 0.0
        return text, conf
    except (subprocess.TimeoutExpired, OSError):
        return None


def run_ocr(image_bytes: bytes, *, kind: str = "") -> tuple[str, float]:
    """Run OCR: RapidOCR primary, Tesseract fallback for empty text."""
    result = run_rapidocr(image_bytes)
    if result is None:
        result = run_tesseract(image_bytes)
    if result is not None:
        return result
    return "", 0.0


def filter_lines(text: str, raw_lines: list[str] | None = None) -> tuple[str, float]:
    """Drop low-confidence or non-ASCII lines. Returns (filtered_text, mean_conf)."""
    if raw_lines is None:
        return text, 0.9 if text.strip() else 0.0

    kept: list[str] = []
    kept_conf: list[float] = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        ascii_ratio = sum(
            1 for c in stripped if c.isprintable() and (c.isalnum() or c in " .,;:!?()-+/=%$")
        ) / max(len(stripped), 1)
        if ascii_ratio < OCR_LINE_ASCII_MIN:
            continue
        kept.append(stripped)
        kept_conf.append(1.0)

    text = "\n".join(kept)
    conf = sum(kept_conf) / len(kept_conf) if kept_conf else 0.0
    return text, conf


def chunk_ocr_text(text: str) -> list[str]:
    """Split OCR text into chunks at ~900 chars if > 1200 chars total."""
    if len(text) <= OCR_CHUNK_CHARS:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + OCR_CHUNK_SPLIT
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Find a line break near end
        br = text.rfind("\n", start + OCR_CHUNK_SPLIT - 100, end)
        if br < 0:
            br = text.find("\n", end - 50, end + 50)
        if br < 0:
            br = end
        chunks.append(text[start:br].strip())
        start = br + 1
    return chunks


@dataclass
class OcrSession:
    """State for a single OCR phase-2 run."""

    seen_hashes: dict[str, tuple[str, float]] = field(default_factory=dict)
    read_count: int = 0
    skipped_count: int = 0
    unread_count: int = 0
    empty_count: int = 0
    duplicate_count: int = 0


def ocr_image(
    ref: ImageRef,
    raw: bytes,
    session: OcrSession,
    *,
    max_long: int = 960,
) -> tuple[str, float, str]:
    """Process one image through pre-check, OCR, and filtering.

    Returns (text, confidence, status_note).
    """
    h = ref.sha256
    if h in session.seen_hashes:
        cached = session.seen_hashes[h]
        session.duplicate_count += 1
        return cached[0], cached[1], "duplicate"

    if should_skip(ref, raw):
        return "", 0.0, "skipped pre-check"

    try:
        normalized = _normalize_image(raw, max_long=max_long)
    except Exception:
        return "", 0.0, "skipped normalize"

    text, conf = run_ocr(normalized, kind=ref.kind)
    text, conf = filter_lines(text)

    note = ""
    if text.strip():
        session.read_count += 1
        note = "read"
    else:
        session.unread_count += 1
        note = "ocr returned empty"

    if text.strip():
        session.seen_hashes[h] = (text, conf)
    return text, conf, note
