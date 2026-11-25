"""Image listing at ingest: list every image and scanned page, report N images not read."""

from __future__ import annotations

import hashlib
import io
import posixpath
import zipfile
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"})
MIN_SIDE_PX = 64


@dataclass(frozen=True)
class ImageRef:
    rel_key: str
    locator: str
    kind: str  # pptx_media | pdf_image | pdf_scan_page | image_file
    w: int
    h: int
    nbytes: int
    sha256: str


def _get_image_dims(raw: bytes) -> tuple[int, int]:
    try:
        from PIL import Image  # type: ignore[import-untyped]

        with Image.open(io.BytesIO(raw)) as img:
            return img.size
    except ImportError:
        return (0, 0)


def list_pptx_images(raw: bytes, rel_key: str) -> list[ImageRef]:
    """List every media image in a .pptx archive."""
    zf = zipfile.ZipFile(io.BytesIO(raw))
    try:
        # Map slide N -> set of normalized media paths
        slide_images: dict[int, set[str]] = {}
        {n.lower(): n for n in zf.namelist()}
        for name in zf.namelist():
            if not name.endswith(".rels"):
                continue
            base = posixpath.basename(name)
            if not base.startswith("slide") or not base.endswith(".xml.rels"):
                continue
            try:
                slide_num = int(base[len("slide") : -len(".xml.rels")])
            except ValueError:
                continue
            slide_images.setdefault(slide_num, set())
            rels_text = zf.read(name).decode("utf-8", errors="replace")
            for line in rels_text.splitlines():
                if 'Target="' not in line:
                    continue
                idx = line.index('Target="') + len('Target="')
                end = line.index('"', idx)
                target = line[idx:end]
                norm = target.lstrip("/").lower()
                if norm.startswith(".."):
                    parts = target.split("/")
                    if len(parts) >= 3:
                        norm = "/".join(parts[-2:]).lower()
                slide_images[slide_num].add(norm)
        # Build rel_key -> (slide_num, first_appearance) for dedupe
        seen_hashes: set[str] = set()
        refs: list[ImageRef] = []
        for name in sorted(zf.namelist()):
            lower = name.lower()
            if lower.startswith("ppt/media/") and lower.endswith(tuple(IMAGE_EXTS)):
                raw_data = zf.read(name)
                h = hashlib.sha256(raw_data).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                w, h_dim = _get_image_dims(raw_data)
                if w < MIN_SIDE_PX and h_dim < MIN_SIDE_PX:
                    continue
                # Find first slide using this image
                norm_name = lower.lstrip("/").lower()
                first_slide = None
                for sn, paths in sorted(slide_images.items()):
                    if norm_name in paths:
                        first_slide = sn
                        break
                locator = f"slide {first_slide} / {name}" if first_slide is not None else name
                refs.append(
                    ImageRef(
                        rel_key=rel_key,
                        locator=locator,
                        kind="pptx_media",
                        w=w,
                        h=h_dim,
                        nbytes=len(raw_data),
                        sha256=h,
                    )
                )
    finally:
        zf.close()
    return refs


def list_pdf_images(
    path: Path,
    rel_key: str,
    *,
    text_chars_by_page: dict[int, int],
    file_sha256: str = "",
) -> list[ImageRef]:
    """List images in a PDF using pypdfium2."""
    try:
        import pypdfium2 as pdfium  # type: ignore[import-untyped]
    except ImportError:
        return []
    refs: list[ImageRef] = []
    try:
        doc = pdfium.PdfDocument(path)
        try:
            num_pages = len(doc)
            for page_idx in range(num_pages):
                page_num = page_idx + 1
                page_w, page_h = 612.0, 792.0
                page_data: list[tuple[int, bytes, int, int]] = []
                try:
                    page = doc[page_idx]
                    try:
                        page_w = page.get_width()
                        page_h = page.get_height()
                        objects = list(page.get_objects())
                        for idx, img in enumerate(objects):
                            try:
                                raw_data = bytes(img.get_data())
                                pw, ph = img.get_px_size()
                                page_data.append((idx, raw_data, pw, ph))
                            except Exception:
                                pass
                    finally:
                        page.close()
                except Exception:
                    pass
                if not page_data:
                    continue
                tchars = text_chars_by_page.get(page_num, 0)
                is_scan = tchars < 20 and any(
                    pw * ph >= page_w * page_h * 0.5 for _, _, pw, ph in page_data
                )
                if is_scan:
                    scan_h = hashlib.sha256(
                        f"scan:{rel_key}:{page_num}:{file_sha256}".encode()
                    ).hexdigest()
                    refs.append(
                        ImageRef(
                            rel_key=rel_key,
                            locator=f"page {page_num}",
                            kind="pdf_scan_page",
                            w=int(page_w),
                            h=int(page_h),
                            nbytes=0,
                            sha256=scan_h,
                        )
                    )
                else:
                    seen: set[str] = set()
                    for idx, raw_data, pw, ph in page_data:
                        h = hashlib.sha256(raw_data).hexdigest()
                        if h in seen:
                            continue
                        seen.add(h)
                        refs.append(
                            ImageRef(
                                rel_key=rel_key,
                                locator=f"page {page_num} / image {idx}",
                                kind="pdf_image",
                                w=pw,
                                h=ph,
                                nbytes=len(raw_data),
                                sha256=h,
                            )
                        )
        finally:
            doc.close()
    except Exception:
        pass
    return refs


def image_file_ref(path: Path, rel_key: str) -> ImageRef | None:
    """Create one ImageRef for a standalone image file."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    w, h = _get_image_dims(raw)
    if w < MIN_SIDE_PX and h < MIN_SIDE_PX:
        return None
    return ImageRef(
        rel_key=rel_key,
        locator=path.name,
        kind="image_file",
        w=w,
        h=h,
        nbytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
