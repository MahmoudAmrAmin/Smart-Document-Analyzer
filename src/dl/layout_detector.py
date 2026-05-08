"""
Layout detection for PDF documents.
Strategy:
  1. Try LayoutParser with pretrained PubLayNet model
  2. If unavailable → rule-based layout detection
     (no torch/detectron2 needed)

Detected region types: title, paragraph, table, list, figure
"""

import os
import sys
import re
from typing import List, Dict
from dataclasses import dataclass, field

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, ROOT_DIR)


# ── Data structures ────────────────────────────────────
@dataclass
class LayoutRegion:
    region_type: str        # title | paragraph | table | list | figure
    text:        str
    page:        int = 0
    confidence:  float = 1.0
    bbox:        List[float] = field(default_factory=list)


@dataclass
class DocumentLayout:
    filename:    str
    regions:     List[LayoutRegion]
    method:      str        # "layoutparser" or "rules"
    page_count:  int = 0

    @property
    def titles(self) -> List[LayoutRegion]:
        return [r for r in self.regions if r.region_type == "title"]

    @property
    def paragraphs(self) -> List[LayoutRegion]:
        return [r for r in self.regions if r.region_type == "paragraph"]

    @property
    def tables(self) -> List[LayoutRegion]:
        return [r for r in self.regions if r.region_type == "table"]

    def summary(self) -> dict:
        from collections import Counter
        counts = Counter(r.region_type for r in self.regions)
        return {
            "method":     self.method,
            "pages":      self.page_count,
            "regions":    len(self.regions),
            "breakdown":  dict(counts)
        }


# ══════════════════════════════════════════════════════
#  LAYOUTPARSER DETECTOR
# ══════════════════════════════════════════════════════

def _detect_with_layoutparser(pdf_path: str) -> DocumentLayout:
    """
    Uses LayoutParser + PubLayNet pretrained model.
    Converts PDF pages to images, runs model, extracts regions.
    """
    import layoutparser as lp
    import fitz
    from PIL import Image
    import io
    import numpy as np

    # PubLayNet model — detects: text, title, list, table, figure
    model = lp.Detectron2LayoutModel(
        config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
        label_map={0: "text", 1: "title", 2: "list",
                   3: "table", 4: "figure"}
    )

    doc     = fitz.open(pdf_path)
    regions = []

    for page_num, page in enumerate(doc):
        # Render page to image
        pix       = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        image     = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(image)

        # Detect layout
        layout = model.detect(img_array)

        for block in layout:
            region_type = block.type.lower()
            if region_type == "text":
                region_type = "paragraph"

            # Extract text from the bounding box area of the page
            x1, y1, x2, y2 = (block.block.x_1, block.block.y_1,
                               block.block.x_2, block.block.y_2)
            # Scale bbox from image coords to PDF coords
            scale_x = page.rect.width  / image.width
            scale_y = page.rect.height / image.height
            rect    = fitz.Rect(
                x1 * scale_x, y1 * scale_y,
                x2 * scale_x, y2 * scale_y
            )
            text = page.get_text(clip=rect).strip()

            if text and len(text) > 5:
                regions.append(LayoutRegion(
                    region_type=region_type,
                    text=text,
                    page=page_num + 1,
                    confidence=float(block.score),
                    bbox=[x1, y1, x2, y2]
                ))

    doc.close()
    return DocumentLayout(
        filename=os.path.basename(pdf_path),
        regions=regions,
        method="layoutparser",
        page_count=len(doc)
    )


# ══════════════════════════════════════════════════════
#  RULE-BASED FALLBACK DETECTOR
# ══════════════════════════════════════════════════════

def _is_title(line: str) -> bool:
    """Heuristic: short, no period at end, possibly title-cased or ALL CAPS."""
    line = line.strip()
    if not line:
        return False
    words = line.split()
    if len(words) > 12:
        return False
    if line.endswith(".") or line.endswith(","):
        return False
    if line.isupper() and len(words) >= 1:
        return True
    if line.istitle() and len(words) <= 8:
        return True
    if re.match(r"^\d+[\.\)]\s+\w", line) and len(words) <= 10:
        return True   # numbered heading
    return False


def _is_table_row(line: str) -> bool:
    """Heuristic: contains multiple tab/pipe separators."""
    return line.count("\t") >= 2 or line.count("|") >= 2


def _is_list_item(line: str) -> bool:
    """Heuristic: starts with bullet or number."""
    return bool(re.match(r"^[\-\•\*\–]\s+\w", line.strip()) or
                re.match(r"^\d+[\.\)]\s+\w", line.strip()))


def _detect_with_rules(text: str, filename: str) -> DocumentLayout:
    """
    Rule-based layout detection from plain text.
    No ML model needed — uses heuristics on line structure.
    """
    lines   = text.split("\n")
    regions = []
    page    = 1

    # Track page breaks (we inserted --- Page N --- markers)
    para_buffer = []

    def flush_paragraph():
        nonlocal para_buffer
        content = " ".join(para_buffer).strip()
        if content and len(content.split()) > 5:
            regions.append(LayoutRegion(
                region_type="paragraph",
                text=content,
                page=page
            ))
        para_buffer = []

    for line in lines:
        stripped = line.strip()

        # Detect page marker
        page_match = re.match(r"---\s*Page\s*(\d+)", stripped)
        if page_match:
            flush_paragraph()
            page = int(page_match.group(1))
            continue

        if not stripped:
            flush_paragraph()
            continue

        if _is_table_row(stripped):
            flush_paragraph()
            regions.append(LayoutRegion(
                region_type="table",
                text=stripped,
                page=page
            ))

        elif _is_list_item(stripped):
            flush_paragraph()
            regions.append(LayoutRegion(
                region_type="list",
                text=stripped,
                page=page
            ))

        elif _is_title(stripped):
            flush_paragraph()
            regions.append(LayoutRegion(
                region_type="title",
                text=stripped,
                page=page
            ))

        else:
            para_buffer.append(stripped)

    flush_paragraph()

    return DocumentLayout(
        filename=filename,
        regions=regions,
        method="rules",
        page_count=page
    )


# ══════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════

def detect_layout(pdf_path: str = None,
                  text: str = None,
                  filename: str = "") -> DocumentLayout:
    """
    Detect document layout.
    Tries LayoutParser only if detectron2 is confirmed available.
    Falls back to rule-based detection automatically.
    """
    fname = filename or (os.path.basename(pdf_path) if pdf_path else "document")

    # Check if detectron2 is actually available before trying
    detectron2_available = False
    if pdf_path and os.path.exists(pdf_path):
        try:
            import detectron2
            import layoutparser as lp
            # Confirm the specific model attribute exists
            if hasattr(lp, "Detectron2LayoutModel"):
                detectron2_available = True
        except (ImportError, Exception):
            pass

    # Try LayoutParser only if confirmed available
    if detectron2_available:
        try:
            print(f"  Trying LayoutParser for: {fname}")
            layout = _detect_with_layoutparser(pdf_path)
            print(f"  ✅ LayoutParser: {len(layout.regions)} regions")
            return layout
        except Exception as e:
            print(f"  ⚠️  LayoutParser failed ({e}) → using rules")

    # Rule-based — direct, no attempt message
    if text is None:
        if pdf_path:
            import fitz
            doc  = fitz.open(pdf_path)
            text = "".join(page.get_text() for page in doc)
            doc.close()
        else:
            raise ValueError("Provide either pdf_path or text")

    layout = _detect_with_rules(text, fname)
    print(f"  ✅ {fname}: {len(layout.regions)} regions "
          f"{layout.summary()['breakdown']}")
    return layout


def detect_layouts_batch(documents: dict,
                         pdf_folder: str = None) -> dict:
    """
    Run layout detection on all documents.

    Args:
        documents:  {filename: clean_text}
        pdf_folder: path to PDFs folder for LayoutParser attempt

    Returns:
        {filename: DocumentLayout}
    """
    layouts = {}
    for filename, text in documents.items():
        print(f"\n  Layout detection: {filename}")
        pdf_path = None
        if pdf_folder:
            candidate = os.path.join(pdf_folder, filename)
            if os.path.exists(candidate):
                pdf_path = candidate

        layouts[filename] = detect_layout(
            pdf_path=pdf_path,
            text=text,
            filename=filename
        )
    return layouts