from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from PIL import ImageFont


_FONT_CANDIDATES = [
    # User/system override first.
    os.environ.get("YOLO_CHESS_FONT", ""),
    # Debian/Ubuntu/Colab.
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    # Arch/Nix/common paths.
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/run/current-system/sw/share/X11/fonts/TTF/DejaVuSans.ttf",
    "/run/current-system/sw/share/fonts/truetype/DejaVuSans.ttf",
    "/run/current-system/sw/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def find_cyrillic_font_path() -> str | None:
    """Return a font file path that supports Cyrillic, if one can be found.

    Matplotlib ships DejaVu Sans in many environments, even when the system has
    no global font package installed. That is why we try font_manager.findfont()
    after checking common OS/Nix paths.
    """
    for candidate in _FONT_CANDIDATES:
        if not candidate:
            continue
        p = Path(candidate)
        if p.exists() and p.is_file():
            return str(p)

    for family in ("DejaVu Sans", "Liberation Sans", "Arial"):
        try:
            path = font_manager.findfont(family, fallback_to_default=True)
            if path and Path(path).exists():
                return path
        except Exception:
            pass

    return None


def get_matplotlib_cyrillic_font(size: int | None = None) -> FontProperties:
    path = find_cyrillic_font_path()
    if path is not None:
        return FontProperties(fname=path, size=size)
    return FontProperties(family="DejaVu Sans", size=size)


def configure_matplotlib_cyrillic() -> FontProperties:
    """Configure matplotlib for Russian labels and return FontProperties.

    Use this once in notebooks before drawing. The returned font can also be
    passed explicitly to ax.text()/set_title() for maximum reliability.
    """
    font_path = find_cyrillic_font_path()
    if font_path is not None:
        try:
            font_manager.fontManager.addfont(font_path)
        except Exception:
            pass
        name = FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = [name, "DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return get_matplotlib_cyrillic_font()


def get_pil_cyrillic_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = find_cyrillic_font_path()
    if path is not None:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()
