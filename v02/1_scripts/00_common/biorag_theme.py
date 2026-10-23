"""Shared plotnine theme and palette for every biorag-guard-sc figure.

One theme, one palette, one font-size scale across the whole figure set
(``figure-agent.md``). Colours are not chosen by eye: the three categorical
hues are slots 1-3 of a palette validated for colour-vision deficiency, and
they clear the all-pairs gates in both light and dark modes (worst CVD dE 9.2,
worst normal-vision dE 24.0). Slot 3 sits below 3:1 contrast on a light
surface, so every figure that uses it also carries a direct label or a legend
glyph paired with linetype or shape - colour never carries meaning alone.
"""

from __future__ import annotations

from pathlib import Path

from plotnine import (
    element_blank,
    element_line,
    element_rect,
    element_text,
    theme,
    theme_minimal,
)

# Categorical slots 1-3, validated. Assigned in fixed order, never cycled.
ATTACK_COLORS: dict[str, str] = {
    "relabel": "#2a78d6",      # blue
    "duplicate": "#eb6834",    # orange
    "interpolate": "#1baf7a",  # aqua
}

# Paired with colour so identity never depends on hue alone.
ATTACK_SHAPES: dict[str, str] = {"relabel": "o", "duplicate": "s", "interpolate": "^"}
ATTACK_LINETYPES: dict[str, str] = {
    "relabel": "solid",
    "duplicate": "dashed",
    "interpolate": "dotted",
}

ATTACK_ORDER: list[str] = ["relabel", "duplicate", "interpolate"]

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e1"
BASELINE = "#8a8985"


def biorag_theme(base_size: float = 9) -> theme:
    """Recessive grid, quiet axes, text in ink tokens rather than series colour.

    Args:
        base_size: base font size in points.

    Returns:
        A plotnine theme.
    """
    return theme_minimal(base_size=base_size) + theme(
        text=element_text(color=INK),
        axis_text=element_text(color=INK_SOFT, size=base_size - 1),
        axis_title=element_text(color=INK, size=base_size),
        plot_title=element_text(color=INK, size=base_size + 2, weight="bold", ha="left"),
        plot_subtitle=element_text(color=INK_SOFT, size=base_size, ha="left"),
        plot_caption=element_text(color=INK_SOFT, size=base_size - 2, ha="left"),
        panel_grid_major=element_line(color=GRID, size=0.3),
        panel_grid_minor=element_blank(),
        panel_background=element_rect(fill="white", color="none"),
        plot_background=element_rect(fill="white", color="none"),
        strip_text=element_text(color=INK, size=base_size - 1, weight="bold"),
        legend_title=element_text(color=INK, size=base_size - 1),
        legend_text=element_text(color=INK_SOFT, size=base_size - 1),
        figure_size=(6.5, 4),
        dpi=150,
    )


def write_caption(
    out_dir: Path, name: str, title: str, subtitle: str, caption: str
) -> Path:
    """Write a figure's prose to a sibling ``.txt`` instead of onto the canvas.

    Only the title, the legend and the axis titles belong inside a figure. The
    subtitle and caption describe it, and a describing sentence set in 7 pt and
    hand-wrapped to the canvas is unreadable at print size, silently clipped
    when it grows, and impossible to copy into a manuscript. It lives here.

    Args:
        out_dir: directory the figure's artefacts are written to.
        name: figure stem, e.g. ``fig01``.
        title: the figure title, repeated here so the file stands alone.
        subtitle: the conditions the figure was drawn under.
        caption: the explanation. Hand-wrapping is removed; a blank line in the
            source is kept as a paragraph break.

    Returns:
        Path to the written file.
    """
    flowed = "\n\n".join(
        " ".join(block.split()) for block in caption.split("\n\n") if block.strip()
    )
    body = (
        f"{name}\n"
        f"{'=' * len(name)}\n\n"
        f"Title\n-----\n{' '.join(title.split())}\n\n"
        f"Subtitle\n--------\n{' '.join(subtitle.split())}\n\n"
        f"Caption\n-------\n{flowed}\n"
    )
    path = out_dir / f"{name}.txt"
    path.write_text(body, encoding="utf-8")
    return path
