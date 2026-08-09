"""Render docs/measured-corrections.png — the size of each bias this framework removes.

Every number is a measurement taken while building and auditing the pipeline, on a real
equity panel, and each one is the difference between what a flawed pipeline reports and what
survives once the flaw is removed. None of them is a return claim; they are the corrections,
not the results.

Run: python examples/make_corrections_figure.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "docs" / "measured-corrections.png"
BG, INK, DIM, FLAW, KEPT, RULE = "#0d1117", "#e6edf3", "#8b949e", "#ef4444", "#7c3aed", "#30363d"

# label, what the flawed pipeline reports, what survives, unit, relative bar length
ROWS = [
    ("Factors chosen with hindsight", "+2.00", "-1.24", "bps/day, same panel and cost", 1.00),
    ("Limit-locked legs priced as fillable", "+6.05", "+0.38", "% forward return of those legs", 0.94),
    ("Universe filtered on whole history", "391", "77", "eligible names, first year", 0.80),
    ("Overlapping labels scored as independent", "-5.79", "-2.25", "t-statistic on pure noise", 0.61),
]


def font(px: int, bold: bool = False):
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), ("arialbd.ttf" if bold else "arial.ttf")):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    W, H, ss = 1160, 700, 2
    img = Image.new("RGB", (W * ss, H * ss), BG)
    d = ImageDraw.Draw(img)
    d.text((60 * ss, 44 * ss), "Four ways a backtest inflates itself",
           font=font(38 * ss, True), fill=INK)
    d.text((60 * ss, 100 * ss),
           "Each measured on a real equity panel, then removed from the pipeline.",
           font=font(21 * ss), fill=DIM)

    y = 180 * ss
    for label, flawed, kept, unit, frac in ROWS:
        d.text((60 * ss, y), label, font=font(24 * ss, True), fill=INK)
        d.text((60 * ss, y + 33 * ss), unit, font=font(17 * ss), fill=DIM)
        bx = 610 * ss
        bw = int(380 * ss * frac)
        d.rounded_rectangle([bx, y + 2 * ss, bx + bw, y + 26 * ss], radius=12 * ss, fill=FLAW)
        d.text((bx + bw + 14 * ss, y + 1 * ss), flawed, font=font(24 * ss, True), fill=FLAW)
        kw = max(int(bw * 0.16), 26 * ss)
        d.rounded_rectangle([bx, y + 40 * ss, bx + kw, y + 64 * ss], radius=12 * ss, fill=KEPT)
        d.text((bx + kw + 14 * ss, y + 39 * ss), kept, font=font(24 * ss, True), fill=KEPT)
        y += 112 * ss
        d.line([60 * ss, y - 22 * ss, (W - 60) * ss, y - 22 * ss], fill=RULE, width=2)

    d.text((60 * ss, (H - 58) * ss),
           "Red is what the flaw reports. Violet is what survives. Neither is a return claim.",
           font=font(20 * ss), fill=DIM)
    img.resize((W, H), Image.LANCZOS).save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
