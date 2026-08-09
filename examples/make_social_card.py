"""Render docs/social-preview.png — the result stated as the two numbers that decide it.

The card this replaces showed a funnel ending in a count of factors that survived costs, taken
from a run nothing on disk could regenerate. What replaces it is the measurement from the full
456-factor scan, which is both reproducible and a better claim: training-side selection carries
real information, and the ten best factors it finds still lose to the universe they came from.

Neither of the usual stories fits that pair, which is exactly why it is worth a card.

Run: python examples/make_social_card.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "docs" / "social-preview.png"
BG, INK, DIM, SLATE, ACCENT, WARN = "#0d1117", "#e6edf3", "#8b949e", "#64748b", "#7c3aed", "#f0883e"

MARK = [
    (72, 128, 250, 128, SLATE, 54), (72, 384, 278, 384, SLATE, 54),
    (330, 64, 330, 448, SLATE, 48), (330, 64, 392, 64, SLATE, 48),
    (330, 448, 392, 448, SLATE, 48), (72, 256, 464, 256, ACCENT, 64),
]

# label, value, sublabel, bar fraction of the worst case, colour
ROWS = [
    ("all 456 factors", "−0.71%", "mean test excess per hold", 1.00, SLATE),
    ("the training-window top 10", "−0.24%", "selected without seeing the test window", 0.34, WARN),
]


def font(px: int, bold: bool = False):
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), ("arialbd.ttf" if bold else "arial.ttf")):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    W, H, ss = 1280, 640, 2
    img = Image.new("RGB", (W * ss, H * ss), BG)
    d = ImageDraw.Draw(img)

    s = (150 * ss) / 512
    ox, oy = 74 * ss, 60 * ss
    for x1, y1, x2, y2, col, w in MARK:
        width = w * s
        d.line([ox + x1 * s, oy + y1 * s, ox + x2 * s, oy + y2 * s], fill=col, width=int(round(width)))
        r = width / 2
        for cx, cy in ((ox + x1 * s, oy + y1 * s), (ox + x2 * s, oy + y2 * s)):
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)

    d.text((246 * ss, 66 * ss), "Perception-XAlpha Lite", font=font(38 * ss, True), fill=INK)
    d.text((246 * ss, 120 * ss), "456 published factors, ranked on training data only.",
           font=font(24 * ss), fill=DIM)

    d.text((80 * ss, 214 * ss), "Selection works. It is not enough.",
           font=font(46 * ss, True), fill=INK)

    y = 300 * ss
    for label, value, sub, frac, colour in ROWS:
        d.text((80 * ss, y), label, font=font(25 * ss, True), fill=INK)
        d.text((80 * ss, y + 34 * ss), sub, font=font(18 * ss), fill=DIM)
        bx, bw = 560 * ss, 380 * ss
        d.rounded_rectangle([bx, y + 2 * ss, bx + int(bw * frac), y + 40 * ss],
                            radius=19 * ss, fill=colour)
        d.text((bx + bw + 24 * ss, y + 2 * ss), value, font=font(38 * ss, True), fill=INK)
        y += 104 * ss

    d.text((80 * ss, (H - 96) * ss),
           "Choosing cleanly is worth +0.47pp a hold — and still loses to the universe it drew from.",
           font=font(22 * ss, True), fill=ACCENT)
    d.text((80 * ss, (H - 62) * ss),
           "Spearman ρ = +0.48 train-to-test across all 456. Before any cost.",
           font=font(19 * ss), fill=DIM)
    d.text((80 * ss, (H - 32) * ss),
           "Research-only. No orders, no profitability claim.",
           font=font(17 * ss), fill=SLATE)

    img.resize((W, H), Image.LANCZOS).save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
