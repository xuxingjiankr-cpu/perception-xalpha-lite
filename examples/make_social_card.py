"""Render docs/social-preview.png — the pipeline stated as its own funnel.

The card shows what the framework did to a published factor library rather than a headline
number: how many candidates went in, how many reached a shortlist, how many survived real
costs, and how many were promoted. That last number is zero, and putting it on the card is
deliberate — a tool that reports its own hit rate is making a stronger claim than one that
advertises a return.

Run: python examples/make_social_card.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "docs" / "social-preview.png"
BG, INK, DIM, SLATE, ACCENT = "#0d1117", "#e6edf3", "#8b949e", "#64748b", "#7c3aed"

MARK = [
    (72, 128, 250, 128, SLATE, 54), (72, 384, 278, 384, SLATE, 54),
    (330, 64, 330, 448, SLATE, 48), (330, 64, 392, 64, SLATE, 48),
    (330, 448, 392, 448, SLATE, 48), (72, 256, 464, 256, ACCENT, 64),
]

# value, label, sublabel, bar width fraction
STAGES = [
    ("456", "searched", "published formulaic factors", 1.00),
    ("10", "shortlisted", "ranked on training data only", 0.34),
    ("4", "survived costs", "30 bps round trip, point-in-time", 0.18),
    ("4", "under forward observation", "frozen spec, scored from Aug 2026", 0.18),
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
    ox, oy = 74 * ss, 66 * ss
    for x1, y1, x2, y2, col, w in MARK:
        width = w * s
        d.line([ox + x1 * s, oy + y1 * s, ox + x2 * s, oy + y2 * s], fill=col, width=int(round(width)))
        r = width / 2
        for cx, cy in ((ox + x1 * s, oy + y1 * s), (ox + x2 * s, oy + y2 * s)):
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)

    d.text((246 * ss, 74 * ss), "Perception-XAlpha Lite", font=font(40 * ss, True), fill=INK)
    d.text((246 * ss, 130 * ss), "One run against a published factor library.",
           font=font(25 * ss), fill=DIM)

    y = 216 * ss
    for index, (value, label, sub, frac) in enumerate(STAGES):
        colour = ACCENT if index >= 2 else SLATE
        d.text((80 * ss, y - 6 * ss), value, font=font(46 * ss, True),
               fill=INK)
        d.text((214 * ss, y + 2 * ss), label, font=font(28 * ss, True), fill=INK)
        d.text((214 * ss, y + 38 * ss), sub, font=font(19 * ss), fill=DIM)
        bx = 700 * ss
        d.rounded_rectangle([bx, y + 6 * ss, bx + int(480 * ss * frac), y + 44 * ss],
                            radius=19 * ss, fill=colour)
        y += 86 * ss

    d.text((80 * ss, (H - 76) * ss),
           "It reports what survived, and what is still being tested.",
           font=font(23 * ss, True), fill=ACCENT)
    d.text((80 * ss, (H - 42) * ss),
           "Research-only. No orders, no profitability claim.",
           font=font(19 * ss), fill=DIM)

    img.resize((W, H), Image.LANCZOS).save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
