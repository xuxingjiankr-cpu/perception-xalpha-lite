# Logo drop-in spec

The README header is already wired for a logo. Save the artwork as `docs/logo.svg` (or
`docs/logo.png`) and it appears immediately — no other edit needed.

## What to hand the designer

**Where it appears:** top of the README, left of the project name, and as the GitHub social
preview card. It is a small mark, not a hero image — the equity-curve figure directly below it
is doing the persuading, so the logo has to survive being small and next to a chart without
competing with it.

**Required sizes**

| use | size | file |
|---|---|---|
| README header | renders at 84 px tall | `docs/logo.svg` preferred, `logo.png` at 2x (168 px) acceptable |
| GitHub social preview | 1280 × 640 px, mark centred with generous margin | `docs/social-preview.png` |
| favicon / avatar (optional) | 512 × 512 px square | `docs/logo-square.png` |

**Constraints that matter more than style**

- **Must read on both light and dark backgrounds.** GitHub users are roughly split, and a mark
  that vanishes in dark mode looks broken to half the audience. Either use a mid-tone palette
  that works on both, or supply two files and the README can switch with
  `<picture><source media="(prefers-color-scheme: dark)">`.
- **Legible at 84 px tall and at 32 px square.** Fine hairlines, thin serifs and small internal
  text disappear. Test by shrinking before approving.
- **SVG strongly preferred** over raster: it stays sharp on high-DPI screens and keeps the
  repository small.
- **No candlesticks, bulls, bears, rockets, money symbols or upward arrows.** This project's
  entire position is that it refuses to promise returns; a mark implying gains contradicts the
  README two lines below it and undercuts the one property competitors cannot copy.

**Concepts that fit the thesis** — the project is about *rejecting* false discoveries, not
finding treasure. Directions worth exploring: a sieve or filter; a signal separated from noise;
a falsification mark (something crossed out deliberately); a boundary or gate; the letter X
from XAlpha built out of a split or partition. Something austere and technical reads as more
credible here than something energetic.

## Wiring, once the file exists

Replace the plain `# Perception-XAlpha Lite` heading with:

```html
<h1>
  <img src="docs/logo.svg" alt="" height="84" align="left" />
  Perception-XAlpha Lite
</h1>
```

For a light/dark pair:

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/logo-dark.svg">
  <img src="docs/logo.svg" alt="Perception-XAlpha Lite" height="84">
</picture>
```

The social preview is set in **Settings → General → Social preview**, not in the README.
