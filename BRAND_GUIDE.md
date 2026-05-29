# HNIVision — Brand Guide (Phase 1)

## Identity
- **Project name:** HNIVision (always one word, "HNI" green + "Vision" slate)
- **Tagline:** "Five computer vision methods. One framework for human-nature interaction."
- **Concept:** Eye-Leaf — computer vision applied to human-nature interaction
- **Author:** Mingze Chen (陈铭泽) · Nature AI Lab
- **GitHub:** https://github.com/LabMingzeChen/HNIVision
- **License:** MIT

## Color palette

| Role | Name | Hex | Usage |
|------|------|-----|-------|
| Primary | Forest | `#1F5C3C` | Logo, headings, primary buttons |
| Accent | Coral | `#E8593C` | CTAs, highlights, the "interaction" focal dot |
| Secondary | Sky | `#5B9BD5` | Data/charts (detection bounding boxes use this) |
| Text | Slate | `#2A3441` | Body text, wordmark "Vision" portion |
| Surface | Cream | `#F8F6F1` | Off-white backgrounds, leaf veins |

### Method-specific accent colors (used in diagrams)
| Method | Color |
|--------|-------|
| Labeling | `#1F5C3C` (forest) |
| Classification | `#3B6D11` (sage) |
| Detection | `#185FA5` (sky-deep) |
| Segmentation | `#854F0B` (earth) |
| Vision-LLM | `#993556` (rose) |

## Logo variants (assets/logo/)

| File | When to use |
|------|-------------|
| `mark-primary.svg` | GitHub social preview, PyPI card, docs site header |
| `mark-primary-dark.svg` | Dark theme docs, dark backgrounds |
| `mark-mono.svg` | Print materials, paper figures |
| `favicon.svg` (+ 32/64 PNG) | Browser tab icon |
| `lockup-horizontal.svg` | README header (full version with tagline) |
| `lockup-horizontal-compact.svg` | Email signature, slide masters |
| `lockup-vertical.svg` | Slide title pages, posters, narrow contexts |

## Banner variants (assets/banner/)

| File | Dimensions | When to use |
|------|------------|-------------|
| `hero-banner.png` | 2400×800 | README top, retina displays |
| `hero-banner-1200.png` | 1200×400 | Fallback for slow connections |
| `hero-banner-social.png` | 1200×630 | Open Graph / Twitter card |

## Typography
- **Display & wordmark:** Helvetica / Arial (web-safe fallback for SVG)
- **Body text in docs:** Inter or system sans-serif
- **Code:** JetBrains Mono / SF Mono
- **Two weights only:** 400 regular, 700 bold (no semi-bold 600 — looks weak)

## Voice
- **Tone:** Academic but accessible. Sentence case everywhere (never Title Case, never ALL CAPS).
- **One-liner pitch:** "Five computer vision methods. One framework for human-nature interaction."
- **Avoid:** "AI-powered", "next-generation", "revolutionary" — these read as hype to the academic audience.
- **Prefer:** "open-source", "benchmark", "unified", "reproducible".
