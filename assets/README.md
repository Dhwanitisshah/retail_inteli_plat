# Demo assets

Photos and generated images used to build the illustrated project
walkthrough: **[SAMGRAHA Walkthrough](https://claude.ai/artifact/FRPQemet67X3Tigs88vQ6J)**.

## demo_photos/

Real store photography (convenience store CCTV monitor, D-Mart checkout
counters, backroom warehouse racking) plus a few AI-generated stock mockups
showing what a detection-overlay feed looks like conceptually
(`02`-`05`, clearly not this platform's real output). Used purely as
illustration in the walkthrough page, mapped one-to-one to the pipeline
module that would read that kind of camera view. Not used by, or required
to run, the actual platform in `../app/` and `../main.py`.

## demo_outputs/

`heatmap_raw_checkout.png` and `heatmap_overlay_checkout.png` are **not**
mockups -- they're rendered by `scripts/generate_heatmap_demo.py`, which
feeds synthetic (but plausible) shopper positions through the platform's
actual `app/vision/heatmap.py::HeatmapAccumulator` and calls the same
`render_png_bytes()` the live dashboard serves at
`/api/heatmap/<camera>/traffic.png`, then alpha-blends the result onto
`demo_photos/08_checkout_wide_aisle.png`. Regenerate with:

```bash
python scripts/generate_heatmap_demo.py
```
