# Studio-photo source boundary checks

The automatic aluminum profile produces candidates for review, not human or
current Blender scene approval. The 2026-11-02 QC batch exposed several distinct
source-coordinate errors. Keep their checks separate:

* Plate-view recognition can register modest framing changes from outer rails
  and caps, but must validate disjoint internal camera/screen evidence against
  the frozen atlas. The existing score and ambiguity thresholds still apply.
  Registration is not a general mirror detector.
* A front rim uses its own opaque material along each local inward normal.
  Another edge's wider alpha shoulder cannot move sampling into glass. Compute
  normals with transparent exterior before cropping the distance field, and
  integrate numerator and support against the same zero exterior. Additional
  transparent canvas must not change the result. Every editable pixel must
  satisfy the existing support threshold; no low-alpha pixel is silently frozen.
* A noisy hardware convex hull may be refined by at most one pixel against its
  actual capsule boundary. All original edge support, alpha, uniqueness and
  hardware-count gates remain. Already supported contours take precedence.
* New full processing tracks both antenna bands. A local RGB material direction
  can recover a boundary hidden by channel averaging only when adjacent paired
  edges establish that direction and both source edges support the new column.
  Missing opaque terminal evidence remains a failure. The historical lower-only
  repair explicitly retains its upper mapping.
* Curved UV seam chains use a continuous transverse coordinate. Nearest-segment
  projection can jump at a segment bisector and produce a diagonal color block.
  The original seam-difference gate remains, with an additional gate on adjacent
  floating correction values. This is not a guarantee about quantized PNG
  differences at saturated source highlights.
* Top and bottom frame faces also need a spatially continuous gamut limit.
  Independent per-pixel clipping can turn faint real grain into a saturated
  flat patch. Propagate source-derived headroom only within the same material's
  additive correction, with a 0.2 L/pixel limit for these faint frame textures.
  Do not filter RGB, alter paper/alpha, or synthesize already overexposed detail.
* At a lower rounded corner, a straight bottom plane alone can retain an
  attached stand fragment. Restrict any curved contact removal to columns with
  a measured distant stand connection, and independently verify the local
  raw RGB boundary. Existing exposed corners and reflections on the body are
  not stand pixels.


* Back-paper tails need independent side and terminal source evidence. In a raw
  padded matte, use the measured physical body end rather than canvas height.
  Preserve dark paper, print and antialias pixels before removing a support;
  a white-only connected component is not a complete paper mask. This helper
  does not certify arbitrary padded inputs through the entire geometry stage.
* Bottom-paper closure is still an unresolved boundary case. A clear source
  cross-edge plus independent lateral metal can expose an overlong protection
  mask, but a rejection guard is not a complete paper segmentation method.
  RGB residual segmentation experiments also connected neighboring metal
  reflections. Do not merge those masks, grow a rectangle, or claim all paper
  antialias pixels are certified from a numerical lighting pass.

The second QC iteration adds source-fit coverage weighting to newly generated
rear-perimeter jobs. Soft mixed edge pixels must not vote as strongly as opaque
same-material samples when smoothing an already supported correction field.
With `D = G(reliable)` and `q = G(fit & reliable) / D`, the result is
`(1-q) * legacy_field + G(raw_shift * (fit & reliable)) / D`. Where no nearby
opaque fit exists, the supported legacy field remains exactly unchanged. The
original source-support and output gates still apply; this operates on additive
correction, never by blurring source RGB. Jobs without `correction_smoothing`
retain the legacy formula. New mode use is limited to source-lightness contour
domains with the source-derived contour-depth lift profile.

The bottom-paper end proposal must not stop at the first signed descent.
A genuine dark paper foot or a short native antialias plateau can look like a
terminal edge and cause that rule to expose real paper. The 2026-11-02 v2
experiments removed the visible Pro17 rectangle but failed these known-alpha
counterexamples, and were not merged. Current two-dimensional side/end evidence
also leaves mixed edge pixels unresolved. This is a limitation of the current
method, not proof that all such photographs lack usable source information.
Pro17's numerical candidate therefore remains a visual failure. The existing
audit is intentionally only a rejection guard, not paper-boundary certification.

When geometry, paper analysis or final canvas trimming fails, save the actual
available matte and pre-trim geometry with image size, source hash, failure stage
and coordinate-space label. Re-raise the failure; do not substitute an earlier
run's completed image or manufacture a candidate to complete a review page.

Run `python scripts/test_all.py` for all bundled suites. Real-image replay tools
are `scripts/regression_side_antenna_endpoints.py`,
`scripts/regression_side_materials.py`, and
`scripts/regression_cloud_rear_gamut.py`. Keep raw-image runs, cached-geometry
replays, source insufficiency, visual review, and user acceptance distinct.

Runtime provenance fingerprints the complete shipped Python modules and frozen
recognition/seam resources, including delegated material helpers, without
requiring an installed package to contain Git metadata.

## 2026-11-03：无纸端证据的整高侧轨必须拒绝

底面 `source_strip_edges` 只证明两侧纸边，不证明纸张沿侧轨延续至画布底。没有解析横向纸端（`unresolved_no_marker`）时，自动流程现在明确失败，不能把独立过保护审计的 `unresolved` 当成可放行证据。Pro17 的已知矩形暗块因此被阻断，但真实纸端逐像素闭合仍未修复；此变更不生成新蒙版，不放宽门限，也不要求为缺红印本身重拍。未外推的白纸分量仍由已有源边界审计处理。
