# Original RGB evidence for side antenna endpoints

R3 retains the original 0.5 grayscale/pixel edge threshold. R2's global 0.25
experiment is not promoted: some newly admitted endpoint pairs disagree with
adjacent column contours by 5–6 native pixels. The endpoint shape median now
includes missing opaque columns as zero evidence, avoiding selection bias.
Strong exceptions require both material contrast >=6 RMS gray levels and
median paired gradient >=2 gray levels/pixel; weaker signals require shape
correlation >=0.65. No terminal opaque column is extrapolated.

If a frozen geometry image cannot provide a full-span pair, the automatic path
may independently map its own original RGB into exactly those geometry
coordinates, before alpha premultiplication and its early 8-bit quantization.
It copies the frozen alpha and cannot alter output RGB or geometry. The
reference still has one bicubic coordinate reprojection; it is not a native
source-coordinate contour tracer. Raw source, cached reference and geometry
hashes, side identity, dimensions and alpha must match. Reference pairs must
also agree with the geometry's central material seed within 2.5 pixels.
Lighting revalidates against the same bound reference. Existing final pixel,
structure, effects and seam gates remain active.

This is an experimental capability, not a full generation fix. Pro sku5 right
can provide its complete pair through original RGB, while its left still
fails. Pro sku3 right and Pro Max sku1 left are unresolved even with original
RGB. Independent negative suites still contain false accepts. User acceptance
and fixed-geometry/model review remain separate. Do not merge or deploy solely
because unit tests pass. Preserve the running S1 service and other research.
