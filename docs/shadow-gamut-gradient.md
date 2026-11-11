# Narrow side shadow and gamut continuity

The prior fixed gamut displacement slope limited a dark rail's correction by
nearby bright metal's much smaller lift budget. It also produced varying gray
patches along the rail. The source photo was not blurred, but its low frequency
photographic shadow survived as a broad band beside bright metal.

For side main metal, gamut cap propagation now permits the gradient of the
already fitted additive L correction. Only additional gamut reductions are
propagated. RGB and source Lab texture are never filtered. Other materials,
paper and opposite-sign fields still block propagation; no canvas wrapping is
allowed. The feature is opt-in for side main metal only.

Side plastic antenna domains use their own source references and the same
normal sampling scale as the metal rail. A broad plastic fit next to a narrow
metal fit otherwise reverses some endpoint contrast after illumination. Their
original 0.72 strength remains; only the fit scale and lift budget change.

Validation must include the existing final paired antenna checks, native and
4x source/output comparisons, and silver, orange and blue examples. Numerical
checks are separate from user acceptance and model UV verification. The
independent experiment report retains rejected intermediate variants.
