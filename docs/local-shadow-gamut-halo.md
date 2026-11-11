# Local dark halos around source highlights on a shadowed side rail

The complaint screenshot is matched to the completed S1 web test
20261103-150751-3adec3, Pro sku1 left side. At the two circled rail locations,
the source contains small bright and dark defects. Gamut safety caps the lift
of those bright pixels, and S1 spreads that restriction into adjacent dark
metal. The broad dark halo is already present in lighting, before alignment.

The new opt-in source feature budget changes only that cap propagation. A
3x3 source median and a robust source residual threshold detect narrow source
impulses; these are evidence fields, never replacement RGB. At these features,
at most their measured residual contrast may contribute to cap path cost. Ordinary
grain retains the previous constraint. Actual per-pixel gamut safety remains.

The relief is disabled near other materials and ramps smoothly beyond a guard
of max(12, source width * .1) pixels. It is further limited to the upper half
of the configured positive lift budget, avoiding changes to ordinary small
corrections and darkening. Paper, alpha, material barriers and all quality
thresholds are unchanged. The option applies only to left/right main metal.

Four frozen-geometry original cases cover silver, orange and blue. Recompute
lighting, structure alignment and seams with fresh material proposals. Keep
rejected experiments and their quality failures separate. This does not count
as rerunning segmentation from JPEG, or as visual acceptance or model-UV proof.
The frozen S1 service and generation-failure branch are not modified.
