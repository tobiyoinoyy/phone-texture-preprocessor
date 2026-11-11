# Resolve unmarked bottom-paper closure from the source

Previously, a bottom label without a recognized red marker was rejected as soon
as its sides were fitted, before the independent transverse-edge detector ran.
Pro Max sku2 in the 261102 batch therefore failed even though the source has a
measurable paper end.

The new branch only applies to that unresolved, side-fitted strip. It reuses the
existing source cross-edge measurement without changing its contrast, support,
slope, or residual thresholds. It additionally requires both paper margins
above the edge, broad agreement with independently sampled lateral metal below
it, and no later reappearance of the paper margins. Printed strokes, shared
metal shadows, continuing labels, different material and transparent lateral
samples remain failures. The overprotection audit remains a rejection guard;
its unresolved result is not used as proof of a complete mask.

The mask is clipped to the measured end plus the existing one-pixel sensor
transition allowance. Only terminal pixels quantization-identical to the
independent metal estimate may be removed from that allowance. No RGB or alpha
pixels are edited by this resolver. Fitted side evidence is not replaced, and
no paper is added outside it. Source hashes, edge support, post-edge witnesses,
lower continuation rows and output mask hashes accompany the result.

Validation includes sharp and sloped closed labels, printed text, a one-pixel
antialiased paper edge, and negative evidence cases. Original six-face source
reruns and native/6x local reviews are kept outside the repository patch. A
processed candidate still requires visual acceptance and model-UV inspection.
The Pro Max sku1 paired-antenna-endpoint failure is not changed by this patch.
