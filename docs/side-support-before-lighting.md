# Separate the physical side boundary before lighting

The actual automatic order already was BiRefNet_dynamic -> source-derived
support removal -> deskew/trim -> source landmarks -> lighting -> alignment.
The issue is the extra one-pixel outward allowance on the lower side rail.
Around two support-contact corners it can retain bright stand/background RGB;
later gamut-cap propagation then produces broad dark halos in adjacent metal.

For left/right only, when the independent raw RGB and alpha boundary estimates
agree within one native source pixel, and raw boundary residual is within one
pixel, use fractional coverage at that measured lower edge without the extra
outward allowance. Do not move the measured edge inward. RGB is never painted,
alpha never increases and the core at least one pixel inside the measured edge
is retained. Curved segmentation corners remain in the mask. Ambiguous estimates
retain the conservative proposal and explicitly record unresolved status.

Geometry review and explicit user acceptance remain required. This does not
certify occluded phone RGB at the support contact; small contact-edge colour
may remain. A one-pixel canvas-width change requires new source landmarks and
subsequent structure/seam checks. Never reuse old alignment coordinates after
the change. Keep the frozen user S1 service unchanged.
