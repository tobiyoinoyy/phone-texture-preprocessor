"""Use a reproducible photo contact for nonblocking front/back geometry."""
from dataclasses import asdict

import numpy as np
from PIL import Image

CONTACT_FAILURES = (
    'Raw body axis exceeds bounded matte neighborhood',
    'Raw body axis disagrees with independent opposite silhouette',
    'Plate lower boundary is uncertain',
    'No bounded-residual source edge consensus',
    'Weak source RGB lower boundary',
)


def recover_plate(matte, face, original_error):
    from scripts.geometry_postprocess import process_array
    from .auto_geometry import trim_visible_canvas

    if face not in ('front', 'back') or not str(original_error).startswith(CONTACT_FAILURES):
        raise ValueError('Not a recoverable front/back source-contact failure')
    candidate, check = process_array(np.asarray(matte.convert('RGBA')), confidence_floor=.85)
    width, height = check.output_size
    support = check.support
    if (not check.support_applied or support is None or support.threshold_hits < 3 or
            support.confidence < .85 or not 1.84 <= height/width <= 2.30 or abs(check.rotation_degrees) > 5):
        raise ValueError('Repeatable alternate stand-contact evidence is insufficient')
    image, trim = trim_visible_canvas(Image.fromarray(candidate))
    a = np.asarray(image.getchannel('A'))
    if not np.any(a == 0) or not np.any(a >= 240):
        raise ValueError('Alternate stand-contact candidate has no usable alpha')
    proof = dict(method='three_threshold_source_matte_stand_contact',
                 review_reasons=['matte_contact_candidate_needs_lower_edge_review'],
                 strict_geometry_verified=False, original_error=str(original_error),
                 source_size=list(matte.size), output_size=list(image.size), stand_contact=asdict(support),
                 visible_canvas_trim=trim, rotation_ccw_degrees=check.rotation_degrees, rgb_painted=False)
    return image, proof
