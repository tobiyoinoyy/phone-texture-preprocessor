"""Normalize a photographed top to a calibrated per-model metal-frame thickness."""
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .top_frame import _calibrate_top, verify_top_mask


PROFILE = Path(__file__).with_name('top_body_profiles.json')


def _source_lower_rail(rgb, upper, thickness, extent):
    from .auto_geometry import consensus_line
    left, right = extent
    width = right - left
    columns = np.r_[np.arange(int(left+.12*width),int(left+.24*width)),
                    np.arange(int(left+.76*width),int(left+.88*width))]
    expected = upper + np.array([0,thickness])
    radius = max(6,int(np.ceil(.012*width)))
    rows = np.rint(np.polyval(expected,columns+.5)[:,None]+np.arange(-radius,radius+1)).astype(int)
    if rows.min()<0 or rows.max()>=rgb.shape[0]:
        raise ValueError('Template lower rail reaches outside the photograph')
    gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype('float32')
    gradient = np.gradient(cv2.GaussianBlur(gray,(0,0),.8),axis=0)
    strength = gradient[rows,columns[:,None]]
    peaks = strength.argmax(axis=1)
    measured = rows[np.arange(len(columns)),peaks]+.5
    line, keep, residual = consensus_line(columns+.5,measured,2.5,np.tan(np.deg2rad(4)))
    supported = keep & (strength.max(axis=1)>3) & (peaks>0) & (peaks<2*radius)
    halves = [float(part.mean()) for part in np.array_split(supported,2)]
    if min(halves)<.7:
        raise ValueError('Template lower rail lacks two-sided source RGB support')
    return line, dict(method='template_seeded_dark_metal_to_background_edge',
                      source_x_ranges=[[int(columns[0]),int(columns[len(columns)//2-1])],
                                       [int(columns[len(columns)//2]),int(columns[-1])]],
                      half_support_fractions=halves,residual_p95_px=residual,
                      template_lower_rail=expected.tolist(),measured_lower_rail=line.tolist())


def normalize_top(matte, raw, model, stage_sink=None, *, previous_error=None):
    profiles = json.loads(PROFILE.read_text())['profiles']
    if model not in profiles:
        raise ValueError('No calibrated top thickness for this model')
    profile = profiles[model]
    a = np.asarray(matte.convert('RGBA'))
    rgb = np.asarray(raw.convert('RGB'))
    upper, lower, measured = _calibrate_top(a, rgb, allow_taper=True)
    left, right = measured['x_extent']
    width = right - left
    ends = np.array([left + .5, right - .5])
    reference_thickness = profile['thickness_to_width_ratio'] * width
    lower, lower_proof = _source_lower_rail(rgb,upper,reference_thickness,(left,right))
    measured['bottom_rail'] = lower.tolist()
    observed_thickness = np.polyval(lower, ends) - np.polyval(upper, ends)
    sampling_ratio = observed_thickness / reference_thickness
    if np.min(sampling_ratio) < .85 or np.max(sampling_ratio) > 1.15:
        raise ValueError('Photographed top thickness is outside calibrated sampling range')

    # Remove the camera and support in source coordinates before resampling.
    # The original curved endcaps keep their photographed coverage.
    yy = np.arange(a.shape[0])[:, None]
    xx = np.arange(a.shape[1])[None, :] + .5
    source_top = np.ceil(np.polyval(upper, xx))
    source_bottom = np.polyval(lower, xx)
    coverage = np.clip(np.minimum(yy + 1, source_bottom) - np.maximum(yy, source_top), 0, 1)
    cropped = a.copy()
    cropped[..., 3] = np.minimum(cropped[..., 3], np.rint(coverage * 255).astype('uint8'))
    source_check = verify_top_mask(a, cropped, upper, lower, measured)

    out_width = profile['output_width_px']
    thickness = profile['thickness_px']
    # Two transparent edge rows retain native antialias coverage.
    source_x = left + (np.arange(out_width) + .5) / out_width * width
    source_upper = np.polyval(upper, source_x)
    source_lower = np.polyval(lower, source_x)
    fraction = (np.arange(thickness + 2)[:, None] + .5 - 1) / thickness
    map_x = np.broadcast_to(source_x[None, :] - .5, fraction.shape[:1] + (out_width,)).astype('float32')
    map_y = (source_upper[None, :] + fraction * (source_lower - source_upper)[None, :] - .5).astype('float32')
    pixels = cropped.astype('float32') / 255
    pixels[..., :3] *= pixels[..., 3, None]
    sampled = cv2.remap(pixels, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    colors = np.zeros_like(sampled[..., :3])
    np.divide(sampled[..., :3], sampled[..., 3, None], out=colors, where=sampled[..., 3, None] > 1e-6)
    result = Image.fromarray(np.clip(np.rint(np.dstack([colors, sampled[..., 3]]) * 255), 0, 255).astype('uint8'))
    alpha = np.asarray(result.getchannel('A'))
    interior = alpha[4:-4, int(.12*out_width):int(.88*out_width)]
    if not interior.size or np.mean(interior >= 192) < .98:
        raise ValueError('Normalized top lost photographed interior coverage')
    if stage_sink:
        stage_sink('source_matte', matte)
        stage_sink('after_support', Image.fromarray(cropped))
        stage_sink('after_rotation', result)
        stage_sink('after_crop', result)

    proof = dict(measured, method='same_model_fixed_thickness_source_remap',
                 previous_failure=previous_error, model=model,
                 normalized_width_px=out_width, nominal_thickness_px=thickness,
                 thickness_basis='median_of_complete_same_model_top_measurements',
                 calibration_samples=profile['calibration_samples'],
                 profile_sha256=hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
                 source_lower_rail_evidence=lower_proof,
                 source_thickness_range_px=observed_thickness.tolist(),
                 source_to_target_sampling_ratio=sampling_ratio.tolist(),
                 source_size=list(matte.size), output_size=list(result.size),
                 source_geometry_checks=source_check,
                 transform_checks=dict(status='passed', method='source_bound_parallel_destination_remap',
                                       camera_removed_before_resampling=True,
                                       opaque_interior_fraction=float(np.mean(interior >= 192))),
                 destination_rails=[[0, 1], [0, 1 + thickness]],
                 camera_policy='exclude_entire_camera_island_and_lenses',
                 rotation_ccw_degrees=float(np.degrees(np.arctan(upper[0]))),
                 rgb_painted=False, source_pixels_resampled=True,
                 review_reasons=['fixed_thickness_perspective_candidate'])
    return result, proof
