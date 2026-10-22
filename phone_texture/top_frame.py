"""Deterministic top-body calibration for the iPhone 17 studio capture profile.

The two exposed body corners, not the camera island's broad upper silhouette,
anchor the rails. A bounded corner fit is checked against raw RGB and against
held-out corner samples. Unsupported captures fail; no agent supplies points.
"""
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares


class TopCalibrationError(ValueError):
    pass


def calibrate_top(rgba, raw_rgb):
    if raw_rgb is None or raw_rgb.shape != rgba[..., :3].shape:
        raise TopCalibrationError('TOP_SOURCE_REQUIRED: original RGB required')
    alpha = rgba[..., 3]
    labels, count = ndimage.label(alpha >= 128)
    if not count:
        raise TopCalibrationError('TOP_BODY_NOT_FOUND')
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    mask = labels == sizes.argmax()
    _, xs = np.nonzero(mask); x0, x1 = int(xs.min()), int(xs.max()) + 1
    width = x1 - x0
    if width < 350:
        raise TopCalibrationError('TOP_RESOLUTION_UNSUPPORTED: body width must be at least 350 px')
    # These dimensionless bands define the capture profile, never per-SKU coordinates.
    distance = np.arange(max(3, int(.006 * width)), int(.035 * width))
    train = np.arange(len(distance)) % 3 != 0
    gradients = np.linalg.norm(np.gradient(ndimage.gaussian_filter1d(
        raw_rgb.astype(np.float32), .8, axis=0), axis=0), axis=2)
    corners = []; levels = []
    for end in (0, 1):
        xx = x0 + distance if end == 0 else x1 - 1 - distance
        if not mask[:, xx].any(axis=0).all():
            raise TopCalibrationError('TOP_CORNER_NOT_VISIBLE')
        observed = [np.argmax(mask[:, xx], axis=0),
                    len(mask) - np.argmax(mask[::-1, xx], axis=0)]
        edge_levels = []
        for edge, sign, yy in zip(('upper', 'lower'), (1, -1), observed):
            # Decay of the rounded end into the flat frame. The island starts
            # farther inward; no island pixels participate in this estimate.
            def residual(v):
                return v[0] + sign * v[1] * np.exp(-distance / v[2]) - yy
            fit = least_squares(lambda v: residual(v)[train],
                [float(np.median(yy)) - sign * 5, 35, .01 * width],
                bounds=([float(yy.min()) - 25, 0, .003 * width],
                        [float(yy.max()) + 25, 100, .03 * width]))
            errors = np.abs(residual(fit.x))
            if (not fit.success or fit.x[1] < 3 or
                    not .0031 * width < fit.x[2] < .0299 * width or
                    np.percentile(errors[train], 95) > 1.5 or
                    np.percentile(errors[~train], 95) > 2):
                raise TopCalibrationError('TOP_CORNER_AMBIGUOUS: unsupported body corner shape')
            sample_y = np.clip(yy[:, None] + np.arange(-3, 4), 0, len(mask)-1)
            support = gradients[sample_y, xx[:, None]].max(axis=1) > 3
            if support.mean() < .8:
                raise TopCalibrationError('TOP_CORNER_RGB_UNSUPPORTED')
            edge_levels.append(float(fit.x[0]))
            corners.append(dict(end='left' if end == 0 else 'right', edge=edge,
                source_x=xx.tolist(), source_y=yy.tolist(), plateau=float(fit.x[0]),
                decay_px=float(fit.x[2]), residual_p95_px=float(np.percentile(errors[train], 95)),
                held_out_error_p95_px=float(np.percentile(errors[~train], 95)),
                raw_rgb_supported_fraction=float(support.mean())))
        levels.append(edge_levels)
    upper = np.polyfit([x0+.5, x1-.5], [v[0] for v in levels], 1)
    lower = np.polyfit([x0+.5, x1-.5], [v[1] for v in levels], 1)
    thickness = np.array([v[1]-v[0] for v in levels])
    if (np.min(thickness) < .07*width or np.max(thickness) > .17*width or
            abs(np.diff(thickness)[0]) > 4 or
            max(abs(upper[0]), abs(lower[0])) > np.tan(np.deg2rad(4))):
        raise TopCalibrationError('TOP_BODY_RAILS_DISAGREE')
    # Raw surface transition across the body is independent of the silhouette
    # fit. Both halves must witness it; a short camera edge cannot suffice.
    xx = np.arange(int(x0+.15*width), int(x1-.15*width))
    yy = np.rint(np.polyval(upper, xx+.5)).astype(int)
    strength = gradients[np.clip(yy[:,None]+np.arange(-3,4),0,len(mask)-1), xx[:,None]].max(axis=1)
    halves = [float(v.mean()) for v in np.array_split(strength > 3, 2)]
    if min(halves) < .8:
        raise TopCalibrationError('TOP_BODY_TRANSITION_UNSUPPORTED')
    return upper, lower, dict(method='top_body_exposed_corners_v1',
        capture_profile='iphone17-studio', x_extent=[x0,x1], corners=corners,
        raw_body_transition_support=halves, top_rail=upper.tolist(), bottom_rail=lower.tolist(),
        camera_policy='exclude_entire_camera_island_and_lenses', agent_required=False)


def verify_top_mask(rgba, corrected, upper, lower, proof):
    """Reject camera retention and body loss in original coordinates."""
    h,w = rgba.shape[:2]; xx=np.arange(w)[None,:]+.5; yy=np.arange(h)[:,None]
    # Entire region outside the measured body plane, including camera soft alpha.
    camera = (yy < np.ceil(np.polyval(upper,xx))) & (rgba[...,3] > 0)
    body = ((yy >= np.polyval(upper,xx)+4) &
            (yy+1 <= np.polyval(lower,xx)-4) & (rgba[...,3] >= 192))
    if not body.any():
        raise TopCalibrationError('TOP_BODY_WITNESS_EMPTY')
    if np.any(corrected[...,3][camera] != 0):
        raise TopCalibrationError('TOP_CAMERA_RETAINED: camera projection enters top calibration')
    if not np.array_equal(corrected[body], rgba[body]):
        raise TopCalibrationError('TOP_BODY_ERODED: interior body pixels changed')
    if not np.array_equal(corrected[...,:3], rgba[...,:3]):
        raise TopCalibrationError('TOP_RGB_CHANGED: geometry must not paint pixels')
    return dict(camera_excluded_pixels=int(camera.sum()), camera_retained_pixels=0,
                body_witness_pixels=int(body.sum()), body_changed_pixels=0,
                status='passed', scope='source_geometry_only')


def verify_top_transform(original, rotated, cropped, upper, lower, angle, box):
    from PIL import Image
    residual=np.abs(np.degrees(np.arctan([upper[0],lower[0]]))-angle)
    if residual.max()>.15:
        raise TopCalibrationError('TOP_RESIDUAL_TILT')
    h,w=original.shape[:2];xx=np.arange(w)[None,:]+.5;yy=np.arange(h)[:,None]
    camera=(yy+1<=np.polyval(upper,xx)-5)&(original[...,3]>0)
    body=(yy>=np.polyval(upper,xx)+6)&(yy+1<=np.polyval(lower,xx)-6)&(original[...,3]>=254)
    # Erode interior witnesses before projecting, so bicubic transition pixels
    # are not mislabeled as certain body/background pixels.
    body=ndimage.binary_erosion(body,iterations=3)
    masks=[]
    for region in (camera,body):
        masks.append(np.asarray(Image.fromarray(region.astype(np.uint8)*255).rotate(
            angle,resample=Image.Resampling.NEAREST,expand=True))>0)
    a=np.array(rotated)[...,3]
    if masks[0].shape!=a.shape or np.any(a[masks[0]]):
        raise TopCalibrationError('TOP_CAMERA_AFTER_ROTATION')
    if not masks[1].any() or np.any(a[masks[1]]<128):
        raise TopCalibrationError('TOP_BODY_AFTER_ROTATION')
    x0,y0,x1,y1=box
    if not np.array_equal(np.array(cropped),np.array(rotated)[y0:y1,x0:x1]):
        raise TopCalibrationError('TOP_CROP_CHANGED_CONTENT')
    outside=masks[1].copy();outside[y0:y1,x0:x1]=False
    if outside.any():raise TopCalibrationError('TOP_CROP_LOST_BODY')
    return dict(camera_after_rotation_pixels=0,body_lost_after_rotation_pixels=0,
                body_lost_after_crop_pixels=0,residual_rail_degrees=residual.tolist(),status='passed')
