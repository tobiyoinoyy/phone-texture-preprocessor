"""Side identity from the raised camera end in fixed-pose studio captures."""
import cv2
import numpy as np


def classify(rgb):
    height, width = rgb.shape[:2]
    image = cv2.resize(rgb, (800, round(height * 800 / width)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    background = float(np.median(np.r_[gray[int(.3*height):int(.6*height), :35].ravel(),
                                      gray[int(.3*height):int(.6*height), -35:].ravel()]))
    readings = []
    for shift in (15, 25, 35):
        foreground = gray < background - shift
        support = foreground[:, 60:740].sum(axis=1)
        rows = np.flatnonzero((support > 440) & (np.arange(height) > .3*height) &
                             (np.arange(height) < .65*height))
        if len(rows) < 7 or len(rows) > .085*width:
            continue
        middle_row = rows[np.argmax(support[rows])]
        columns = np.flatnonzero(foreground[middle_row])
        left, right = int(columns.min()), int(columns.max()) + 1
        body_width = right - left
        if not .65*width < body_width < .96*width:
            continue
        x = np.arange(left, right)
        search_top = max(0, int(rows.min() - .10*body_width))
        patch = foreground[search_top:rows.max()+1, x]
        if patch.any(axis=0).mean() < .98:
            continue
        top = patch.argmax(axis=0) + search_top
        center = (x > left + .4*body_width) & (x < left + .6*body_width)
        rail = np.polyfit(x[center], top[center], 1)
        residual = float(np.percentile(abs(np.polyval(rail, x[center]) - top[center]), 95))
        if residual > 2.5:
            continue
        raised = np.polyval(rail, x) - top
        elevated = (raised > max(3, .006*body_width)) & (raised < .12*body_width)
        end_regions = [(x > left + .04*body_width) & (x < left + .38*body_width),
                       (x > left + .62*body_width) & (x < right - .04*body_width)]
        counts = [int(np.count_nonzero(elevated & end)) for end in end_regions]
        observed = [count > .065*body_width for count in counts]
        if sum(observed) != 1:
            continue
        face = 'right' if observed[0] else 'left'
        readings.append(dict(threshold=background-shift, face=face, end_support_pixels=counts,
                             body_bounds=[left, int(rows.min()), right, int(rows.max())+1],
                             upper_rail=rail.tolist(), upper_residual_p95_px=residual,
                             camera_x=[int(x[elevated].min()), int(x[elevated].max())+1]))
    identities = {reading['face'] for reading in readings}
    face = next(iter(identities)) if len(identities) == 1 and len(readings) >= 2 else None
    return dict(face=face, method='fixed_studio_pose_camera_end',
                pose='horizontal_phone_camera_projection_above_body',
                direction_rule='camera_left_is_right_side_camera_right_is_left_side',
                background_gray=background, normalized_image_size=[width, height],
                readings=readings,
                reason=None if face else 'Camera end lacks repeated one-sided source contrast')
