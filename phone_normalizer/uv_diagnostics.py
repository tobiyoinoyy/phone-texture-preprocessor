"""Visual geometry audit; guide marks are never written onto delivered textures."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default(size=size)


def _panel(canvas, image, area, boxes):
    x, y, width, height = area
    factor = min(width / image.width, height / image.height)
    size = (max(1, round(image.width*factor)), max(1, round(image.height*factor)))
    image = image.convert('RGBA').resize(size, Image.Resampling.LANCZOS)
    origin = (x + (width-size[0])//2, y + (height-size[1])//2)
    draw = ImageDraw.Draw(canvas)
    for yy in range(y, y+height, 16):
        for xx in range(x, x+width, 16):
            color = '#e5e7eb' if ((xx-x)//16 + (yy-y)//16)%2 else '#f3f4f6'
            draw.rectangle((xx, yy, min(xx+15,x+width-1), min(yy+15,y+height-1)), fill=color)
    canvas.paste(image, origin, image)
    sx, sy = size[0] / max(1, boxes.pop('_width')), size[1] / max(1, boxes.pop('_height'))
    for name, (box, color) in boxes.items():
        b = [origin[0]+box[0]*sx, origin[1]+box[1]*sy,
             origin[0]+box[2]*sx, origin[1]+box[3]*sy]
        draw.rectangle(b, outline=color, width=3)


def alignment_diagnostic(candidate, result, info):
    """Side-by-side silhouette audit, explicitly excluding semantic validation."""
    canvas = Image.new('RGB', (1400, 1000), '#ffffff')
    draw = ImageDraw.Draw(canvas)
    font, small = _font(23), _font(18)
    gate = info['geometry_gate']
    color = '#b45309' if gate != 'passed' else '#166534'
    draw.text((32, 20), f"{info['face'].upper()} | {gate.upper()}", fill=color, font=font)
    draw.text((32, 60), 'SOURCE: detected body (blue)', fill='#1f2937', font=small)
    draw.text((720, 60), 'OUTPUT: target (orange), mapped body (blue)', fill='#1f2937', font=small)
    box = info['candidate_body_bbox']
    _panel(canvas, candidate, (32, 95, 648, 660), {
        '_width': candidate.width, '_height': candidate.height,
        'body': (box, '#2563eb'),
    })
    matrix = np.asarray(info['candidate_to_final_matrix'])
    p = matrix @ np.array([[box[0],box[2]], [box[1],box[3]], [1.,1.]])
    mapped = [p[0,0],p[1,0],p[0,1],p[1,1]]
    target = info['reference_body_bbox']; crop = info['crop_xyxy']
    target = [target[0]-crop[0],target[1]-crop[1],target[2]-crop[0],target[3]-crop[1]]
    _panel(canvas, result, (720, 95, 648, 660), {
        '_width': result.width, '_height': result.height,
        'target': (target, '#f97316'), 'mapped': (mapped, '#2563eb'),
    })
    sx, sy = info['requested_scale_xy']
    lines = [
        f"Requested scale X/Y: {sx:.4f} / {sy:.4f} | anisotropy: {100*info['anisotropy']:.2f}% | limit: {100*info['max_anisotropy']:.2f}%",
        'Alpha edge spread (L/T/R/B px): ' + ', '.join(f'{v:.1f}' for v in info['body_evidence']['threshold_edge_spread_px']),
        'Reasons: ' + (', '.join(info['geometry_gate_reasons']) or 'geometry checks passed'),
        'Buttons / ports / cameras: NOT CHECKED. Geometry pass is not UV acceptance.',
        'Rejected output is an aspect-preserving REVIEW PREVIEW, not an aligned texture.' if gate != 'passed'
        else 'Review on the target 3D model before accepting this texture.',
    ]
    y = 790
    for line in lines:
        words, current = line.split(), ''
        for word in words:
            proposed = (current + ' ' + word).strip()
            if draw.textlength(proposed, font=small) > 1336 and current:
                draw.text((32, y), current, fill='#374151', font=small)
                y += 27
                current = word
            else:
                current = proposed
        draw.text((32, y), current, fill='#374151', font=small)
        y += 30
    return canvas


def crop_diagnostic(image, face, evidence):
    canvas = Image.new('RGB', (1000, 1000), '#ffffff')
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 24), f'{face.upper()} | CROP ONLY - NO UV TEMPLATE', fill='#1f2937', font=_font(23))
    draw.text((32, 65), 'Blue: estimated body. Protrusions remain in the source image.', fill='#374151', font=_font(18))
    _panel(canvas, image, (32, 110, 936, 700), {
        '_width': image.width, '_height': image.height,
        'body': (evidence['bbox'], '#2563eb'),
    })
    lines = [
        'Body edge spread (px): ' + ', '.join(f'{x:.1f}' for x in evidence['threshold_edge_spread_px']),
        'Review: ' + (', '.join(evidence['review_reasons']) or 'no silhouette warning'),
        'No button/port correspondence or UV alignment has been validated.',
    ]
    for i, line in enumerate(lines):
        draw.text((32, 850+i*35), line, fill='#374151', font=_font(18))
    return canvas
