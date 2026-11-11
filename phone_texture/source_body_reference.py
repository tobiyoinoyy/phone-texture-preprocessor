"""Source-only body reference for explicitly requested review candidates.

No side image, antenna target, hardware location, SKU or expected length is an
input to this module. The existing raw RGB/alpha rail is mapped, never refitted.
This provides a measured reference candidate, not semantic/visual approval.
Build only inside the root's global low-priority processing guard.
"""
import hashlib
import json
import math
from pathlib import Path

MODE = 'source_body_reference_review'
METHOD = 'same_source_raw_rail_reference_candidate_v1'
GEOMETRY_METHOD = 'source_exposed_plate_rgb_alpha_consensus_v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    try:
        digest=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024), b''):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as error:
        raise ValueError('Body reference source unavailable: '+str(path)) from error


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=_sha(path))


def _seal(record):
    return _canonical({k:v for k,v in record.items() if k!='evidence_sha256'})


def _geometry_contract(geometry):
    _require(geometry.get('method')==GEOMETRY_METHOD, 'No supported source-only exposed rail method')
    _require(geometry.get('rgb_painted') is False, 'Body reference requires original source RGB')
    for key in ('source_size', 'output_size'):
        value=geometry.get(key)
        _require(isinstance(value, list) and len(value)==2 and
                 all(type(v) is int and v>0 for v in value), 'Invalid geometry '+key)
    sw,sh=geometry['source_size']; w,h=geometry['output_size']
    rotation=geometry.get('rotation_ccw_degrees')
    _require(isinstance(rotation,(int,float)) and math.isfinite(rotation), 'Invalid source rotation')
    _,expanded=rotation_reverse(geometry['source_size'], rotation)
    crop=geometry.get('canvas_trim_xyxy')
    visible=geometry.get('visible_canvas_trim',{}).get('crop_xyxy')
    for box,size in ((crop,expanded),(visible,[(crop[2]-crop[0]) if isinstance(crop,list) and len(crop)==4 else 0,
                                           (crop[3]-crop[1]) if isinstance(crop,list) and len(crop)==4 else 0])):
        _require(isinstance(box,list) and len(box)==4 and all(type(v) is int for v in box), 'Invalid source crop')
        _require(0<=box[0]<box[2]<=size[0] and 0<=box[1]<box[3]<=size[1], 'Source crop outside its canvas')
    _require([visible[2]-visible[0],visible[3]-visible[1]]==[w,h], 'Visible source crop dimensions differ')
    body=geometry.get('body_x'); rail=geometry.get('bottom_rail'); alpha=geometry.get('alpha_lower_rail')
    for value in (body,rail,alpha):
        _require(isinstance(value,list) and len(value)==2 and
                 all(isinstance(v,(int,float)) and math.isfinite(v) for v in value), 'Missing source rail provenance')
    _require(0<=body[0]<body[1]<=sw, 'Raw body columns outside source')
    bands=geometry.get('clear_bands')
    _require(isinstance(bands,list) and len(bands)==2 and 0<bands[0]<bands[1]<.5,
             'Exposed two-flank rail evidence required; unsupported geometry branch')
    _require(type(geometry.get('exposed_samples')) is int and geometry['exposed_samples']>=24,
             'Too few original exposed rail samples')
    # This is the original raw RGB/alpha consensus limit, not a side-target fit.
    _require(max(abs((rail[0]-alpha[0])*x+rail[1]-alpha[1]) for x in body)<=3.00000001,
             'Raw RGB rail no longer agrees with independent alpha rail')
    for x in body:
        _require(0<=rail[0]*(x+.5)+rail[1]<=sh, 'Raw rail outside source')
    return geometry


def at(matrix, x, y):
    a, b, c, d, e, f = matrix
    return a*x+b*y+c, d*x+e*y+f


def inverse(matrix):
    a, b, c, d, e, f = matrix
    det = a*e-b*d
    _require(abs(det)>1e-12, 'Singular geometry coordinate transform')
    return [e/det, -b/det, (b*f-e*c)/det, -d/det, a/det, (d*c-a*f)/det]


def rotation_reverse(size, degrees):
    """Pillow expand=True, rotated-edge -> raw-edge; no image resampling."""
    w, h = size
    if abs(degrees)<1e-5:
        return [1., 0., 0., 0., 1., 0.], [w, h]
    angle = -math.radians(degrees % 360.)
    matrix = [round(math.cos(angle), 15), round(math.sin(angle), 15), 0.,
              round(-math.sin(angle), 15), round(math.cos(angle), 15), 0.]
    matrix[2], matrix[5] = at(matrix, -w/2., -h/2.)
    matrix[2] += w/2.; matrix[5] += h/2.
    corners = [at(matrix, x, y) for x, y in ((0,0),(w,0),(w,h),(0,h))]
    nw = math.ceil(max(p[0] for p in corners))-math.floor(min(p[0] for p in corners))
    nh = math.ceil(max(p[1] for p in corners))-math.floor(min(p[1] for p in corners))
    matrix[2], matrix[5] = at(matrix, -(nw-w)/2., -(nh-h)/2.)
    return matrix, [nw, nh]


def coordinate_map(geometry):
    reverse, expanded = rotation_reverse(geometry['source_size'], geometry['rotation_ccw_degrees'])
    forward = inverse(reverse)
    trim = geometry['canvas_trim_xyxy']; visible = geometry['visible_canvas_trim']['crop_xyxy']
    shift = [trim[0]+visible[0], trim[1]+visible[1]]
    def raw_edge_to_geometry_index(x, y):
        u, v = at(forward, x, y)
        return u-shift[0]-.5, v-shift[1]-.5
    def geometry_index_to_raw_index(x, y):
        u, v = at(reverse, x+shift[0]+.5, y+shift[1]+.5)
        return u-.5, v-.5
    errors = []
    for x, y in ((0.,0.), (geometry['output_size'][0]-1.,0.),
                 (0.,geometry['output_size'][1]-1.),
                 ((geometry['output_size'][0]-1.)/2., (geometry['output_size'][1]-1.)/2.)):
        rx, ry = geometry_index_to_raw_index(x, y)
        gx, gy = raw_edge_to_geometry_index(rx+.5, ry+.5)
        errors.append(max(abs(gx-x), abs(gy-y)))
    _require(max(errors)<1e-8, 'Raw/geometry coordinate roundtrip failed')
    proof = dict(raw_space='EXIF-transposed source pixel-edge coordinates',
        geometry_space='immutable RGBA pixel-centre indices',
        rotated_edge_to_raw_edge=reverse, raw_edge_to_rotated_edge=forward,
        expanded_size=expanded, combined_crop_shift=shift,
        maximum_roundtrip_error_px=max(errors), no_side_transpose=True,
        nearest_fixed_point_path_used=False)
    return raw_edge_to_geometry_index, geometry_index_to_raw_index, proof


def native_step_support(raw, line, columns, to_raw, *, inside_below=False):
    """Native RGB steps around a proposed source line; never moves a line."""
    import numpy as np
    offsets = [-8, -6, -4, 4, 6, 8]
    rows = []
    for x in columns:
        y = line[0]*float(x)+line[1]
        coordinates = [to_raw(float(x), y+offset) for offset in offsets]
        indices = np.rint(coordinates).astype(int)
        if (indices[:,0].min()<0 or indices[:,1].min()<0 or
                indices[:,0].max()>=raw.shape[1] or indices[:,1].max()>=raw.shape[0]):
            rows.append(dict(column=int(x), geometry_y=float(y), valid=False, reason='outside_native_source'))
            continue
        colors = raw[indices[:,1],indices[:,0],:3].astype(float)
        above, below = colors[:3], colors[3:]
        av, bv = np.median(above,axis=0), np.median(below,axis=0)
        noise = max(1., float(np.median(np.linalg.norm(np.concatenate([above-av,below-bv]),axis=1))))
        delta = bv-av
        rows.append(dict(column=int(x), geometry_y=float(y), valid=True,
            native_indices_xy=indices.tolist(), native_RGB=colors.tolist(),
            delta_below_minus_above_RGB=delta.tolist(), RGB_step=float(np.linalg.norm(delta)),
            local_RGB_noise=noise, step_to_noise=float(np.linalg.norm(delta)/noise),
            inside_RGB=(bv if inside_below else av).tolist(),
            outside_RGB=(av if inside_below else bv).tolist()))
    valid = [row for row in rows if row['valid']]
    if valid:
        vectors = np.asarray([row['delta_below_minus_above_RGB'] for row in valid])
        direction = np.median(vectors,axis=0); norm = float(np.linalg.norm(direction))
        agreement = (vectors@direction)/(np.linalg.norm(vectors,axis=1)*norm+1e-12)
        for row, value in zip(valid, agreement):
            row['direction_agreement'] = float(value)
        strong = [row['step_to_noise']>=3 and row['direction_agreement']>=.8 for row in valid]
        fraction = sum(strong)/len(strong)
    else:
        direction, fraction = np.zeros(3), 0.
    return dict(samples=rows, valid_columns=len(valid), diagnostic_supported_fraction=fraction,
        median_delta_RGB=direction.tolist(), native_RGB_resampled=False,
        diagnostic_only=True, semantic_body_boundary_certified=False,
        scoring=dict(step_to_noise=3, direction_agreement=.8, sample_offsets=offsets))


def propose_effective_back_reference(rgba, raw, geometry, label_mask):
    """Reusable source-only candidate. No side position/hardware as fit input.

    Historical raw RGB rail supplies a hypothesis, not an approved crop. Its
    native transitions and the top endpoint remain explicit review evidence.
    """
    import numpy as np
    forward, to_raw, mapping = coordinate_map(geometry)
    h, w = rgba.shape[:2]; opaque = rgba[...,3]>=128
    ys = np.arange(int(h*.25),int(h*.65))
    _require(len(ys)>0 and opaque[ys].any(axis=1).all(), 'Incomplete source shaft alpha')
    left = int(np.median(np.argmax(opaque[ys],axis=1)))
    right = int(np.median(w-1-np.argmax(opaque[ys,::-1],axis=1)))
    _require(right-left>16, 'No usable source shaft width')
    columns = np.arange(left+int((right-left)*.25),left+int((right-left)*.75))
    columns = columns[~label_mask[int(h*.8):,columns].any(axis=0)]
    _require(len(columns)>=10, 'No independent source top columns outside label')
    _require(opaque[:,columns].any(axis=0).all(), 'Incomplete source top alpha')
    top = float(np.median(np.argmax(opaque[:,columns],axis=0)))
    line = geometry.get('bottom_rail')
    _require(line is not None and len(line)==2, 'No saved raw-space bottom rail')
    raw_x = geometry['body_x']
    # bottom_rail was fitted to raw pixel-edge coordinates: do NOT add .5 to y.
    points = [forward(x+.5, line[0]*(x+.5)+line[1]) for x in raw_x]
    _require(abs(points[1][0]-points[0][0])>1, 'Degenerate mapped raw rail')
    slope = (points[1][1]-points[0][1])/(points[1][0]-points[0][0])
    mapped = [float(slope), float(points[0][1]-slope*points[0][0])]
    bottom = float(mapped[0]*(left+right)/2.+mapped[1])
    _require(0<=top<bottom<h, 'Saved source rail outside current geometry')
    flank_columns = {}
    evidence = {}
    for name, fractions in [('left',(.18,.30)),('right',(.70,.82))]:
        xx = np.unique(np.rint(np.linspace(left+(right-left)*fractions[0],
                                         left+(right-left)*fractions[1],17)).astype(int))
        keep = []
        for x in xx:
            y = int(round(mapped[0]*x+mapped[1]))
            if 0<=x<w and not label_mask[max(0,y-12):min(h,y+13),x].any():
                keep.append(int(x))
        flank_columns[name] = keep
        evidence[name] = native_step_support(raw,mapped,keep,to_raw)
    top_columns = [int(x) for x in np.unique(np.rint(np.linspace(left+.25*(right-left),
                                                                left+.75*(right-left),17)))]
    top_columns = [x for x in top_columns if not label_mask[max(0,int(top)-12):min(h,int(top)+13),x].any()]
    top_evidence = native_step_support(raw,[0.,top],top_columns,to_raw,inside_below=True)
    supported = all(v['valid_columns']>=8 and v['diagnostic_supported_fraction']>=.75
                    for v in evidence.values())
    return dict(method='same_source_raw_rail_reference_candidate_v1',
        status='reference_candidate_requires_review', body_y=[top,bottom],
        shaft_x=[left,right], original_bottom_rail_raw_edges=line,
        mapped_bottom_rail_geometry_indices=mapped, mapped_raw_rail_endpoints=points,
        coordinate_map=mapping, bottom_native_flanks=evidence, top_native=top_evidence,
        bottom_has_two_supported_native_flanks=supported,
        source_geometry_review_reasons=geometry.get('review_reasons',[]),
        fit_uses_side_antenna_or_hardware=False, geometry_changed=False,
        reference_certified=False, automatic_remap_allowed=False,
        notes='Native RGB step support is not semantic proof that a rail is the physical body end.')


def _supported(proposal):
    witnesses=[proposal['top_native'], *proposal['bottom_native_flanks'].values()]
    return len(witnesses)==3 and all(v['valid_columns']>=8 and
        v['diagnostic_supported_fraction']>=.75 for v in witnesses)


def build(raw_path, back_path, geometry_item, label_path=None):
    """Measure a bound back-only proposal; raises if source support is inadequate.

    geometry_item is this back image's immutable geometry-report row, including
    its raw/output hashes. Absence/failure is an explicit caller warning with
    the existing reference retained, never a guessed fallback length.
    """
    import numpy as np
    from PIL import Image, ImageOps
    _require(geometry_item.get('face')=='back', 'Only own back source may supply this reference')
    geometry_item=json.loads(json.dumps(geometry_item,allow_nan=False))
    geometry=_geometry_contract(geometry_item['geometry'])
    raw_binding,back_binding=_binding(raw_path),_binding(back_path)
    _require(raw_binding['sha256']==geometry_item['source_sha256'], 'Raw geometry source identity differs')
    _require(back_binding['sha256']==geometry_item['output_sha256'], 'Back geometry identity differs')
    with Image.open(raw_path) as image:
        raw=np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
    with Image.open(back_path) as image:
        _require(image.mode=='RGBA' and image.format=='PNG', 'Immutable back must be RGBA PNG')
        rgba=np.asarray(image)
    _require(list(raw.shape[1::-1])==geometry['source_size'], 'EXIF-oriented raw dimensions differ')
    _require(list(rgba.shape[1::-1])==geometry['output_size'], 'Current back dimensions differ')
    label_binding=None
    if label_path is None:
        label=np.zeros(rgba.shape[:2],bool)
    else:
        label_binding=_binding(label_path)
        with Image.open(label_path) as image:
            label=np.asarray(image.convert('L'))>0
        _require(label.shape==rgba.shape[:2], 'Label exclusion differs from source coordinates')
    proposal=propose_effective_back_reference(rgba,raw,geometry,label)
    _require(_supported(proposal), 'Source body reference lacks top or two native flank witnesses')
    record=dict(schema='source_body_reference_review_v1', method=METHOD,
        coordinate_mode=MODE, source=back_binding, raw_source=raw_binding,
        label_exclusion=label_binding, geometry_item=geometry_item,
        geometry_item_sha256=_canonical(geometry_item),
        implementation_sha256=_sha(__file__), body_y=proposal['body_y'],
        proposal=proposal, reference_supported=True, reference_review_required=True,
        reference_certified=False, remap_verified=False, visual_approved=False,
        publishable=False, source_geometry_changed=False,
        warning='机身长度采用本机源图边界参考；参考与最终天线位置仍待视觉核对')
    record['evidence_sha256']=_seal(record)
    validate(record,back_path)
    return record


def validate(record, back_measurement_path):
    """Bind the same reference to immutable pre-lighting geometry at both gates.

    The record hash catches accidental edits; this is evidence provenance, not
    an authentication signature. Source bytes and generating implementation
    are checked every time. A relit image must use its existing measurement
    source binding; the caller's old geometry/alpha comparison still applies.
    """
    _require(record.get('schema')=='source_body_reference_review_v1' and
             record.get('coordinate_mode')==MODE and record.get('method')==METHOD,
             'Unsupported body reference evidence')
    _require(record.get('evidence_sha256')==_seal(record), 'Body reference evidence changed')
    _require(record.get('implementation_sha256')==_sha(__file__), 'Body reference implementation changed')
    for key in ('source','raw_source','label_exclusion'):
        binding=record.get(key)
        if key=='label_exclusion' and binding is None:
            continue
        _require(isinstance(binding,dict) and _sha(binding['path'])==binding['sha256'],
                 'Body reference bound source changed: '+key)
    _require(_sha(back_measurement_path)==record['source']['sha256'], 'Body reference belongs to another back')
    item=record['geometry_item']
    _require(record['geometry_item_sha256']==_canonical(item), 'Body geometry evidence changed')
    _require(item.get('face')=='back' and item['source_sha256']==record['raw_source']['sha256'] and
             item['output_sha256']==record['source']['sha256'], 'Raw/back geometry provenance differs')
    geometry=_geometry_contract(item['geometry'])
    proposal=record['proposal']
    _require(proposal.get('method')==METHOD and proposal.get('fit_uses_side_antenna_or_hardware') is False and
             proposal.get('geometry_changed') is False, 'Reference is not source-only')
    _require(_supported(proposal), 'Source body reference lacks top or two native flank witnesses')
    _require(record.get('reference_supported') is True and record.get('reference_review_required') is True and
             all(record.get(k) is False for k in ('reference_certified','remap_verified','visual_approved',
                                                  'publishable','source_geometry_changed')),
             'Review reference must not claim final acceptance')
    forward,_,mapping=coordinate_map(geometry)
    _require(_canonical(mapping)==_canonical(proposal['coordinate_map']), 'Source mapping evidence differs')
    rail=geometry['bottom_rail']
    _require(rail==proposal['original_bottom_rail_raw_edges'], 'Source rail was replaced')
    points=[forward(x+.5,rail[0]*(x+.5)+rail[1]) for x in geometry['body_x']]
    _require(abs(points[1][0]-points[0][0])>1, 'Degenerate mapped rail')
    slope=(points[1][1]-points[0][1])/(points[1][0]-points[0][0])
    intercept=points[0][1]-slope*points[0][0]
    _require(max(abs(a-b) for a,b in zip([slope,intercept],proposal['mapped_bottom_rail_geometry_indices']))<1e-8,
             'Body rail mapping differs from source geometry')
    shaft=proposal['shaft_x']; body=record['body_y']; w,h=geometry['output_size']
    _require(len(shaft)==2 and 0<=shaft[0]<shaft[1]<w, 'Body shaft outside geometry')
    _require(isinstance(body,list) and len(body)==2 and all(math.isfinite(v) for v in body) and
             0<=body[0]<body[1]<h and body==proposal['body_y'], 'Body reference range invalid')
    _require(abs(body[1]-(slope*sum(shaft)/2+intercept))<1e-8, 'Body end differs from fixed source rail')
    top_samples=[s for s in proposal['top_native']['samples'] if s.get('valid')]
    _require(top_samples and all(abs(s['geometry_y']-body[0])<1e-8 for s in top_samples),
             'Body top differs from its native witnesses')
    return list(body)


def targets(record, antenna_rows, side_body, side_height):
    """Shared proposal/review equation; receives only already chosen reference.

    Side coordinates map a fixed back reference and never change its length.
    Hardware and actual final-pixel checks stay in the existing structure gate.
    """
    back0,back1=record['body_y']
    _require(type(side_height) is int and side_height>1 and len(side_body)==2 and
             all(math.isfinite(v) for v in side_body) and
             0<=side_body[0]<side_body[1]<side_height, 'Invalid side body coordinates')
    rows=list(antenna_rows)
    _require(len(rows)==4 and all(math.isfinite(v) for v in rows) and
             all(back0<rows[i]<rows[i+1]<back1 for i in range(3)),
             'Rear antennas must lie in the source body reference')
    return [(row-back0)/(back1-back0)*(side_body[1]-side_body[0])+side_body[0] for row in rows]

