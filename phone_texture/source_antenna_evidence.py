"""Hash-bound original RGB evidence in the frozen geometry coordinates.

This reference measures source edges. It never changes output RGB or alpha.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from .geometry_review import digest


def create(source, geometry_path, geometry, face, target):
    original=ImageOps.exif_transpose(Image.open(source)).convert('RGB')
    if list(original.size)!=geometry['source_size']:
        raise ValueError('Original antenna evidence dimensions differ')
    reference=original.rotate(geometry['rotation_ccw_degrees'],resample=Image.Resampling.BICUBIC,expand=True)
    reference=reference.crop(geometry['canvas_trim_xyxy'])
    reference=reference.transpose(Image.Transpose.ROTATE_90 if face=='left' else Image.Transpose.ROTATE_270)
    reference=reference.crop(geometry['visible_canvas_trim']['crop_xyxy'])
    actual=Image.open(geometry_path).convert('RGBA')
    if actual.size!=reference.size:raise ValueError('Original antenna evidence coordinates differ')
    a=np.dstack([np.array(reference),np.array(actual)[...,3]])
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(a).save(target)
    return dict(method='original_rgb_before_alpha_premultiplication_v1',path=str(target),sha256=digest(target),
        geometry_sha256=digest(geometry_path),raw_source=str(source),raw_source_sha256=digest(source),
        geometry=geometry,face=face)


def load(context, geometry_path, face=None):
    if context.get('method')!='original_rgb_before_alpha_premultiplication_v1':
        raise ValueError('Unknown original antenna evidence method')
    if face is not None and context.get('face')!=face:
        raise ValueError('Original antenna evidence belongs to another face')
    if digest(geometry_path)!=context['geometry_sha256'] or digest(context['raw_source'])!=context['raw_source_sha256']:
        raise ValueError('Original antenna evidence belongs to changed source/geometry')
    if digest(context['path'])!=context['sha256']:
        raise ValueError('Original antenna evidence changed')
    image=Image.open(context['path']).convert('RGBA');actual=Image.open(geometry_path).convert('RGBA')
    if image.size!=actual.size or not np.array_equal(np.array(image)[...,3],np.array(actual)[...,3]):
        raise ValueError('Original antenna evidence changed geometry or alpha')
    if context.get('photometric_path'):
        if digest(context['photometric_path'])!=context['photometric_sha256']:
            raise ValueError('Photometric source evidence changed')
        with np.load(context['photometric_path'],allow_pickle=False) as cached:
            data=cached['data']
        if data.shape!=np.array(actual).shape or not np.isfinite(data).all() or not np.array_equal(data[...,3],np.array(actual)[...,3]):
            raise ValueError('Photometric source evidence changed coordinates or alpha')
        from .photometric_field import AnalyticalRGBA
        return AnalyticalRGBA(data)
    return image


def create_photometric(context,geometry_path,rows,face):
    if context.get('photometric_path'):raise ValueError('Photometric source evidence cannot be nested')
    from .photometric_recovery import recover
    reference,proof=recover(load(context,geometry_path,face),rows)
    target=Path(context['path']).with_name(Path(context['path']).stem+'-photometric.npz')
    np.savez_compressed(target,data=np.asarray(reference.convert('RGBA')))
    result=dict(context,photometric_path=str(target),photometric_sha256=digest(target),
        photometric_proof=proof,normalization_rows=list(rows))
    return load(result,geometry_path,face),result
