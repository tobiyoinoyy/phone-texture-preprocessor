"""Enforce the user's explicit sticker-only protection policy on visible pixels."""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image


def check_sticker_only(config, source, protected, allowed=None):
    if config.get('protection_policy', 'reviewed_regions') != 'sticker_only':
        return None
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    if config.get('source_sha256') != digest:
        raise ValueError('Sticker-only policy requires current source SHA256')
    visible = np.array(Image.open(source).convert('RGBA'))[...,3] > 0
    if protected.shape != visible.shape:
        raise ValueError('Protection dimensions differ')
    path = config.get('sticker_mask')
    if not path:
        raise ValueError('Sticker-only policy needs an explicit source-sized sticker mask (empty if no sticker)')
    paper = np.array(Image.open(path).convert('L')) > 0
    if paper.shape != visible.shape:
        raise ValueError('Sticker mask dimensions differ')
    paper &= visible
    extra = protected & visible & ~paper
    missing = paper & ~protected
    if extra.any() or missing.any():
        raise ValueError(f'Sticker-only protection violated: {extra.sum()} extra protected pixels, {missing.sum()} unprotected paper pixels')
    if allowed is not None and np.any(paper & allowed):
        raise ValueError('Sticker pixels must not be editable')
    return {'policy':'sticker_only','sticker_mask_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'protected_visible_pixels':int(paper.sum()),'extra_protected_pixels':0}
