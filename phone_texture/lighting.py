"""Typed adapters to the bundled lighting engines, shared by every entry point."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry_review import digest, write
from .pipeline import fresh


def engine(material):
    names = {
        'aluminum': ('phone-aluminum-frame-restore', 'harmonize_aluminum.py'),
        'stainless': ('phone-six-view-restore', 'composite_natural_highlights.py'),
    }
    if material not in names:
        raise ValueError('请选择 aluminum 或 stainless 材质')
    skill, script = names[material]
    path = Path(__file__).resolve().parents[1] / 'skills' / skill / 'scripts' / script
    spec = importlib.util.spec_from_file_location('_phone_lighting_' + material, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(material, job_path):
    """Run reviewed parameters; no material classification or image generation."""
    job_path = Path(job_path).resolve()
    job = json.loads(job_path.read_text())
    module = engine(material)
    if material == 'aluminum':
        return module.run(job_path)
    # A stainless job contains only the compositing API arguments and records_dir.
    records = Path(job.pop('records_dir')).resolve()
    fresh(records)
    report = module.composite(**job)
    records.mkdir(parents=True)
    src = np.asarray(Image.open(job['source']).convert('RGBA'))
    highlighted = np.asarray(Image.open(job['highlight_mask']).convert('L')) > 0
    protected = (np.asarray(Image.open(job['protect_mask']).convert('L')) > 0
                 if job.get('protect_mask') else np.zeros(src.shape[:2], bool))
    allowed = highlighted & (src[..., 3] > 0) & ~protected
    for name, mask in [('editable', allowed), ('protected', protected)]:
        Image.fromarray(mask.astype('uint8') * 255).save(records / (name + '.png'))
    write(records / 'validation.json', report)
    return report


def dependencies(value):
    """Bind file inputs as well as parameter text for resumable jobs."""
    result = {}

    def walk(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key not in ('output', 'output_dir', 'records_dir'):
                    walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)
        elif isinstance(item, str):
            try:
                path = Path(item).expanduser()
                if path.is_file():
                    result[str(path.resolve())] = digest(path)
            except (OSError, ValueError):
                pass

    walk(value)
    return result
