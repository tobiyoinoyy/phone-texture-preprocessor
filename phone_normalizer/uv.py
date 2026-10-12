"""Six-view precision cropping and optional registration to a numeric UV profile.

The legacy Normalizer remains compatible. This entry point preserves connected
soft alpha and applies the validated six-view orientation before registration.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from scripts.precise_geometry import normalize_candidate
from scripts.segmentation import normalize_image, validate_geometry_size
from .api import IMAGE_EXTENSIONS, Normalizer, NormalizerConfig
from .uv_profile import align_to_profile, load_profile, orient_candidate


FACES = ("front", "back", "left", "right", "top", "bottom")


def _validate_settings(config: NormalizerConfig, profile: Optional[dict]) -> None:
    if not config.tight_crop:
        raise ValueError("The precise six-view pipeline requires tight cropping")
    if profile is not None and (config.crop_threshold, config.geometry_threshold) != (128, 32):
        raise ValueError("UV profiles require crop_threshold=128 and geometry_threshold=32")
    if profile is not None:
        for rec in profile["faces"].values():
            validate_geometry_size(tuple(rec["reference_size"]), config.max_geometry_pixels)


def normalize_face(
    image: Image.Image,
    face: str,
    *,
    backend: Any,
    config: Optional[NormalizerConfig] = None,
    profile: Optional[dict] = None,
    max_anisotropy: float = 0.05,
    diagnostic_sink: Optional[Callable[[Image.Image], None]] = None,
) -> tuple[Image.Image, dict]:
    """Process a named face, retaining source RGB through segmentation.

    ``backend`` is caller-owned and reused across faces. A profile must be the
    validated dictionary returned by :func:`load_profile`. Result dimensions alone
    do not establish that ports, buttons, and cameras match the UV mesh.
    """
    face = str(face).lower()
    if face not in FACES:
        raise ValueError(f"Unknown face {face!r}; expected {', '.join(FACES)}")
    config = config or NormalizerConfig(crop_threshold=128)
    _validate_settings(config, profile)
    validate_geometry_size(image.size, config.max_geometry_pixels)
    # Reuse the core's EXIF, source-RGB preservation and backend validation,
    # while leaving all geometric work to the precise pipeline below.
    raw, segmentation = normalize_image(
        image, backend=backend, remove_support=False, deskew=False,
        tight_crop=False,
    )
    candidate, geometry = normalize_candidate(
        np.asarray(raw), crop_threshold=config.crop_threshold,
        geometry_threshold=config.geometry_threshold,
        confidence_floor=config.confidence_floor,
        remove_support=config.remove_support, deskew=config.deskew,
        max_deskew_degrees=config.max_deskew_degrees,
        max_geometry_pixels=config.max_geometry_pixels,
        require_support_border=config.require_support_border,
        preserve_canvas=profile is not None,
    )
    result = Image.fromarray(candidate)
    if profile is not None:
        result = orient_candidate(result, face, profile)
        oriented_size = result.size
        candidate = result
        result, registration = align_to_profile(
            candidate, face, profile, max_geometry_pixels=config.max_geometry_pixels,
            max_anisotropy=max_anisotropy,
        )
        if diagnostic_sink is not None:
            from .uv_diagnostics import alignment_diagnostic
            diagnostic_sink(alignment_diagnostic(candidate, result, registration))
    else:
        if face == "left":
            result = result.transpose(Image.Transpose.ROTATE_90)
        elif face == "right":
            result = result.transpose(Image.Transpose.ROTATE_270)
        oriented_size = result.size
        registration = None
        from .body_geometry import measure_body
        geometry["body_evidence"] = measure_body(result)
        if diagnostic_sink is not None:
            from .uv_diagnostics import crop_diagnostic
            diagnostic_sink(crop_diagnostic(result, face, geometry["body_evidence"]))
    return result, {
        "pipeline": "stable-six-view-v2", "face": face,
        "model": segmentation.model, "device": segmentation.device,
        "model_input_size": segmentation.model_input_size,
        "input_size": segmentation.input_size,
        "oriented_size": list(oriented_size), "output_size": list(result.size),
        "geometry": geometry, "registration": registration,
        "geometric_resampling_passes": int(abs(geometry["rotation_degrees"]) >= 1e-5)
            + (registration["source_resampling_passes_in_final_image"] if registration else 0),
        "uv_fit_validation": "not_performed",
    }


def _six_sources(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"Input must be a folder containing the six named faces: {directory}")
    sources: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        face = path.stem.lower()
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS or face not in FACES:
            continue
        if face in sources:
            raise ValueError(f"Duplicate face {face}: {sources[face].name}, {path.name}")
        sources[face] = path
    missing = [face for face in FACES if face not in sources]
    if missing:
        raise ValueError(f"Missing named faces: {', '.join(missing)}")
    return sources


def _write_new_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def process_six_views(
    input_dir: Any,
    output_dir: Any,
    *,
    uv_profile: Optional[Any] = None,
    config: Optional[NormalizerConfig] = None,
    backend_instance: Optional[Any] = None,
    progress: Optional[Callable[[str], None]] = None,
    max_anisotropy: float = 0.05,
) -> dict:
    """Process one six-view set without overwriting any existing artifact.

Unknown filenames are ignored; missing or duplicate named faces are rejected
before inference. Failures stop the set and are reported in manifest.json;
completed faces remain available. Re-run into a new output directory.
    """
    import math
    if isinstance(max_anisotropy, bool) or not math.isfinite(max_anisotropy) or not 0 <= max_anisotropy <= 1:
        raise ValueError("max_anisotropy must be a finite number between 0 and 1")
    source_dir = Path(input_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if source_dir == destination:
        raise ValueError("Use a separate output folder; source files must remain unchanged")
    sources = _six_sources(source_dir)
    config = config or NormalizerConfig(crop_threshold=128)
    profile = (load_profile(uv_profile, max_geometry_pixels=config.max_geometry_pixels)
               if uv_profile is not None else None)
    # This profile's body coordinates were calibrated using the fixed alpha
    # thresholds. Reject mismatched settings instead of silently changing UVs.
    _validate_settings(config, profile)
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Output must be a directory: {destination}")
    metadata_dir = destination / "metadata"
    if metadata_dir.exists() and not metadata_dir.is_dir():
        raise ValueError(f"Metadata path must be a directory: {metadata_dir}")
    diagnostics_dir = destination / "diagnostics"
    review_dir = destination / "review"
    for folder in (diagnostics_dir, review_dir):
        if folder.exists() and not folder.is_dir():
            raise ValueError(f"Output subfolder must be a directory: {folder}")
    artifacts = [destination / "manifest.json"]
    artifacts += [destination / f"{face}.png" for face in FACES]
    artifacts += [metadata_dir / f"{face}.json" for face in FACES]
    artifacts += [folder / f"{face}.png" for folder in (diagnostics_dir, review_dir) for face in FACES]
    for path in artifacts:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Output already exists; choose a new output folder: {path}")
    # Read headers before opening a model session, including the last face.
    for source in sources.values():
        with Image.open(source) as opened:
            validate_geometry_size(opened.size, config.max_geometry_pixels)
    destination.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    if profile is not None:
        review_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "schema_version": 1, "pipeline": "stable-six-view-v2",
        "profile_id": profile["profile_id"] if profile is not None else None,
        "settings": asdict(config), "success": False, "faces": {},
        "uv_fit_validation": "not_performed",
        "geometry_gate": "not_performed", "review_required_faces": [],
        "max_anisotropy": max_anisotropy,
    }
    try:
        with Normalizer(config, backend_instance=backend_instance) as normalizer:
            for face in FACES:
                if progress is not None:
                    progress(face)
                source = sources[face]
                diagnostic_path = diagnostics_dir / f"{face}.png"
                def save_diagnostic(image):
                    with diagnostic_path.open("xb") as handle:
                        image.save(handle, format="PNG")
                with Image.open(source) as opened:
                    result, info = normalize_face(
                        opened, face, backend=normalizer.backend,
                        config=config, profile=profile, max_anisotropy=max_anisotropy,
                        diagnostic_sink=save_diagnostic,
                    )
                registration = info["registration"]
                rejected = registration is not None and registration["geometry_gate"] != "passed"
                if rejected:
                    report["review_required_faces"].append(face)
                output = (review_dir if rejected else destination) / f"{face}.png"
                info["diagnostic"] = str(diagnostic_path)
                with output.open("xb") as handle:
                    result.save(handle, format="PNG")
                info.update({"source": str(source), "output": str(output)})
                _write_new_json(metadata_dir / f"{face}.json", info)
                report["faces"][face] = info
        report["success"] = True
        if profile is not None:
            report["geometry_gate"] = "review_required" if report["review_required_faces"] else "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _write_new_json(destination / "manifest.json", report)
    return report
