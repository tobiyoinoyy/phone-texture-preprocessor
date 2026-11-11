"""Portable, explicitly pinned review candidates; no image processing.

Only the current producer's declared evidence inventory is read during export.
Replay reads the bundle and the six current raw files, never an origin path.
Hashes certify byte identity, not visual approval or new material evidence.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import tempfile
import zlib


SCHEMA = 'phone_texture_review_bundle/v1'
EVIDENCE_SCHEMA = 'current_review_evidence/v1'
FACES = ('front', 'back', 'left', 'right', 'top', 'bottom')
OPTION_KEYS = {'model', 'protection_policy', 'review_fallback', 'lighting_profile',
               'bottom_alignment', 'source_body_reference_review', 'side_candidate_review'}
UNAPPROVED = ('visual_review_passed', 'visual_approved', 'model_uv_verified',
              'selected_for_delivery', 'structural_alignment_certified',
              'strict_structure_verified', 'strict_endpoint_evidence_complete',
              'material_classification_complete')
FACE_VERIFIED = ('source_identity_verified', 'pixel_integrity_verified', 'output_selection_verified')
PREFIX = 'bundle://'


class BundleError(ValueError):
    pass


def _require(value, message):
    if not value:
        raise BundleError(message)


def _digest(value):
    _require(isinstance(value, str) and len(value) == 64
             and all(c in '0123456789abcdef' for c in value), 'Invalid SHA-256')
    return value


def _bytes(value):
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    except (TypeError, ValueError) as error:
        raise BundleError('Evidence must be finite JSON data') from error


def _clone(value):
    return json.loads(_bytes(value))


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate JSON member: ' + key)
        result[key] = value
    return result


def _json(data):
    try:
        return json.loads(data, object_pairs_hook=_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(BundleError('Nonfinite JSON number')))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BundleError('Invalid bundle JSON') from error


def _absolute(path, *, exists=True, directory=False):
    _require(isinstance(path, (str, Path)) and str(path), 'Explicit filesystem path required')
    path = Path(path).expanduser()
    _require('..' not in path.parts, 'Parent traversal is not permitted')
    path = path.absolute()
    for part in (path, *path.parents):
        _require(not part.is_symlink(), 'Symlink is not permitted: ' + str(part))
    if exists:
        _require(path.is_dir() if directory else path.is_file(), 'Required path missing: ' + str(path))
    return path


def _relative(value):
    _require(isinstance(value, str) and value and '\\' not in value and '\x00' not in value,
             'Invalid relative bundle path')
    path = PurePosixPath(value)
    _require(not path.is_absolute() and path.as_posix() == value
             and all(part not in ('', '.', '..') and ':' not in part for part in path.parts),
             'Unsafe relative bundle path: ' + value)
    return value


def _inside(root, relative):
    return _absolute(root.joinpath(*PurePosixPath(_relative(relative)).parts))


def _read(path, expected=None):
    path = _absolute(path)
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags)
    try:
        _require(stat.S_ISREG(os.fstat(fd).st_mode), 'Evidence must be a regular file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            data = stream.read()
    finally:
        os.close(fd)
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None:
        _require(digest == _digest(expected), 'Stale source/evidence SHA: ' + str(path))
    return data, digest


def _identity(path, expected=None):
    """Hash opaque assets in bounded chunks; never retain a case in memory."""
    path = _absolute(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    digest = hashlib.sha256(); size = 0
    try:
        _require(stat.S_ISREG(os.fstat(fd).st_mode), 'Evidence must be a regular file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk); size += len(chunk)
    finally:
        os.close(fd)
    result = digest.hexdigest()
    if expected is not None:
        _require(result == _digest(expected), 'Stale source/evidence SHA: ' + str(path))
    return result, size


def _copy(source, target, expected):
    source = _absolute(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(source, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    digest = hashlib.sha256()
    try:
        _require(stat.S_ISREG(os.fstat(fd).st_mode), 'Evidence must be a regular file')
        with os.fdopen(fd, 'rb', closefd=False) as incoming, target.open('xb') as outgoing:
            for chunk in iter(lambda: incoming.read(1024*1024), b''):
                digest.update(chunk); outgoing.write(chunk)
    finally:
        os.close(fd)
    _require(digest.hexdigest() == _digest(expected), 'Source changed during copy')
    _identity(target, expected)


def _png(data):
    # Header validation only; the producer owns all measured pixel contracts.
    _require(len(data) >= 33 and data[:8] == b'\x89PNG\r\n\x1a\n'
             and data[8:12] == b'\x00\x00\x00\r' and data[12:16] == b'IHDR',
             'Candidate is not a PNG')
    width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', data[16:29])
    _require(width > 0 and height > 0 and depth == 8 and color == 6
             and compression == filtering == 0 and interlace in (0, 1)
             and zlib.crc32(data[12:29]) & 0xffffffff == struct.unpack('>I', data[29:33])[0],
             'Candidate must have a valid 8-bit RGBA PNG header')
    return {'width': width, 'height': height, 'mode': 'RGBA', 'header_only': True}


def _options(value):
    _require(isinstance(value, dict) and set(value) == OPTION_KEYS, 'Complete explicit processing options required')
    _require(value['model'] in ('iphone17pro', 'iphone17promax')
             and value['protection_policy'] in ('none', 'sticker_only')
             and value['review_fallback'] in ('fail', 'preserve-source')
             and value['lighting_profile'] in ('baseline', 'accepted-shadow-lift-v1', 'accepted-shadow-lift-v2')
             and value['bottom_alignment'] in ('appearance', 'off')
             and type(value['source_body_reference_review']) is bool
             and type(value['side_candidate_review']) is bool, 'Unsupported processing options')
    return _clone(value)


def _sku(value):
    _require(isinstance(value, str) and value not in ('', '.', '..')
             and Path(value).name == value and '/' not in value and '\\' not in value,
             'Unsafe SKU')
    return value


def _source_map(mapping):
    _require(isinstance(mapping, dict) and set(mapping) == set(FACES), 'Six explicit current raw-file paths required')
    paths = {face: _absolute(path) for face, path in mapping.items()}
    _require(len(set(paths.values())) == 6, 'Raw faces must refer to six distinct files')
    return paths, {face: _identity(path)[0] for face, path in paths.items()}


def _flags(record, label):
    _require(isinstance(record, dict), label + ' missing')
    for field in UNAPPROVED:
        if field in record:
            _require(record[field] is False, label + ' elevated approval: ' + field)


def _bound_report(report, assets, raw_hashes, label, report_loader):
    _flags(report, label)
    for field in ('quality_checks_complete', 'six_faces_processed'):
        if field in report:
            _require(report[field] is False, label + ' elevated certification: ' + field)
    _require(report.get('eligible_for_visual_review') is True and report.get('hard_failures') == [],
             label + ' is not a completed hard-check candidate')
    for field in ('source_sha256', 'candidate_sha256'):
        _digest(report.get(field))
    for field, hash_field in (('source', 'source_sha256'), ('candidate', 'candidate_sha256')):
        _require(isinstance(report.get(field), str) and assets.get(report[field]) == report[hash_field],
                 label + ' unbound ' + field)
    artifacts = report.get('artifacts')
    _require(isinstance(artifacts, dict) and {'result', 'preparation', 'candidate'} <= set(artifacts),
             label + ' required evidence artifacts missing')
    for name, path in artifacts.items():
        _require(isinstance(path, str) and path in assets, label + ' evidence asset missing: ' + name)
    _require(assets[artifacts['candidate']] == report['candidate_sha256'], label + ' artifact/output mismatch')
    _require(report_loader(artifacts['result']) == report,
             label + ' semantic checks differ from the bound current report')
    for field in ('artifact_sha256', 'portable_assets_sha256', 'input_sha256', 'implementation_sha256', 'code_sha256'):
        mapping = report.get(field, {})
        _require(isinstance(mapping, dict), label + ' malformed ' + field)
        for path, digest in mapping.items():
            _digest(digest)
            # Raw origins need not remain accessible: the package includes the
            # intake snapshots and replay checks the current six raw hashes.
            _require(assets.get(path) == digest or digest in raw_hashes.values()
                     or (field in ('implementation_sha256', 'code_sha256') and digest in assets.values()),
                     label + ' declared dependency is absent from bundle: ' + str(path))


def _semantics(row, evidence, raw_hashes, final_hashes, assets, report_loader):
    _require(isinstance(row, dict) and row.get('six_faces_generated') is True
             and row.get('status') in ('source_preserved_needs_review', 'full_stages_effect_warning_needs_review',
                                       'processed_needs_visual_review'), 'A completed generated candidate row is required')
    _sku(row.get('sku'))
    _flags(row, 'Candidate')
    _require(row.get('visual_review_passed') is False and row.get('model_uv_verified') is False,
             'Candidate must retain explicit unapproved visual/UV state')
    _require(isinstance(evidence, dict) and set(evidence) == {'schema', 'options', 'assets', 'faces'}
             and evidence['schema'] == EVIDENCE_SCHEMA, 'Current producer evidence schema is missing or unknown')
    _options(evidence['options'])
    _require(row.get('final_sha256') == final_hashes and set(final_hashes) == set(FACES), 'Final six-face selection differs')
    _require(set(evidence.get('faces', {})) == set(FACES), 'Six current per-face checks required')
    stages = row.get('per_face_stages')
    _require(isinstance(stages, dict) and set(stages) == set(FACES), 'Six explicit selected stages required')
    _require(set(raw_hashes) == set(FACES), 'Six original-source identities required')
    for value in raw_hashes.values():
        _require(_digest(value) in assets.values(), 'Raw intake snapshot omitted from evidence inventory')
    for face in FACES:
        fact = evidence['faces'][face]
        _require(isinstance(fact, dict) and set(fact) == {'source_sha256', 'output_sha256', 'checks'},
                 'Malformed current face evidence: ' + face)
        _require(_digest(fact['source_sha256']) in assets.values()
                 and fact['output_sha256'] == final_hashes[face], 'Current source/output binding differs: ' + face)
        checks = fact['checks']
        _require(isinstance(checks, dict) and checks.get('hard_failures') == []
                 and all(checks.get(name) is True for name in FACE_VERIFIED),
                 'Current hard checks incomplete or failed: ' + face)
        info = stages[face]
        _require(isinstance(info, dict) and info.get('source_sha256') == fact['source_sha256']
                 and info.get('selected_sha256') == final_hashes[face], 'Selected stage is not current: ' + face)
        if info.get('selected_stage') == 'structure_review':
            _require(evidence['options']['side_candidate_review'] is True, 'Structure review was not explicitly enabled')
            for name in ('strict_structure_verified', 'material_classification_complete', 'quality_checks_complete', 'six_faces_processed'):
                _require(row.get(name) is False, 'Review remap cannot claim complete certification: ' + name)
            light, structure = checks.get('lighting'), checks.get('structure')
            _bound_report(light, assets, raw_hashes, face + ' lighting', report_loader)
            _bound_report(structure, assets, raw_hashes, face + ' structure', report_loader)
            _require(light['source_sha256'] == fact['source_sha256']
                     and structure['source_sha256'] == light['candidate_sha256']
                     and structure['candidate_sha256'] == final_hashes[face], 'Fresh lighting/remap chain differs: ' + face)
        elif info.get('selected_stage') == 'lighting_review':
            light = checks.get('lighting')
            _bound_report(light, assets, raw_hashes, face + ' lighting', report_loader)
            _require(light['source_sha256'] == fact['source_sha256']
                     and light['candidate_sha256'] == final_hashes[face], 'Fresh lighting chain differs: ' + face)


def _portable(value, references, origins):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            transformed = _portable(key, references, origins)
            transformed_item = _portable(item, references, origins)
            _require(transformed not in result or result[transformed] == transformed_item,
                     'Path rewriting would merge conflicting evidence keys')
            result[transformed] = transformed_item
        return result
    if isinstance(value, list):
        return [_portable(item, references, origins) for item in value]
    if isinstance(value, str):
        if value in references:
            return PREFIX + references[value]
        if Path(value).is_absolute():
            token = 'origin:' + hashlib.sha256(value.encode()).hexdigest()
            origins[token] = value
            return token
    return value


def _restore(value, root, inventory):
    if isinstance(value, dict):
        return {_restore(key, root, inventory): _restore(item, root, inventory) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore(item, root, inventory) for item in value]
    if isinstance(value, str) and value.startswith(PREFIX):
        relative = _relative(value[len(PREFIX):])
        _require(relative in inventory, 'Reference is absent from asset inventory')
        return str(root.joinpath(*PurePosixPath(relative).parts))
    return value


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
    _read(path, hashlib.sha256(data).hexdigest())


def package(row, case, bundle_dir):
    """Export a current case. The manifest is published only after all checks."""
    case = _absolute(case, directory=True)
    target = _absolute(bundle_dir, exists=False)
    _require(not target.exists() and target != case and target not in case.parents,
             'Bundle destination must be fresh and cannot contain the case')
    row = _clone(row)
    _require(_absolute(row.get('process'), directory=True) == case, 'Candidate belongs to another case')
    sku = _sku(row.get('sku'))
    evidence = row.pop('portable_review_evidence', None)
    _require(isinstance(evidence, dict), 'Current producer evidence is required')
    options = _options(evidence.get('options'))
    intake_data, intake_sha = _read(case/'work/intake.json')
    intake = _json(intake_data)
    _require(isinstance(intake, dict) and set(intake.get('skus', {})) == {sku}, 'Current intake SKU differs')
    raw = intake['skus'][sku]
    _require(isinstance(raw, dict) and set(raw) == set(FACES), 'Current intake is incomplete')
    raw_hashes = {}; read_bindings = {str(case/'work/intake.json'): intake_sha}
    for face in FACES:
        item = raw[face]; expected = _digest(item.get('sha256'))
        snapshot = _absolute(item.get('snapshot'))
        _require(snapshot.is_relative_to(case/'work/input'/sku), 'Snapshot is outside the current intake')
        for path in (_absolute(item.get('source')), snapshot):
            _identity(path, expected); read_bindings[str(path)] = expected
        raw_hashes[face] = expected
    declared = evidence.get('assets')
    _require(isinstance(declared, dict) and declared, 'Current asset inventory is empty')
    runtime = Path(__file__).absolute().parents[1]
    assets = {}; blobs = {}; references = {}; origins = {}; asset_origins = {}
    for original, expected in declared.items():
        _require(isinstance(original, str) and Path(original).is_absolute(), 'Asset origin must be explicit')
        _digest(expected)
        path = _absolute(original)
        _require(path.is_relative_to(case) or path.is_relative_to(runtime), 'Asset is outside current case/runtime: ' + original)
        digest, size = _identity(path, expected)
        assets[original] = digest; read_bindings[str(path)] = digest
        relative = 'assets/' + digest + '.blob'
        references[original] = relative
        blobs.setdefault(relative, dict(source=path, sha256=digest, size_bytes=size))
        asset_origins.setdefault(relative, []).append(original)
    candidate = _absolute(row.get('candidate'), directory=True)
    _require(candidate.is_relative_to(case), 'Candidate is outside current case')
    final_hashes = row.get('final_sha256')
    _require(isinstance(final_hashes, dict) and set(final_hashes) == set(FACES), 'Six final hashes required')
    for expected in final_hashes.values():
        _digest(expected)
    face_records = {}; face_bytes = {}
    for face in FACES:
        path = candidate/sku/(face+'.png'); data, digest = _read(path, final_hashes[face])
        read_bindings[str(path)] = digest
        relative = 'faces/' + face + '.png'
        face_records[face] = dict(path=relative, sha256=digest, size_bytes=len(data), png=_png(data))
        face_bytes[relative] = dict(source=path, sha256=digest, size_bytes=len(data))
        # Keep an explicitly inventoried stage artifact as an asset reference.
        references.setdefault(str(path), relative)
    _semantics(row, evidence, raw_hashes, final_hashes, assets,
               lambda path: _json(_read(path, assets[path])[0]))
    semantic = _portable({'row': row, 'evidence': evidence}, references, origins)
    manifest = dict(schema=SCHEMA, sku=sku, options=options, source_sha256=raw_hashes,
        faces=face_records, assets={relative: dict(sha256=item['sha256'], size_bytes=item['size_bytes'])
                                   for relative, item in blobs.items()}, semantic=semantic,
        origin_provenance=dict(case=str(case), paths=origins, asset_origins=asset_origins,
            path_policy='descriptive_only_never_opened_on_replay', code_blobs_executable=False,
            intake_sha256=intake_sha, current_producer_required=True),
        approval_added=False, image_processing_executed=False)
    manifest_data = _bytes(manifest); manifest_sha = hashlib.sha256(manifest_data).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+target.name+'-staging-', dir=target.parent))
    try:
        for relative, item in {**blobs, **face_bytes}.items():
            _copy(item['source'], staging/relative, item['sha256'])
        for path, expected in read_bindings.items():
            _identity(path, expected)
        _write(staging/'manifest.json', manifest_data)
        _require(not target.exists(), 'Bundle destination appeared during export')
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return dict(schema=SCHEMA, path=str(target), manifest_sha256=manifest_sha,
                sku=sku, source_sha256=raw_hashes, final_sha256=_clone(final_hashes),
                visual_approval_added=False)


def _verified(bundle_dir, current_source, *, sku, options, expected_manifest_sha256):
    root = _absolute(bundle_dir, directory=True)
    manifest_data, digest = _read(root/'manifest.json', _digest(expected_manifest_sha256))
    manifest = _json(manifest_data)
    _require(isinstance(manifest, dict) and manifest.get('schema') == SCHEMA, 'Unknown review bundle schema')
    _require(manifest.get('sku') == _sku(sku), 'Bundle SKU differs; renaming requires a new explicit contract')
    _require(manifest.get('options') == _options(options), 'Bundle model/processing configuration differs')
    _require(manifest.get('approval_added') is False and manifest.get('image_processing_executed') is False,
             'Bundle claims an unsupported operation')
    raw_paths, raw_hashes = _source_map(current_source)
    _require(manifest.get('source_sha256') == raw_hashes, 'Current six originals differ from bundle')
    asset_records, faces = manifest.get('assets'), manifest.get('faces')
    _require(isinstance(asset_records, dict) and asset_records and isinstance(faces, dict)
             and set(faces) == set(FACES), 'Bundle inventory incomplete')
    inventory = {}; files = {}; final_hashes = {}
    for relative, item in asset_records.items():
        _relative(relative)
        _require(isinstance(item, dict) and set(item) == {'sha256', 'size_bytes'}
                 and relative == 'assets/' + _digest(item.get('sha256')) + '.blob', 'Invalid content-addressed asset')
        path = _inside(root, relative); value, size = _identity(path, item['sha256'])
        _require(type(item['size_bytes']) is int and size == item['size_bytes'], 'Asset size mismatch')
        inventory[relative] = value; files[str(path)] = value
    for face, item in faces.items():
        _require(isinstance(item, dict) and set(item) == {'path', 'sha256', 'size_bytes', 'png'}
                 and item.get('path') == 'faces/' + face + '.png', 'Face path is not canonical')
        path = _inside(root, item['path']); data, value = _read(path, _digest(item['sha256']))
        _require(type(item['size_bytes']) is int and item['size_bytes'] == len(data) and item['png'] == _png(data),
                 'Candidate byte/header identity differs')
        inventory[item['path']] = value; files[str(path)] = value; final_hashes[face] = value
    expected_files = set(inventory) | {'manifest.json'}
    actual_files = set()
    for directory, directories, names in os.walk(root, followlinks=False):
        for name in directories + names:
            _require(not (Path(directory)/name).is_symlink(), 'Symlink in bundle tree')
        for name in names:
            actual_files.add((Path(directory)/name).relative_to(root).as_posix())
    _require(actual_files == expected_files, 'Unlisted or missing files in bundle')
    provenance = manifest.get('origin_provenance')
    _require(isinstance(provenance, dict)
             and provenance.get('path_policy') == 'descriptive_only_never_opened_on_replay'
             and provenance.get('code_blobs_executable') is False,
             'Unsupported origin provenance policy')
    origins = provenance.get('asset_origins')
    _require(isinstance(origins, dict) and set(origins) == set(asset_records), 'Asset origin aliases incomplete')
    references = {}
    for relative, aliases in origins.items():
        _require(isinstance(aliases, list) and aliases
                 and all(isinstance(path, str) and Path(path).is_absolute() for path in aliases),
                 'Malformed descriptive asset aliases')
        for path in aliases:
            _require(path not in references or references[path] == relative, 'Conflicting descriptive asset alias')
            references[path] = relative
    def report_loader(path):
        # Descriptive old strings are used as rewrite keys only, never opened.
        original = _json(_read(path, files[path])[0])
        return _restore(_portable(original, references, {}), root, inventory)
    semantic = _restore(manifest.get('semantic'), root, inventory)
    _require(isinstance(semantic, dict) and set(semantic) == {'row', 'evidence'}, 'Semantic review record missing')
    row, evidence = semantic['row'], semantic['evidence']
    _require(row.get('sku') == sku and evidence.get('options') == options, 'Semantic identity/config differs')
    declared = evidence.get('assets')
    _require(isinstance(declared, dict) and declared and all(files.get(path) == value for path, value in declared.items()),
             'Semantic assets are not covered by the verified bundle')
    _semantics(row, evidence, raw_hashes, final_hashes, declared, report_loader)
    _require(set(asset_records) == {Path(path).relative_to(root).as_posix() for path in declared},
             'Evidence inventory omits a packaged asset')
    files[str(root/'manifest.json')] = digest
    for path, expected in files.items():
        _identity(path, expected)
    for face, path in raw_paths.items():
        _identity(path, raw_hashes[face])
    descriptor = dict(schema=SCHEMA, path=str(root), manifest_sha256=digest, sku=sku,
        options=_clone(options), source_sha256=raw_hashes, final_sha256=final_hashes,
        status=row['status'], six_faces_generated=True, approval_added=False,
        historical_paths_read=False, evidence_assets=len(asset_records))
    return descriptor, manifest, row, evidence, files, raw_paths


def verify(bundle_dir, current_source, *, sku, options, expected_manifest_sha256):
    """Read-only verification. The expected manifest SHA is mandatory."""
    return _verified(bundle_dir, current_source, sku=sku, options=options,
                     expected_manifest_sha256=expected_manifest_sha256)[0]


def replay(bundle_dir, current_source, *, sku, options, case, expected_manifest_sha256):
    """Copy a verified moved bundle into a fresh case, preserving review state."""
    target = _absolute(case, exists=False)
    _require(not target.exists(), 'Replay requires a fresh case')
    descriptor, manifest, row, evidence, files, raw_paths = _verified(
        bundle_dir, current_source, sku=sku, options=options,
        expected_manifest_sha256=expected_manifest_sha256)
    bundle = Path(descriptor['path'])
    _require(target != bundle and target not in bundle.parents and bundle not in target.parents,
             'Replay case and bundle must be independent')
    _require(all(target != path and target not in path.parents for path in raw_paths.values()),
             'Replay case must not contain current original files')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+target.name+'-replay-', dir=target.parent))
    try:
        for path, expected in files.items():
            relative = Path(path).relative_to(bundle)
            _copy(path, staging/'work/review-bundle'/relative, expected)
        intake = dict(source_root=str(next(iter(raw_paths.values())).parent), delivery_goal='model_textures',
                      goal_reason=None, skus={sku: {}})
        for face, source in raw_paths.items():
            suffix = source.suffix.lower() or '.raw'
            relative = Path('work/input')/sku/(face+suffix)
            _copy(source, staging/relative, descriptor['source_sha256'][face])
            intake['skus'][sku][face] = dict(source=str(source), snapshot=str(target/relative),
                                           sha256=descriptor['source_sha256'][face])
        inventory = {**manifest['assets'], **{item['path']: item for item in manifest['faces'].values()}}
        restored = _restore(manifest['semantic'], target/'work/review-bundle', inventory)
        result = restored['row']
        for face, item in manifest['faces'].items():
            _copy(bundle/item['path'], staging/'candidate'/sku/(face+'.png'), item['sha256'])
            result['per_face_stages'][face]['selected_stage_file'] = str(target/'candidate'/sku/(face+'.png'))
        source_parents = {path.parent for path in raw_paths.values()}
        result.update(process=str(target), candidate=str(target/'candidate'),
            source=str(next(iter(source_parents))) if len(source_parents) == 1 else str(target/'work/input'/sku),
            portable_review_evidence=restored['evidence'], review_bundle_reused=True,
            pipeline_rerun=False, pipeline_executed=False,
            events=[dict(stage='verified_portable_review_continuation', seconds=0)],
            review_bundle_provenance=dict(descriptor, current_raw_sources={f: str(p) for f, p in raw_paths.items()},
                historical_paths_read=False, byte_copy_only=True, pipeline_stages_executed=[],
                original_fresh_evidence_reused=True, fresh_pixel_checks_executed=False,
                current_raw_hashes_rechecked=True))
        _write(staging/'work/intake.json', _bytes(intake))
        _write(staging/'work/stage-report.json', _bytes(result))
        _write(staging/'result.json', _bytes(result))
        for path, expected in files.items():
            _identity(path, expected)
        for face, path in raw_paths.items():
            _identity(path, descriptor['source_sha256'][face])
        _require(not target.exists(), 'Replay case appeared during copy')
        staging.rename(target)
        return result
    except BaseException:
        shutil.rmtree(staging)
        raise
