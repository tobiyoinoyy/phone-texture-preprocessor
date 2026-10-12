"""Guard texture installation against the actual Blender scene used to calibrate.

Run capture() in Blender (e.g. via MCP) before deriving seam curves. Store the
returned JSON beside calibration data. Always call bind_images() with that same
record when installing textures; a new export or file name is not scene identity.
"""
import hashlib
import json


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def require_match(expected, current):
    if expected['sha256'] != fingerprint(expected['payload']):
        raise ValueError('Calibration scene record is corrupt')
    if current['sha256'] != fingerprint(current['payload']):
        raise ValueError('Current scene record is corrupt')
    if expected['sha256'] != current['sha256']:
        raise ValueError('Current Blender geometry, UV or mapping differs; recalibrate before binding textures')


def capture(obj):
    mesh = obj.data
    if obj.mode != 'OBJECT' or not mesh.uv_layers.active:
        raise ValueError('Object mode and an active UV map are required')
    materials = []
    for mat in mesh.materials:
        if not mat or not mat.use_nodes:
            raise ValueError('All calibrated material slots must use nodes')
        nodes = []
        for node in mat.node_tree.nodes:
            item = {'name': node.name, 'type': node.type}
            # Image identity can change with SKU; coordinates and sampling cannot.
            if node.type == 'TEX_IMAGE':
                item.update(projection=node.projection, extension=node.extension,
                            interpolation=node.interpolation)
            if node.type in ('MAPPING', 'UVMAP', 'TEX_COORD', 'VECT_MATH', 'VECTOR_MATH'):
                for prop in ('uv_map', 'vector_type', 'operation', 'from_instancer'):
                    if hasattr(node, prop):
                        item[prop] = getattr(node, prop)
                item['inputs'] = {}
                for socket in node.inputs:
                    if hasattr(socket, 'default_value'):
                        value = socket.default_value
                        item['inputs'][socket.identifier] = value if isinstance(value, (str, bool, int, float, type(None))) else list(value)
            nodes.append(item)
        materials.append({'name': mat.name, 'nodes': nodes, 'links': sorted(
            [l.from_node.name, l.from_socket.identifier, l.to_node.name, l.to_socket.identifier]
            for l in mat.node_tree.links)})
    payload = {'vertices': [list(v.co) for v in mesh.vertices],
               'polygons': [[list(p.vertices), p.material_index] for p in mesh.polygons],
               'uv': {uv.name: [list(v.uv) for v in uv.data] for uv in mesh.uv_layers},
               'active_uv': mesh.uv_layers.active.name,
               'render_uv': [uv.name for uv in mesh.uv_layers if uv.active_render],
               'matrix_world': [list(row) for row in obj.matrix_world],
               'materials': materials}
    return {'sha256': fingerprint(payload), 'payload': payload}


def bind_images(obj, expected, images_by_material):
    """images_by_material maps material -> image-node name -> loaded bpy Image.

    Load images separately first. No material mutation occurs if any check fails.
    This guard does not replace visual inspection of the evaluated final model.
    """
    require_match(expected, capture(obj))
    assignments = []
    for material, nodes in images_by_material.items():
        mat = next(m for m in obj.data.materials if m.name == material)
        for name, image in nodes.items():
            node = mat.node_tree.nodes[name]
            if node.type != 'TEX_IMAGE' or image is None:
                raise ValueError('Expected a loaded image and image texture node')
            assignments.append((node, image))
    for node, image in assignments:
        node.image = image
    return len(assignments)
