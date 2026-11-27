"""Build a genuinely open, five-panel rectangular paper bag with welded handles.

Geometry only: the generated mesh has no support, pins, attachments, or top cap.
The entire open bag starts lying on its broad side, retaining a rectangular
cavity. Paper constitutive parameters belong to the simulation configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
WIDTH, DEPTH, HEIGHT = .28, .20, .30
HANDLE_RADIUS, HANDLE_RISE, HANDLE_WIDTH = .07, .09, .014


def boundary_loops(faces):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[count == 1]
    graph = {}
    for a, b in boundary:
        graph.setdefault(int(a), []).append(int(b))
        graph.setdefault(int(b), []).append(int(a))
    if any(len(row) != 2 for row in graph.values()):
        raise ValueError('Boundary is not a collection of closed simple loops')
    unseen, loops = set(graph), []
    while unseen:
        start = min(unseen)
        loop, previous, current = [], None, start
        while current not in loop:
            loop.append(current)
            unseen.discard(current)
            following = [item for item in graph[current] if item != previous]
            previous, current = current, following[0]
        if current != start:
            raise ValueError('Invalid loop traversal')
        loops.append(loop)
    return loops, int((count > 2).sum()), unique


def nonneighbor_intersections(points, faces):
    """Triangle separating-axis test, including axes for coplanar triangles.

    Faces sharing a topological vertex are omitted. Their regular grid and
    welded root construction is checked separately; this does not certify
    dynamic continuous contact, which belongs to the native IPC solver.
    """
    triangles = points[faces]
    lower, upper = triangles.min(axis=1), triangles.max(axis=1)
    pairs = []
    for first in range(len(faces)):
        candidates = np.flatnonzero(np.all(upper[first] >= lower[first + 1:] - 1e-11, axis=1)
            & np.all(upper[first + 1:] >= lower[first] - 1e-11, axis=1)) + first + 1
        for second in candidates:
            if not np.intersect1d(faces[first], faces[second]).size:
                pairs.append((first, int(second)))
    intersections = []
    for first, second in pairs:
        a, b = triangles[first], triangles[second]
        ea, eb = np.roll(a, -1, axis=0) - a, np.roll(b, -1, axis=0) - b
        na, nb = np.cross(ea[0], ea[1]), np.cross(eb[0], eb[1])
        axes = np.vstack([na, nb, np.cross(ea[:, None], eb[None]).reshape(-1, 3),
            np.cross(na, ea), np.cross(nb, eb)])
        norm = np.linalg.norm(axes, axis=1)
        axes = axes[norm > 1e-14] / norm[norm > 1e-14, None]
        pa, pb = a @ axes.T, b @ axes.T
        separated = (pa.max(axis=0) < pb.min(axis=0) - 1e-10) | (pb.max(axis=0) < pa.min(axis=0) - 1e-10)
        if not separated.any():
            intersections.append([first, second])
    return {'method': 'AABB broadphase then triangle SAT with coplanar axes',
        'tested_nonadjacent_pairs': len(pairs), 'intersection_count': len(intersections),
        'intersections': intersections, 'shared_vertex_pairs_excluded': True}


def build_geometry():
    points, faces, lookup, groups, face_groups, grids = [], [], {}, {}, {}, {}

    def vertex(point):
        key = tuple(np.round(point, 12))
        if key not in lookup:
            lookup[key] = len(points)
            points.append(list(map(float, point)))
        return lookup[key]

    def surface(name, coordinates, reverse=False):
        grid = np.array([[vertex(p) for p in row] for row in coordinates], dtype=np.int32)
        first = len(faces)
        for j in range(grid.shape[0] - 1):
            for i in range(grid.shape[1] - 1):
                a, b, c, d = grid[j, i], grid[j, i + 1], grid[j + 1, i + 1], grid[j + 1, i]
                # Alternate diagonals avoids a single diagonal bias.
                pair = [[a, b, c], [a, c, d]] if (i + j) % 2 == 0 else [[a, b, d], [b, c, d]]
                faces.extend([triangle[::-1] if reverse else triangle for triangle in pair])
        groups[name] = np.unique(grid).astype(np.int32)
        face_groups[name + '_face_indices'] = np.arange(first, len(faces), dtype=np.int32)
        grids[name + '_grid'] = grid
        return grid

    # Root segments have exactly the ribbon width, avoiding coincident roots
    # and avoiding tiny triangles from inserting points into a uniform grid.
    xs = np.array([-.14, -.12, -.10, -.077, -.063, -.04, -.02, 0., .02, .04, .063, .077, .10, .12, .14])
    ys, zs = np.linspace(-DEPTH / 2, DEPTH / 2, 11), np.linspace(0., HEIGHT, 16)
    surface('front', [[(x, -DEPTH / 2, z) for x in xs] for z in zs])
    surface('back', [[(x, DEPTH / 2, z) for x in xs] for z in zs], reverse=True)
    surface('left', [[(-WIDTH / 2, y, z) for y in ys] for z in zs], reverse=True)
    surface('right', [[(WIDTH / 2, y, z) for y in ys] for z in zs])
    surface('bottom', [[(x, y, 0.) for x in xs] for y in ys], reverse=True)
    body_face_indices = np.arange(len(faces), dtype=np.int32)
    body = np.unique(np.asarray(faces)).astype(np.int32)
    theta = np.linspace(0., np.pi, 21)
    handles = {}
    for side, y, reverse in [('front', -DEPTH / 2, True), ('back', DEPTH / 2, False)]:
        path = np.column_stack([-HANDLE_RADIUS * np.cos(theta), np.full(theta.shape, y), HEIGHT + HANDLE_RISE * np.sin(theta)])
        normal = np.column_stack([-HANDLE_RISE * np.cos(theta), np.zeros(theta.shape), HANDLE_RADIUS * np.sin(theta)])
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        coordinates = path[:, None, :] + np.array([-.5, .5])[None, :, None] * HANDLE_WIDTH * normal[:, None, :]
        # The rectangle-grid convention is (row, column); the arch orientation
        # produces the opposite normal to the front/back vertical panels.
        grid = surface('handle_' + side, coordinates, reverse=reverse)
        groups['handle_' + side + '_root_left'] = grid[0]
        groups['handle_' + side + '_root_right'] = grid[-1]
        groups['grasp_' + ('upper' if side == 'back' else 'lower')] = grid[9:12].ravel()
        handles[side] = {'grid': grid.tolist(), 'centerline_length_m': float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
            'width_m': HANDLE_WIDTH, 'rise_above_mouth_m': HANDLE_RISE,
            'left_root_m': path[0].tolist(), 'right_root_m': path[-1].tolist(),
            'root_vertices_welded_to_body': bool(set(np.r_[grid[0], grid[-1]]) <= set(body))}

    upright, faces = np.asarray(points), np.asarray(faces, dtype=np.int32)
    # Rx(+90 degrees): upright y becomes flat z; upright z becomes -flat y.
    rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    translation = np.array([0., 0., DEPTH / 2])
    flat = upright @ rotation.T + translation
    groups['body'] = body
    loops, _, _ = boundary_loops(faces[body_face_indices])
    if len(loops) != 1:
        raise RuntimeError('Body must have exactly one open mouth')
    groups['rim'] = np.asarray(loops[0], dtype=np.int32)
    groups['top_handles'] = np.unique(np.r_[groups['handle_front'], groups['handle_back']]).astype(np.int32)
    return flat, upright, faces, groups, face_groups, grids, body_face_indices, rotation, translation, handles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--areal-density', type=float, default=.12,
        help='Assumed paper kg/m2 solely for reporting area-derived mass; no material is assigned')
    args = parser.parse_args()
    if not np.isfinite(args.areal_density) or args.areal_density <= 0:
        raise ValueError('Areal density must be finite and positive')
    assets = HERE / 'assets'
    assets.mkdir(exist_ok=True)
    flat, upright, faces, groups, face_groups, grids, body_ids, rotation, translation, handles = build_geometry()
    mesh = trimesh.Trimesh(flat, faces, process=False)
    body_mesh = trimesh.Trimesh(flat, faces[body_ids], process=False)
    loops, nonmanifold, edges = boundary_loops(faces)
    edge_lengths = np.linalg.norm(flat[edges[:, 1]] - flat[edges[:, 0]], axis=1)
    intersection = nonneighbor_intersections(flat, faces)
    checks = {
        'one_connected_shell': len(mesh.split(only_watertight=False)) == 1,
        'three_open_boundary_loops_mouth_and_handles': len(loops) == 3,
        'no_nonmanifold_edges': nonmanifold == 0,
        'consistent_outward_winding': bool(mesh.is_winding_consistent and body_mesh.is_winding_consistent),
        'one_body_mouth_boundary': len(boundary_loops(faces[body_ids])[0]) == 1,
        'positive_triangle_areas': bool(np.all(mesh.area_faces > 1e-12)),
        'exact_rectangular_body_area': bool(np.isclose(body_mesh.area, WIDTH * DEPTH + 2 * HEIGHT * (WIDTH + DEPTH), atol=1e-12)),
        'roots_share_actual_body_vertices': all(h['root_vertices_welded_to_body'] for h in handles.values()),
        'no_nonadjacent_triangle_intersections': intersection['intersection_count'] == 0,
        'proper_rigid_initial_rotation': bool(np.allclose(rotation @ rotation.T, np.eye(3)) and np.isclose(np.linalg.det(rotation), 1.)),
        'side_lies_at_z_zero': bool(np.isclose(flat[:, 2].min(), 0.)),
        'upper_handle_is_above_lower_handle': bool(flat[groups['grasp_upper'], 2].mean() > flat[groups['grasp_lower'], 2].mean()),
        'vertex_budget_below_2000': len(flat) < 2000,
    }
    if not all(checks.values()):
        raise RuntimeError('Geometry checks failed: ' + json.dumps(checks))
    obj = assets / 'paper_bag.obj'
    obj.write_text('# Open rectangular kraft-paper bag; metre units; no top cap or rigid support.\n'
        + ''.join('v %.12g %.12g %.12g\n' % tuple(p) for p in flat)
        + ''.join('f %d %d %d\n' % tuple(f + 1) for f in faces))
    np.savez_compressed(assets / 'paper_bag.npz', points=flat, points_flat=flat, points_upright=upright,
        faces=faces, body_face_indices=body_ids, rotation_upright_to_flat=rotation,
        translation_upright_to_flat=translation, **groups, **face_groups, **grids)
    metadata = {
        'schema_version': 1, 'description': 'Open rectangular five-panel paper bag with two welded arch ribbon handles; lying on its broad side',
        'units': 'm', 'dimensions_upright_m': {'width_x': WIDTH, 'depth_y': DEPTH, 'height_z': HEIGHT},
        'cavity_volume_m3': WIDTH * DEPTH * HEIGHT,
        'shape_note': 'Lying means the rectangular open bag is on its broad side; the cavity is not collapsed to zero depth.',
        'top_cap': False, 'hidden_supports': False, 'fixed_soft_vertices': 0,
        'flat_bounds_m': mesh.bounds.tolist(), 'upright_bounds_m': [upright.min(axis=0).tolist(), upright.max(axis=0).tolist()],
        'suggested_world_position_m': [-.18, .22, .742],
        'initial_rotation': {'axis': 'x', 'degrees': 90., 'matrix': rotation.tolist(), 'translation_after_rotation_m': translation.tolist(),
            'meaning': 'upright y=-0.10 broad side lies at local z=0; mouth faces local -Y'},
        'vertices': len(flat), 'triangles': len(faces), 'body_triangles': len(body_ids),
        'surface_area_m2': float(mesh.area), 'body_area_m2': float(body_mesh.area), 'handles_area_m2': float(mesh.area - body_mesh.area),
        'mass_from_assumed_areal_density': {'areal_density_kg_m2': args.areal_density,
            'mass_kg': float(mesh.area * args.areal_density), 'scope': 'Area-derived estimate only; actual native mass depends on configured density and full thickness'},
        'edge_length_range_m': [float(edge_lengths.min()), float(edge_lengths.max())],
        'handles': handles, 'groups': {name: ids.tolist() for name, ids in groups.items()},
        'face_groups': {name: ids.tolist() for name, ids in face_groups.items()},
        'topology': {'connected_components': 1, 'nonmanifold_edges': nonmanifold, 'boundary_loop_lengths': list(map(len, loops)),
            'body_mouth_rim': groups['rim'].tolist(), 'euler_number': int(mesh.euler_number), 'winding_consistent': bool(mesh.is_winding_consistent)},
        'static_geometry_audit': {'checks': checks, 'all_checks_passed': all(checks.values()), 'intersections': intersection,
            'scope': 'Static CPU geometry and topology only; not dynamic physical task success'},
        'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [obj, assets / 'paper_bag.npz']},
    }
    (assets / 'paper_bag_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({key: metadata[key] for key in ['vertices', 'triangles', 'surface_area_m2', 'mass_from_assumed_areal_density', 'flat_bounds_m', 'topology', 'static_geometry_audit']}, indent=2))


if __name__ == '__main__':
    main()
