"""Read-only geometric diagnostics for the paper-bag / soft-teddy scenario.

All lengths are metres. No Genesis imports, simulator writes, attachments, or
success assumptions are made here. Geometry can reject an invalid simulation;
geometry alone cannot prove contact forces, grasp stability, or task success.

For pose/containment, ``bag_reference`` is the *upright local* reference mesh,
not its initial world placement. Its x/y/z axes define ``inner_bounds`` and its
positive z direction defines upright. Current vertices retain the same order.
Use bag-body indices only for the fit (exclude handles). Rest/current meshes
for strain diagnostics may instead use any common rigid coordinate system.

Run this module with --self-test for small analytic CPU checks.
"""
from __future__ import annotations

import argparse
import json
import unittest
from typing import Any

import numpy as np


def _points(value: Any, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or not len(a):
        raise ValueError(f"{name} must be a nonempty (N, 3) array")
    return a


def _pair(reference: Any, current: Any) -> tuple[np.ndarray, np.ndarray]:
    a, b = _points(reference, "reference"), _points(current, "current")
    if a.shape != b.shape:
        raise ValueError("Reference/current meshes must have matching shape and vertex order")
    return a, b


def _indices(value: Any, count: int, width: int | None = None) -> np.ndarray:
    a = np.asarray(value)
    shape_ok = (a.ndim == 1) if width is None else (a.ndim == 2 and a.shape[1] == width)
    if not shape_ok or not a.size or not np.issubdtype(a.dtype, np.integer):
        expected = "nonempty integer vector" if width is None else f"nonempty (N, {width}) integer array"
        raise ValueError(f"Indices must be a {expected}")
    if a.min() < 0 or a.max() >= count:
        raise ValueError("Vertex index out of bounds")
    return a.astype(np.int64, copy=False)


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    a = np.asarray(values, dtype=float).reshape(-1)
    valid = a[np.isfinite(a)]
    return {
        "finite_count": int(len(valid)), "nonfinite_count": int(len(a) - len(valid)),
        "min": float(valid.min()) if len(valid) else None,
        "median": float(np.median(valid)) if len(valid) else None,
        "max": float(valid.max()) if len(valid) else None,
    }


def finite_geometry(vertices: Any) -> dict[str, Any]:
    a = _points(vertices, "vertices")
    valid = np.isfinite(a).all(axis=1)
    return {"all_finite": bool(valid.all()), "vertex_count": int(len(a)),
            "nonfinite_vertex_count": int((~valid).sum())}


def triangle_metrics(rest: Any, current: Any, triangles: Any,
                     area_epsilon: float = 1e-14, edge_epsilon: float = 1e-12) -> dict[str, Any]:
    """Measure unsigned area and unique-edge length ratios without a rigid fit.

    A triangle rotating in 3D is not an inversion. Unsigned areas cannot detect
    shell self-intersection or distinguish such rotation from local folding.
    """
    a, b = _pair(rest, current)
    f = _indices(triangles, len(a), 3)
    if min(area_epsilon, edge_epsilon) <= 0:
        raise ValueError("Geometry tolerances must be positive")
    rest_tri, cur_tri = a[f], b[f]
    rest_area = np.linalg.norm(np.cross(rest_tri[:, 1] - rest_tri[:, 0],
                                        rest_tri[:, 2] - rest_tri[:, 0]), axis=1) / 2
    cur_area = np.linalg.norm(np.cross(cur_tri[:, 1] - cur_tri[:, 0],
                                       cur_tri[:, 2] - cur_tri[:, 0]), axis=1) / 2
    edges = np.unique(np.sort(np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), axis=1), axis=0)
    rest_len = np.linalg.norm(a[edges[:, 1]] - a[edges[:, 0]], axis=1)
    cur_len = np.linalg.norm(b[edges[:, 1]] - b[edges[:, 0]], axis=1)
    area_ratio = np.divide(cur_area, rest_area, out=np.full_like(cur_area, np.nan),
                           where=rest_area > area_epsilon)
    edge_ratio = np.divide(cur_len, rest_len, out=np.full_like(cur_len, np.nan),
                           where=rest_len > edge_epsilon)
    return {
        "all_finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
        "triangle_count": int(len(f)), "unique_edge_count": int(len(edges)),
        "degenerate_rest_triangle_count": int((rest_area <= area_epsilon).sum()),
        "degenerate_rest_edge_count": int((rest_len <= edge_epsilon).sum()),
        "collapsed_current_triangle_count": int((cur_area <= area_epsilon).sum()),
        "rest_surface_area_m2": float(rest_area.sum()) if np.isfinite(rest_area).all() else None,
        "current_surface_area_m2": float(cur_area.sum()) if np.isfinite(cur_area).all() else None,
        "area_ratio": _summary(area_ratio), "edge_length_ratio": _summary(edge_ratio),
        "limitations": "Unsigned shell areas/edge lengths do not detect self-intersection, tearing, or collision correctness.",
    }


def signed_tetra_volumes(vertices: Any, tetrahedra: Any) -> np.ndarray:
    a = _points(vertices, "vertices")
    t = _indices(tetrahedra, len(a), 4)
    p = a[t]
    return np.einsum("ij,ij->i", p[:, 1] - p[:, 0],
                     np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6


def tetra_metrics(rest: Any, current: Any, tetrahedra: Any,
                  volume_epsilon: float = 1e-15,
                  collapse_ratio: float = 0.05) -> dict[str, Any]:
    """Signed current/rest volume ratios detect orientation reversal.

    Rest tetrahedra may have either winding. A ratio below zero indicates an
    inversion relative to the corresponding rest tetrahedron.
    """
    a, b = _pair(rest, current)
    t = _indices(tetrahedra, len(a), 4)
    if volume_epsilon <= 0 or collapse_ratio < 0:
        raise ValueError("Invalid volume/collapse tolerance")
    vr, vc = signed_tetra_volumes(a, t), signed_tetra_volumes(b, t)
    good = np.abs(vr) > volume_epsilon
    ratio = np.divide(vc, vr, out=np.full_like(vc, np.nan), where=good)
    inverted = np.isfinite(ratio) & (ratio < 0)
    collapsed = np.isfinite(ratio) & (np.abs(ratio) <= collapse_ratio)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    total_rest, total_current = float(np.abs(vr).sum()), float(np.abs(vc).sum())
    return {
        "all_finite": finite, "tetrahedron_count": int(len(t)),
        "degenerate_rest_tetrahedron_count": int((~good).sum()),
        "inverted_tetrahedron_count": int(inverted.sum()),
        "collapsed_tetrahedron_count": int(collapsed.sum()),
        "collapse_ratio_threshold": float(collapse_ratio),
        "signed_volume_ratio": _summary(ratio),
        "rest_absolute_volume_m3": total_rest if np.isfinite(total_rest) else None,
        "current_absolute_volume_m3": total_current if np.isfinite(total_current) else None,
        "total_absolute_volume_ratio": total_current / total_rest
            if finite and total_rest > volume_epsilon else None,
        "limitations": "A positive volume ratio alone does not rule out excessive shear, self-intersection, or incorrect forces.",
    }


def fit_rigid_transform(reference: Any, current: Any, indices: Any = None) -> dict[str, Any]:
    """Return proper least-squares transform: current = reference @ R.T + t.

    At least three non-collinear selected points are required. No scale or
    reflection is fitted, so deformation appears in the residual.
    """
    a, b = _pair(reference, current)
    if indices is not None:
        idx = _indices(indices, len(a))
        a, b = a[idx], b[idx]
    if len(a) < 3 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Rigid fit requires at least three finite point pairs")
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    aa, bb = a - ca, b - cb
    if np.linalg.matrix_rank(aa) < 2 or np.linalg.matrix_rank(bb) < 2:
        raise ValueError("Rigid fit is undefined for collinear/coincident points")
    u, singular, vt = np.linalg.svd(aa.T @ bb)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = cb - rotation @ ca
    residual = np.linalg.norm(a @ rotation.T + translation - b, axis=1)
    return {"rotation": rotation.tolist(), "translation_m": translation.tolist(),
            "rms_residual_m": float(np.sqrt(np.mean(residual**2))),
            "max_residual_m": float(residual.max()), "point_count": int(len(a)),
            "singular_values": singular.tolist()}


def bag_pose(reference: Any, current: Any, body_indices: Any,
             reference_up: Any = (0, 0, 1), world_up: Any = (0, 0, 1)) -> dict[str, Any]:
    """Estimate bag orientation from its deforming body, excluding handles."""
    fit = fit_rigid_transform(reference, current, body_indices)
    local_up, target_up = np.asarray(reference_up, float), np.asarray(world_up, float)
    if local_up.shape != (3,) or target_up.shape != (3,) or not np.isfinite([local_up, target_up]).all():
        raise ValueError("Up axes must be finite three-vectors")
    if min(np.linalg.norm(local_up), np.linalg.norm(target_up)) <= 0:
        raise ValueError("Up axes must be nonzero")
    current_up = np.asarray(fit["rotation"]) @ (local_up / np.linalg.norm(local_up))
    cosine = float(np.clip(current_up @ (target_up / np.linalg.norm(target_up)), -1, 1))
    return {**fit, "estimated_up_axis": current_up.tolist(), "up_dot_world_up": cosine,
            "tilt_from_upright_degrees": float(np.degrees(np.arccos(cosine))),
            "method": "Proper least-squares rigid fit of bag body vertices; deformation is reported as residual."}


def approximate_containment(teddy_current: Any, bag_reference: Any, bag_current: Any,
                            body_indices: Any, inner_bounds: Any, *,
                            clearance: float = 0.0,
                            max_fit_residual: float = 0.02) -> dict[str, Any]:
    """Test all teddy vertices against the estimated open-bag interior box.

    ``inner_bounds`` is [[xmin, ymin, zmin], [xmax, ymax, zmax]] in upright
    reference coordinates; zmax is the mouth plane. Fully inserted requires
    teddy vertices below that plane. The bag's top remains physically open.
    This is explicitly an approximation, not a mesh/collision predicate.
    """
    teddy = _points(teddy_current, "teddy_current")
    bounds = np.asarray(inner_bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError("inner_bounds must be finite ordered (2, 3) bounds")
    if not np.isfinite([clearance, max_fit_residual]).all() or min(clearance, max_fit_residual) < 0:
        raise ValueError("Clearance/residual tolerance must be finite and nonnegative")
    low, high = bounds[0] + clearance, bounds[1] - clearance
    if np.any(high <= low):
        raise ValueError("Clearance consumes the interior box")
    fit = fit_rigid_transform(bag_reference, bag_current, body_indices)
    local = (teddy - np.asarray(fit["translation_m"])) @ np.asarray(fit["rotation"])
    valid = np.isfinite(local).all(axis=1)
    signed_clearance = np.minimum(local - low, high - local)
    inside = valid & np.all(signed_clearance >= 0, axis=1)
    reliable = bool(fit["rms_residual_m"] <= max_fit_residual)
    all_inside = bool(inside.all())
    return {
        "method": "Approximate interior box transformed by a rigid fit to current bag-body vertices",
        "all_teddy_vertices_finite": bool(valid.all()),
        "all_vertices_inside_fitted_box": all_inside,
        "fraction_vertices_inside_fitted_box": float(inside.mean()),
        "outside_vertex_count": int((~inside).sum()),
        "body_fit_within_residual_tolerance": reliable,
        "approximate_containment_pass": bool(all_inside and reliable),
        "clearance_m": float(clearance), "max_body_fit_rms_residual_m": float(max_fit_residual),
        "minimum_signed_box_clearance_m": float(np.min(signed_clearance)) if valid.all() else None,
        "teddy_bounds_in_bag_reference_m": [local.min(axis=0).tolist(), local.max(axis=0).tolist()]
            if valid.all() else None,
        "inner_bounds_reference_m": bounds.tolist(), "bag_body_fit": fit,
        "limitations": [
            "The fitted box does not follow local wall buckling or mouth collapse.",
            "Vertex sampling does not prove containment of curved/interpolated surfaces or exclude wall intersection.",
            "A static frame cannot establish released-grasp stability, contact forces, or successful insertion history.",
        ],
    }


def _edge_topology(faces: np.ndarray) -> dict[str, Any]:
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    canonical = np.sort(directed, axis=1)
    unique, inverse, counts = np.unique(canonical, axis=0, return_inverse=True, return_counts=True)
    balance = np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
    boundary = unique[counts == 1]
    graph: dict[int, set[int]] = {}
    for first, second in unique:
        graph.setdefault(int(first), set()).add(int(second))
        graph.setdefault(int(second), set()).add(int(first))
    unseen, components = set(graph), 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            fresh = graph[stack.pop()] & unseen
            unseen.difference_update(fresh)
            stack.extend(fresh)
    return {"boundary_edges": boundary, "directed_edges": directed,
            "boundary_edge_count": int(len(boundary)),
            "nonmanifold_edge_count": int((counts > 2).sum()),
            "inconsistent_internal_edge_count": int(((counts == 2) & (balance != 0)).sum()),
            "connected_component_count": components,
            "watertight_consistent": bool(np.all(counts == 2) and np.all(balance == 0) and components == 1)}


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _simple_polygon_2d(polygon: np.ndarray, tolerance: float) -> bool:
    """Reject crossing/touching non-neighbour edges, including collinear overlap."""
    count = len(polygon)
    edges = np.roll(polygon, -1, axis=0) - polygon
    if np.any(np.linalg.norm(edges, axis=1) <= tolerance):
        return False

    def on_segment(a, b, p):
        return bool(abs(_cross2(b - a, p - a)) <= tolerance * np.linalg.norm(b - a)
                    and np.all(p >= np.minimum(a, b) - tolerance)
                    and np.all(p <= np.maximum(a, b) + tolerance))

    for i in range(count):
        a, b = polygon[i], polygon[(i + 1) % count]
        for j in range(i + 1, count):
            if j == i + 1 or (i == 0 and j == count - 1):
                continue
            c, d = polygon[j], polygon[(j + 1) % count]
            if np.any(np.maximum(a, b) + tolerance < np.minimum(c, d)) or np.any(np.maximum(c, d) + tolerance < np.minimum(a, b)):
                continue
            o1, o2 = _cross2(b - a, c - a), _cross2(b - a, d - a)
            o3, o4 = _cross2(d - c, a - c), _cross2(d - c, b - c)
            if (o1 * o2 < 0 and o3 * o4 < 0) or any((
                    on_segment(a, b, c), on_segment(a, b, d),
                    on_segment(c, d, a), on_segment(c, d, b))):
                return False
    return True


def _winding_and_distance(points: np.ndarray, triangles: np.ndarray,
                          point_chunk: int, triangle_chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """Generalized solid-angle winding and exact point/triangle distance."""
    winding = np.zeros(len(points), dtype=float)
    distance = np.full(len(points), np.inf)
    for start in range(0, len(points), point_chunk):
        sample = points[start:start + point_chunk]
        angle = np.zeros(len(sample))
        best = np.full(len(sample), np.inf)
        for first in range(0, len(triangles), triangle_chunk):
            tri = triangles[first:first + triangle_chunk]
            a = tri[None, :, 0] - sample[:, None]
            b = tri[None, :, 1] - sample[:, None]
            c = tri[None, :, 2] - sample[:, None]
            la, lb, lc = [np.linalg.norm(v, axis=2) for v in (a, b, c)]
            numerator = np.einsum("qfi,qfi->qf", a, np.cross(b, c))
            denominator = la * lb * lc + np.einsum("qfi,qfi->qf", a, b) * lc \
                + np.einsum("qfi,qfi->qf", b, c) * la + np.einsum("qfi,qfi->qf", c, a) * lb
            angle += (2 * np.arctan2(numerator, denominator)).sum(axis=1)
            ab, ac = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
            normal = np.cross(ab, ac)
            aa, bb, cc = np.einsum("fi,fi->f", ab, ab), np.einsum("fi,fi->f", ab, ac), np.einsum("fi,fi->f", ac, ac)
            ap_ab = np.einsum("qfi,fi->qf", -a, ab)
            ap_ac = np.einsum("qfi,fi->qf", -a, ac)
            determinant = aa * cc - bb**2
            bary1 = (cc * ap_ab - bb * ap_ac) / determinant
            bary2 = (aa * ap_ac - bb * ap_ab) / determinant
            face_inside = (bary1 >= 0) & (bary2 >= 0) & (bary1 + bary2 <= 1)
            plane_distance = np.abs(np.einsum("qfi,fi->qf", a, normal)) / np.linalg.norm(normal, axis=1)
            candidate = np.where(face_inside, plane_distance, np.inf)
            for edge_a, edge_b in ((a, b), (b, c), (c, a)):
                edge = edge_b - edge_a
                parameter = np.clip(-np.einsum("qfi,qfi->qf", edge_a, edge)
                    / np.einsum("qfi,qfi->qf", edge, edge), 0, 1)
                edge_distance = np.linalg.norm(edge_a + parameter[:, :, None] * edge, axis=2)
                candidate = np.minimum(candidate, edge_distance)
            best = np.minimum(best, candidate.min(axis=1))
        winding[start:start + len(sample)] = angle / (4 * np.pi)
        distance[start:start + len(sample)] = best
    return winding, distance


def actual_bag_containment(teddy_points: Any, bag_points_source_order: Any,
                           body_faces_source_order: Any, ordered_rim: Any, *,
                           max_rim_planarity: float = 0.005,
                           boundary_tolerance: float = 1e-6,
                           winding_tolerance: float = 1e-4,
                           point_chunk: int = 64, triangle_chunk: int = 512) -> dict[str, Any]:
    """Test teddy vertices inside the current bag surface, virtually closing its mouth.

    Bag vertices and body-face indices must use the same *source vertex order*.
    ``ordered_rim`` is a non-repeated cyclic ordering of the body's sole boundary.
    Handles must be excluded from body faces. The cap exists only in this
    diagnostic, never in the simulator. Invalid/non-simple/strongly warped rims
    produce a rejected report instead of a misleading containment result.

    This establishes sampled-vertex geometric enclosure only. Native collision
    validity, no inversion, upright placement, release and a stable dwell are
    independent requirements for any physical task-success claim.
    """
    bear, bag = _points(teddy_points, "teddy_points"), _points(bag_points_source_order, "bag_points_source_order")
    faces = _indices(body_faces_source_order, len(bag), 3)
    rim_ids = _indices(ordered_rim, len(bag))
    if len(rim_ids) < 3 or len(np.unique(rim_ids)) != len(rim_ids):
        raise ValueError("ordered_rim must contain at least three unique vertices, without repeated closure")
    tolerances = [max_rim_planarity, boundary_tolerance, winding_tolerance]
    if not np.isfinite(tolerances).all() or min(tolerances) < 0 or winding_tolerance >= .5:
        raise ValueError("Invalid rim, boundary or winding tolerance")
    if not isinstance(point_chunk, int) or not isinstance(triangle_chunk, int) or min(point_chunk, triangle_chunk) < 1:
        raise ValueError("Chunk sizes must be positive integers")
    report: dict[str, Any] = {
        "method": "Solid-angle winding of actual deformed bag body plus a diagnostic centroid-fan mouth cap",
        "surface_audit_pass": False, "actual_containment_pass": False,
        "all_teddy_vertices_inside": False, "inside_vertex_fraction": None,
        "outside_or_boundary_vertex_count": None, "minimum_surface_distance_m": None,
        "minimum_signed_surface_clearance_m": None, "absolute_winding_range": None,
        "surface_watertight": False, "max_rim_planarity_m": None,
        "rim_planarity_tolerance_m": float(max_rim_planarity),
        "boundary_tolerance_m": float(boundary_tolerance), "rejection_reasons": [],
        "cap_is_simulated": False,
        "limitations": [
            "The virtual mouth cap is an analysis surface; it adds no collision, support, or force to the bag.",
            "Small rim warping uses the actual piecewise fan; warping beyond tolerance, self-crossing rims, and non-star-shaped fans are rejected.",
            "All sampled vertices inside does not prove every teddy edge/face stays inside a folded or concave bag.",
            "Topological watertightness is not a geometric self-intersection test; native collision validity must be independently checked.",
            "This single-frame enclosure test does not prove grasp correctness, upright placement, release, or stability.",
        ],
    }
    if not (np.isfinite(bear).all() and np.isfinite(bag).all()):
        report["rejection_reasons"].append("Nonfinite teddy or bag vertices")
        return report
    report["teddy_bounds_world_m"] = [bear.min(axis=0).tolist(), bear.max(axis=0).tolist()]
    body_ids = np.unique(faces)
    report["bag_body_bounds_world_m"] = [bag[body_ids].min(axis=0).tolist(), bag[body_ids].max(axis=0).tolist()]
    topology = _edge_topology(faces)
    report["body_topology"] = {k: v for k, v in topology.items() if not isinstance(v, np.ndarray)}
    rim_edges = np.column_stack((rim_ids, np.roll(rim_ids, -1)))
    expected_boundary = {tuple(edge) for edge in np.sort(rim_edges, axis=1)}
    actual_boundary = {tuple(edge) for edge in topology["boundary_edges"]}
    if actual_boundary != expected_boundary:
        report["rejection_reasons"].append("Ordered rim does not equal the body's sole edge boundary")
    if topology["nonmanifold_edge_count"] or topology["inconsistent_internal_edge_count"] or topology["connected_component_count"] != 1:
        report["rejection_reasons"].append("Bag body is nonmanifold, inconsistently oriented, or disconnected")
    rim = bag[rim_ids]
    center = rim.mean(axis=0)
    _, singular, vt = np.linalg.svd(rim - center, full_matrices=False)
    report["max_rim_planarity_m"] = float(np.max(np.abs((rim - center) @ vt[2])))
    if singular[1] <= boundary_tolerance:
        report["rejection_reasons"].append("Rim has collapsed to a line or point")
    if report["max_rim_planarity_m"] > max_rim_planarity:
        report["rejection_reasons"].append("Rim warping exceeds the permitted cap planarity tolerance")
    projected = (rim - center) @ vt[:2].T
    simple = _simple_polygon_2d(projected, max(boundary_tolerance, 1e-12))
    report["projected_rim_is_simple"] = bool(simple)
    if not simple:
        report["rejection_reasons"].append("Projected rim is self-crossing, self-touching, or has degenerate edges")
    fan_areas = np.array([_cross2(a, b) / 2 for a, b in zip(projected, np.roll(projected, -1, axis=0))])
    fan_tolerance = max(boundary_tolerance**2, 1e-14)
    fan_valid = bool(np.all(fan_areas > fan_tolerance) or np.all(fan_areas < -fan_tolerance))
    report["centroid_fan_valid"] = fan_valid
    report["projected_mouth_area_m2"] = float(abs(fan_areas.sum()))
    if not fan_valid:
        report["rejection_reasons"].append("Rim centroid does not define a nonoverlapping interior fan")
    if report["rejection_reasons"]:
        return report
    body_directions = {tuple(edge) for edge in topology["directed_edges"]}
    same_direction = [tuple(edge) in body_directions for edge in rim_edges]
    if not (all(same_direction) or not any(same_direction)):
        report["rejection_reasons"].append("Rim direction is inconsistent with body-face winding")
        return report
    cap_edges = rim_edges[:, ::-1] if all(same_direction) else rim_edges
    cap = np.column_stack((cap_edges, np.full(len(rim_ids), len(bag), dtype=np.int64)))
    closed_points = np.concatenate((bag, center[None]), axis=0)
    closed_faces = np.concatenate((faces, cap), axis=0)
    closed_topology = _edge_topology(closed_faces)
    report["closed_surface_topology"] = {k: v for k, v in closed_topology.items() if not isinstance(v, np.ndarray)}
    report["surface_watertight"] = closed_topology["watertight_consistent"]
    triangles = closed_points[closed_faces]
    areas2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if not report["surface_watertight"] or np.any(areas2 <= 1e-14):
        report["rejection_reasons"].append("Capped surface is not consistently watertight or contains collapsed triangles")
        return report
    report["surface_audit_pass"] = True
    winding, distances = _winding_and_distance(bear, triangles, point_chunk, triangle_chunk)
    inside = (np.abs(np.abs(winding) - 1) <= winding_tolerance) & (distances > boundary_tolerance)
    report.update({
        "all_teddy_vertices_inside": bool(inside.all()),
        "actual_containment_pass": bool(inside.all()),
        "inside_vertex_fraction": float(inside.mean()),
        "outside_or_boundary_vertex_count": int((~inside).sum()),
        "boundary_vertex_count": int((distances <= boundary_tolerance).sum()),
        "minimum_surface_distance_m": float(distances.min()),
        "minimum_signed_surface_clearance_m": float(np.where(inside, distances, -distances).min()),
        "absolute_winding_range": [float(np.abs(winding).min()), float(np.abs(winding).max())],
        "winding_tolerance": float(winding_tolerance),
    })
    return report


def table_contact_metrics(vertices: Any, table_height: float = 0.74, *,
                          tolerance: float = 0.003, contact_radius: float = 0.0) -> dict[str, Any]:
    """Test sampled vertices against the table's infinite horizontal plane.

    This requires callers to include only vertices over the actual tabletop.
    ``contact_radius`` may account for a shell's one-sided thickness.
    """
    a = _points(vertices, "vertices")
    if not np.isfinite([table_height, tolerance, contact_radius]).all() or min(tolerance, contact_radius) < 0:
        raise ValueError("Invalid plane, tolerance, or contact radius")
    finite = bool(np.isfinite(a).all())
    gap = a[:, 2] - table_height - contact_radius
    penetration = np.maximum(-gap, 0)
    return {"all_finite": finite, "table_height_m": float(table_height),
            "tolerance_m": float(tolerance), "contact_radius_m": float(contact_radius),
            "minimum_signed_gap_m": float(gap.min()) if finite else None,
            "maximum_penetration_m": float(penetration.max()) if finite else None,
            "vertices_beyond_tolerance_count": int(np.count_nonzero(gap < -tolerance)),
            "within_table_penetration_tolerance": bool(finite and np.all(gap >= -tolerance)),
            "scope": "Sampled vertices over a horizontal tabletop plane; not a general triangle collision audit."}


def audit_scene(*, bag_reference: Any, bag_rest: Any, bag_current: Any,
                bag_triangles: Any, bag_body_indices: Any, inner_bounds: Any,
                teddy_rest: Any, teddy_current: Any, teddy_tetrahedra: Any,
                table_height: float = 0.74, table_tolerance: float = 0.003,
                max_upright_tilt_degrees: float = 20.0,
                min_paper_edge_ratio: float = 0.8, max_paper_edge_ratio: float = 1.2,
                min_tetra_volume_ratio: float = 0.1, max_tetra_volume_ratio: float = 3.0,
                max_bag_fit_residual: float = 0.02) -> dict[str, Any]:
    """Combine conservative diagnostics, with caller-adjustable thresholds.

    Thresholds are engineering diagnostics, not experimentally calibrated
    material limits. ``geometric_checks_pass`` must not be renamed success.
    This checks a single frame and does not test stability or action history.
    """
    paper = triangle_metrics(bag_rest, bag_current, bag_triangles)
    teddy = tetra_metrics(teddy_rest, teddy_current, teddy_tetrahedra)
    bag_table = table_contact_metrics(bag_current, table_height, tolerance=table_tolerance)
    teddy_table = table_contact_metrics(teddy_current, table_height, tolerance=table_tolerance)
    try:
        pose = bag_pose(bag_reference, bag_current, bag_body_indices)
        containment = approximate_containment(teddy_current, bag_reference, bag_current,
            bag_body_indices, inner_bounds, max_fit_residual=max_bag_fit_residual)
    except (ValueError, np.linalg.LinAlgError) as exc:
        pose = {"error": str(exc)}
        containment = {"approximate_containment_pass": False, "error": str(exc)}
    edges, volumes = paper["edge_length_ratio"], teddy["signed_volume_ratio"]
    checks = {
        "finite_geometry": bool(paper["all_finite"] and teddy["all_finite"]),
        "valid_rest_meshes": bool(paper["degenerate_rest_triangle_count"] == 0
            and paper["degenerate_rest_edge_count"] == 0 and teddy["degenerate_rest_tetrahedron_count"] == 0),
        "paper_edge_ratios_within_bounds": bool(edges["nonfinite_count"] == 0
            and edges["min"] is not None and min_paper_edge_ratio <= edges["min"]
            and edges["max"] <= max_paper_edge_ratio),
        "no_collapsed_paper_triangles": bool(paper["collapsed_current_triangle_count"] == 0),
        "no_inverted_or_collapsed_tetrahedra": bool(teddy["inverted_tetrahedron_count"] == 0
            and teddy["collapsed_tetrahedron_count"] == 0),
        "tetra_volume_ratios_within_bounds": bool(volumes["nonfinite_count"] == 0
            and volumes["min"] is not None and min_tetra_volume_ratio <= volumes["min"]
            and volumes["max"] <= max_tetra_volume_ratio),
        "bag_approximately_upright": bool("error" not in pose
            and pose["tilt_from_upright_degrees"] <= max_upright_tilt_degrees
            and pose["rms_residual_m"] <= max_bag_fit_residual),
        "teddy_approximately_inside": containment["approximate_containment_pass"],
        "bag_table_penetration_within_tolerance": bag_table["within_table_penetration_tolerance"],
        "teddy_table_penetration_within_tolerance": teddy_table["within_table_penetration_tolerance"],
    }
    return {"geometric_checks_pass": all(checks.values()), "checks": checks,
        "paper_deformation": paper, "teddy_deformation": teddy, "bag_pose": pose,
        "approximate_containment": containment, "bag_table": bag_table, "teddy_table": teddy_table,
        "thresholds": {"max_upright_tilt_degrees": max_upright_tilt_degrees,
            "paper_edge_ratio_bounds": [min_paper_edge_ratio, max_paper_edge_ratio],
            "tetra_volume_ratio_bounds": [min_tetra_volume_ratio, max_tetra_volume_ratio],
            "max_bag_fit_rms_residual_m": max_bag_fit_residual},
        "scope": "Read-only single-frame geometric checks; not proof of physics fidelity, grasp validity, or task success."}


class _AnalyticChecks(unittest.TestCase):
    def setUp(self):
        self.tet = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.topology = np.array([[0, 1, 2, 3]])
        self.box = np.array([[x, y, z] for x in (-.14, .14) for y in (-.10, .10) for z in (0, .30)])

    def test_rigid_motion_preserves_strain_and_fit(self):
        rotation = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
        cur = self.tet @ rotation.T + [2, 3, 4]
        self.assertAlmostEqual(tetra_metrics(self.tet, cur, self.topology)["signed_volume_ratio"]["min"], 1)
        self.assertAlmostEqual(triangle_metrics(self.tet, cur, np.array([[0, 1, 2]]))["edge_length_ratio"]["max"], 1)
        fit = fit_rigid_transform(self.tet, cur)
        np.testing.assert_allclose(fit["rotation"], rotation, atol=1e-14)
        self.assertLess(fit["rms_residual_m"], 1e-14)

    def test_inverted_and_collapsed_tetrahedra(self):
        inverted = self.tet.copy()
        inverted[3, 2] = -1
        result = tetra_metrics(self.tet, inverted, self.topology)
        self.assertEqual(result["inverted_tetrahedron_count"], 1)
        self.assertAlmostEqual(result["signed_volume_ratio"]["min"], -1)
        inverted[3, 2] = 0
        self.assertEqual(tetra_metrics(self.tet, inverted, self.topology)["collapsed_tetrahedron_count"], 1)
        reversed_topology = self.topology[:, [0, 2, 1, 3]]
        self.assertAlmostEqual(tetra_metrics(self.tet, self.tet, reversed_topology)["signed_volume_ratio"]["min"], 1)

    def test_outside_bear_and_rotation(self):
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        offset = np.array([-.18, .22, .742])
        cur = self.box @ rotation.T + offset
        bounds = [[-.14, -.1, 0], [.14, .1, .3]]
        inside = np.array([[0, 0, .15], [.1, 0, .20]]) @ rotation.T + offset
        call = lambda points: approximate_containment(points, self.box, cur, np.arange(8), bounds)
        self.assertTrue(call(inside)["approximate_containment_pass"])
        outside = np.concatenate((inside, [[-.18, .22, 1.052]]))
        self.assertFalse(call(outside)["approximate_containment_pass"])
        self.assertEqual(call(outside)["outside_vertex_count"], 1)

    def test_flat_bag_and_table_penetration(self):
        flat_rotation = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
        flat = self.box @ flat_rotation.T
        self.assertAlmostEqual(bag_pose(self.box, flat, np.arange(8))["tilt_from_upright_degrees"], 90)
        self.assertFalse(table_contact_metrics([[0, 0, .73]])["within_table_penetration_tolerance"])
        self.assertTrue(table_contact_metrics([[0, 0, .739]])["within_table_penetration_tolerance"])

    def test_nonfinite_and_degenerate_data_do_not_pass(self):
        broken = self.tet.copy()
        broken[0, 0] = np.nan
        self.assertFalse(tetra_metrics(self.tet, broken, self.topology)["all_finite"])
        self.assertFalse(table_contact_metrics(broken)["within_table_penetration_tolerance"])
        with self.assertRaises(ValueError):
            fit_rigid_transform(np.zeros((3, 3)), np.zeros((3, 3)))
        result = triangle_metrics(self.tet, self.tet, np.array([[0, 0, 1]]))
        self.assertEqual(result["degenerate_rest_triangle_count"], 1)

    def test_actual_containment_rejects_outside_and_boundary(self):
        faces = np.array([[0, 2, 6], [0, 6, 4], [0, 4, 5], [0, 5, 1],
                          [2, 3, 7], [2, 7, 6], [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5]])
        rim = np.array([1, 5, 7, 3])
        inside = np.array([[0., 0, .15], [.1, -.05, .25]])
        valid = actual_bag_containment(inside, self.box, faces, rim, point_chunk=1, triangle_chunk=3)
        self.assertTrue(valid["actual_containment_pass"], valid)
        self.assertTrue(valid["surface_watertight"])
        self.assertAlmostEqual(valid["minimum_surface_distance_m"], .04)
        reversed_result = actual_bag_containment(inside, self.box, faces, rim[::-1])
        self.assertTrue(reversed_result["actual_containment_pass"])
        outside = actual_bag_containment([[0, 0, .31], [0, 0, .3], [0, 0, .15]], self.box, faces, rim)
        self.assertFalse(outside["actual_containment_pass"])
        self.assertEqual(outside["outside_or_boundary_vertex_count"], 2)
        self.assertEqual(outside["boundary_vertex_count"], 1)
        self.assertAlmostEqual(outside["inside_vertex_fraction"], 1 / 3)

    def test_actual_containment_rejects_invalid_rim_and_surface(self):
        faces = np.array([[0, 2, 6], [0, 6, 4], [0, 4, 5], [0, 5, 1],
                          [2, 3, 7], [2, 7, 6], [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5]])
        rim = np.array([1, 5, 7, 3])
        inside = [[0., 0, .15]]
        warped = self.box.copy()
        warped[1, 2] += .06
        self.assertFalse(actual_bag_containment(inside, warped, faces, rim)["surface_audit_pass"])
        self.assertFalse(actual_bag_containment(inside, self.box, faces[:-1], rim)["surface_audit_pass"])
        crossed = self.box.copy()
        crossed[[5, 7]] = crossed[[7, 5]]
        result = actual_bag_containment(inside, crossed, faces, rim)
        self.assertFalse(result["surface_audit_pass"])
        self.assertFalse(result["projected_rim_is_simple"])
        broken_winding = faces.copy()
        broken_winding[0] = broken_winding[0, ::-1]
        self.assertFalse(actual_bag_containment(inside, self.box, broken_winding, rim)["surface_audit_pass"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run analytic CPU checks (no simulation)")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(_AnalyticChecks)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    print(json.dumps({"module": "validate_physics", "usage": "Import explicit-array diagnostic functions, or run --self-test."}))
