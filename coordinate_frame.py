"""Optional constant physics-origin translation; planning stays in canonical world coordinates.

Only initial placements and read-only observations cross this boundary. Velocities,
forces, torques about a body/TCP, rotations, local vertices and gravity are unchanged.
No integrated state is repositioned and no installed simulator code is modified.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CoordinateFrame:
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        origin = np.asarray(self.origin_m, dtype=float)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("physics_origin_m must contain three finite coordinates")
        object.__setattr__(self, "origin_m", tuple(float(x) for x in origin))

    @classmethod
    def from_config(cls, config):
        return cls(config.get("simulation", {}).get("physics_origin_m", (0., 0., 0.)))

    @property
    def active(self):
        return any(value != 0. for value in self.origin_m)

    def points_to_physics(self, points):
        points = np.asarray(points, dtype=float)
        if points.shape[-1:] != (3,):
            raise ValueError("Position arrays must end in three coordinates")
        return points - self.origin_m if self.active else points.copy()

    def points_to_world(self, points):
        points = np.asarray(points, dtype=float)
        if points.shape[-1:] != (3,):
            raise ValueError("Position arrays must end in three coordinates")
        return points + self.origin_m if self.active else points.copy()

    def transforms_to_world(self, transforms):
        result = np.asarray(transforms, dtype=float).copy()
        if result.shape[-2:] != (4, 4):
            raise ValueError("Affine transforms must end in shape (4, 4)")
        if self.active:
            result[..., :3, 3] += self.origin_m
        return result

    def report(self):
        return {"enabled": self.active, "physics_origin_m": list(self.origin_m),
                "mapping": "physics_position = canonical_world_position - physics_origin_m",
                "planning_and_saved_observations": "canonical world coordinates",
                "rotations_forces_local_geometry_and_materials_changed": False,
                "runtime_state_repositioning": False}


def frame_for(source):
    """Get a scene's immutable frame, also accepting entities and frame objects."""
    if isinstance(source, CoordinateFrame):
        return source
    scene = getattr(source, "scene", source)
    return getattr(scene, "teddy_coordinate_frame", CoordinateFrame())


def to_world(points, source):
    return frame_for(source).points_to_world(points)


def to_physics(points, source):
    return frame_for(source).points_to_physics(points)


def transforms_to_world(transforms, source):
    return frame_for(source).transforms_to_world(transforms)


def _correct_native_ground(native_scene, frame):
    """Correct this scene's sole horizontal floor before native World.init.

Genesis 1.4.3 builds IPC ground using geom.init_pos (link-local), omitting
the Plane entity translation. Keep both native current and rest geometry
consistent with the already-translated Genesis/rendered floor.
"""
    import uipc
    objects = native_scene.objects()
    planes = [obj for i in range(objects.created_count())
              if (obj := objects.find(i)) is not None and obj.name().startswith("rigid_plane_")]
    if len(planes) != 1 or len(planes[0].geometries().ids()) != 1:
        raise RuntimeError("Translated scene requires exactly one native floor plane")
    geometry_id = planes[0].geometries().ids()[0]
    slots = native_scene.geometries().find(geometry_id)
    expected = frame.points_to_physics((0., 0., 0.))
    rows = []
    # Verify both before mutating either. Fail if upstream plane handling changes.
    geometries = [slot.geometry() for slot in slots]
    for geometry in geometries:
        uid = np.asarray(uipc.view(geometry.meta().find("implicit_geometry_uid"))).reshape(-1)
        normal = np.asarray(uipc.view(geometry.instances().find("N"))).reshape(-1, 3)
        point = np.asarray(uipc.view(geometry.instances().find("P"))).reshape(-1, 3)
        if uid.tolist() != [1] or normal.shape != (1, 3) or point.shape != (1, 3) \
                or not np.allclose(normal, [[0., 0., 1.]], rtol=0, atol=1e-14) \
                or not np.allclose(point, 0., rtol=0, atol=1e-14):
            raise RuntimeError("Unexpected native floor; origin adapter must be reviewed")
    for label, geometry in zip(("current", "rest"), geometries):
        point = uipc.view(geometry.instances().find("P"))
        point[...] = expected.reshape(point.shape)
        readback = np.asarray(uipc.view(geometry.instances().find("P"))).reshape(3).copy()
        if not np.array_equal(readback, expected):
            raise RuntimeError("Native floor origin readback failed")
        rows.append({"geometry": label, "physics_point_m": readback.tolist(),
                     "canonical_point_m": frame.points_to_world(readback).tolist()})
    return {"verified": True, "native_object": planes[0].name(), "geometries": rows,
            "scope": "Initial native plane point only, before World.init"}


def install_coordinate_frame(scene, frame):
    """Attach the frame before any entities/cameras are added or the scene is built."""
    if scene.is_built or hasattr(scene, "teddy_coordinate_frame"):
        raise RuntimeError("Coordinate frame must be installed once on an unbuilt scene")
    scene.teddy_coordinate_frame = frame
    report = frame.report()
    scene.teddy_coordinate_frame_report = report
    if not frame.active:
        report["native_floor"] = {"verified": True, "changed": False}
        return report
    from importlib.metadata import version
    if version("genesis-world") != "1.4.3" or version("pyuipc") != "0.0.29":
        raise RuntimeError("Native floor adapter requires the audited Genesis/libuipc versions")
    coupler = scene.sim.coupler
    original = coupler._add_rigid_geoms_to_ipc

    def add_rigid_geometries():
        original()
        report["native_floor"] = _correct_native_ground(coupler._ipc_scene, frame)

    coupler._add_rigid_geoms_to_ipc = add_rigid_geometries
    report["native_floor"] = {"verified": False, "changed": False}
    return report
