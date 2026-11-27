"""Scene-local effective isotropic kraft-paper shell for Genesis 1.4.3 / pyuipc 0.0.29.

Genesis' FEM.Cloth is used as the surface-mesh entity interface. Its default
anisotropic cloth law is replaced, before IPC initialization, by native
NeoHookeanShell. No engine files, positions, velocities or constraints change.

This is an uncalibrated elastic engineering approximation to dry kraft paper,
not a material identification from the reference photograph. It does not
simulate orthotropy, yield, tearing, moisture, or new permanent creases.
Contact thickness is the physical one-sided radius, r=t/2. Native membrane energy and mass use full thickness t.
"""
from __future__ import annotations

from importlib.metadata import version
import numpy as np

# Engineering starting values, not measurements of the pictured bag. Full
# thickness and density jointly imply 200 g/m^2; IPC receives half-thickness.
DEFAULT_KRAFT = {
    "E": 800e6,
    "nu": 0.30,
    "rho": 800.0,
    "full_thickness": 0.00025,
}


def paper_constants(E, nu, rho, full_thickness):
    """Return the isotropic, small-strain plane-stress constants in SI units."""
    E, nu, rho, t = map(float, (E, nu, rho, full_thickness))
    if not np.isfinite([E, nu, rho, t]).all() or min(E, rho, t) <= 0 or not 0 < nu < .5:
        raise ValueError("Paper shell requires finite E/rho/thickness > 0 and 0 < nu < 0.5")
    return {
        "young_modulus_pa": E,
        "poisson_ratio": nu,
        "density_kg_m3": rho,
        "full_thickness_m": t,
        "ipc_one_sided_radius_m": t / 2,
        "lame_lambda_pa": E * nu / (1 - nu * nu),
        "shear_mu_pa": E / (2 * (1 + nu)),
        "bending_stiffness_nm": E * t**3 / (12 * (1 - nu * nu)),
        "areal_density_kg_m2": rho * t,
        "small_strain_extensional_stiffness_n_m": E * t / (1 - nu * nu),
        "small_strain_shear_stiffness_n_m": E * t / (2 * (1 + nu)),
    }


def audit_native_shell(geometry, constants):
    """Read native material attributes without altering the geometry."""
    import uipc

    def values(collection, name):
        attribute = getattr(geometry, collection)().find(name)
        if attribute is None:
            raise RuntimeError(f"Native paper attribute missing: {collection}.{name}")
        return np.asarray(uipc.view(attribute))

    uid = values("meta", "constitution_uid").reshape(-1)
    # Extra-constitution IDs are a vector-valued attribute; JSON preserves
    # their nested layout without relying on an unsupported ragged view.
    extra = geometry.meta().to_json().get("extra_constitution_uids", {})
    extra_ids = np.asarray(extra.get("attribute", {}).get("values", []), dtype=int).reshape(-1)
    density = values("meta", "mass_density").reshape(-1)
    radius = values("vertices", "thickness").reshape(-1)
    volume = values("vertices", "volume").reshape(-1)
    fixed = values("vertices", "is_fixed").reshape(-1)
    dynamic = values("vertices", "is_dynamic").reshape(-1)
    lam = values("triangles", "lambda").reshape(-1)
    mu = values("triangles", "mu").reshape(-1)
    bending = values("edges", "bending_stiffness").reshape(-1)
    points = np.asarray(uipc.view(geometry.positions())).reshape(-1, 3)
    faces = np.asarray(uipc.view(geometry.triangles().topo())).reshape(-1, 3)
    tri = points[faces]
    area = float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2)

    def matches(array, target):
        return bool(array.size and np.isfinite(array).all() and np.allclose(array, target, rtol=1e-10, atol=1e-14))

    checks = {
        "neo_hookean_shell_uid_11": bool(uid.size == 1 and uid[0] == 11),
        "only_discrete_shell_bending_uid_17": bool(extra_ids.size == 1 and extra_ids[0] == 17),
        "native_density_matches": matches(density, constants["density_kg_m3"]),
        "native_radius_is_half_physical_thickness": matches(radius, constants["ipc_one_sided_radius_m"]),
        "native_lame_lambda_matches_plane_stress": matches(lam, constants["lame_lambda_pa"]),
        "native_shear_mu_matches_isotropic_law": matches(mu, constants["shear_mu_pa"]),
        "native_bending_matches_physical_plate_value": matches(bending, constants["bending_stiffness_nm"]),
        "no_fixed_vertices": bool(fixed.size == len(points) and np.all(fixed == 0)),
        "all_vertices_dynamic": bool(dynamic.size == len(points) and np.all(dynamic == 1)),
        "positive_finite_surface_area": bool(np.isfinite(area) and area > 0),
        "native_volume_uses_full_thickness": bool(np.isfinite(volume).all() and np.all(volume > 0)
            and np.isclose(volume.sum(), area * constants["full_thickness_m"], rtol=1e-10, atol=1e-14)),
    }
    result = {
        "verified": all(checks.values()), "checks": checks,
        "constitution_uid": uid.tolist(), "extra_constitution_uids": extra_ids.tolist(),
        "vertices": len(points), "triangles": len(faces), "surface_area_m2": area,
        "native_density_kg_m3": density.tolist(),
        "native_radius_range_m": [float(radius.min()), float(radius.max())],
        "native_lame_lambda_range_pa": [float(lam.min()), float(lam.max())],
        "native_shear_mu_range_pa": [float(mu.min()), float(mu.max())],
        "native_bending_range_nm": [float(bending.min()), float(bending.max())],
        "native_rest_volume_m3": float(volume.sum()),
        "native_rest_mass_kg": float(volume.sum() * density[0]),
        "expected_rest_mass_kg": float(area * constants["areal_density_kg_m2"]),
        "fixed_vertex_count": int(np.count_nonzero(fixed)),
    }
    if not result["verified"]:
        raise RuntimeError(f"Native kraft-paper readback failed: {result}")
    return result


def configure_paper_shell(scene, *, shell_entity):
    """Install before scene.build(); return a report populated during build.

    Exactly one shell entity is supported. Volumetric teddy bodies retain the
    original native StableNeoHookean constitution and all contact handling.
    Do not combine this adapter with the cloth shear-coefficient adapter.
    """
    if scene.is_built or version("genesis-world") != "1.4.3" or version("pyuipc") != "0.0.29":
        raise RuntimeError("Kraft-paper adapter requires an unbuilt Genesis 1.4.3 / pyuipc 0.0.29 scene")
    from genesis.engine.couplers import IPCCoupler
    from genesis.engine.materials.FEM.cloth import Cloth
    from genesis.engine.couplers.ipc_coupler.utils import read_ipc_geometry_metadata
    from uipc.backend import SceneVisitor
    from uipc.constitution import NeoHookeanShell
    from uipc.geometry import SimplicialComplexSlot

    coupler = scene.sim.coupler
    if not isinstance(coupler, IPCCoupler):
        raise RuntimeError("Kraft-paper adapter requires the native IPC coupler")
    shells = [e for e in coupler.fem_solver.entities if isinstance(e.material, Cloth)]
    if len(shells) != 1 or shells[0] is not shell_entity:
        raise ValueError("Exactly the named paper entity must be the scene's only FEM shell")
    if getattr(coupler, "_paper_shell_adapter_installed", False) or coupler._ipc_nks is not None:
        raise RuntimeError("An IPC shell constitution or kraft-paper adapter was already installed")
    material = shell_entity.material
    constants = paper_constants(material.E, material.nu, material.rho, 2 * material.thickness)
    if material.bending_stiffness is None or not np.isclose(material.bending_stiffness,
            constants["bending_stiffness_nm"], rtol=1e-10, atol=1e-14):
        raise ValueError("Set paper bending_stiffness to E*t**3/(12*(1-nu**2)), with physical full thickness t")
    report = {
        "enabled": True, "verified": False, "calibrated": False,
        "material": "uncalibrated effective isotropic elastic approximation to dry kraft paper",
        "native_membrane": "NeoHookeanShell", "native_membrane_uid": 11,
        "native_bending": "DiscreteShellBending", "native_bending_uid": 17,
        "scope": "Scene-local native constitution selection before IPC World initialization",
        "engine_installation_modified": False, "global_classes_modified": False,
        "soft_state_mutations": False, "attachments_or_fixed_soft_vertices": False,
        "independent_cloth_shear_override": False, "constants": constants,
        "friction_coefficient_input": float(material.friction_mu),
        "calibration_status": "Engineering assumptions; reference photo provides no measured material parameters",
        "rest_shape": "Manufactured open rectangular form; existing folds belong to initial rest geometry",
        "handles": "Same shell material as bag body; flat paper strip approximation, not twisted cord",
        "source_notes": "MATERIAL_NOTES.txt",
        "limitations": ["Uncalibrated effective isotropic paper; no orthotropic fibre directions, yielding, tearing or new permanent creases",
            "No moisture dependence, viscoelastic relaxation, calibrated damping, glue delamination or air pressure",
            "2D isotropic neo-Hookean law matches small-strain plane stress; finite-strain 3D plane-stress reduction is not solved"],
        "entities": [],
    }
    native_add = coupler._add_fem_entities_to_ipc

    def add_with_isotropic_paper():
        if coupler._ipc_nks is not None or report["entities"]:
            raise RuntimeError("Native paper registration attempted more than once")
        membrane = NeoHookeanShell()
        if membrane.uid() != 11:
            raise RuntimeError("Unexpected native NeoHookeanShell UID")
        coupler._ipc_nks = membrane
        coupler._ipc_constitution_tabular.insert(membrane)
        # Genesis sees the already installed constitution and invokes its
        # compatible apply_to(mesh, moduli, mass_density, thickness) signature.
        native_add()
        for slot in SceneVisitor(coupler._ipc_scene).geometries():
            if not isinstance(slot, SimplicialComplexSlot):
                continue
            geometry = slot.geometry()
            metadata = read_ipc_geometry_metadata(geometry)
            if metadata is None or metadata[0] != "cloth":
                continue
            _, environment, entity_index = metadata
            if coupler.fem_solver.entities[entity_index] is not shell_entity:
                raise RuntimeError("Unexpected shell entity during native kraft-paper readback")
            readback = audit_native_shell(geometry, constants)
            report["entities"].append({"entity_index": int(entity_index),
                "environment": int(environment), **readback})
        report["verified"] = bool(len(report["entities"]) == scene.sim._B
            and all(row["verified"] for row in report["entities"]))
        if not report["verified"]:
            raise RuntimeError("Kraft paper was not verified in every environment")

    coupler._add_fem_entities_to_ipc = add_with_isotropic_paper
    coupler._paper_shell_adapter_installed = True
    return report
