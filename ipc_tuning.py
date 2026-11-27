"""Scene-local choice of a native FEM linear-system preconditioner.

Only `linear_system/fem_preconditioner` is changed before IPC World.init.
Geometry, mass, constitutive laws, friction, time step, solver residual
tolerances, iteration limits and CUDA graph mode remain exactly as configured.
MAS is a numerical acceleration candidate, not a change to the physical model.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import math


SETTING = "linear_system/fem_preconditioner"


def _config_scalar(config, key):
    attribute = config.find(key)
    if attribute is None or attribute.size() != 1:
        raise RuntimeError(f"Expected one native IPC configuration value for {key}")
    value = attribute.view()[0]
    # JSON values must be ordinary Python scalar types, not NumPy scalars.
    return value.item() if hasattr(value, "item") else value


def set_native_preconditioner(native_scene, selection):
    """Apply and audit the one permitted native configuration edit.

    The caller must use this before World.init. This helper creates neither a
    world nor an engine and supports a CPU-only configuration readback check.
    """
    if selection not in ("diag", "mas"):
        raise ValueError("FEM preconditioner must be 'diag' or 'mas'")
    import uipc

    config = native_scene.config()
    before = deepcopy(config.to_json())
    original = _config_scalar(config, SETTING)
    if original not in ("diag", "mas"):
        raise RuntimeError(f"Unexpected native FEM preconditioner {original!r}")
    tolerance = float(_config_scalar(config, "linear_system/tol_rate"))
    if not math.isclose(tolerance, 1e-4, rel_tol=0, abs_tol=1e-15):
        raise ValueError("This scene's audited native linear residual tolerance must remain 1e-4")
    # AttributeSlotString.view() is a const span. uipc.view(slot) is the
    # documented mutable StringSpan and accepts only an explicit string.
    uipc.view(config.find(SETTING))[0] = selection
    after = deepcopy(config.to_json())
    protected_before = {k: v for k, v in before.items() if k != SETTING}
    protected_after = {k: v for k, v in after.items() if k != SETTING}
    readback = _config_scalar(config, SETTING)
    unchanged = protected_before == protected_after
    if readback != selection or not unchanged:
        raise RuntimeError("Native preconditioner edit changed an unrelated IPC setting")
    raw = json.dumps(protected_after, sort_keys=True, separators=(",", ":"))
    keys = ("dt", "linear_system/solver", "linear_system/tol_rate",
        "linear_system/use_cuda_graph", "linear_system/check_interval",
        "newton/max_iter", "newton/velocity_tol", "newton/transrate_tol",
        "newton/ccd_tol", "newton/semi_implicit/enable", "line_search/max_iter",
        "contact/d_hat", "contact/eps_velocity", "contact/friction/enable",
        "contact/enable", "sanity_check/enable")
    return {
        "verified": True, "setting": SETTING, "previous": original, "selected": selection,
        "readback": readback, "all_other_native_configuration_attributes_unchanged": unchanged,
        "protected_native_configuration_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "protected_readback": {key: _config_scalar(config, key) for key in keys},
        "native_configuration_before": before, "native_configuration_after": after,
    }


def configure_ipc_tuning(scene, *, fem_preconditioner="mas"):
    """Install before scene.build(); return a report populated by native init."""
    if scene.is_built or version("genesis-world") != "1.4.3" or version("pyuipc") != "0.0.29":
        raise RuntimeError("IPC tuning requires an unbuilt Genesis 1.4.3 / pyuipc 0.0.29 scene")
    if fem_preconditioner not in ("diag", "mas"):
        raise ValueError("FEM preconditioner must be 'diag' or 'mas'")
    from genesis.engine.couplers import IPCCoupler
    coupler = scene.sim.coupler
    if not isinstance(coupler, IPCCoupler) or coupler._ipc_scene is not None:
        raise RuntimeError("Install IPC tuning before native IPC initialization")
    if getattr(coupler, "_plastic_ipc_tuning_installed", False):
        raise RuntimeError("IPC tuning was already installed on this scene")
    report = {
        "enabled": True, "verified": False, "requested_fem_preconditioner": fem_preconditioner,
        "scope": "One native configuration attribute before World.init",
        "engine_installation_modified": False, "global_classes_modified": False,
        "physics_or_tolerances_changed": False,
        "performance_validated": False,
        "interpretation": "MAS preconditions all nonempty FEM bodies; setup and application overhead can outweigh reduced PCG iterations",
    }
    native_init = coupler._init_ipc

    def init_with_preconditioner():
        if report["verified"]:
            raise RuntimeError("Native IPC tuning initialization was repeated")
        native_init()
        # Genesis has just created Engine/World/Scene, but World.init happens
        # later in _finalize_ipc, after constitutions and contacts are added.
        result = set_native_preconditioner(coupler._ipc_scene, fem_preconditioner)
        if not math.isclose(float(result["protected_readback"]["dt"]),
                float(scene.sim.substep_dt), rel_tol=0, abs_tol=1e-15):
            raise RuntimeError("Native time step disagrees with the Genesis substep")
        report.update(result)

    coupler._init_ipc = init_with_preconditioner
    coupler._plastic_ipc_tuning_installed = True
    return report
