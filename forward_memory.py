"""Bound FEM history memory in this forward-only scene without changing integration.

Genesis 1.4.3 FEMSolver.save_ckpt stores three GPU tensors under every new
checkpoint name even when Simulator.requires_grad is False. The simulator also
uses that method to roll the final local frame into frame zero. This instance-only
adapter omits history creation and keeps exactly that required copy_frame call.
It must never be installed on a differentiable scene.
"""
from __future__ import annotations

from importlib.metadata import version


def install_forward_memory(scene):
    """Install before build; return a small live, JSON-serializable report.

    No installed Genesis module/class is changed. Existing differentiable scenes
    are rejected before any mutation, and a later change to requires_grad also
    fails before modifying a solver frame. All local frame arrays continue to be
    handled by Genesis's own copy_frame kernel.
    """
    sim = scene.sim
    if sim.requires_grad:
        raise RuntimeError("Forward FEM memory adapter requires requires_grad=False")
    if scene.is_built or hasattr(scene, "teddy_forward_memory_report"):
        raise RuntimeError("Forward FEM memory adapter must be installed once before build")
    if version("genesis-world") != "1.4.3":
        raise RuntimeError("Forward FEM memory adapter requires audited Genesis 1.4.3")
    solver = sim.fem_solver
    if not callable(getattr(solver, "save_ckpt", None)) or not callable(getattr(solver, "copy_frame", None)):
        raise RuntimeError("Expected FEM checkpoint and frame-copy API is unavailable")
    if getattr(solver, "_ckpt", {}):
        raise RuntimeError("Cannot install forward FEM memory adapter with existing checkpoints")

    report = {
        "enabled": True,
        "scope": "this scene's FEM solver instance only",
        "requires_grad": False,
        "genesis_version": "1.4.3",
        "history_checkpoint_tensors_retained": False,
        "frame_rollover": "FEMSolver.copy_frame(sim.substeps_local, 0)",
        "native_copy_frame_kernel_unchanged": True,
        "physics_parameters_changed": False,
        "rollover_count": 0,
        "retained_checkpoint_count": 0,
    }

    def save_forward_frame(ckpt_name):
        # The simulator still calls this at exactly its normal checkpoint boundary.
        # ckpt_name identifies reverse-mode history, which this scene never uses.
        if sim.requires_grad:
            raise RuntimeError("Forward FEM memory adapter cannot be used with requires_grad=True")
        if solver.is_active:
            # This is the original method's only integration-state mutation.
            solver.copy_frame(solver.sim.substeps_local, 0)
            report["rollover_count"] += 1
            report["retained_checkpoint_count"] = len(solver._ckpt)

    solver.save_ckpt = save_forward_frame
    scene.teddy_forward_memory_report = report
    return report
