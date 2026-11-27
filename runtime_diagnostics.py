"""Bounded, observation-only diagnostics for the GPU plastic-bag scene.

No initialization, solver settings, forces, constraints or soft states change.
Use positions_no_grad in place of FEMEntity.get_state().pos for an ordinary
forward run: Genesis 1.4.3 otherwise retains every queried GPU state for AD.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import math
from pathlib import Path
import time


def positions_no_grad(entity):
    """Copy actual FEM positions and release only this temporary AD query.

    The source FEM state is read through Genesis' existing get_state API.
    QueriedStates.discard is bookkeeping only; it does not touch solver state.
    A fresh NumPy copy owns the returned storage after the query is discarded.
    Differentiable runs must retain their queries and are deliberately rejected.
    """
    if entity.sim.requires_grad:
        raise RuntimeError("positions_no_grad cannot be used for differentiable simulation")
    queries = getattr(entity, "_queried_states", None)
    discard = getattr(queries, "discard", None)
    if not callable(discard):
        raise RuntimeError("Expected Genesis QueriedStates.discard for bounded FEM observation")
    state = entity.get_state()
    try:
        return state.pos.detach().cpu().numpy().reshape(-1, 3).copy()
    finally:
        discard(state)


def _bounded_json(value, *, depth=0):
    """Preserve small native records while bounding unexpected large payloads."""
    if depth > 8:
        return "<nested value omitted>"
    if isinstance(value, dict):
        items = list(value.items())
        result = {str(k): _bounded_json(v, depth=depth + 1) for k, v in items[:64]}
        if len(items) > 64:
            result["_omitted_keys"] = len(items) - 64
        return result
    if isinstance(value, (list, tuple)):
        # Latest status messages are the most useful when the native collection
        # grows. Keep a bounded tail without clearing the engine's own history.
        result = [_bounded_json(v, depth=depth + 1) for v in value[-32:]]
        if len(value) > 32:
            return {"total_entries": len(value), "latest_entries": result}
        return result
    if isinstance(value, str):
        return value if len(value) <= 1600 else value[:1600] + "…"
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:1600]


def cuda_observation():
    """Read device and allocator counters; never initialize a simulation."""
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return {"cuda_initialized": False, "fallback_performed": False}
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "cuda_initialized": True, "device_index": int(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_free_bytes": int(free), "device_total_bytes": int(total),
        "torch_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "torch_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "scope": "Device-wide free/total plus this process's PyTorch allocator; IPC and graphics allocations are outside PyTorch counters",
    }


class RuntimeDiagnostics:
    """Collect a bounded tail plus whole-run counters at a modest interval.

    Construct after scene.build(). Call sample() after scene.step(), optionally
    passing the wall time spent in scene.step alone. It returns None between
    samples. Native Engine.frame_stats and status are observations, not timers
    or convergence-setting changes. Save report() at completion; passing a
    jsonl_path additionally streams bounded rows for monitoring while running.
    """
    def __init__(self, scene, *, every_steps=20, max_records=2048, jsonl_path=None):
        if not scene.is_built:
            raise ValueError("Build the scene before collecting runtime diagnostics")
        if not isinstance(every_steps, int) or every_steps < 1:
            raise ValueError("every_steps must be a positive integer")
        if not isinstance(max_records, int) or max_records < 1:
            raise ValueError("max_records must be a positive integer")
        self.scene = scene
        self.engine = scene.sim.coupler._ipc_engine
        if self.engine is None or self.engine.backend_name() != "cuda":
            raise RuntimeError("Runtime diagnostics expect the native IPC CUDA engine")
        self.every_steps = every_steps
        self.records = deque(maxlen=max_records)
        self.total_samples = 0
        self.last_sample_step = None
        self.last_observed_step = None
        self.observed_native_steps = 0
        self.native_convergence_failure_count = 0
        self.native_line_search_warning_count = 0
        self.native_line_search_warnings = deque(maxlen=16)
        self.native_convergence_failures = deque(maxlen=16)
        self.started = time.monotonic()
        self.last_status_digest = None
        self.error_seen = False
        self.status_changes = 0
        self.peak_counts = {}
        self.minimum_free_bytes = None
        self.maximum_torch_allocated_bytes = 0
        self.maximum_fem_retained_queries = 0
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            # Refuse to mix diagnostic evidence from two simulations.
            self.jsonl_path.open("x").close()
        self.initial_cuda = cuda_observation()

    def sample(self, simulation_time_s, *, step_wall_seconds=None, force=False):
        step = int(self.scene.sim.cur_step_global)
        frame = self.engine.frame_stats()
        convergence_failure = bool(step > 0 and (not isinstance(frame, dict) or not frame
            or not frame.get("completed", False) or not frame.get("converged", False)
                 or frame.get("hit_newton_limit", False)))
        line_warning = bool(step > 0 and isinstance(frame, dict) and frame.get("hit_line_search_limit", False))
        if step > 0 and step != self.last_observed_step:
            self.observed_native_steps += 1
            self.native_convergence_failure_count += int(convergence_failure)
            self.native_line_search_warning_count += int(line_warning)
            if line_warning:
                self.native_line_search_warnings.append({"physics_step": step,
                    "simulation_time_s": float(simulation_time_s), "native_frame_stats": _bounded_json(frame)})
            if convergence_failure:
                self.native_convergence_failures.append({"physics_step": step,
                    "simulation_time_s": float(simulation_time_s), "native_frame_stats": _bounded_json(frame)})
            self.last_observed_step = step
            for key in ("newton_iterations", "line_search_trials", "linear_solver_iterations"):
                value = frame.get(key) if isinstance(frame, dict) else None
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    self.peak_counts[key] = max(self.peak_counts.get(key, 0), value)
        if self.last_sample_step == step:
            return None
        if not force and not convergence_failure and not line_warning and step % self.every_steps:
            return None
        now = time.monotonic()
        status = self.engine.status()
        status_json = status.to_json()
        status_blob = json.dumps(status_json, sort_keys=True, default=str)
        status_digest = hashlib.sha256(status_blob.encode()).hexdigest()
        status_changed = status_digest != self.last_status_digest
        has_error = bool(status.has_error())
        self.error_seen |= has_error
        self.status_changes += int(status_changed)
        gpu = cuda_observation()
        free = gpu.get("device_free_bytes")
        if free is not None:
            self.minimum_free_bytes = free if self.minimum_free_bytes is None else min(free, self.minimum_free_bytes)
        self.maximum_torch_allocated_bytes = max(self.maximum_torch_allocated_bytes,
            gpu.get("torch_allocated_bytes", 0))
        retained = {}
        for entity in self.scene.sim.fem_solver.entities:
            queues = getattr(getattr(entity, "_queried_states", None), "states", {})
            retained[entity.name] = sum(len(values) for values in queues.values())
        self.maximum_fem_retained_queries = max(self.maximum_fem_retained_queries, sum(retained.values()))
        row = {
            "physics_step": step, "simulation_time_s": float(simulation_time_s),
            "wall_elapsed_s": now - self.started,
            "scene_step_wall_s": float(step_wall_seconds) if step_wall_seconds is not None else None,
            "native_frame_stats": _bounded_json(frame), "native_engine_has_error": has_error,
            "native_convergence_failure": convergence_failure,
            "native_line_search_warning": line_warning,
            "native_status_changed": status_changed, "native_status_sha256": status_digest,
            "native_status": _bounded_json(status_json) if status_changed or has_error else None,
            "cuda": gpu, "retained_fem_query_states": retained,
        }
        self.records.append(row)
        self.total_samples += 1
        self.last_sample_step = step
        self.last_status_digest = status_digest
        if self.jsonl_path is not None:
            with self.jsonl_path.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        return row

    def report(self):
        return {
            "method": "Read-only native frame statistics/status and CUDA allocator counters",
            "solver_or_physics_parameters_changed": False, "native_backend": "cuda",
            "sampling_interval_steps": self.every_steps, "samples_collected": self.total_samples,
            "retained_record_count": len(self.records), "maximum_retained_records": self.records.maxlen,
            "initial_cuda": self.initial_cuda, "minimum_device_free_bytes": self.minimum_free_bytes,
            "maximum_torch_allocated_bytes": self.maximum_torch_allocated_bytes,
            "maximum_retained_fem_query_states": self.maximum_fem_retained_queries,
            "native_engine_error_seen": self.error_seen, "native_status_changes": self.status_changes,
            "observed_native_steps": self.observed_native_steps,
            "native_convergence_failure_count": self.native_convergence_failure_count,
            "native_line_search_warning_count": self.native_line_search_warning_count,
            "latest_native_line_search_warnings": list(self.native_line_search_warnings),
            "convergence_policy": "Require completed, converged, no Newton-limit and no engine errors every step. Record sticky line-search-limit separately: it can occur in an intermediate Newton iteration that subsequently converges. Such warnings are not a claim of exact energy descent.",
            "latest_native_convergence_failures": list(self.native_convergence_failures),
            "peak_observed_native_iteration_counts": self.peak_counts,
            "jsonl_path": str(self.jsonl_path) if self.jsonl_path is not None else None,
            "records": list(self.records),
            "limitations": ["Native convergence covers each distinct step where sample() is called; invoke it after every step for complete coverage",
                "scene_step timing excludes observations, camera rendering, video encoding and motion planning",
                "CUDA allocator queries do not measure GPU utilization or attribute device-wide memory to this process"],
        }
