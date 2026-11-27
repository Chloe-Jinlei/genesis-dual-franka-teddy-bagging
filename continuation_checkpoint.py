"""Preserve native IPC and controller evidence without advancing or restoring.

This is an exporter, not an exact-resume implementation.  A successful export
verifies the native dump against current observations and copies its workspace;
it does not establish that a rebuilt coupled scene can continue identically.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import tempfile

import numpy as np

from coordinate_frame import CoordinateFrame
from runtime_diagnostics import positions_no_grad

FEM_DIRECTORY = Path('dump/cuda/finite_element/finite_element_method.cu')


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    result = np.asarray(value)
    if result.dtype.kind not in 'biufc':
        raise ValueError('Checkpoint arrays must have numeric or boolean dtype')
    return result.copy()


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {'unserialized_type': type(value).__name__, 'representation': repr(value)}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def workspace_manifest(directory):
    """Hash every regular file; reject symbolic links rather than following them."""
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Native workspace contains a symbolic link: {path}')
        if path.is_file():
            result[str(path.relative_to(directory))] = {'size_bytes': path.stat().st_size, 'sha256': _sha(path)}
    return result


def copy_workspace_verified(source, destination):
    """Copy files without deleting either tree; verify source stability and bytes."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir() or source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Native workspace copy must use two separate nonnested directories')
    before = workspace_manifest(source)
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for relative, record in before.items():
        src, dst = source/relative, destination/relative
        if dst.is_symlink():
            raise ValueError(f'Checkpoint destination contains a symbolic link: {dst}')
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_file() or dst.stat().st_size != record['size_bytes'] or _sha(dst) != record['sha256']:
            shutil.copy2(src, dst)
            copied += 1
    if workspace_manifest(source) != before:
        raise ValueError('Native source workspace changed during preservation')
    for relative, record in before.items():
        path = destination/relative
        if path.stat().st_size != record['size_bytes'] or _sha(path) != record['sha256']:
            raise ValueError(f'Native checkpoint copy failed verification: {relative}')
    return before, copied


def read_native_f64(path):
    """Read libuipc's 16-byte buffer header and its exact float64 payload."""
    raw = Path(path).read_bytes()
    if len(raw) < 16:
        raise ValueError('Native buffer is shorter than its header')
    payload_bytes = struct.unpack('<Q', raw[8:16])[0]
    if payload_bytes != len(raw)-16 or payload_bytes % 8:
        raise ValueError('Native buffer length/header mismatch')
    return np.frombuffer(raw, dtype='<f8', offset=16).copy()


def verify_native_observation(workspace, native_frame, time_s, dt, substeps, bag, bear, origin):
    """Verify frame timing and canonical-to-physics positions, without writes."""
    if isinstance(substeps, bool) or int(substeps) != substeps or substeps < 1:
        raise ValueError('Invalid substep count for checkpoint timing')
    native_dt = float(dt)/int(substeps)
    if not np.isfinite([time_s, native_dt]).all() or native_dt <= 0:
        raise ValueError('Invalid native frame timing')
    expected_frame = int(round(float(time_s)/native_dt))
    if native_frame != expected_frame or abs(native_frame*native_dt-float(time_s)) > 1e-7:
        raise ValueError('Native frame and task simulation time disagree')
    workspace = Path(workspace)
    state_path = workspace/f'dump/common/sim_engine.cpp/state.{native_frame}.json'
    state = json.loads(state_path.read_text())
    if state.get('frame') != native_frame:
        raise ValueError('Native state marker has a different frame')
    native = read_native_f64(workspace/FEM_DIRECTORY/f'q.{native_frame}')
    frame = CoordinateFrame(tuple(origin))
    observed = {'bag': frame.points_to_physics(bag), 'bear': frame.points_to_physics(bear)}
    if any(p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() for p in observed.values()):
        raise ValueError('Current canonical soft observations are invalid')
    if native.size != sum(p.size for p in observed.values()) or not np.isfinite(native).all():
        raise ValueError('Native FEM size differs from the two observed soft bodies')
    native = native.reshape(-1, 3)
    # Native constitution ordering can differ from entity creation order. Both
    # complete-body orders are checked; vertices within a body are never shuffled.
    matches = []
    for order in (('bag', 'bear'), ('bear', 'bag')):
        expected = np.concatenate([observed[name] for name in order])
        error = float(np.max(np.abs(native-expected)))
        if error <= 1e-10:
            matches.append((order, error))
    if len(matches) != 1:
        raise ValueError('Native FEM q does not uniquely match canonical observations minus physics origin')
    for name in ('q_prev', 'q_v'):
        buffer = read_native_f64(workspace/FEM_DIRECTORY/f'{name}.{native_frame}')
        if buffer.size != native.size or not np.isfinite(buffer).all():
            raise ValueError(f'Native FEM {name} is missing or inconsistent')
    return {'verified': True, 'native_frame': int(native_frame), 'native_dt_s': native_dt,
        'observed_time_s': float(time_s), 'native_soft_body_order': list(matches[0][0]),
        'native_q_vs_canonical_minus_origin_max_error_m': matches[0][1],
        'position_comparison_tolerance_m': 1e-10,
        'coordinate_conversion': 'physics = canonical - origin; only roundoff tolerance allowed',
        'q_previous_and_velocity_buffers_present': True}


def _save_arrays(path, arrays):
    np.savez_compressed(path, **arrays)


def _controller_observation(exp, arrays):
    config = exp.config
    frame = CoordinateFrame.from_config(config)
    for name in ('bag', 'bear'):
        arrays[name] = frame.points_to_world(positions_no_grad(getattr(exp, name)))
    metadata = {'time_s': float(exp.time), 'physics_origin_m': list(frame.origin_m),
        'coordinates_of_bag_bear': 'canonical world',
        'coordinates_of_native_and_coupling_buffers': 'physics frame',
        'robot_names': {}, 'flags': {}, 'events': _json(getattr(exp, 'events', [])),
        'rotations_xyzw': {}, 'control_targets_from_last_completed_phase': {},
        'active_command_recorded': False, 'active_plan_recorded': False}
    for side, robot in exp.robots.items():
        arrays['q_'+side] = _array(robot.get_qpos()).reshape(-1)
        arrays['dq_'+side] = _array(robot.get_dofs_velocity()).reshape(-1)
        arrays['control_force_'+side] = _array(robot.get_dofs_control_force()).reshape(-1)
        arrays['kp_'+side] = _array(robot.get_dofs_kp()).reshape(-1)
        arrays['kv_'+side] = _array(robot.get_dofs_kv()).reshape(-1)
        low, high = robot.get_dofs_force_range()
        arrays['force_limit_low_'+side], arrays['force_limit_high_'+side] = _array(low), _array(high)
        metadata['robot_names'][side] = getattr(robot, 'name', side)
        metadata['control_targets_from_last_completed_phase'][side] = {
            'q': _json(exp.qs[side]), 'finger_half_opening_m': float(exp.finger[side])}
        metadata['rotations_xyzw'][side] = exp.rotations[side].as_quat().tolist()
    for name in ('must_hold_bag', 'must_hold_right_handle', 'must_hold_teddy', 'require_bear_below_left_hand'):
        metadata['flags'][name] = bool(getattr(exp, name, False))
    for name in ('left_grasp_material_ids', 'right_grasp_material_ids'):
        arrays[name] = _array(getattr(exp, name))
    for name, value in getattr(exp, 'rest', {}).items():
        arrays['rest_'+name] = _array(value)
    command = getattr(exp, 'checkpoint_last_command', None)
    if isinstance(command, dict):
        for quantity, per_side in command.items():
            for side, value in per_side.items():
                arrays[f'command_{quantity}_{side}'] = _array(value)
        metadata['active_command_recorded'] = True
    plan = getattr(exp, 'checkpoint_active_plan', None)
    if plan is not None:
        metadata['active_plan'] = {'duration_s': float(plan.duration),
            'finger_start': _json(plan.finger_start), 'finger_end': _json(plan.finger_end),
            'time_scaling': 'MotionPlan minimum_jerk(time/duration)'}
        for side, curve in plan.curves.items():
            arrays[f'active_spline_x_{side}'] = _array(curve.x)
            arrays[f'active_spline_c_{side}'] = _array(curve.c)
        metadata['active_plan_recorded'] = True
    metadata['phase'] = {'label': getattr(exp, 'checkpoint_phase_label', None),
        'elapsed_s': getattr(exp, 'checkpoint_phase_elapsed_s', None),
        'completed': getattr(exp, 'checkpoint_phase_completed', None),
        'latest_event': _json(exp.events[-1]) if getattr(exp, 'events', None) else None}
    for name in ('contact_guard', 'staged_hand_monitor'):
        obj = getattr(exp, name, None)
        if obj is not None:
            metadata[name+'_scalar_state'] = {}
            for key, value in vars(obj).items():
                if isinstance(value, np.ndarray):
                    arrays[f'{name}_{key}'] = value.copy()
                elif value is None or isinstance(value, (str, bool, int, float, dict, list, tuple)):
                    metadata[name+'_scalar_state'][key] = _json(value)
    return metadata


def _coupling_observation(coupler, arrays):
    data = coupler._coupling_data
    report = {'array_keys': [], 'links': [], 'additional_arrays': []}
    for key, value in vars(data).items():
        if isinstance(value, np.ndarray):
            arrays['coupling_'+key] = value.copy()
            report['array_keys'].append(key)
    for link in data.links:
        report['links'].append({'entity': link.entity.name, 'link': link.name, 'index': int(link.idx)})
    for link, values in getattr(coupler, '_abd_transforms_by_link', {}).items():
        key = 'abd_aim_transforms_link_'+str(link.idx)
        arrays[key] = _array(values)
        report['additional_arrays'].append(key)
    for link, values in getattr(coupler, '_abd_data_by_link', {}).items():
        for index, entry in enumerate(values):
            for field in ('transform', 'velocity'):
                key = f'abd_link_{link.idx}_env_{index}_{field}'
                arrays[key] = _array(getattr(entry, field))
                report['additional_arrays'].append(key)
    for entity, value in getattr(coupler, '_articulation_data_by_entity', {}).items():
        for key, item in vars(value).items():
            if isinstance(item, np.ndarray):
                name = f'articulation_{entity.name}_{key}'
                arrays[name] = item.copy()
                report['additional_arrays'].append(name)
    return report


def export_checkpoint(exp, label):
    """Best-effort evidence export at a completed step boundary; never restore.

    Returns success=False with errors if native/state/copy verification fails.
    Available controller observations remain preserved even if native dump fails.
    Optional rigid checkpoint failures are explicit limitations, not resume claims.
    """
    report = {'success': False, 'label': str(label), 'exact_resume_certified': False,
        'restore_implemented': False, 'simulation_advanced': False,
        'physical_state_mutated': False, 'errors': [], 'warnings': [],
        'created_utc': datetime.now(timezone.utc).isoformat()}
    record = None
    try:
        base = Path(exp.out)/'native_checkpoint'
        records = base/'records'
        records.mkdir(parents=True, exist_ok=True)
        safe_label = re.sub(r'[^A-Za-z0-9_.-]', '_', str(label))[:100] or 'checkpoint'
        record = Path(tempfile.mkdtemp(prefix=safe_label+'_', dir=records))
        report['record_directory'] = str(record)
        arrays = {'time_s': np.asarray(float(exp.time))}
        try:
            controller = _controller_observation(exp, arrays)
        finally:
            # Preserve whatever was read even if a later observation fails.
            _save_arrays(record/'controller_state.npz', arrays)
        (record/'controller.json').write_text(json.dumps(_json(controller), indent=2)+'\n')
        (record/'config.json').write_text(json.dumps(_json(exp.config), indent=2)+'\n')
        c = exp.scene.sim.coupler
        native_frame = int(c._ipc_world.frame())
        report.update(native_frame=native_frame, time_s=float(exp.time),
            controller_active_command_recorded=controller['active_command_recorded'],
            controller_active_plan_recorded=controller['active_plan_recorded'])
        if not controller['active_command_recorded']:
            report['warnings'].append('No active command sample was provided; last completed phase targets are not assumed current.')
        coupling_arrays = {}
        report['coupling'] = _coupling_observation(c, coupling_arrays)
        _save_arrays(record/'coupling_arrays.npz', coupling_arrays)
        try:
            checkpoint = c.rigid_solver.__getstate__()
            rigid_arrays = {key: _array(value) for key, value in checkpoint.arrays.items()}
            _save_arrays(record/'rigid_solver_arrays.npz', rigid_arrays)
            (record/'rigid_solver_configs.json').write_text(json.dumps(_json(checkpoint.configs), indent=2)+'\n')
            report['rigid_solver'] = {'exported': True, 'array_count': len(rigid_arrays),
                'array_names': sorted(rigid_arrays), 'restoration_tested': False,
                'scope': 'Installed solver checkpoint arrays; includes available declared state/scratch buffers, not a whole-scene resume certificate'}
        except Exception as exc:
            report['rigid_solver'] = {'exported': False, 'error': str(exc)}
            report['warnings'].append('Rigid solver checkpoint arrays unavailable; exact whole-scene state is incomplete.')
        simulator = exp.scene.sim
        clocks = {'cur_substep_global': int(getattr(simulator, '_cur_substep_global', -1))}
        if hasattr(simulator, '_steps'):
            _save_arrays(record/'simulator_clocks.npz', {'steps': _array(simulator._steps)})
        (record/'simulator_clocks.json').write_text(json.dumps(clocks, indent=2)+'\n')
        if not c._ipc_world.dump():
            raise ValueError('Native World.dump returned false')
        source = Path(c._ipc_engine.workspace()).resolve()
        report['native_source_workspace'] = str(source)
        report['native_observation_verification'] = verify_native_observation(
            source, native_frame, exp.time, exp.config['simulation']['dt'],
            exp.config['simulation'].get('substeps', 1), arrays['bag'], arrays['bear'],
            controller['physics_origin_m'])
        manifest, count = copy_workspace_verified(source, base/'workspace')
        if int(c._ipc_world.frame()) != native_frame or float(exp.time) != report['time_s']:
            raise ValueError('Simulation advanced during checkpoint export')
        (record/'native_workspace_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        report.update(native_workspace_copy=str((base/'workspace').resolve()),
            native_workspace_file_count=len(manifest), native_files_copied=count,
            native_workspace_manifest_sha256=_sha(record/'native_workspace_manifest.json'))
        archive = Path(exp.out)/'source'
        if archive.is_dir():
            (record/'source_manifest.json').write_text(json.dumps(workspace_manifest(archive), indent=2)+'\n')
        for filename in ('input_sha256.json', 'rest.npz'):
            path = Path(exp.out)/filename
            if path.is_file():
                report.setdefault('run_reference_files', {})[filename] = {'sha256': _sha(path), 'path': str(path.resolve())}
        report['success'] = True
        report['status'] = 'native_and_observation_export_verified_no_restore'
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
        report['status'] = 'export_incomplete'
    report['limitations'] = [
        'No restoration or resumed-step equivalence test was performed.',
        'Native dump + robot/controller/coupling observations are preservation evidence, not a certified exact restart.',
        'Undeclared native/Genesis caches, callbacks, sensor state and Python controller state may remain unexported.',
        'Archived inputs support replay from the initial state; replay is distinct from resuming this midpoint.']
    if record is not None:
        try:
            report['record_file_manifest'] = workspace_manifest(record)
            (record/'report.json').write_text(json.dumps(_json(report), indent=2)+'\n')
            # Latest is a small pointer; prior records and native frame files remain.
            pointer = record.parents[1]/'latest.json'
            pointer.write_text(json.dumps(_json(report), indent=2)+'\n')
        except Exception as exc:
            report['success'] = False
            report['errors'].append('Cannot persist final exporter report: '+str(exc))
    return report
