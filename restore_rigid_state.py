"""Restore exported rigid/coupling evidence after native IPC recovery, without stepping.

This is a scene-specific, checked restart helper. It does not certify exact
whole-scene restart or restore a Python task controller. The original exporter
did not save the rigid forward-kinematics flags; their end-step values are
inferred from the audited Genesis 1.4.3 integration path and reported explicitly.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def restore_rigid_state(exp, source_record):
    """Restore a same-build exp from source_record after IPC world.recover/retrieve.

    Returns a JSON-serializable report. Caller MUST require success=True before
    simulation continues. No scene.step(), native recover(), reset(), or force
    reapplication occurs here. All writes are initialization of the resumed run.
    exp.config, exp.scene and exp.robots must already represent the frozen scene.
    """
    report = {'success': False, 'source_record': str(Path(source_record).resolve()),
        'exact_resume_certified': False, 'simulation_advanced': False,
        'native_recovery_performed_here': False, 'coupling_forces_reapplied': False,
        'errors': [], 'warnings': [], 'writes_started': False,
        'limitations': ['Rigid FK flags inferred from completed Genesis IPC rigid-post-coupling substep; not exported originally',
            'No resumed-step equivalence comparison against the original uninterrupted process',
            'Python task/controller objects and unexported caches are not restored by this helper']}
    try:
        import torch
        from genesis.engine.states import KinematicSolverCheckpoint

        record = Path(source_record)
        source_report = json.loads((record / 'report.json').read_text())
        controller = json.loads((record / 'controller.json').read_text())
        source_config = json.loads((record / 'config.json').read_text())
        if source_config != exp.config:
            raise ValueError('Rebuilt scene config differs from the frozen checkpoint config')
        if version('genesis-world') != '1.4.3' or version('pyuipc') != '0.0.29':
            raise ValueError('Restart helper requires audited Genesis 1.4.3 / pyuipc 0.0.29')
        if not source_report.get('success') or not source_report.get('rigid_solver', {}).get('exported'):
            raise ValueError('The source record does not report a verified native and rigid export')
        filenames = ['rigid_solver_arrays.npz', 'rigid_solver_configs.json', 'coupling_arrays.npz',
            'controller_state.npz', 'controller.json', 'config.json', 'simulator_clocks.json', 'simulator_clocks.npz']
        for name in filenames:
            expected = source_report['record_file_manifest'].get(name)
            if not expected or _sha(record / name) != expected['sha256']:
                raise ValueError('Checkpoint file hash mismatch: ' + name)
        rigid = dict(np.load(record / 'rigid_solver_arrays.npz', allow_pickle=False))
        coupling = dict(np.load(record / 'coupling_arrays.npz', allow_pickle=False))
        observed = dict(np.load(record / 'controller_state.npz', allow_pickle=False))
        clocks = json.loads((record / 'simulator_clocks.json').read_text())
        clock_arrays = dict(np.load(record / 'simulator_clocks.npz', allow_pickle=False))
        scene, sim = exp.scene, exp.scene.sim
        solver, coupler = sim.rigid_solver, sim.coupler
        if not scene.is_built or sim.requires_grad or sim.substeps_local != 1:
            raise ValueError('Helper requires a built, forward-only scene with local frame count one')
        if int(coupler._ipc_world.frame()) != int(source_report['native_frame']):
            raise ValueError('Caller must recover native IPC to the exported frame before rigid restoration')
        if int(clocks['cur_substep_global']) != int(source_report['native_frame']):
            raise ValueError('Saved simulator clock and native frame disagree')
        fresh = solver.__getstate__()
        if set(rigid) != set(fresh.arrays):
            raise ValueError('Rebuilt rigid array names differ from the checkpoint')
        for name, value in rigid.items():
            target = _array(fresh.arrays[name])
            if value.shape != target.shape or value.dtype.kind != target.dtype.kind:
                raise ValueError('Rigid array shape/type differs: ' + name)
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite saved rigid array: ' + name)
        if int(source_report['rigid_solver']['array_count']) != len(rigid):
            raise ValueError('Saved rigid array count mismatch')
        actual_links = [{'entity': link.entity.name, 'link': link.name, 'index': int(link.idx)}
            for link in coupler._coupling_data.links]
        if actual_links != source_report['coupling']['links']:
            raise ValueError('Rebuilt coupling links/order differ from the frozen scene')
        # Collect and validate every coupling destination before any restoration.
        writes, matched = [], set()
        for key, value in vars(coupler._coupling_data).items():
            if isinstance(value, np.ndarray):
                name = 'coupling_' + key
                if name not in coupling:
                    raise ValueError('Missing coupling array: ' + name)
                writes.append((name, value, coupling[name])); matched.add(name)
        for link, values in coupler._abd_transforms_by_link.items():
            name = 'abd_aim_transforms_link_' + str(link.idx)
            if name not in coupling:
                raise ValueError('Missing ABD target: ' + name)
            value = coupling[name]
            if value.shape != np.asarray(values).shape:
                raise ValueError('ABD target shape mismatch: ' + name)
            for env_idx, destination in enumerate(values):
                writes.append((name + ':' + str(env_idx), destination, value[env_idx]))
            matched.add(name)
        for link, values in coupler._abd_data_by_link.items():
            for env_idx, entry in enumerate(values):
                for field in ('transform', 'velocity'):
                    name = f'abd_link_{link.idx}_env_{env_idx}_{field}'
                    if name not in coupling:
                        raise ValueError('Missing native ABD mirror: ' + name)
                    writes.append((name, getattr(entry, field), coupling[name])); matched.add(name)
        for entity, data in coupler._articulation_data_by_entity.items():
            for key, value in vars(data).items():
                if isinstance(value, np.ndarray):
                    name = f'articulation_{entity.name}_{key}'
                    if name not in coupling:
                        raise ValueError('Missing articulation mirror: ' + name)
                    writes.append((name, value, coupling[name])); matched.add(name)
        if matched != set(coupling):
            raise ValueError('Coupling arrays not matched to the rebuilt scene: ' + str(sorted(set(coupling) ^ matched)))
        for name, destination, value in writes:
            if np.asarray(destination).shape != value.shape or not np.isfinite(value).all():
                raise ValueError('Invalid coupling restoration buffer: ' + name)
        if clock_arrays['steps'].shape != tuple(sim._steps.shape):
            raise ValueError('Simulator per-environment clock shape mismatch')
        # In this exact engine, IPC rigid substep_post_coupling ends in substep(),
        # which assigns both flags to not _enable_mujoco_compatibility. No scene
        # reset or set_qpos is used here; saved derived arrays remain unchanged.
        fk_flag = not bool(solver._enable_mujoco_compatibility)
        report['fk_flags'] = {'position': fk_flag, 'velocity': fk_flag,
            'source': 'inferred from Genesis rigid_solver.substep final assignment',
            'original_flags_exported': False, 'mujoco_compatibility': bool(solver._enable_mujoco_compatibility)}
        restored = KinematicSolverCheckpoint(arrays=rigid, configs=fresh.configs, kinds=fresh.kinds,
            is_forward_pos_updated=fk_flag, is_forward_vel_updated=fk_flag)
        report['writes_started'] = True
        solver.__setstate__(restored)
        for name, destination, value in writes:
            destination[:] = value
        sim._steps[:] = torch.as_tensor(clock_arrays['steps'], dtype=sim._steps.dtype, device=sim._steps.device)
        sim._cur_substep_global = int(clocks['cur_substep_global'])
        torch.cuda.synchronize()
        reread = solver.__getstate__()
        mismatches = [name for name in rigid if not np.array_equal(_array(reread.arrays[name]), rigid[name])]
        if mismatches:
            raise ValueError('Rigid array readback differs: ' + str(mismatches))
        if any(not np.array_equal(np.asarray(destination), value) for _, destination, value in writes):
            raise ValueError('Coupling array readback differs')
        per_robot = {}
        for side, robot in exp.robots.items():
            q = _array(robot.get_qpos()).reshape(-1)
            dq = _array(robot.get_dofs_velocity()).reshape(-1)
            q_error = float(np.max(np.abs(q - observed['q_' + side].reshape(-1))))
            dq_error = float(np.max(np.abs(dq - observed['dq_' + side].reshape(-1))))
            # Full rigid INFO arrays contain gains and force bounds. Verify the
            # public API sees those exact values; silently changing them would
            # hide an incomplete state restore, so mismatch is a hard failure.
            low, high = robot.get_dofs_force_range()
            quantities = {'kp': robot.get_dofs_kp(), 'kv': robot.get_dofs_kv(),
                'force_limit_low': low, 'force_limit_high': high}
            control_errors = {key: float(np.max(np.abs(_array(value).reshape(-1)
                - observed[key + '_' + side].reshape(-1)))) for key, value in quantities.items()}
            if max([q_error, dq_error, *control_errors.values()]) > 1e-12:
                raise ValueError(f'Robot {side} public state/gains/limits do not match the saved observation')
            per_robot[side] = {'q_max_error_rad': q_error, 'dq_max_error_rad_s': dq_error,
                'gain_and_force_limit_errors': control_errors}
        if not np.array_equal(_array(sim._steps), clock_arrays['steps']):
            raise ValueError('Simulator clock readback differs')
        report.update(success=True, rigid_array_count=len(rigid), coupling_array_count=len(coupling),
            restored_native_frame=int(coupler._ipc_world.frame()), restored_global_substep=sim._cur_substep_global,
            per_environment_steps=_array(sim._steps).tolist(), robot_readback=per_robot,
            rigid_array_exact_readback=True, coupling_array_exact_readback=True,
            frozen_config_matches=True, file_hashes_verified=filenames,
            static_config_source='Fresh identical-config scene; original static objects were exported only as repr')
    except Exception as exc:
        report['errors'].append(type(exc).__name__ + ': ' + str(exc))
    return report
