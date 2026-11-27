"""Release-only diagnostic from an exported native state; no full-task promotion.

The archived failed trial remains unchanged. State readback is verified, but
bitwise equivalence to an uninterrupted coupled simulation is not certified.
"""
from __future__ import annotations
import argparse
import json
import shutil
import signal
import time
import traceback
from pathlib import Path
import numpy as np
from run import Experiment, HERE, write_json
from scene import read_config, positions
from physical_monitor import tetra_volumes
from continuation_checkpoint import read_native_f64, FEM_DIRECTORY


class ReleaseExperiment(Experiment):
    def __init__(self, config, args):
        self.restoring = True
        super().__init__(config, args)
        self.source = Path(args.source).resolve()
        old = json.loads((self.source/'report.json').read_text())
        checkpoint = old['latest_checkpoint']
        if not checkpoint['success']:
            raise RuntimeError('Source checkpoint export failed')
        record = Path(checkpoint['record_directory'])
        self.saved = np.load(record/'controller_state.npz')
        rest = np.load(self.source/'rest.npz')
        for name in self.rest:
            if not np.array_equal(self.rest[name], rest[name]):
                raise RuntimeError('Rest mesh/order mismatch: '+name)
        coupling = self.scene.sim.coupler
        workspace = Path(coupling._ipc_engine.workspace())
        source_workspace = self.source/'native_checkpoint/workspace'
        shutil.copytree(source_workspace/'dump', workspace/'dump', dirs_exist_ok=True)
        frame = int(checkpoint['native_frame'])
        if (not coupling._ipc_world.recover(frame) or coupling._ipc_world.frame()!=frame
                or not coupling._ipc_world.is_valid()):
            raise RuntimeError('Native recovery failed')
        coupling._ipc_world.retrieve()
        coupling._retrieve_fem_states()
        from restore_rigid_state import restore_rigid_state
        restoration = restore_rigid_state(self, record)
        write_json(self.out/'restoration.json',restoration)
        if not restoration['success']:
            raise RuntimeError('Rigid/coupling restore failed: '+str(restoration['errors']))
        # Compare actual recovered native ABD states, not just Python mirrors.
        coupling._retrieve_rigid_states()
        saved_coupling=np.load(record/'coupling_arrays.npz')
        abd_errors={}
        for link,entries in coupling._abd_data_by_link.items():
            for i,entry in enumerate(entries):
                for field in ('transform','velocity'):
                    key=f'abd_link_{link.idx}_env_{i}_{field}'
                    abd_errors[key]=float(np.max(np.abs(getattr(entry,field)-saved_coupling[key])))
        if max(abd_errors.values())>1e-12:
            raise RuntimeError('Native ABD state readback differs: '+str(abd_errors))
        restoration['native_abd_max_error']=max(abd_errors.values())
        self.time = float(self.saved['time_s'])
        errors = {name:float(np.max(np.abs(positions(getattr(self,name))-self.saved[name])))
                  for name in ('bag','bear')}
        if max(errors.values())>1e-10:
            raise RuntimeError('Recovered soft positions disagree: '+str(errors))
        # Re-dump without advancing and compare native position/velocity histories.
        if not coupling._ipc_world.dump():
            raise RuntimeError('Recovery verification dump failed')
        native_errors = {}
        for name in ('q','q_prev','q_v'):
            a=read_native_f64(source_workspace/FEM_DIRECTORY/f'{name}.{frame}')
            b=read_native_f64(workspace/FEM_DIRECTORY/f'{name}.{frame}')
            native_errors[name]=float(np.max(np.abs(a-b)))
        if max(native_errors.values())>1e-12:
            raise RuntimeError('Recovered native history differs: '+str(native_errors))
        self.start_time=self.time
        self.previous_bear=positions(self.bear).copy()
        self.baseline_ratios=tetra_volumes(self.previous_bear,self.monitor.tets)/self.monitor.rest_volumes
        self.inherited_bad=np.flatnonzero(self.baseline_ratios<.05)
        if np.any(self.baseline_ratios<=0) or len(self.inherited_bad)!=1:
            raise RuntimeError('Unexpected inherited failure geometry')
        self.first_recovered_time=None
        self.minimum_release_ratio=float(self.baseline_ratios.min())
        self.must_hold_bag=True
        self.must_hold_teddy=False
        self.restoring=False
        self.qs={s:self.saved['command_q_'+s].copy() for s in self.robots}
        self.finger={'left':float(self.saved['command_finger_left']),
                     'right':float(self.saved['q_right'][7:9].mean())}
        self.rotations={s:self.planner.fk(s,self.qs[s])[1] for s in self.robots}
        self.report.update(requested_task='release_saved', task_passed=False,
            source_failed_run=str(self.source), source_task_passed=old['task_passed'],
            source_error=old['error'], resumed_from_simulation_time_s=self.time,
            restoration={**restoration,'soft_position_max_errors_m':errors,
                'native_fem_history_max_errors':native_errors,
                'native_frame':frame,'state_readback_verified':True,
                'uninterrupted_step_equivalence_certified':False},
            physical_object_motion='Native checkpoint initialization, then native FEM/contact only',
            release_exception={'scope':'Only preexisting collapsed tetrahedron during first 0.8 s of release',
                'tetrahedron_indices':self.inherited_bad.tolist(),
                'initial_ratios':self.baseline_ratios[self.inherited_bad].tolist(),
                'strictly_positive_required':True,'all_other_tetrahedra_minimum_ratio':.05,
                'original_full_task_failure_preserved':True})
        if args.direct_release:
            self.report['release_exception']={
                'scope':'User requested direct release without further quality testing',
                'quality_thresholds_observed_only':True,
                'original_full_task_failure_preserved':True,
                'physically_validated_continuation':False}
        self.events=[]
        self.phase='restored_before_release'
        self.capture(self.phase)
        self.save(self.phase,native=True)
        write_json(self.out/'restoration.json',self.report['restoration'])
        print('RESTORED '+json.dumps(self.report['restoration']),flush=True)

    def capture(self,label=None):
        if not self.restoring: super().capture(label)

    def save(self,label,native=False):
        if not self.restoring: super().save(label,native)

    def observe(self):
        bag,bear=positions(self.bag),positions(self.bear)
        row=self.monitor.measure(self.time,bag,bear); self.metrics.append(row)
        speed=float(np.linalg.norm(bear-self.previous_bear,axis=1).mean()/self.dt)
        self.bear_speed_tail.append(speed);self.bear_speed_tail=self.bear_speed_tail[-100:]
        self.previous_bear=bear.copy()
        ratio=tetra_volumes(bear,self.monitor.tets)/self.monitor.rest_volumes
        self.minimum_release_ratio=min(self.minimum_release_ratio,float(ratio.min()))
        if self.first_recovered_time is None and ratio.min()>=.05:
            self.first_recovered_time=self.time
        invalid=ratio<.05
        if self.time-self.start_time<=.8+1e-9 and self.first_recovered_time is None:
            invalid[self.inherited_bad]=False
        self.report['release_observed_inverted_tetrahedra_max']=max(
            self.report.get('release_observed_inverted_tetrahedra_max',0),int(np.count_nonzero(ratio<=0)))
        if (not row[5] or (not self.args.direct_release and (np.any(ratio<=0) or invalid.any()
                or max(row[2:4])>.002 or row[4]>1.12))):
            raise RuntimeError('Release physical gate failed: '+str(row))
        data=self.scene.sim.coupler._coupling_data
        grip={'time_s':self.time}
        for side in self.robots:
            ids=[i for i,l in enumerate(data.links) if l.entity.name==side+'_franka'
                 and l.name in ('left_finger','right_finger')]
            force=np.einsum('ij,ij->i',data.out_forces[0,ids],data.aim_transforms[0,ids,:3,1])
            points=bag[self.left_grasp_material_ids] if side=='left' else bear
            distance=float(np.linalg.norm(points-self.tcp(side)[0],axis=1).min())
            good=bool(np.isfinite(force).all() and len(force)==2 and force.min()>=.05)
            self.grip_dropout_seconds[side]=0. if good else self.grip_dropout_seconds[side]+self.dt
            grip[side]={'required':side=='left','pad_opening_reaction_n':force.tolist(),
                        'nearest_material_m':distance,'two_sided_force_proxy_pass':good}
            if not self.args.direct_release and side=='left' and (distance>.025 or self.grip_dropout_seconds[side]>=.075-1e-12):
                raise RuntimeError('Left handle support lost during release: '+str(grip))
        self.grip_trace.append(grip)

    def run_release(self):
        # Brake the residual arm target velocities over 0.25 s with continuous
        # velocity/acceleration. Right fingers start at their measured opening.
        q0={s:self.saved['command_q_'+s].copy() for s in self.robots}
        v0={s:self.saved['command_qd_'+s].copy() for s in self.robots}
        f0=self.finger['right']; duration=.8; brake=.25
        if self.args.direct_release:
            self.set_grip_force('right',self.config['robots']['finger_force_limit'])
            self.robots['right'].set_dofs_kp([self.config['robots']['finger_kp']]*2,dofs_idx_local=[7,8])
            self.report['direct_opening_controller']={
                'target_half_opening_m':.04,'force_limit_n':self.config['robots']['finger_force_limit'],
                'finger_kp':self.config['robots']['finger_kp'],'finger_kv':200.}
        self.phase='release_right_gripper';self.checkpoint_phase_label=self.phase
        self.checkpoint_phase_completed=False;self.checkpoint_active_plan=None
        for t in np.arange(self.dt,duration+self.dt/2,self.dt):
            u=min(t/brake,1.); x=brake*(u-u**3+.5*u**4); dx=1-3*u*u+2*u**3
            s=min(t/duration,1.); h=10*s**3-15*s**4+6*s**5
            dh=(30*s*s-60*s**3+30*s**4)/duration
            command={'q':{k:q0[k]+v0[k]*x for k in q0},
                'qd':{k:v0[k]*dx for k in q0},
                'finger':{'left':self.finger['left'],'right':f0+(.04-f0)*h},
                'finger_velocity':{'left':0.,'right':(.04-f0)*dh}}
            if self.args.direct_release:
                command['finger']['right']=.04
                command['finger_velocity']['right']=0.
            self.checkpoint_phase_elapsed_s=float(t);self.step(command)
        self.qs={s:command['q'][s].copy() for s in self.robots}
        self.finger=command['finger'].copy()
        self.checkpoint_phase_completed=True
        self.events.append({'phase':self.phase,'time_s':self.time})
        self.capture(self.phase);self.save(self.phase,native=True)
        self.report['release_action_completed']=True
        self.hold(3.,'observe_after_release')
        from validate_physics import actual_bag_containment,bag_pose
        bag=positions(self.bag)[self.source_to_runtime];bear=positions(self.bear)
        actual=actual_bag_containment(bear,bag,self.asset['faces'][self.asset['body_face_indices']],self.asset['rim'])
        q=self.robots['right'].get_qpos().detach().cpu().numpy().reshape(-1)
        self.report.update(actual_containment=actual,
            final_bag_pose=bag_pose(self.asset['points_upright'],bag,self.asset['body']),
            actual_final_right_fingers_m=q[7:9].tolist(),
            right_tcp_to_bear_nearest_m=float(np.linalg.norm(bear-self.tcp('right')[0],axis=1).min()),
            status='release_diagnostic_completed',task_passed=False)

    def finish(self,error=None):
        self.report.update(release_minimum_tetrahedron_ratio=getattr(self,'minimum_release_ratio',None),
            inherited_compression_recovered_at_s=getattr(self,'first_recovered_time',None))
        super().finish(error)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default=str(HERE/'runs/paper_09_full_authorized'))
    p.add_argument('--output',default='paper_10_release_saved')
    p.add_argument('--wall-seconds',type=float,default=900)
    p.add_argument('--direct-release',action='store_true')
    p.add_argument('--viewer',action='store_true');p.add_argument('--width',type=int,default=800)
    p.add_argument('--height',type=int,default=600)
    a=p.parse_args();a.task='release_saved'
    config=read_config(Path(a.source)/'config.json')
    def stop(sig,frame): raise TimeoutError('External stop '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    exp=None
    try:
        exp=ReleaseExperiment(config,a);exp.run_release()
    except BaseException as exc:
        if exp is not None:exp.finish(exc)
        else:
            out=HERE/'runs'/a.output;out.mkdir(exist_ok=True,parents=True)
            write_json(out/'initialization_failure.json',{'error':repr(exc),'traceback':traceback.format_exc()})
            print(traceback.format_exc(),flush=True)
        return 2
    exp.finish();return 0

if __name__=='__main__':raise SystemExit(main())
