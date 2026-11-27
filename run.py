"""GPU paper-bag manipulation. All motion is finite-force joint control.

No soft pins, grasp welds, state teleports, or prescribed object poses.
Checkpoints preserve evidence and support replay; exact midpoint restore is not claimed.
"""
from __future__ import annotations
import argparse, json, shutil, signal, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from motion import Planner, MotionPlan, DOWN, TCP
from scene import build, positions, read_config, physics_metadata
from physical_monitor import PhysicalMonitor
from runtime_diagnostics import RuntimeDiagnostics
from contact_grasp import cloth_grasp_frame
from continuation_checkpoint import export_checkpoint

HERE=Path(__file__).resolve().parent

def serial(o):
    if isinstance(o,np.ndarray): return o.tolist()
    if isinstance(o,np.generic): return o.item()
    if isinstance(o,Path): return str(o)
    raise TypeError(type(o).__name__)

def write_json(path,data):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(data,default=serial,indent=2)+'\n');temp.replace(path)

class Experiment:
    def __init__(self,c,args):
        self.config,self.args=c,args
        self.out=HERE/'runs'/(args.output or datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.out.mkdir(parents=True,exist_ok=False)
        write_json(self.out/'config.json',c)
        source=self.out/'source';source.mkdir()
        for p in HERE.glob('*.py'): shutil.copy2(p,source/p.name)
        shutil.copy2(HERE/'run.sh',source/'run.sh')
        shutil.copytree(HERE/'assets',source/'assets',symlinks=False)
        shutil.copytree((HERE/'vendor').resolve(),source/'vendor',symlinks=False,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo'))
        self.wall=time.monotonic();self.time=0.;self.events=[];self.phase='build'
        self.report={'status':'building','task_passed':False,'requested_task':args.task,
            'exact_midpoint_resume_implemented':False,'physical_object_motion':'native FEM/contact only'}
        self.planner=Planner(c);self.finger={s:.04 for s in ('left','right')}
        self.rotations={s:DOWN for s in self.finger}
        self.qs={s:self.planner.home.copy() for s in self.finger}
        # Initial robot pose is established before native IPC initialization.
        initial=c.get('initial_robot_targets',{})
        for s,item in initial.items():
            self.rotations[s]=Rotation.from_quat(item['rotation_xyzw'])
            q,e,a=self.planner.solve(s,item['position'],self.rotations[s])
            if e>.0005 or a>.002: raise RuntimeError('Unreachable initial pose')
            self.qs[s]=q
        self.scene,self.robots,self.bag,self.bear,self.cameras=build(c,viewer=args.viewer,
            width=args.width,height=args.height,initial_q={s:np.r_[q,.04,.04] for s,q in self.qs.items()})
        self.rest={'bag':positions(self.bag),'bear':positions(self.bear),
            'bag_faces':np.asarray(self.bag.elems),'bear_tets':np.asarray(self.bear.elems)}
        np.savez_compressed(self.out/'rest.npz',**self.rest)
        self.asset=np.load(HERE/'assets/paper_bag.npz')
        err,self.source_to_runtime=cKDTree(self.rest['bag']).query(self.asset['points']+c['bag']['position'])
        if err.max()>1e-6: raise RuntimeError('Paper mesh correspondence lost')
        self.left_grasp_material_ids=self.source_to_runtime[self.asset['grasp_upper']]
        self.right_grasp_material_ids=self.source_to_runtime[self.asset['grasp_lower']]
        self.runtime=RuntimeDiagnostics(self.scene,every_steps=20,jsonl_path=self.out/'runtime.jsonl')
        self.monitor=PhysicalMonitor(self.rest,c)
        self.metrics=[];self.trace=[];self.grip_trace=[];self.last_saved=-1.;self.last_frame=-1.
        self.last_progress=time.monotonic();self.dt=c['simulation']['dt']
        self.must_hold_bag=False;self.must_hold_teddy=False
        self.grip_dropout_seconds={"left":0.,"right":0.}
        self.previous_bear=self.rest['bear'].copy();self.bear_speed_tail=[]
        self.actual_max_joint_speed=0.;self.actual_min_rigid_clearance=.2
        self.writer=imageio.get_writer(self.out/'simulation.mp4',fps=10,codec='libx264',quality=7)
        self.checkpoint_last_command=None;self.checkpoint_active_plan=None
        self.report.update(status='running',physics=physics_metadata(c,self.bag,self.bear),
            material_readback=self.scene.teddy_shell_report,initialization=self.scene.teddy_initialization_report,
            native_solver=self.scene.plastic_tuning_report,rigid_contact=self.scene.teddy_rigid_contact_report)
        write_json(self.out/'scene_build.json',self.report)
        self.capture('initial');self.save('initial',native=True)

    def tcp(self,side):
        from coordinate_frame import to_world
        link=self.robots[side].get_link('hand')
        pos=link.get_pos().detach().cpu().numpy().reshape(3)
        quat=link.get_quat().detach().cpu().numpy().reshape(4)
        rot=Rotation.from_quat(quat[[1,2,3,0]])
        return to_world(pos+rot.apply(TCP),self.scene),rot

    def capture(self,label=None):
        rgb=self.cameras['overview'].render()[0]
        self.writer.append_data(rgb);self.last_frame=self.time
        if label:
            imageio.imwrite(self.out/(label+'.png'),rgb)
            imageio.imwrite(self.out/(label+'_detail.png'),self.cameras['detail'].render()[0])

    def save(self,label,native=False):
        arrays={'bag':positions(self.bag),'bear':positions(self.bear),'time':self.time}
        for s,r in self.robots.items():
            arrays['q_'+s]=r.get_qpos().detach().cpu().numpy()
            arrays['dq_'+s]=r.get_dofs_velocity().detach().cpu().numpy()
        np.savez_compressed(self.out/'latest_state.npz',**arrays)
        write_json(self.out/'progress.json',{'phase':self.phase,'simulation_time_s':self.time,
            'wall_seconds':time.monotonic()-self.wall,'events':self.events})
        if native:
            self.report['latest_checkpoint']=export_checkpoint(self,label)

    def set_grip_force(self,side,force):
        lim=np.array([87,87,87,87,12,12,12,force,force])
        self.robots[side].set_dofs_force_range(-lim,lim)

    def observe(self):
        bag,bear=positions(self.bag),positions(self.bear)
        row=self.monitor.measure(self.time,bag,bear);self.metrics.append(row)
        self.bear_speed_tail.append(float(np.linalg.norm(bear-self.previous_bear,axis=1).mean()/self.dt))
        self.bear_speed_tail=self.bear_speed_tail[-100:];self.previous_bear=bear.copy()
        if not row[5] or row[1]<.05 or max(row[2:4])>.002 or row[4]>1.12:
            raise RuntimeError('Physical gate failed [time,volume,bag penetration,bear penetration,stretch,finite] '+str(row))
        grip={'time_s':self.time}
        data=self.scene.sim.coupler._coupling_data
        grip_errors=[]
        for s,required in [('left',self.must_hold_bag),('right',self.must_hold_teddy)]:
            ids=[i for i,l in enumerate(data.links) if l.entity.name==s+'_franka' and l.name in ('left_finger','right_finger')]
            forces=np.einsum('ij,ij->i',data.out_forces[0,ids],data.aim_transforms[0,ids,:3,1])
            points=bag[self.left_grasp_material_ids] if s=='left' else bear
            distance=float(np.linalg.norm(points-self.tcp(s)[0],axis=1).min())
            distance_limit=.025 if s=='left' else .04
            finite_force=bool(len(forces)==2 and np.isfinite(forces).all())
            force_ok=bool(finite_force and np.min(forces)>=.05)
            self.grip_dropout_seconds[s]=(self.grip_dropout_seconds[s]+self.dt
                if required and not force_ok else 0.)
            grip[s]={'required':required,'pad_opening_reaction_n':forces.tolist(),
                'nearest_material_m':distance,'distance_limit_m':distance_limit,
                'two_sided_force_proxy_pass':force_ok,
                'consecutive_force_dropout_s':self.grip_dropout_seconds[s],
                'force_dropout_limit_s':.075}
            # A one-step STC reaction dip need not mean the material escaped.
            # Distance failure is immediate; sustained force loss is bounded
            # to 75 ms and retained in the raw trace, including the failing row.
            if required and (not np.isfinite(distance) or distance>distance_limit):
                grip_errors.append(f'{s} material left original grasp distance: {grip[s]}')
            elif required and (not finite_force or self.grip_dropout_seconds[s]>=.075-1e-12):
                grip_errors.append(f'{s} sustained loss of two-sided force proxy: {grip[s]}')
        self.grip_trace.append(grip)
        if grip_errors: raise RuntimeError('; '.join(grip_errors))

    def step(self,command):
        if time.monotonic()-self.wall>self.args.wall_seconds: raise TimeoutError('User computation budget reached')
        self.checkpoint_last_command=command
        for s,r in self.robots.items():
            q=np.r_[command['q'][s],command['finger'][s],command['finger'][s]]
            dq=np.r_[command['qd'][s],command['finger_velocity'][s],command['finger_velocity'][s]]
            r.control_dofs_position_velocity(q,dq)
        t=time.monotonic();self.scene.step();self.time+=self.dt
        self.runtime.sample(self.time,step_wall_seconds=time.monotonic()-t)
        if self.runtime.native_convergence_failure_count or self.runtime.error_seen:
            raise RuntimeError('Native IPC solve failed or engine reported an error')
        self.observe()
        qs={s:r.get_qpos().detach().cpu().numpy().reshape(-1) for s,r in self.robots.items()}
        speed=max(float(r.get_dofs_velocity().detach().abs()[:7].max()) for r in self.robots.values())
        self.actual_max_joint_speed=max(self.actual_max_joint_speed,speed)
        if speed>2.175: raise RuntimeError('Measured arm joint speed exceeds conservative Franka limit')
        self.trace.append(np.r_[self.time,*qs['left'],*qs['right']])
        if self.time-self.last_frame>=.099:
            checks=self.planner.clearance({s:q[:7] for s,q in qs.items()},{s:float(q[7:9].mean()) for s,q in qs.items()})
            gap=min(x['distance'] for x in checks.values())
            self.actual_min_rigid_clearance=min(self.actual_min_rigid_clearance,gap)
            if gap<-.002: raise RuntimeError('Measured robot geometry penetration: '+str(checks))
            self.capture()
        if self.time-self.last_saved>=1.:
            self.save('periodic');self.last_saved=self.time
        if time.monotonic()-self.last_progress>15:
            print(json.dumps({'phase':self.phase,'sim_s':round(self.time,3),'wall_s':round(time.monotonic()-self.wall,1)}),flush=True)
            self.last_progress=time.monotonic()

    def execute(self,plan,label):
        self.phase=label;self.checkpoint_phase_label=label;self.checkpoint_phase_elapsed_s=0.
        self.checkpoint_phase_completed=False;self.checkpoint_active_plan=plan
        print('PHASE '+label+' duration='+str(plan.duration),flush=True)
        for t in np.arange(self.dt,plan.duration+self.dt/2,self.dt):
            self.checkpoint_phase_elapsed_s=float(t);self.step(plan.sample(t))
        final=plan.sample(plan.duration)
        self.qs={s:final['q'][s].copy() for s in self.qs};self.finger=final['finger'].copy()
        self.rotations={s:self.planner.fk(s,self.qs[s])[1] for s in self.qs}
        self.checkpoint_phase_completed=True
        self.events.append({'phase':label,'time_s':self.time})
        self.capture(label);self.save(label,native=True)

    def hold(self,duration,label,fingers=None):
        curves={s:CubicSpline([0,1],np.array([q,q]),axis=0) for s,q in self.qs.items()}
        self.execute(MotionPlan(curves,duration,self.finger.copy(),fingers or self.finger.copy()),label)

    def move(self,side,target,rotation,label,duration=3.,margin=.006):
        plan=self.planner.trajectory(self.qs,{side:target},rotations={side:rotation},samples=24,
            duration=duration,finger_start=self.finger.copy(),finger_end=self.finger.copy(),
            max_joint_velocity=.65,max_joint_acceleration=1.3,clearance_margin=margin,validate_samples=61)
        self.execute(plan,label)

    def grasp_bear(self):
        """Observe, close with finite force, and lift under the original physical guards."""
        bear=positions(self.bear)
        ref=np.asarray(self.config['bear']['position'])+self.config['task']['bear_grasp_local']
        ids=np.linalg.norm(self.rest['bear']-ref,axis=1)<.023
        target=ref+np.mean(bear[ids]-self.rest['bear'][ids],axis=0)
        self.move('right',target+[0,0,.12],DOWN,'bear_pregrasp')
        self.move('right',target,DOWN,'bear_approach',2.)
        self.set_grip_force('right',self.config['task']['bear_finger_force_limit'])
        # At kp=8000 a zero-gap target saturates the motor through contact,
        # leaving its velocity damping ineffective. Reduce only the right
        # finger position gain so existing damping can slow the closure.
        self.robots['right'].set_dofs_kp([200.,200.],dofs_idx_local=[7,8])
        self.robots['right'].set_dofs_kv([200.,200.],dofs_idx_local=[7,8])
        self.report['right_grasp_controller']={'finger_kp':200.,'finger_kv':200.,
            'force_limit_n':self.config['task']['bear_finger_force_limit'],
            'post_close_settle_s':.6,'force_proxy_dropout_limit_s':.075}
        self.hold(.8,'bear_close',{**self.finger,'right':0.})
        self.must_hold_teddy=True
        self.hold(.6,'bear_grasp_settle')
        p,r=self.tcp('right');self.move('right',p+[0,0,.38],r,'lift_bear',4.)

    def run(self):
        self.hold(.3,'settle')
        if self.args.task=='settle': return
        if self.args.task=='bear_grasp':
            self.grasp_bear()
            self.hold(.5,'bear_grasp_probe_hold')
            self.report['bear_grasp_probe_passed']=True
            return
        bag=positions(self.bag)
        grasp=cloth_grasp_frame(bag,self.rest['bag_faces'],self.left_grasp_material_ids,
            preferred_approach=(0,1,0),preferred_closing=(0,0,-1),shell_radius=self.config['bag']['thickness'])
        write_json(self.out/'initial_handle_grasp.json',grasp.as_dict())
        pre=grasp.position-grasp.rotation.apply([0,0,.10])
        self.move('left',pre,grasp.rotation,'left_handle_pregrasp')
        # Read actual material again after approach; do not use a stale target.
        grasp=cloth_grasp_frame(positions(self.bag),self.rest['bag_faces'],self.left_grasp_material_ids,
            preferred_approach=(0,1,0),preferred_closing=(0,0,-1),shell_radius=self.config['bag']['thickness'])
        self.move('left',grasp.position,grasp.rotation,'left_handle_approach',2.)
        self.set_grip_force('left',self.config['task']['handle_force_limit'])
        self.hold(.6,'left_handle_close',{**self.finger,'left':0.})
        self.must_hold_bag=True
        p,r=self.tcp('left')
        p[2]=self.config['task'].get('lift_handle_height',1.30)
        self.move('left',p,r,'lift_bag',4.)
        if self.args.task=='lift': return
        # Move the physically held handle while rotating the wrist; bag follows contact/gravity.
        target=np.asarray(self.config['task'].get('upright_handle_target',[-.18,.10,1.145]))
        upright_rotation=Rotation.from_quat(self.config['task'].get('upright_rotation_xyzw',DOWN.as_quat()))
        self.move('left',target,upright_rotation,'upright_bag',4.)
        self.hold(.8,'upright_settle')
        current=positions(self.bag);bottom=current[self.source_to_runtime[self.asset['bottom']]]
        clearance=float(bottom[:,2].min()-self.config['table']['height'])
        if clearance>0.002:
            p,r=self.tcp('left');p[2]-=max(0.,clearance-.001)
            self.move('left',p,r,'place_bag_on_table',2.)
        self.hold(.5,'bag_standing')
        # A low corner alone is not a seated bottom. Observe the body's actual
        # bottom centre and yaw, then bring the held handle above its own wall.
        # Keeping the old XY target while lowering can pull the handle across
        # the mouth and sustain a tilted bag. Only robot motor targets change.
        from validate_physics import bag_pose
        local_grasp=self.asset['points_upright'][self.asset['grasp_upper']].mean(axis=0)
        seating=[]
        for attempt in range(4):
            current=positions(self.bag)[self.source_to_runtime]
            pose=bag_pose(self.asset['points_upright'],current,self.asset['body'])
            bottom=current[self.asset['bottom']]
            gap_min=float(bottom[:,2].min()-self.config['table']['height'])
            gap90=float(np.quantile(bottom[:,2],.90)-self.config['table']['height'])
            if gap90<=.006 and pose['tilt_from_upright_degrees']<=8.: break
            if pose['rms_residual_m']>.035:
                raise RuntimeError('Bag shape is too distorted for physical placement feedback: '+str(pose))
            # Establish contact before lateral alignment; never chase the base
            # of an airborne bag with repeated horizontal corrections.
            if gap_min>.006:
                p,r=self.tcp('left');p[2]-=gap_min-.001
                self.move('left',p,r,f'seat_recontact_{attempt+1}',2.)
                self.hold(.5,f'seat_recontact_settle_{attempt+1}')
                current=positions(self.bag)[self.source_to_runtime]
                pose=bag_pose(self.asset['points_upright'],current,self.asset['body'])
                bottom=current[self.asset['bottom']]
                gap90=float(np.quantile(bottom[:,2],.90)-self.config['table']['height'])
            matrix=np.asarray(pose['rotation'])
            yaw=float(np.arctan2(matrix[1,0],matrix[0,0]))
            upright_offset=Rotation.from_euler('z',yaw).apply(local_grasp)
            p,r=self.tcp('left')
            desired=np.asarray(pose['translation_m'])+upright_offset
            desired[2]=self.config['table']['height']+.001+local_grasp[2]
            shift=desired[:2]-p[:2]
            distance=float(np.linalg.norm(shift))
            target=p.copy()
            if distance>.012:
                # A bounded move gives a fresh observation before a large
                # lateral correction can displace the bag unpredictably.
                target[:2]+=shift*min(1.,.18/distance)
                target[2]=np.clip(desired[2],p[2]-.070,p[2]+.025)
                action='align_held_handle'
            else:
                target[2]-=min(.025,max(.004,gap90-.003))
                action='add_handle_slack'
            seating.append({'attempt':attempt+1,'action':action,
                'body_pose_before':pose,'bottom_gap90_before_m':gap90,
                'observed_tcp_m':p,'ideal_upright_handle_m':desired,
                'commanded_tcp_m':target,'soft_state_mutated':False})
            self.report['seating_feedback']=seating
            write_json(self.out/'seating_feedback.json',seating)
            self.move('left',target,r,f'seat_bag_{attempt+1}',3.)
            self.hold(.7,f'seat_settle_{attempt+1}')
        pose=bag_pose(self.asset['points_upright'],positions(self.bag)[self.source_to_runtime],self.asset['body'])
        self.report['standing_bag_pose']=pose
        bottom=positions(self.bag)[self.source_to_runtime[self.asset['bottom']]]
        gap90=float(np.quantile(bottom[:,2],.90)-self.config['table']['height'])
        self.report['standing_bottom_gap_90percentile_m']=gap90
        if pose['tilt_from_upright_degrees']>8. or pose['rms_residual_m']>.035 or gap90>.008:
            raise RuntimeError('Bag did not become sufficiently upright: '+str(pose))
        if self.args.task=='upright': return
        self.grasp_bear()
        # Native teddy geometry determines the gripper-to-body offset and clearance.
        bag=positions(self.bag);bear=positions(self.bear)
        rim=bag[self.source_to_runtime[self.asset['rim']]]
        p,r=self.tcp('right');center=bear.mean(axis=0)
        dest=p.copy();dest[:2]+=rim.mean(axis=0)[:2]-center[:2]
        dest[2]+=max(0.,rim[:,2].max()+.065-bear[:,2].min())
        self.move('right',dest,r,'transport_bear_above_opening',4.)
        bag=positions(self.bag);bear=positions(self.bear);p,r=self.tcp('right')
        p[2]-=max(0.,bear[:,2].min()-(self.config['table']['height']+.035))
        self.move('right',p,r,'lower_bear_inside',4.,margin=.004)
        self.must_hold_teddy=False
        self.hold(.8,'release_bear',{**self.finger,'right':.04})
        p,r=self.tcp('right');self.move('right',p+[0,0,.30],r,'withdraw_right',3.)
        self.hold(2.,'final_settle')
        from validate_physics import approximate_containment,actual_bag_containment,bag_pose
        bag=positions(self.bag)[self.source_to_runtime]
        containment=approximate_containment(positions(self.bear),self.asset['points_upright'],bag,
            self.asset['body'],[[-.14,-.10,0.],[.14,.10,.30]])
        self.report['containment']=containment
        actual=actual_bag_containment(positions(self.bear),bag,self.asset['faces'][self.asset['body_face_indices']],self.asset['rim'])
        pose=bag_pose(self.asset['points_upright'],bag,self.asset['body'])
        bottom_gap=float(bag[self.asset['bottom'],2].min()-self.config['table']['height'])
        bottom_gap90=float(np.quantile(bag[self.asset['bottom'],2],.90)-self.config['table']['height'])
        final_speed=float(max(self.bear_speed_tail[-40:]))
        self.report.update(actual_containment=actual,final_bag_pose=pose,final_bottom_gap_m=bottom_gap,final_bottom_gap90_m=bottom_gap90,
            final_bear_mean_vertex_speed_peak_m_s=final_speed,
            validation_scope='One uncalibrated numerical run, finite force grasp proxy, native contact and measured geometry; no physical experiment calibration')
        self.report['task_passed']=bool(actual['actual_containment_pass'] and pose['tilt_from_upright_degrees']<8
            and -.002<=bottom_gap<=.004 and bottom_gap90<.008 and final_speed<.025 and not self.runtime.error_seen
            and self.runtime.native_convergence_failure_count==0)
        self.report['status']='numerical_task_passed' if self.report['task_passed'] else 'motion_completed_validation_failed'

    def finish(self,error=None):
        if error is not None:
            self.report.update(status='stopped',error=repr(error),traceback=traceback.format_exc())
        elif self.report['status']=='running': self.report['status']='requested_stage_completed'
        self.report.update(last_phase=self.phase,simulation_time_s=self.time,
            wall_seconds=time.monotonic()-self.wall,events=self.events,runtime=self.runtime.report(),
            actual_max_joint_speed_rad_s=self.actual_max_joint_speed,
            actual_min_sampled_rigid_clearance_m=self.actual_min_rigid_clearance)
        try: self.save('finish',native=True)
        except Exception as exc: self.report['preservation_error']=repr(exc)
        try: self.capture('final')
        except Exception as exc: self.report['final_render_error']=repr(exc)
        try: self.writer.close()
        except Exception as exc: self.report['video_close_error']=repr(exc)
        np.savez_compressed(self.out/'trajectory.npz',joint_trace=np.asarray(self.trace),physical_metrics=np.asarray(self.metrics))
        write_json(self.out/'grip_trace.json',self.grip_trace)
        write_json(self.out/'report.json',self.report)
        print('RESULT '+json.dumps({'status':self.report['status'],'phase':self.phase,'time_s':self.time,'output':str(self.out),'error':repr(error)}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(HERE/'config.json'))
    p.add_argument('--task',choices=['settle','lift','upright','bear_grasp','full'],default='full')
    p.add_argument('--output');p.add_argument('--viewer',action='store_true')
    p.add_argument('--wall-seconds',type=float,default=3000)
    p.add_argument('--width',type=int,default=800);p.add_argument('--height',type=int,default=600)
    a=p.parse_args()
    if a.output is None: a.output=datetime.now().strftime('%Y%m%d-%H%M%S')
    def stop(sig,frame): raise TimeoutError(f'External stop signal {sig}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    exp=None
    try:
        exp=Experiment(read_config(a.config),a);exp.run()
    except BaseException as exc:
        if exp is not None:
            exp.finish(exc)
            return 2
        else:
            out=HERE/'runs'/a.output
            if out.is_dir(): write_json(out/'initialization_failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'task_passed':False})
            print('INITIALIZATION_FAILED '+traceback.format_exc(),flush=True)
            raise
    else:
        exp.finish()
        return 0 if a.task!='full' or exp.report['task_passed'] else 3

if __name__=='__main__':raise SystemExit(main())
