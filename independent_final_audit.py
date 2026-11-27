"""Bounded, read-only CPU audit of an already completed paper-bag run.

Usage: .venv/bin/python independent_final_audit.py RUN_DIRECTORY
Writes only RUN_DIRECTORY/independent_final_audit.json. Never starts Genesis,
advances/restores a simulation, changes thresholds, or promotes a failed run.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, struct, sys, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from validate_physics import (bag_pose, triangle_metrics, tetra_metrics,
    actual_bag_containment, table_contact_metrics)


def summary(values):
    a=np.asarray(values,float).reshape(-1)
    return {'min':float(a.min()),'median':float(np.median(a)),
        'p95':float(np.quantile(a,.95)),'max':float(a.max()),
        'rms':float(np.sqrt(np.mean(a*a))),'count':len(a)} if len(a) and np.isfinite(a).all() else {'error':'empty/nonfinite values'}


def read_native(path):
    raw=Path(path).read_bytes()
    if len(raw)<16 or struct.unpack('<Q',raw[8:16])[0]!=len(raw)-16 or (len(raw)-16)%24:
        raise ValueError('Native double-vector buffer header/size mismatch: '+str(path))
    a=np.frombuffer(raw,dtype='<f8',offset=16).copy().reshape(-1,3)
    if not np.isfinite(a).all(): raise ValueError('Nonfinite native vector buffer')
    return a


def boundary_faces(tets):
    t=np.asarray(tets,int)
    faces=np.concatenate([t[:,v] for v in ([0,1,2],[0,3,1],[0,2,3],[1,3,2])])
    _,first,counts=np.unique(np.sort(faces,axis=1),axis=0,return_index=True,return_counts=True)
    if np.any(counts>2):raise ValueError('Nonmanifold teddy tetrahedral faces')
    return faces[first[counts==1]]


def triangles_intersect(a,b,tolerance=1e-9):
    """Triangle SAT, including axes needed for coplanar pairs."""
    ea=np.roll(a,-1,axis=0)-a;eb=np.roll(b,-1,axis=0)-b
    na=np.cross(ea[0],ea[1]);nb=np.cross(eb[0],eb[1])
    axes=np.vstack((na,nb,np.cross(ea[:,None],eb[None]).reshape(-1,3),np.cross(na,ea),np.cross(nb,eb)))
    length=np.linalg.norm(axes,axis=1);axes=axes[length>1e-14]/length[length>1e-14,None]
    if not len(axes):raise ValueError('Degenerate triangle in intersection audit')
    pa=a@axes.T;pb=b@axes.T
    return bool(not np.any((pa.max(axis=0)<pb.min(axis=0)-tolerance)|(pb.max(axis=0)<pa.min(axis=0)-tolerance)))


def intersection_audit(points_a,faces_a,points_b,faces_b,*,seconds=12.,max_pairs=20000,self_mode=False):
    """Bounded exact triangle tests after AABB rejection; no rtree dependency.

    A non-completed report is inconclusive, never a clean intersection audit.
    Self mode excludes shared-vertex face pairs; this limitation is explicit.
    """
    start=time.monotonic();a=np.asarray(points_a)[faces_a];b=np.asarray(points_b)[faces_b]
    alo,ahi=a.min(axis=1),a.max(axis=1);blo,bhi=b.min(axis=1),b.max(axis=1)
    tested=0;hits=[];completed=True;reason=None
    for i in range(len(a)):
        if time.monotonic()-start>seconds:completed=False;reason='CPU time budget';break
        ids=np.flatnonzero(np.all(ahi[i]>=blo-1e-9,axis=1)&np.all(bhi>=alo[i]-1e-9,axis=1))
        for j in ids:
            if self_mode and (j<=i or np.intersect1d(faces_a[i],faces_b[j]).size):continue
            if tested>=max_pairs or time.monotonic()-start>seconds:
                completed=False;reason='candidate/time budget';break
            tested+=1
            if triangles_intersect(a[i],b[j]):hits.append([int(i),int(j)])
        if not completed:break
    return {'method':'AABB then triangle separating-axis test including coplanar axes',
        'completed':completed,'stop_reason':reason,'tested_candidate_pairs':tested,
        'intersecting_pair_count':len(hits),'first_intersections':hits[:100],
        'no_intersections_verified':bool(completed and not hits),'seconds':time.monotonic()-start,
        'geometric_tolerance_m':1e-9,'shared_vertex_pairs_excluded':self_mode,
        'scope':'Floating-point geometric audit at one frame; no shell thickness or continuous-motion certificate.'}


def mouth_metrics(rim,bear):
    center=rim.mean(axis=0);_,_,vt=np.linalg.svd(rim-center,full_matrices=False)
    xy=(rim-center)@vt[:2].T;edges=np.roll(xy,-1,axis=0)-xy
    lengths=np.linalg.norm(edges,axis=1)
    if np.any(lengths<1e-12):return {'error':'Collapsed mouth edge'}
    hull=ConvexHull(xy);hxy=xy[hull.vertices];he=np.roll(hxy,-1,axis=0)-hxy
    normals=np.column_stack((-he[:,1],he[:,0]));normals/=np.linalg.norm(normals,axis=1)[:,None]
    support=xy@normals.T;bear_support=(bear-center)@vt[:2].T@normals.T
    nearest=np.linalg.norm(xy+np.clip(-np.sum(xy*edges,axis=1)/(lengths*lengths),0,1)[:,None]*edges,axis=1)
    area=.5*abs(np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1)))
    return {'projected_polygon_area_m2':float(area),'nominal_rest_area_m2':.28*.20,
        'area_fraction_of_nominal':float(area/(.28*.20)),
        'minimum_convex_hull_width_m':float(np.ptp(support,axis=0).min()),
        'centroid_to_nearest_rim_edge_m':float(nearest.min()),
        'centroid_centered_inscribed_circle_diameter_m':float(2*nearest.min()),
        'teddy_projected_widths_range_m':[float(np.ptp(bear_support,axis=0).min()),float(np.ptp(bear_support,axis=0).max())],
        'maximum_rim_planarity_error_m':float(abs((rim-center)@vt[2]).max()),
        'scope':'Convex-hull width may overstate concave usable width. Inscribed-circle interpretation requires the separately checked simple interior-centroid rim. Final teddy projection is not the insertion-time cross-section.'}


def native_audit(run,report,last,config):
    cp=report.get('latest_checkpoint',{});workspace=run/'native_checkpoint/workspace'
    if not workspace.exists():workspace=Path(cp.get('native_workspace_copy','/does-not-exist'))
    frame=cp.get('native_frame')
    if frame is None:frame=int(round(float(last['time'])/config['simulation']['dt']*config['simulation'].get('substeps',1)))
    folder=workspace/'dump/cuda/finite_element/finite_element_method.cu'
    q=read_native(folder/f'q.{frame}');v=read_native(folder/f'q_v.{frame}')
    origin=np.asarray(config['simulation'].get('physics_origin_m',[0,0,0]))
    orders=[]
    for order in [('bag','bear'),('bear','bag')]:
        expected=np.concatenate([last[k] for k in order])-origin
        if expected.shape==q.shape:
            error=float(abs(q-expected).max())
            if error<=1e-10:orders.append((order,error))
    if len(orders)!=1 or q.shape!=v.shape:raise ValueError('Native q does not uniquely match saved soft state')
    order,error=orders[0];offset=0;speeds={}
    for key in order:
        speeds[key]=summary(np.linalg.norm(v[offset:offset+len(last[key])],axis=1));offset+=len(last[key])
    state=json.loads((workspace/f'dump/common/sim_engine.cpp/state.{frame}.json').read_text())
    if state['frame']!=frame:raise ValueError('Native state frame mismatch')
    out={'verified':True,'frame':frame,'workspace':str(workspace),'body_order':list(order),
        'q_matches_final_npz_max_error_m':error,'native_velocity_m_s':speeds,
        'exact_resume_certified':False,'scope':'Native exported final velocities; one frame cannot prove stable dwell.'}
    # Independent displacement speeds if a neighbouring preserved frame exists.
    earlier=[]
    for p in folder.glob('q.*'):
        try:f=int(p.name.split('.')[-1])
        except ValueError:continue
        if f<frame:earlier.append((f,p))
    if earlier:
        previous,path=max(earlier);old=read_native(path)
        dt=config['simulation']['dt']/config['simulation'].get('substeps',1)*(frame-previous)
        if old.shape==q.shape and dt>0:
            out['last_preserved_interval_s']=dt;out['last_preserved_displacement_speed_m_s']=summary(np.linalg.norm(q-old,axis=1)/dt)
    return out


def archived_planner(run,config):
    path=run/'source/motion.py'
    spec=importlib.util.spec_from_file_location('independent_archived_motion',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    config=json.loads(json.dumps(config));config['robots']['mjcf_path']=str(run/'source/assets/panda_tactile.xml')
    return module.Planner(config)


def robot_metrics(planner,state):
    qs={s:np.asarray(state['q_'+s]).reshape(-1) for s in ['left','right']}
    tcp,rot=planner.fk('right',qs['right'][:7]);bear=state['bear']
    # Individual measured finger joint positions are preserved, without assuming symmetry.
    for side in qs:
        planner.d.qpos[planner.qindices[side]]=qs[side][:7]
        for finger in [1,2]:
            planner.d.qpos[planner.m.joint(f'{side}_finger_joint{finger}').qposadr[0]]=qs[side][6+finger]
    import mujoco
    mujoco.mj_forward(planner.m,planner.d)
    links={}
    for name in ['right_hand','right_left_finger','right_right_finger']:
        point=planner.d.xpos[planner.m.body(name).id].copy()
        links[name]={'origin_world_m':point.tolist(),'nearest_teddy_vertex_m':float(np.linalg.norm(bear-point,axis=1).min())}
    return {'right_tcp_world_m':tcp.tolist(),'right_quaternion_xyzw':rot.as_quat().tolist(),
        'nearest_teddy_vertex_to_tcp_m':float(np.linalg.norm(bear-tcp,axis=1).min()),
        'actual_right_finger_joint_positions_m':qs['right'][7:9].tolist(),
        'actual_nominal_jaw_opening_m':float(qs['right'][7:9].sum()),'hand_and_finger_origins':links,
        'scope':'MuJoCo FK of actual saved robot joints. Origin/TCP distances are separation proxies, not exact hand surface distances or contact attribution.'}


def stage_audit(run,planner,height):
    records={}
    for folder in (run/'native_checkpoint/records').glob('*'):
        meta=folder/'controller.json';statefile=folder/'controller_state.npz'
        if not(meta.exists() and statefile.exists()):continue
        m=json.loads(meta.read_text());phase=m.get('phase',{})
        if not phase.get('completed'):continue
        label=phase.get('label')
        if label not in ['bear_approach','bear_close','lift_bear','transport_bear_above_opening','lower_bear_inside','release_bear','withdraw_right','final_settle']:continue
        with np.load(statefile) as a:
            b=a['bear'];row={'time_s':m['time_s'],'record_directory':str(folder),
                'bear_min_table_gap_m':float(b[:,2].min()-height),'bear_centroid_world_m':b.mean(axis=0).tolist(),
                'robot':robot_metrics(planner,a)}
        if label not in records or row['time_s']>records[label]['time_s']:records[label]=row
    result={'records':records,'lift_evidence_available':all(x in records for x in ['bear_close','lift_bear']),
        'release_evidence_available':all(x in records for x in ['release_bear','withdraw_right','final_settle'])}
    if result['lift_evidence_available']:
        before,after=records['bear_close'],records['lift_bear']
        delta=np.asarray(after['bear_centroid_world_m'])-before['bear_centroid_world_m']
        hand=np.asarray(after['robot']['right_tcp_world_m'])-before['robot']['right_tcp_world_m']
        result.update(bear_centroid_lift_m=float(delta[2]),tcp_lift_m=float(hand[2]),
            bear_clear_of_table_after_lift=bool(after['bear_min_table_gap_m']>.01),
            centroid_minus_tcp_displacement_m=(delta-hand).tolist())
    result['scope']='Recorded phase endpoints support observed pickup/release; net finger forces are not target-specific contact proof.'
    return result


def audit(run,intersection_seconds=12.):
    run=Path(run).resolve();started=time.monotonic()
    if not(run/'report.json').exists():raise ValueError('Completed report.json is required; running trials are not audited')
    report=json.loads((run/'report.json').read_text());config=json.loads((run/'config.json').read_text())
    last=np.load(run/'latest_state.npz');rest=np.load(run/'rest.npz');asset=np.load(run/'source/assets/paper_bag.npz')
    error,mapping=cKDTree(rest['bag']).query(asset['points']+config['bag']['position'])
    if error.max()>1e-6 or len(np.unique(mapping))!=len(mapping):raise ValueError('Source/runtime bag mapping is not bijective')
    bag=last['bag'][mapping];bear=last['bear'];faces=asset['faces'][asset['body_face_indices']];height=config['table']['height']
    out={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'run_directory':str(run),
        'scope':'Independent completed-run CPU evidence audit; no simulation advancement or physical state changes.',
        'reported_status':report.get('status'),'reported_task_passed':report.get('task_passed',False),
        'full_task_success_promoted':False,'simulation_time_s':float(last['time']),'errors':[],
        'source_mapping_max_error_m':float(error.max()),
        'bag_pose':bag_pose(asset['points_upright'],bag,asset['body']),
        'paper_deformation':triangle_metrics(rest['bag'],last['bag'],rest['bag_faces']),
        'teddy_deformation':tetra_metrics(rest['bear'],bear,rest['bear_tets']),
        'actual_containment':actual_bag_containment(bear,bag,faces,asset['rim']),
        'bag_table':table_contact_metrics(bag,height,contact_radius=config['bag']['thickness']),
        'bear_table':table_contact_metrics(bear,height)}
    bottom=bag[asset['bottom'],2]-height;out['bottom_gap_m']=summary(bottom)
    out['bottom_gap_m']['p90']=float(np.quantile(bottom,.9))
    for name,call in [('mouth',lambda:mouth_metrics(bag[asset['rim']],bear)),
                      ('native',lambda:native_audit(run,report,last,config))]:
        try:out[name]=call()
        except Exception as e:out[name]={'error':repr(e)};out['errors'].append(name+': '+repr(e))
    try:
        planner=archived_planner(run,config);out['final_robot']=robot_metrics(planner,last)
        out['stage_evidence']=stage_audit(run,planner,height)
    except Exception as e:out['errors'].append('robot/stages: '+repr(e))
    try:
        teddy_surface=boundary_faces(rest['bear_tets'])
        out['bag_teddy_surface_intersections']=intersection_audit(bag,faces,bear,teddy_surface,seconds=intersection_seconds)
        out['bag_self_intersections']=intersection_audit(bag,faces,bag,faces,seconds=intersection_seconds,self_mode=True)
    except Exception as e:out['errors'].append('surface_intersections: '+repr(e))
    runtime=report.get('runtime',{})
    out['native_convergence']={k:runtime.get(k) for k in ['native_backend','observed_native_steps','native_convergence_failure_count','native_line_search_warning_count','native_engine_error_seen']}
    records=runtime.get('records',[])
    out['native_convergence']['all_persisted_samples_converged']=bool(records and all(r.get('native_frame_stats',{}).get('converged',False) and r.get('native_frame_stats',{}).get('completed',False) and not r.get('native_engine_has_error',True) for r in records))
    out['native_convergence']['persisted_sample_count']=len(records)
    if (run/'grip_trace.json').exists():
        grip=json.loads((run/'grip_trace.json').read_text());out['grasp_proxy']={}
        for side in ['left','right']:
            rows=[g[side] for g in grip if g.get(side,{}).get('required')]
            out['grasp_proxy'][side]={'required_samples':len(rows),'minimum_opening_reaction_n':min(min(g['pad_opening_reaction_n']) for g in rows) if rows else None,'maximum_nearest_material_m':max(g['nearest_material_m'] for g in rows) if rows else None}
    out['limitations']=['This tool never promotes a stopped/failed run to success.','Native contact convergence is not experimental material calibration.','Containment includes a virtual analysis-only mouth cap; it adds no simulated support.','Final native speed is one frame; a stable time window requires temporal evidence.','Hand separation is an FK origin/TCP proxy; target-specific contact identity is not exported.','Self-intersection audit excludes shared-vertex pairs; incomplete bounded audits are inconclusive.','Meaningful usable mouth aperture is reported without inventing a calibrated success threshold.']
    out['evidence_sha256']={name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in ['report.json','latest_state.npz','rest.npz','config.json']}
    out['elapsed_seconds']=time.monotonic()-started
    target=run/'independent_final_audit.json';target.write_text(json.dumps(out,indent=2)+'\n')
    return target,out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_directory')
    parser.add_argument('--intersection-seconds',type=float,default=12.,help='CPU budget for EACH of two triangle audits')
    args=parser.parse_args()
    if not 0<args.intersection_seconds<=30:parser.error('intersection-seconds must be >0 and <=30')
    path,result=audit(args.run_directory,args.intersection_seconds)
    print(json.dumps({'output':str(path),'reported_status':result['reported_status'],'reported_task_passed':result['reported_task_passed'],
        'actual_containment':result['actual_containment']['actual_containment_pass'],'elapsed_seconds':result['elapsed_seconds'],'errors':result['errors']},indent=2))
