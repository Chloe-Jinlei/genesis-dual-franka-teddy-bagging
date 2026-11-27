"""CPU-only diagnostic rendering of a COMPLETED run's actual saved soft meshes.

No simulation, object transforms, mesh smoothing, or physical material changes.
Transparency is only a visualization aid. Existing images are never overwritten.
"""
from __future__ import annotations
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree


def surface_from_tetrahedra(tetrahedra):
    t=np.asarray(tetrahedra,int)
    faces=np.concatenate([t[:,ids] for ids in ([0,1,2],[0,3,1],[0,2,3],[1,3,2])])
    _,first,counts=np.unique(np.sort(faces,axis=1),axis=0,return_index=True,return_counts=True)
    if np.any(counts>2):raise ValueError('Nonmanifold tetrahedron face topology')
    return faces[first[counts==1]]


def fresh_path(directory,stem,suffix):
    target=directory/(stem+suffix)
    if not target.exists():return target
    for index in range(2,10000):
        target=directory/(stem+f'_{index:02d}'+suffix)
        if not target.exists():return target
    raise RuntimeError('No unused diagnostic filename')


def render(run,dpi=160):
    run=Path(run).resolve()
    if not(run/'report.json').is_file():raise ValueError('Completed report.json is required; no live-run rendering')
    report=json.loads((run/'report.json').read_text())
    if report.get('status') in ['building','running']:raise ValueError('Run has not completed')
    config=json.loads((run/'config.json').read_text())
    state=np.load(run/'latest_state.npz');rest=np.load(run/'rest.npz')
    source=np.load(run/'source/assets/paper_bag.npz')
    errors,indices=cKDTree(rest['bag']).query(source['points']+config['bag']['position'])
    if errors.max()>1e-6 or len(np.unique(indices))!=len(indices):raise ValueError('Ambiguous source-to-runtime bag correspondence')
    bag=np.asarray(state['bag'])[indices];teddy=np.asarray(state['bear'])
    if not(np.isfinite(bag).all() and np.isfinite(teddy).all()):raise ValueError('Cannot render nonfinite saved geometry')
    bag_faces=source['faces'];teddy_faces=surface_from_tetrahedra(rest['bear_tets'])
    height=float(config['table']['height']);points=np.concatenate((bag,teddy))
    lower=points.min(axis=0);upper=points.max(axis=0)
    xylo=lower[:2]-.06;xyhi=upper[:2]+.06
    table=np.array([[xylo[0],xylo[1],height],[xyhi[0],xylo[1],height],
                    [xyhi[0],xyhi[1],height],[xylo[0],xyhi[1],height]])
    center=(lower+upper)/2;center[2]=(height+upper[2])/2
    span=max(float(np.ptp(points[:,:2],axis=0).max())+.12,float(upper[2]-height)+.10,.35)
    fig=plt.figure(figsize=(13.5,7.5),facecolor='#f6f3ee')
    views=[(24,-58,'Perspective view'),(89,-90,'View through the opening')]
    for number,(elevation,azimuth,title) in enumerate(views,1):
        ax=fig.add_subplot(1,2,number,projection='3d',facecolor='#f6f3ee')
        ax.add_collection3d(Poly3DCollection([table],facecolor='#c7b69d',edgecolor='#a48d6d',alpha=.22,linewidth=.6))
        ax.add_collection3d(Poly3DCollection(bag[bag_faces],facecolor='#a86f35',edgecolor='#a47b49',alpha=.115,linewidth=.10))
        ax.add_collection3d(Poly3DCollection(teddy[teddy_faces],facecolor='#b98243',edgecolor='#806047',alpha=1.,linewidth=.10))
        rim=bag[source['rim']];rim=np.concatenate((rim,rim[:1]))
        ax.plot(rim[:,0],rim[:,1],rim[:,2],color='#236977',linewidth=1.7,label='Actual bag mouth')
        ax.set_xlim(center[0]-span/2,center[0]+span/2);ax.set_ylim(center[1]-span/2,center[1]+span/2)
        ax.set_zlim(max(height-.025,center[2]-span/2),max(height+.18,center[2]+span/2))
        ax.set_box_aspect((1,1,1));ax.view_init(elev=elevation,azim=azimuth)
        ax.set_title(title,fontsize=12,color='#26343a',pad=8)
        ax.set_xlabel('X (m)',fontsize=8,labelpad=2);ax.set_ylabel('Y (m)',fontsize=8,labelpad=2)
        ax.set_zlabel('Z (m)',fontsize=8,labelpad=2);ax.tick_params(labelsize=7,pad=0)
        if number==2:
            ax.set_zticks([]);ax.set_zlabel('')
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
            axis.pane.fill=False;axis.pane.set_edgecolor('#d8d5d0')
            axis._axinfo['grid']['color']=(.7,.7,.7,.25)
    status=str(report.get('status','unknown')).replace('_',' ')
    simtime=float(state['time']);passed=bool(report.get('task_passed',False))
    fig.suptitle('Saved geometry diagnostic — transparent bag display',fontsize=17,color='#233840',y=.96)
    fig.text(.5,.905,f'{run.name}  |  simulation {simtime:.3f} s  |  {status}',ha='center',fontsize=10,color='#40545c')
    fig.text(.5,.065,'Transparency is a diagnostic display only. Actual saved vertex positions are unchanged.',ha='center',fontsize=10,color='#344c55')
    fig.text(.5,.034,'This image alone does not certify contact physics or successful insertion.',ha='center',fontsize=9,color='#64747b')
    fig.subplots_adjust(left=.01,right=.99,top=.85,bottom=.12,wspace=.02)
    output=fresh_path(run,'diagnostic_transparent_saved','.png')
    fig.savefig(output,dpi=dpi,facecolor=fig.get_facecolor());plt.close(fig)
    manifest=output.with_suffix('.json')
    manifest.write_text(json.dumps({'run_directory':str(run),'image':str(output),'simulation_time_s':simtime,
        'reported_status':report.get('status'),'reported_task_passed':passed,
        'diagnostic_transparency_only':True,'physical_state_changed':False,'simulation_started':False,
        'bag_vertex_count':len(bag),'bag_triangle_count':len(bag_faces),'teddy_vertex_count':len(teddy),
        'teddy_boundary_triangle_count':len(teddy_faces),'source_correspondence_max_error_m':float(errors.max()),
        'inputs_sha256':{name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in ['latest_state.npz','rest.npz','report.json']},
        'note':'Two CPU Matplotlib views of saved meshes; solid teddy colour is diagnostic, without fur texture or robot rendering.'},indent=2)+'\n')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_directory')
    parser.add_argument('--dpi',type=int,default=160);args=parser.parse_args()
    if not 80<=args.dpi<=300:parser.error('dpi must be between80and300')
    print(render(args.run_directory,args.dpi))
