"""Write a read-only index of finished experiments; never advance simulation."""
from datetime import datetime, timezone
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def main():
    rows=[]
    for directory in sorted((HERE/'runs').iterdir()):
        path=directory/'report.json'
        if not path.is_file():
            continue
        r=json.loads(path.read_text())
        rows.append({'run':directory.name,'requested_task':r.get('requested_task'),
            'status':r.get('status'),'task_passed':r.get('task_passed',False),
            'last_phase':r.get('last_phase'),'simulation_time_s':r.get('simulation_time_s'),
            'error':r.get('error'),'checkpoint_export_success':r.get('latest_checkpoint',{}).get('success'),
            'report':str(path),'video':str(directory/'simulation.mp4'),
            'final_image':str(directory/'final_detail.png')})
    result={'generated_utc':datetime.now(timezone.utc).isoformat(),
        'full_task_verified':any(r['requested_task']=='full' and r['task_passed'] for r in rows),
        'runs':rows,'replay_supported':True,'exact_midpoint_resume_certified':False,
        'continuation_instructions':str(HERE/'CONTINUE_CN.txt'),
        'material_limitations':str(HERE/'MATERIAL_NOTES.txt'),
        'budget':json.loads((HERE/'budget.json').read_text())}
    (HERE/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
