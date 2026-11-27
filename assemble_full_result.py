"""Present existing recordings in full, explicitly labeling the restored seam.

This program only reads saved output and encodes video. It cannot run physics.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import zipfile

HERE=Path(__file__).resolve().parent
OUT=HERE/'deliverables/full_result'
OUT.mkdir(parents=True,exist_ok=False)
front=HERE/'runs/paper_09_full_authorized'
release=HERE/'runs/paper_11_direct_release'
font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
assert font.is_file()

stages=[
 (0.,.5,'初始状态：熊与开口纸袋平躺在桌面'),
 (.5,8.5,'左臂接近并夹住纸袋的一条提手'),
 (8.5,17.6,'左臂提起纸袋，并转至竖直方向'),
 (17.6,28.1,'调整袋底位置，纸袋竖放在桌面'),
 (28.1,36.1,'右臂接近玩具熊并夹住'),
 (36.1,40.2,'右臂提起玩具熊'),
 (40.2,45.6,'将熊移到袋口上方'),
 (45.6,46.6,'恢复保存状态后，右夹爪直接松开'),
 (46.6,49.8,'熊自由下落入袋；左臂继续扶住提手'),
 (49.8,54.8,'最终保存位置：透明袋壁仅用于查看内部'),
]
filters=[
 '[0:v]setpts=PTS-STARTPTS,setsar=1[v0]',
 '[1:v]setpts=PTS-STARTPTS,setsar=1[v1]',
 '[2:v]scale=800:608:force_original_aspect_ratio=decrease,pad=800:608:(ow-iw)/2:(oh-ih)/2:color=0xf6f3ee,setsar=1,trim=duration=5,setpts=PTS-STARTPTS[v2]',
 '[v0][v1][v2]concat=n=3:v=1:a=0,pad=800:720:0:0:color=0x14202b[base]'
]
draw=[]
for i,(start,end,caption) in enumerate(stages):
 textfile=OUT/f'caption_{i:02d}.txt';textfile.write_text(caption)
 draw.append(f"drawtext=fontfile='{font}':textfile='{textfile}':fontcolor=white:fontsize=24:x=22:y=627:enable='gte(t,{start})*lt(t,{end})'")
footer=OUT/'caption_footer.txt'
footer.write_text('GPU 仿真录像｜45.6 秒处衔接恢复状态｜局部网格异常仍保留')
draw.append(f"drawtext=fontfile='{font}':textfile='{footer}':fontcolor=0xc4d0da:fontsize=17:x=22:y=679")
filters.append('[base]'+','.join(draw)+'[out]')
filterfile=OUT/'video_filter.txt';filterfile.write_text(';\n'.join(filters))
video=OUT/'full_process.mp4'
command=['ffmpeg','-hide_banner','-loglevel','error','-nostdin',
 '-i',str(front/'simulation.mp4'),'-i',str(release/'simulation.mp4'),
 '-loop','1','-framerate','10','-t','5','-i',str(release/'diagnostic_transparent_saved.png'),
 '-filter_complex_script',str(filterfile),'-map','[out]','-an','-r','10',
 '-c:v','libx264','-crf','19','-preset','fast','-pix_fmt','yuv420p',
 '-movflags','+faststart',str(video)]
subprocess.run(command,check=True)
shutil.copy2(release/'diagnostic_transparent_saved.png',OUT/'final_inside_bag.png')
shutil.copy2(front/'initial_detail.png',OUT/'initial_scene.png')
shutil.copy2(release/'final_detail.png',OUT/'final_scene.png')
backup=HERE.parent/'teddy_paper_gpu_20261126_saved_20261127T020453Z.tar.gz'
readme='''完整过程交付

full_process.mp4：从初始桌面场景到松手入袋的完整录像，约 55 秒。
包含：夹住提手、提袋、竖放、抓熊、提熊、搬运至袋口、松手、下落入袋。
最后 5 秒展示实际保存几何的透明袋壁视图，便于看清袋内的熊。
final_inside_bag.png：同一最终位置图，未修改顶点或物理状态。

视频由两段已有记录衔接：
  0–45.6 秒：paper_09_full_authorized，初始状态至物理时间 43.765 秒。
  45.6–49.8 秒：paper_11_direct_release，从同一保存状态恢复后直接释放。
  49.8–54.8 秒：静态最终位置图；此段是结果展示，没有推进仿真。
恢复的位置、速度、机器人及耦合数组与保存状态读回一致；没有声称这是同一个
进程不中断跑完的录像。没有生成不存在的运动，也没有补做原计划的下降和撤臂。

结果：右夹爪张开，熊落入袋内，左臂继续扶住纸袋。保存的熊顶点均在袋内。
限制：原运行在局部过度压缩处停止；按后续指令直接松手时，最多 4 个体积单元
发生翻转。所以动作展示已到入袋，物理质量尚未验证通过；原失败记录全部保留。
此次整理没有启动新仿真或追加物理测试。

scene_and_progress.tar.gz：包含场景代码、模型、参数、运行状态、日志、原始录像、
原生检查点、Genesis 源码和依赖版本记录的完整保存包。原有本机虚拟环境保留，
包内不包含整份虚拟环境或离线 wheelhouse。精确不间断续算没有经过认证。
解压后先阅读 RELEASE_RESULT_CN.txt 和 CONTINUE_CN.txt。
'''
(OUT/'说明.txt').write_text(readme)
metadata={'created_utc':datetime.now(timezone.utc).isoformat(),
 'simulation_started':False,'source_runs':[str(front),str(release)],
 'seam_at_video_seconds':45.6,'seam_physics_time_seconds':43.765,
 'video_stage_captions':stages,'release_completed':True,
 'full_physical_task_validated':False,'video':str(video),
 'full_scene_backup':str(backup),
 'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
 [front/'simulation.mp4',release/'simulation.mp4',release/'diagnostic_transparent_saved.png',backup]}}
(OUT/'provenance.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
bundle=OUT.parent/'complete_result_bundle.zip'
with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_STORED) as z:
 for name in ['full_process.mp4','final_inside_bag.png','initial_scene.png','final_scene.png','说明.txt','provenance.json']:
  z.write(OUT/name,arcname='完整结果/'+name)
 z.write(backup,arcname='完整结果/scene_and_progress.tar.gz')
 z.write(Path(str(backup)+'.verification.json'),arcname='完整结果/scene_archive_verification.json')
 z.write(Path(str(backup)+'.sha256'),arcname='完整结果/original_archive.sha256')
print(json.dumps({'video':str(video),'bundle':str(bundle),'bytes':bundle.stat().st_size},ensure_ascii=False))
