#!/usr/bin/env python3
"""Create the deterministic second fallback prompt for Observer failures."""
import argparse,json,shutil
from pathlib import Path
PFX='[video editing] The target video is an edited version of <Video 1>.'
def main():
 p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True); rows=[]
 for obs in sorted(a.source_run.glob('task_*/stages/S1/observation/observation.json'),key=lambda x:int(x.parents[3].name.split('_')[1])):
  d=json.loads(obs.read_text());
  if d.get('success') is not False: continue
  t=obs.parents[3].name.split('_')[1]; src=obs.parents[3]; dst=a.out_dir/f'task_{t}/stages/S1'; b=dst/'bridge_for_next'; b.mkdir(parents=True,exist_ok=True)
  raw=str(d.get('atomic_prompt','')).strip()
  if not raw:
   plan=Path('/root/autodl-tmp/Pipline_inputs/compact_plans_content_only_qwen38_flash_final_20260906/compact_plans_without_camera_or_global_style')/f'task_{int(t):03d}.json'
   if plan.exists():
    x=json.loads(plan.read_text()); raw=str((x.get('video_diffusion_rounds') or [{}])[0].get('instruction','')).strip()
  prompt=PFX+' MUST visibly complete this requested edit throughout the video. '+(' '.join([raw]*3))
  (b/f'task_{t}_S1_prompt_only_optimized_h3_prompt.txt').write_text(prompt+'\n')
  (dst/'fallback_prompt_only.json').write_text(json.dumps({'task_id':t,'atomic_prompt':raw,'source_video':str(src/'media'/f'task_{t}_initial.mp4'),'reference_images_attached_to_final_prompt':[],'final_refinement':None,'reference_aware_prompt_regenerated':False},indent=2))
  (dst/'reference_generation.json').write_text(json.dumps({'status':'video_only_no_reference_required','source_video':str(src/'media'/f'task_{t}_initial.mp4'),'reference_images':[]},indent=2));rows.append(int(t))
 (a.out_dir/'tasks.json').write_text(json.dumps({'tasks':sorted(rows)}));print(json.dumps({'created':len(rows),'tasks':sorted(rows)}))
if __name__=='__main__':main()
