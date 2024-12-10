import decord
import json

from tqdm import tqdm

gaps = json.load(open('/oscar/data/superlab/users/ztang47/models/Vamos2/finetune/data/ptest/bbox/flexible_calc_gaps.json'))

video_folder = '/oscar/data/superlab/datasets/PerceptionTest/videos'

d = {}
for k, g in tqdm(gaps.items()):
    video_path = f'{video_folder}/{k}.mp4'
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    vlen = len(vr)
    n_frames = 1 + int(vlen // (fps * g))
    d[k] = n_frames

with open('flexible_calc_nframes.json', 'w') as f:
    f.write(json.dumps(d))
    f.close()