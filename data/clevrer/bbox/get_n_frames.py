import decord
import json
import pandas as pd
from tqdm import tqdm

def get_vids_(dataset, split):
    L = []
    try:
        fn = f'/oscar/data/superlab/users/ztang47/models/Vamos2/finetune/data/{dataset}/{split}.csv'
        df = pd.read_csv(fn)
        for i, row in df.iterrows():
            L.append(row['video'])
    except:
        pass
    return L

def get_vids(dataset):
    L = get_vids_(dataset, 'train') + get_vids_(dataset, 'val')
    L = list(set(L))
    return L

g = 1

vids = get_vids('clevrer')

video_folder = '/oscar/data/superlab/datasets/clevrer'

d = {}
for k in tqdm(vids):
    x = int(k.split('_')[-1]) // 1000 * 1000
    f = f'video_{str(x).zfill(5)}-{str(x + 1000).zfill(5)}'
    video_path = f'{video_folder}/{f}/{k}.mp4'
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    vlen = len(vr)
    n_frames = 1 + int(vlen // (fps * g))
    d[k] = n_frames

with open('flexible_calc_nframes.json', 'w') as f:
    f.write(json.dumps(d))
    f.close()