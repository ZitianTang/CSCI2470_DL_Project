import numpy as np
import decord

import json
import os

from glob import glob
from tqdm import tqdm

from llama import Tokenizer_llama, Tokenizer_llama3
llama_model_path = 'finetuned/pretrained/llama3/'
tokenizer = Tokenizer_llama3(model_path='/oscar/data/superlab/users/ztang47/models/Vamos/finetune/finetuned/pretrained/llama3/tokenizer.model')

bbox_dir = '/oscar/data/superlab/datasets/PerceptionTest/features/bounding_boxes'

# format_func = 'textual' # boxes, boxes_time
format_func = 'textual_clip'
gap = 0

box_token = '<|box|>'

def parse_object_tracking_boxes(video_path, gap):
    objs = glob(f'{video_path}/*.npz')
    objs.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    P = [
        'Object tracking:',
        f'There are {len(objs)} objects in the video.',
        'Their bounding boxes: ',
    ]
    P = ' '.join(P) + '\n'
    boxes = []
    for obj_pth in objs:
        obj_id = obj_pth.split('/')[-1].split('.')[0]
        obj = np.load(obj_pth)
        P = P + f'(Object {obj_id}) {obj["label"].item()} - '
        fps = obj['fps']
        last = -100000
        for i, f in enumerate(obj["frame_ids"]):
            if f >= last + fps * gap:
                P = P + f'frame {f} {box_token} '
                boxes.append({
                    'object_id': int(obj_id),
                    'frame_id': i
                })
                last = f
        P = P + ';\n'
    return {'text': P, 'box': boxes}

def parse_object_tracking_boxes_time(video_path, gap):
    objs = glob(f'{video_path}/*.npz')
    objs.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    P = [
        'Object tracking:',
        f'There are {len(objs)} objects in the video.',
        'Their bounding boxes: ',
    ]
    P = ' '.join(P) + '\n'
    boxes = []
    for obj_pth in objs:
        obj_id = obj_pth.split('/')[-1].split('.')[0]
        obj = np.load(obj_pth)
        P = P + f'(Object {obj_id}) {obj["label"].item()} - '
        fps = obj['fps']
        last = -100000
        for i, f in enumerate(obj["frame_ids"]):
            if f >= last + fps * gap:
                P = P + f'{box_token} '
                boxes.append({
                    'object_id': int(obj_id),
                    'frame_id': i
                })
                last = f
        P = P + ';\n'
    return {'text': P, 'box': boxes}

def constrain_len(ret):
    if len(tokenizer.encode(ret['text'], bos=False, eos=False)) > 1000:
        return {'text': 'Object Tracking: None', 'box': []}
    else:
        return ret

def parse_object_tracking_integer(video_path, gap):
    objs = glob(f'{video_path}/*.npz')
    objs.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    P = [
        'Object tracking:',
        f'There are in total {len(objs)} objects in the video.',
        'For each object, I will give its bounding boxes at certain frames.',
        'The bounding boxes are in format [x1 y1 x2 y2] with each value between 0 and 100.',
        '[x1 x2] is the boundary along the video width, from the left (0) to the right (100).',
        '[y1 y2] is the boundary along the video height, from the top (0) to the bottom (100).'
    ]
    P = ' '.join(P) + '\n'
    for obj_pth in objs:
        obj_id = obj_pth.split('/')[-1].split('.')[0]
        obj = np.load(obj_pth)
        P = P + f'(Object {obj_id}) {obj["label"].item()} - '
        fps = obj['fps']
        last = -100000
        for f, b in zip(obj["frame_ids"], obj["bounding_boxes"]):
            if f >= last + fps * gap:
                P = P + f'frame {f} [{int(b[0] * 100 + 0.5)} {int(b[1] * 100 + 0.5)} {int(b[2] * 100 + 0.5)} {int(b[3] * 100 + 0.5)}] '
                last = f
        P = P + ';\n'
    return {'text': P, 'box': []}

def parse_object_tracking_integer_clip(video_path, gap):
    objs = glob(f'{video_path}/*.npz')
    objs.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    P = [
        'Object tracking:',
        f'There are in total {len(objs)} objects in the video.',
        'For each object, I will give its bounding boxes at certain frames.',
        'The bounding boxes are in format [x1 y1 x2 y2] with each value between 0 and 100.',
        '[x1 x2] is the boundary along the video width, from the left (0) to the right (100).',
        '[y1 y2] is the boundary along the video height, from the top (0) to the bottom (100).'
    ]
    P = ' '.join(P) + '\n'
    for obj_pth in objs:
        obj_id = obj_pth.split('/')[-1].split('.')[0]
        obj = np.load(obj_pth)
        P = P + f'(Object {obj_id}) {obj["label"].item()} - '
        fps = obj['fps']
        last = -100000
        for f, b in zip(obj["frame_ids"], obj["bounding_boxes"]):
            if f >= last + fps * gap:
                P = P + f'frame {f} [{int(b[0] * 100 + 0.5)} {int(b[1] * 100 + 0.5)} {int(b[2] * 100 + 0.5)} {int(b[3] * 100 + 0.5)}] {box_token} '
                last = f
        P = P + ';\n'
    return {'text': P, 'box': []}

def parse_object_tracking_labels(video_path, gap):
    objs = glob(f'{video_path}/*.npz')
    objs.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    P = [
        'Object tracking:',
        f'There are in total {len(objs)} objects in the video. They are: '
    ]
    P = ' '.join(P)
    for obj_pth in objs:
        obj_id = obj_pth.split('/')[-1].split('.')[0]
        obj = np.load(obj_pth)
        P = P + f'(Object {obj_id}) {obj["label"].item()}, '
        # fps = obj['fps']
        # last = -100000
        # for f, b in zip(obj["frame_ids"], obj["bounding_boxes"]):
        #     if f >= last + fps * gap:
        #         P = P + f'frame {f} [{int(b[0] * 100 + 0.5)} {int(b[1] * 100 + 0.5)} {int(b[2] * 100 + 0.5)} {int(b[3] * 100 + 0.5)}] '
        #         last = f
        # P = P + ';\n'
    P = P[:-2] + '.'
    return {'text': P, 'box': []}

d = {}
# gaps = []
# nframes = []
gaps = json.load(open('flexible_calc_gaps.json'))
for video_path in tqdm(glob(f'{bbox_dir}/*')):
    vid = video_path.split('/')[-1]
    if format_func == 'boxes':
        c_text = constrain_len(parse_object_tracking_boxes(video_path, gaps[vid]))
    elif format_func == 'boxes_time':
        c_text = constrain_len(parse_object_tracking_boxes_time(video_path, gaps[vid]))
    elif format_func == 'textual':
        c_text = constrain_len(parse_object_tracking_integer(video_path, gaps[vid]))
    elif format_func == 'textual_labels':
        c_text = constrain_len(parse_object_tracking_labels(video_path, gaps[vid]))
    elif format_func == 'textual_clip':
        c_text = parse_object_tracking_integer_clip(video_path, gaps[vid])
        d_text = constrain_len(parse_object_tracking_integer(video_path, gaps[vid]))
        if 'None' in d_text['text']:
            c_text = d_text
    d[vid] = c_text
    # break

if format_func == 'flexible_calc':
    with open(f'flexible_calc_gaps.json', 'w') as f:
        f.write(json.dumps(gaps))
        f.close()
else:
    with open(f'bbox_{format_func}.json', 'w') as f:
        f.write(json.dumps(d))
        f.close()