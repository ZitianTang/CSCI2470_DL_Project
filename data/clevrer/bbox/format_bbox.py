import numpy as np

import json
import os

from glob import glob
from tqdm import tqdm

bbox_dir = '/oscar/data/superlab/datasets/clevrer/features/bounding_boxes'

# format_func = 'integer'
format_func = 'textual_labels' # boxes, boxes_time, textual

box_token = '<|box|>'

def parse_object_tracking_labels(video_path):
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
    P = P[:-2] + '.'
    return {'text': P, 'box': []}

def parse_object_tracking_textual(video_path):
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
            if f >= last + fps:
                P = P + f'frame {f} [{int(b[0] * 100 + 0.5)} {int(b[1] * 100 + 0.5)} {int(b[2] * 100 + 0.5)} {int(b[3] * 100 + 0.5)}] '
                last = f
        P = P + ';\n'
    return {'text': P, 'box': []}

def parse_object_tracking_textual_dlr(video_path):
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
            if f >= last + fps:
                P = P + f'frame {f} [{int(b[0] * 100 + 0.5)} {int(b[1] * 100 + 0.5)} {int(b[2] * 100 + 0.5)} {int(b[3] * 100 + 0.5)}] {box_token} '
                last = f
        P = P + ';\n'
    return {'text': P, 'box': []}

def parse_object_tracking_boxes(video_path):
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
            if f >= last + fps:
                P = P + f'frame {f} {box_token} '
                boxes.append({
                    'object_id': int(obj_id),
                    'frame_id': i
                })
                last = f
        P = P + ';\n'
    return {'text': P, 'box': boxes}

def parse_object_tracking_boxes_time(video_path):
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
            if f >= last + fps:
                P = P + f'{box_token} '
                boxes.append({
                    'object_id': int(obj_id),
                    'frame_id': i
                })
                last = f
        P = P + ';\n'
    return {'text': P, 'box': boxes}

# def parse_object_tracking_labels(video_path):
#     objs = glob(f'{video_path}/*.npz')
#     objs.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
#     P = [
#         'Object tracking:',
#         f'There are in total {len(objs)} objects in the video.',
#         'They are'
#     ]
#     P = ' '.join(P)
#     O = []
#     for obj_pth in objs:
#         obj_id = obj_pth.split('/')[-1].split('.')[0]
#         obj = np.load(obj_pth)
#         O.append(f'(Object {obj_id}) {obj["label"].item()}')
#     P = P + ' ' + ', '.join(O) + '.\n'
#     return P

d = {}
for video_path in tqdm(glob(f'{bbox_dir}/*')):
    if format_func == 'boxes':
        c_text = parse_object_tracking_boxes(video_path)
    elif format_func == 'boxes_time':
        c_text = parse_object_tracking_boxes_time(video_path)
    elif format_func == 'textual_dlr':
        c_text = parse_object_tracking_textual_dlr(video_path)
    elif format_func == 'textual':
        c_text = parse_object_tracking_textual(video_path)
    elif format_func == 'textual_labels':
        c_text = parse_object_tracking_labels(video_path)
    # elif format_func == 'labels':
    #     c_text = parse_object_tracking_labels(video_path)
    # else:
    #     c_text = parse_object_tracking_integer(video_path)
    vid = video_path.split('/')[-1]
    d[vid] = c_text
    # break
with open(f'bbox_{format_func}.json', 'w') as f:
    f.write(json.dumps(d))
    f.close()