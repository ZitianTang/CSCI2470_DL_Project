import json

fn = '/oscar/data/superlab/users/ztang47/models/llava_new/LLaVA/llava_caption_v1_5_13b/clevrer.jsonl'
save_fn = 'llava_13b.json'

captions = {}
# Open the .jsonl file
for line in open(fn, 'r'):
    a = json.loads(line)
    vid = a['video_uid']
    fid = a['frame_id']
    cap = ' '.join(a['text'].split(' ')[3:])
    if vid not in captions:
        captions[vid] = []
    else:
        assert fid > captions[vid][-1]['frame_id']
    captions[vid].append({
        'frame_id': fid,
        'caption': cap
    })

data = {}
for vid, L in captions.items():
    data[vid] = 'The video shows: ' + '; '.join([x['caption'] for x in L])

with open(save_fn, 'w') as f:
    f.write(json.dumps(data))
    f.close()