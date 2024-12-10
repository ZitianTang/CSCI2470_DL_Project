import pandas as pd
import json

from tqdm import tqdm

questions = json.load(open('/oscar/data/superlab/users/ztang47/datasets/clevrer/clevrer_mvbench/VideoChat2-IT/clevrer_VideoChat2-IT.json'))

# L = []
# for q in questions:
#     L.append(len(q['options']))
# print(max(L)) # 4


data = []
for q in tqdm(questions):
    Q = {
        'video': q['video_id'],
        'num_option': len(q['options']),
        'qid': f'clevrer_it_{q["video_id"]}_{q["question_id"]}',
        'answer': q['answer'],
        'question': q['question']
    }
    for i in range(Q['num_option']):
        Q[f'a{i}'] = q['options'][i]
    data.append(Q)

df = pd.DataFrame(data, columns=['video', 'num_option', 'qid', 'a0', 'a1', 'a2', 'a3', 'answer', 'question', 'start', 'end'])

df.to_csv('train.csv', index=False)