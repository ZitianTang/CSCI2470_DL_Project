import pandas as pd
import json

df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('val.csv')

video_list = df1['video'].tolist() + df2['video'].tolist()
video_list = list(set(video_list))
video_list.sort()

with open('vid_list.json', 'w') as f:
    f.write(json.dumps(video_list, indent=4))
    f.close()