import torch
from .base_dataset import BaseDataset, video_loader, box_loader
import pandas as pd
import json

from collections import defaultdict

class STAR(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        self.data = json.load(open(f'./data/star/STAR_{split}.json'))
        if args.use_cap:
            self.caption = json.load(open(f'./data/star/llava_{args.cap_model}.json'))
        self.mm_features = {}
        self.mm_texts = {}
        self.mm_used = args.mm_used
        self.max_feats = args.max_feats
        self.dlr_format = args.dlr_format
        if args.use_vis:
            self.mm_texts['video'] = defaultdict(lambda: "Video:<|video|>")
            self.mm_features['video'] = video_loader(f'./data/star/clipvitl14.pth', self.max_feats['video'])
        if args.use_box:
            data = json.load(open(f'./data/star/bbox/bbox_{args.box_format}.json'))
            self.mm_texts['box'] = {k: v['text'] for k, v in data.items()}
            self.mm_features['box'] = box_loader(
                box_folder = '/oscar/data/superlab/datasets/star/features/bounding_boxes',
                _format = args.box_format,
                data = data
            )
        if args.use_pose:
            data = json.load(open(f'./data/star/pose/pose_{args.pose_format}.json'))
            self.mm_texts['keypoints'] = {k: v['text'] for k, v in data.items()}
            self.mm_features['keypoints'] = pose_loader(
                pose_folder = '/oscar/data/superlab/datasets/star/features/skeleton_npz',
                _format = args.pose_format,
                data = data
            )
            
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)'}
        self.num_options = 4
        self.use_cap = args.use_cap
        self.use_box = args.use_box
        print("use cap", args.use_cap)
        print("modality used:", self.mm_used)
        print(f"Num {split} data: {len(self.data)}")
        
    def _get_text(self, idx):
        question = self.data[idx]['question'].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"

        assert len(self.data[idx]['choices']) == 4
        options = [x['choice'] for x in self.data[idx]['choices']]
        answer = options.index(self.data[idx]['answer'])
        vid = self.data[idx]['video_id']
        qid = self.data[idx]['question_id']
        qid = str(vid) + "_" + str(qid)
        # find caption where video id is vid
        caption = ""
        for m in self.mm_used:
            caption = caption + self.mm_texts[m][vid] + '\n'
        if self.use_cap:
            caption = caption + self.caption[vid].replace('In the video we see these scenes', 'This video shows') + '\n'
            
        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        
        a_text = "Answer: The answer is "
        text = {'c_text': caption, 'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        return text, answer

    def __getitem__(self, idx):
        vid = self.data[idx]['video_id']
        # qtype = self.qtype_mapping[self.data['type'].values[idx]]
        # answer = self.data['answer'].values[idx]
        text, answer = self._get_text(idx)
        text_id, label, mm_starts, label_mask = self._get_text_token(text, answer)
        mm_features = {}
        for m in self.mm_used:
            mm_features[m] = [torch.Tensor(x) for x in self.mm_features[m][vid]]
        
        return {"vid": vid, "mm_features": mm_features, "mm_starts": mm_starts, "text": text, "text_id": text_id, "label": label,
                "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": -1}
        # return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
        #         "video_index": video_index, "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": qtype}

    def __len__(self):
        return len(self.data)

class DatasetArgs:
    mm_used = ['box']
    max_feats = {'box': 1, 'video': 10, 'keypoints': 1}
    dlr_format = False
    use_cap = False
    use_box = True
    use_vis = False
    use_pose = False
    box_format = 'textual_labels'
    pose_format = 'textual'
    cap_model = '13b'
    llama_model_path = '/oscar/data/superlab/users/ztang47/models/Vamos2/finetune/finetuned/pretrained/llama3'
    max_seq_len = 1200
    efficiency = None

if __name__ == '__main__':
    args = DatasetArgs()
    from llama import Tokenizer_llama3
    tokenizer = Tokenizer_llama3(model_path=f'{args.llama_model_path}/tokenizer.model')
    dataset = STAR(args, tokenizer=tokenizer, split='train')
    L = []
    from tqdm import tqdm
    for idx in tqdm(list(range(len(dataset)))):
        text, answer = dataset._get_text(idx)
        vqa_id, vqa_prefix_index, vqa_mm_starts = tokenizer.encode_vqa(text=text, max_feats=dataset.max_feats, split=dataset.split, answer_mapping=dataset.answer_mapping, answer=answer, dlr_format=dataset.dlr_format)
        # print(len(vqa_id[0]))
        L.append(len(vqa_id[0]))
    print(max(L), sum(L)/len(L))
    # vis: 102 for train 98 for val, use 120
    # cap: 1006 for train 952 for val, use 1100
    # box: 688 for train 678 for val, use 700
    # cap + box: 1556 for train 1494 for val, use 1600
    # pose: 810 for train 796 for val, use 850
    # box + pose: 1399 for train 1391 for val, use 1400
    # vis + cap + box + pose: 2280 for train 2218 for val, use 2300
    # box_labels: 188 for train and 172 for val, use 200
    # vis + cap + box textual: 1569 for train and 1507 for val, use 1600