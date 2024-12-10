import torch
from .base_dataset import BaseDataset, video_loader, box_loader
# from base_dataset import BaseDataset, video_loader, box_loader
import pandas as pd
import json

from collections import defaultdict

class PTest(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        if split == 'train' and args.efficiency is not None:
            self.data = pd.read_csv(f'./data/ptest/{split}_{args.efficiency}quarter.csv')
        elif split == 'val' and args.subset:
            self.data = pd.read_csv(f'./data/ptest/{split}_subset.csv')
        else:
            self.data = pd.read_csv(f'./data/ptest/{split}.csv')
        if args.use_cap:
            self.caption = json.load(open(f'./data/ptest/llava_{args.cap_model}_n6.json'))
        self.mm_features = {}
        self.mm_texts = {}
        self.mm_used = args.mm_used
        self.max_feats = args.max_feats
        self.dlr_format = args.dlr_format
        if args.use_vis:
            self.mm_texts['video'] = defaultdict(lambda: "Video:<|video|>")
            self.mm_features['video'] = video_loader(f'./data/ptest/clipvitl14.pth', self.max_feats['video'])
        if args.use_box:
            data = json.load(open(f'./data/ptest/bbox/bbox_{args.box_format}.json'))
            self.mm_texts['box'] = {k: v['text'] for k, v in data.items()}
            self.mm_features['box'] = box_loader(
                box_folder = '/oscar/data/superlab/datasets/PerceptionTest/features/bounding_boxes',
                _format = args.box_format,
                data = data
            )
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)'}
        self.num_options = 3
        self.use_cap = args.use_cap
        self.use_box = args.use_box
        print("use cap", args.use_cap)
        print("modality used:", self.mm_used)
        print(f"Num {split} data: {len(self.data)}")
        
    def _get_text(self, idx):
        question = self.data["question"].values[idx].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"

        num_options = self.data['num_option'].values[idx]
        options = [self.data[f'a{i}'].values[idx] for i in range(num_options)]
        vid = self.data['video'].values[idx]
        qid = self.data['qid'].values[idx]
        # find caption where video id is vid
        caption = ""
        for m in self.mm_used:
            caption = caption + self.mm_texts[m][vid] + '\n'
        if self.use_cap:
            caption = caption + self.caption[vid] + '\n' # .split("The video shows: ")[-1].strip()
        # if self.use_pose:
        #     caption = caption + self.pose[vid]

        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        
        a_text = "Answer: The answer is "
        text = {'c_text': caption, 'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        return text

    def __getitem__(self, idx):
        vid = self.data['video'].values[idx]
        answer = self.data['answer'].values[idx]
        text = self._get_text(idx)
        text_id, label, mm_starts, label_mask = self._get_text_token(text, answer)
        mm_features = {}
        for m in self.mm_used:
            mm_features[m] = [torch.Tensor(x) for x in self.mm_features[m][vid]]

        return {"vid": vid, "mm_features": mm_features, "mm_starts": mm_starts, "text": text, "text_id": text_id, "label": label,
                "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": -1}

    def __len__(self):
        return len(self.data)

class DatasetArgs:
    mm_used = ['box']
    max_feats = {'video': 10, 'box': 1}
    dlr_format = False
    use_cap = False
    use_box = True
    use_vis = False
    box_format = 'boxes'
    cap_model = '13b'
    llama_model_path = '/oscar/data/superlab/users/ztang47/models/Vamos2/finetune/finetuned/pretrained/llama3'
    max_seq_len = 1200
    efficiency = None
    subset = False

if __name__ == '__main__':
    args = DatasetArgs()
    from llama import Tokenizer_llama3
    tokenizer = Tokenizer_llama3(model_path=f'{args.llama_model_path}/tokenizer.model')
    dataset = PTest(args, tokenizer=tokenizer, split='val')
    L = []
    from tqdm import tqdm
    for idx in tqdm(list(range(len(dataset)))):
        text = dataset._get_text(idx)
        answer = dataset.data['answer'].values[idx]
        vqa_id, vqa_prefix_index, vqa_mm_starts = tokenizer.encode_vqa(text=text, max_feats=dataset.max_feats, split=dataset.split, answer_mapping=dataset.answer_mapping, answer=answer, dlr_format=dataset.dlr_format)
        # print(len(vqa_id[0]))
        L.append(len(vqa_id[0]))
    print(max(L), sum(L)/len(L))
    # vis + cap + tbox 1873 for train and 2112 for val. Use 2200
    # box_labels: 486 for train and 535 for vall. Use 560
    # box textual: 1114 for train and 1127 for val. Use 1200
    # box projector: 633 for train and 689 for val. Use 700
    # box textual clip: 1238 for train and 1257 for val. Use 1260