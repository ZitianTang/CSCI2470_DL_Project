import torch
from .base_dataset import BaseDataset, box_loader, video_loader
# from base_dataset import BaseDataset, box_loader, video_loader
import pandas as pd
import json

from collections import defaultdict

class CLEVRER(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        if split == 'train' and args.efficiency is not None:
            self.data = pd.read_csv(f'./data/clevrer/{split}_{args.efficiency}quarter.csv')
        else:
            self.data = pd.read_csv(f'./data/clevrer/{split}.csv')
        if args.use_cap:
            self.caption = json.load(open(f'./data/clevrer/llava_{args.cap_model}.json'))
        self.mm_features = {}
        self.mm_texts = {}
        self.mm_used = args.mm_used
        self.max_feats = args.max_feats
        self.dlr_format = args.dlr_format
        if args.use_vis:
            self.mm_texts['video'] = defaultdict(lambda: "Video:<|video|>")
            self.mm_features['video'] = video_loader(f'./data/clevrer/clipvitl14.pth', self.max_feats['video'])
        if args.use_box:
            if args.box_format.endswith('_sam2'):
                box_folder = '/oscar/data/superlab/users/ztang47/models/sam2/tracking_clevrer/bounding_boxes'
            else:
                box_folder = '/oscar/data/superlab/datasets/clevrer/features/bounding_boxes'
            data = json.load(open(f'./data/clevrer/bbox/bbox_{args.box_format}.json'))
            self.mm_texts['box'] = {k: v['text'] for k, v in data.items()}
            self.mm_features['box'] = box_loader(
                box_folder = box_folder,
                _format = args.box_format,
                data = data
            )
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)'}
        # self.num_options = 3
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
    max_feats = {'video': 10, 'box': 9}
    dlr_format = False
    use_cap = False
    use_box = True
    use_vis = False
    box_format = 'boxes'
    cap_model = '13b'
    llama_model_path = '/oscar/data/superlab/users/ztang47/models/Vamos2/finetune/finetuned/pretrained/llama3'
    max_seq_len = 1200
    efficiency = None

if __name__ == '__main__':
    args = DatasetArgs()
    from llama import Tokenizer_llama3
    tokenizer = Tokenizer_llama3(model_path=f'{args.llama_model_path}/tokenizer.model')
    dataset = CLEVRER(args, tokenizer=tokenizer, split='val')
    L = []
    from tqdm import tqdm
    for idx in tqdm(list(range(len(dataset)))):
        text = dataset._get_text(idx)
        answer = dataset.data['answer'].values[idx]
        vqa_id, vqa_prefix_index, vqa_mm_starts = tokenizer.encode_vqa(text=text, max_feats=dataset.max_feats, split=dataset.split, answer_mapping=dataset.answer_mapping, answer=answer, dlr_format=dataset.dlr_format)
        # print(len(vqa_id[0]))
        L.append(len(vqa_id[0]))
    print(max(L), sum(L)/len(L))
    # DLR: 813 589.3587267005568 for train and 780 585.725 for val   Use 820
    # projector box: 388 265.37811667877025 for train and 349 260.556 for val   Use 400
    # projector box * 6: 573 for train and 539 for val. Use 600
    # projector box * 9: 686 for train and 654 for val. Use 700
    # vis embedding: 115 78.36372549019607 for train 100 73.187 for val . Use 120
    # caption 785 574.2202614379084 for train and 703 571.415 for val. Use 800
    # textual box 752 552.140813362382 for train and 719 548.243 for val. Use 800
    # vis + cap + tbox 1418 for train and 1284 for val. Use 1440
    # box_labels: 204 for train and 183 for val, use 220