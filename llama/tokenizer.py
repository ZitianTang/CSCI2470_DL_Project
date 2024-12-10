# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from .llama3_tokenizer import Tokenizer as Tokenizer3
from sentencepiece import SentencePieceProcessor

from logging import getLogger
from typing import List
import os
import torch

logger = getLogger()

mm_tokens = {
    '<|video|>': 'video',
    '<|box|>': 'box',
    '<|keypoints|>': 'keypoints'
}

class Tokenizer_llama3:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.tk_model = Tokenizer3(model_path=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.tk_model.n_words
        self.bos_id: int = self.tk_model.bos_id
        self.eos_id: int = self.tk_model.eos_id
        self.pad_id: int = self.tk_model.pad_id
        self.unk_id: int = self.tk_model.pad_id
        
        self.v_token_id = 10955
        self.q_token_id = 14924
        self.a_token_id = 16533
        self.nl_id = 198
        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tk_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def encode_w_mm_tokens(self, s, max_feats):
        texts = [s]
        for m in mm_tokens:
            new_texts = []
            for t in texts:
                t = t.split(m)
                for x in t:
                    if x.strip() != '':
                        new_texts.append(x)
                    new_texts.append(m)
                new_texts = new_texts[:-1]
            texts = new_texts
        mm_starts = []
        L = [self.bos_id]
        for t in texts:
            if t in mm_tokens:
                m = mm_tokens[t]
                mm_starts.append((m, len(L)))
                L = L + [self.unk_id for _ in range(max_feats[m])]
            else:
                L = L + self.tk_model.encode(t)
        L = L + [self.eos_id]
        return L, mm_starts

    def encode_vqa(self, text=None, max_feats=None, split='train', answer_mapping=None, answer=None, dlr_format=False): # -> List[int]:
        i_text = "Instruction: Predict the answer based on the video and question.\n"
        q_text = text['q_text']
        o_text = text['o_text']
        a_text = text['a_text']
        c_text = text['c_text']

        # if c_text != "":
        #     # q_text = "Video: " + c_text + "\n" + q_text
        #     q_text = c_text + q_text

        # s1 = i_text + 'Video:'
        # t1 = [self.bos_id] + self.tk_model.encode(s1) 
        # video_start = len(t1)


        # s2 = q_text + o_text + a_text

        # bos + i_text + 'Video:' + q_text + o_text + a_text + answer + eos

        if split == 'train':
            # s = i_text + 'Video:' + q_text + o_text + a_text + answer_mapping[answer]
            if dlr_format:
                # print('using dlr_format')
                s = i_text + q_text + o_text + c_text + a_text + answer_mapping[answer]
            else:
                s = i_text + c_text + q_text + o_text + a_text + answer_mapping[answer]
            # print(s)
            w, mm_starts = self.encode_w_mm_tokens(s, max_feats)
            t = [w]
            # print(t[0])
            prefix_index = t[0].index(self.a_token_id) + 5
        else:
            t = []
            for k, v in answer_mapping.items():
                if dlr_format:
                    s = i_text + q_text + o_text + c_text + a_text + v
                else:
                    s = i_text + c_text + q_text + o_text + a_text + v
                w, mm_starts = self.encode_w_mm_tokens(s, max_feats)
                t.append(w)
            prefix_index = t[answer].index(self.a_token_id) + 5
        return t, prefix_index, mm_starts

    def decode(self, t: List[int]) -> str:
        return self.tk_model.decode(t)
    

class Tokenizer_llama:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        self.unk_id: int = self.sp_model.unk_id()
        
        self.v_token_id = 15167
        self.q_token_id = 16492
        self.a_token_id = 22550
        self.nl_id = 13
        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def encode_vqa(self, text=None, max_feats=10, split='train', answer_mapping=None, answer=None) -> List[int]:
        i_text = "Instruction: Predict the answer based on the video and question.\n"
        q_text = text['q_text']
        o_text = text['o_text']
        a_text = text['a_text']
        c_text = text['c_text']

        if c_text != "":
            q_text = "Video: " + c_text + "\n" + q_text

        s1 = i_text + 'Video:'
        t1 = [self.bos_id] + self.sp_model.encode(s1) 
        video_start = len(t1)


        s2 = q_text + o_text + a_text

        if split == 'train':
            s2 = s2 + answer_mapping[answer] 
            t2 = self.sp_model.encode(s2) + [self.eos_id]
            t = [t1 + [self.unk_id for _ in range(max_feats)] + [self.nl_id] + t2]
            prefix_index = t[0].index(self.a_token_id) + 5
        else:
            t = []
            for k, v in answer_mapping.items():
                t2 = self.sp_model.encode(s2 + v) + [self.eos_id]
                t.append(t1 + [self.unk_id for _ in range(max_feats)] + [self.nl_id] + t2)
            prefix_index = t[answer].index(self.a_token_id) + 5
        return t, prefix_index, video_start

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)


# if __name__ == '__main__':
#     def encode_try(s: str):
#         texts = [s]
#         for m in mm_tokens:
#             new_texts = []
#             for t in texts:
#                 t = t.split(m)
#                 for x in t:
#                     if x.strip() != '':
#                         new_texts.append(x)
#                     new_texts.append(m)
#                 new_texts = new_texts[:-1]
#             texts = new_texts
#         return texts
    
#     print(encode_try('This is a sentence with <|video|> token, <|keypoints|> token, and <|box|> <|box|> tokens. The first one is <|video|> and last is <|box|>.'))