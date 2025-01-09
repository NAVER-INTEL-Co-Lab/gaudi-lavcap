import json
import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import LlamaTokenizer
class ASRDataset(Dataset):
    def __init__(self, asr_ann_path, asr_feat_path, llama_path, max_txt_len=128):
        super().__init__()

        self.asr_feat_path = asr_feat_path
        self.asr_annotation = json.load(open(asr_ann_path, "r"))

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_path)
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) ## pad 원래 없음??
        self.llama_tokenizer.padding_side = "right"

        self.max_txt_len = max_txt_len
    
    def load_audio(self, audio_path):
        audio_path = os.path.join(self.asr_feat_path, audio_path + '.npz')
        audio = np.load(audio_path)
        audio = torch.tensor(audio['arr_0'], dtype=torch.float32).squeeze(0)
        return audio

    def __len__(self):
        return len(self.asr_annotation)
    
    def __getitem__(self, index):
        anno = self.asr_annotation[index]
        audio_id = anno['audio_id']
        feature = self.load_audio(audio_id)
        transcription = anno['input']

        labels = f"spoken text: {transcription}</s>"

        return {
            "id": audio_id,
            "spectrogram": feature,
            "asr": f"spoken text: {transcription}",
            "labels": labels,
            "task": "asr",
        }
    