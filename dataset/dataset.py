import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable, Sequence, Optional, Union, Any, Callable, Tuple
import re
from transformers import BERTTokenizer
import spacy


BERT_TOKENIZER = BERTTokenizer.from_pretrained()


class NERDataset(Dataset):
    def load_data(self, data_path: Path):
        with data_path.open("r") as f:
            txt = f.read()
        headers = [re.sub("[\-]", "", h) for h in 
                txt.split("\n\n")[0].split(" ")]
        sentences = [[tok.split(" ") for tok in sent.split("\n")] 
            for sent in txt.split("\n\n")[1:]
            ]
        sentences = [{k: [tok[i] for tok in sent if len(tok) > i] 
                    for i, k in enumerate(headers)
                } 
            for sent in sentences]
        sentences = pd.concat([pd.Series(
            data=sent, 
            dtype='object', 
            index=headers).to_frame().T 
            for sent in sentences if sent.get("DOCSTART")])
        sentences.index = range(sentences.shape[0])
        return sentences

    def __init__(self, 
        data_path: str="./data/conll2003", 
        mode: str='train', 
        target_col: str='O'):
        """
        :param target_col: Takes 'X' for CFG parsing tasks or 'O' for NER tasks
        """
        super(NERDataset, self).__init__()
        data_path = Path(data_path).joinpath(f"{mode}.txt")
        data = self.load_data(data_path)
        self.data = data.loc[:, ["DOCSTART", target_col]]

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return tuple(self.data.loc[idx].values)

    def __len__(self):
        return self.data.shape[0]

    def _get_vocabulary(cased: bool, tokenizer: str='spacy')