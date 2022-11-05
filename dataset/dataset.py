import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable, Sequence, Optional, Union, Any, Callable


class NERDataset(Dataset):
    def load_data(data_path: Path):
        with data_path.open("r") as f:
            txt = f.read()
        headers = txt.split("\n\n")[0].split(" ")
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
            for sent in sentences])
        sentences.index = range(sentences.shape[0])
        return sentences

    def __init__(self, data_path: str="./data/conll2003", mode: str='train'):
        super(NERDataset, self).__init__()
        data_path = Path(data_path).joinpath(f"{mode}.txt")
        self.data = self.load_data(data_path)