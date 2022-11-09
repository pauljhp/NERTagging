import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable, Sequence, Optional, Union, Any, Callable, Tuple, Set
import re
from transformers import BertTokenizer
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
import itertools


SPACY_ENG = English()
SPACY_TOKENIZER = Tokenizer(SPACY_ENG.vocab)


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
        target_col: str='O',
        cased: bool=False,
        tokenizer: Optional[str]=None
        ):
        """
        :param target_col: Takes 'X' for CFG parsing tasks or 'O' for NER tasks
        :param tokenizer: Optional, takes 'spacy', 'bert'
        """
        super(NERDataset, self).__init__()
        data_path = Path(data_path).joinpath(f"{mode}.txt")
        data = self.load_data(data_path)
        self.data = data.loc[:, ["DOCSTART", target_col]]
        self.tokens = set(itertools.chain(*self.data.DOCSTART.values))
        self._vocab = self._get_vocabulary(cased, tokenizer)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return tuple(self.data.loc[idx].values)

    def __len__(self):
        return self.data.shape[0]

    def _get_vocabulary(self, 
        cased: bool, 
        tokenizer: Optional[str]=None) -> Set[str]:
        """get the vocabulary occuring in the dataset
        :param tokenizer: Optional, takes 'spacy', 'bert'"""
        if tokenizer is None:
            if cased:
                return self.tokens
            else:
                return {t.lower() for t in self.tokens}
        elif tokenizer in ['spacy']:
            tokens = {SPACY_TOKENIZER(t) for t in self.tokens}
            return {t.doc.text for t in tokens}
        elif tokenizer in ["bert"]:
            if cased:
                bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
            else:
                bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            return bert_tokenizer(list(self.tokens))
        else:
            raise NotImplementedError(f"unrecognized tokenizer {tokenizer}")
    
    @property
    def vocab(self):
        return self._vocab