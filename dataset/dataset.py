import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import (Iterable, Sequence, Optional, Union, 
    Any, Callable, Tuple, Set, List, Dict)
import re
from transformers import BertTokenizer
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
import itertools
import utils
import datetime as dt


TODAY = dt.datetime.today().strftime("%Y-%m-%d")
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
        tokenizer: Optional[str]=None,
        unk_token: str="<UNK>",
        pad_token: str="<PAD>",
        from_vocab: Optional[str]="./embeddings/spacy_train_uncased_2022-11-13.json",
        # max_token_len: int=128,
        dtype: torch.dtype=torch.long,
        float_dtype: torch.dtype=torch.float32
        ):
        """
        :param target_col: Takes 'X' for CFG parsing tasks or 'O' for NER tasks
        :param tokenizer: Optional, takes 'spacy', 'bert'
        """
        super(NERDataset, self).__init__()
        data_path = Path(data_path).joinpath(f"{mode}.txt")
        data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.float_dtype = float_dtype
        self.data = data.loc[:, ["DOCSTART", target_col]]
        # self.max_token_len = max_token_len
        self.pad_token, self.unk_token = pad_token, unk_token
        self.tokens = set(itertools.chain(*self.data.DOCSTART.values))
        if from_vocab:
            self._tokenidx, self._vocab = self._get_vocabulary(from_vocab, cased, tokenizer)
            self.vocab_size = len(self._vocab) + 2
            self._token_lookup = pd.Series(self._tokenidx).sort_values().index
        else:
            self._vocab = self._get_vocabulary(from_vocab, cased, tokenizer)
            self.vocab_size = len(self._vocab) + 2 # add <UNK> and <PAD>
            self._token_lookup = list(self._vocab) #+ [self.unk_token, self.pad_token]
            self._tokenidx = dict([(tok, i) for i, tok in enumerate(self._token_lookup)])
            self._tokenidx.update({unk_token: len(self._vocab) + 1, 
                pad_token: len(self._vocab) + 2}) # assign -inf to <PAD>
            self._token_lookup.append(unk_token)
            self._token_lookup.append(pad_token)
            with Path(f"./embeddings/{tokenizer}_{mode}_{'cased' if cased else 'uncased'}_{TODAY}").open("w") as f:
                json.dump(self._tokenidx, f)
        targetidx_path = Path("./embeddings/nertag_idx.json")
        if targetidx_path.exists():
            with targetidx_path.open("r") as f:
                self._targetidx = json.load(f)
            self._targets = set(self._targetidx.keys())
            self.ntargets = len(self._targets) - 1 # remove <PAD>
        else:
            self._targets = set(itertools.chain(*self.data.loc[:, target_col]))
            self.ntargets = len(self._targets)
            self._target_lookup = list(self._targets)
            self._targetidx = dict([(tgt, i + 1) for i, tgt 
                in enumerate(self._target_lookup)])
            self._targetidx.update({self.pad_token: 0}) # very small integer for <PAD>
            with targetidx_path.open("w") as f:
                json.dump(self._targetidx, f)
        self.data.loc[:, "DOCSTART"] = self.data.DOCSTART.apply(
            lambda x: self._tokenize(x, cased=cased))
        self.data.loc[:, "target_idx"] = self.data.loc[:, target_col].apply(
            lambda x: torch.tensor([self._targetidx.get(t) for t in x], 
                dtype=self.dtype))
        self.data.loc[:, "text_idx"] = self.data.DOCSTART.apply(
            lambda x: torch.tensor([self._tokenidx.get(t) 
                    if t in self._vocab else self._tokenidx.get(unk_token)
                    for t in x],
                dtype=self.dtype))
        self.data = self.data.loc[
            self.data.target_idx.apply(lambda x: True if len(x) > 0 else False)] # filter out empty entries]

    def get_token_from_idx(self, 
        token_idx: Union[Sequence[int], int]) -> Union[str, List[str]]:
        """get the tokens from their IDs"""
        if isinstance(token_idx, int):
            return self._token_lookup[token_idx]
        elif isinstance(token_idx, Sequence):
            return [self._token_lookup[i] for i in token_idx]
        else:
            raise TypeError(f"token_idx must be int or Sequence[int], but is type {type(token_idx)}")


    def __getitem__(self, idx: int) -> Tuple[torch.tensor]:
        """:returns: features, tag_prob, tags"""
        features, tags = tuple(self.data.loc[idx, ["text_idx", "target_idx"]].values)
        tag_prob = utils.tagidx_to_prob(tags, self.ntargets, self.float_dtype)
        return (features, tag_prob, tags, idx)

    def __len__(self):
        return self.data.shape[0]

    def _tokenize(self, 
        tokens: Union[Sequence[str], str],
        # from_vocab: Optional[str]="./embeddings/spacy_train_uncased_2022-11-13.json",
        cased: bool=False, 
        tokenizer: Optional[str]=None) -> Sequence[str]:
        """tokenize the corpus
        :param tokenizer: Optional, takes 'spacy', 'bert'"""

        if tokenizer is None:
            if cased:
                return tokens
            else:
                return [str(token).lower() for token in tokens]
        elif tokenizer in ['spacy']:
            tokens = [SPACY_TOKENIZER(t) for t in self.tokens]
            if cased:
                return [t.doc.text for t in tokens]
            else:
                return [t.doc.text.lower() for t in tokens]
        elif tokenizer in ["bert"]:
            if cased:
                bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
            else:
                bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            return bert_tokenizer(list(self.tokens))
        else:
            raise NotImplementedError(f"unrecognized tokenizer {tokenizer}")

    def _get_vocabulary(self, 
        from_vocab: Optional[str]="./embeddings/spacy_train_uncased_2022-11-13.json",
        cased: bool=False, 
        tokenizer: Optional[str]=None) -> Tuple[Dict[str, int], Optional[Set[str]]]:
        """get the vocabulary occuring in the dataset
        :param tokenizer: Optional, takes 'spacy', 'bert'"""
        if from_vocab:
            with Path(from_vocab).open("r") as f:
                vocab_dict = json.load(f)
            vocabs = {t for t in vocab_dict.keys() 
                if t not in (self.unk_token, self.pad_token)}
            return vocab_dict, vocabs

        else:
            if tokenizer is None:
                if cased:
                    return self.tokens
                else:
                    return {t.lower() if t not in (self.unk_token, self.pad_token) 
                        else t for t in self.tokens}
            elif tokenizer in ['spacy']:
                tokens = {SPACY_TOKENIZER(t) if t not in (self.unk_token, self.pad_token) 
                        else t for t in self.tokens}
                tokens = {t.doc.text for t in tokens}
                if cased:
                    return tokens
                else:
                    return {t.lower() if t not in (self.unk_token, self.pad_token)
                        else t for t in tokens}
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

    @property
    def tags(self):
        return self._tags

    def tagidx(self, tag: str):
        return self._tagidx.get(tag)
