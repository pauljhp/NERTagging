import torch
import torch.nn as nn
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import utils
import pandas as pd
import pickle
import numpy as np
from collections import OrderedDict


def load_glove_embeddings(glove_dir: str="./embeddings/glove",
    glove_ver: str="glove.6B",
    embedding_dim: int=50,
    **kwargs
    ):
    """load glove embeddings"""
    pkl_p = Path(glove_dir).joinpath(f"{glove_ver}.{embedding_dim}d.pkl")
    txt_p = Path(glove_dir).joinpath(f"{glove_ver}.{embedding_dim}d.txt")
    if pkl_p.exists():
        glove_dict = pickle.load(pkl_p.open("rb"))
        return glove_dict
    else:
        try:
            df = pd.read_csv(txt_p.as_posix(), sep=" ", index_col=0)
            glove_dict = OrderedDict(df.to_dict())
            pickle.dump(glove_dict, pkl_p.open("wb"))
            return glove_dict
        except:
            glove_dict = OrderedDict()
            with txt_p.open("r") as f:
                values = f.read()
                for line in values.split("\n"):
                    val = line.split(" ")
                    if len(val) >= 2:
                        k, v = val[0], val[1:]
                    glove_dict[k] = [float(i) for i in v]
            pickle.dump(glove_dict, pkl_p.open("wb"))
            return glove_dict
        


def get_embedder(embedding: str="torch", **kwargs):
    if embedding in ["torch", "pytorch"]:
        embedder = nn.Embedding(**kwargs)
    elif embedding in ["glove"]:
        glove = load_glove_embeddings(**kwargs)
        vocab_npa = np.array(list(glove.keys()))
        embs_npa = np.array(list(glove.values()))
        vocab_npa = np.insert(vocab_npa, len(vocab_npa), '<UNK>')
        vocab_npa = np.insert(vocab_npa, len(vocab_npa), '<PAD>')
        pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
        unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True) 
        embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))
        embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())
        assert embedding.weight.shape == embs_npa.shape
        return embedding
    else:
        raise NotImplementedError
    return embedder

class WordEmbedding(nn.Module):
    def __init__(self, 
        vocab_size: int,
        embedding_dim: int,
        pad_token_idx: int,
        embedding: str="torch",
        ):
        """
        :param vocab_size: vocabulary size
        :param embedding_dim: dimensionality of the embeddings
        :param embedding: str, takes "torch", "glove", "pytorch" 
            (same as "torch"), and "bert"
        """
        super(WordEmbedding, self).__init__()
        self.embedding = get_embedder(embedding, 
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim,
            padding_idx=pad_token_idx,
            max_norm=1e2,
            norm_type=2.0)
        self.embedding_dim = embedding_dim
        self.pad_token_idx = pad_token_idx
        
    def forward(self, x: torch.tensor):
        x_ = self.embedding(x)
        return x_