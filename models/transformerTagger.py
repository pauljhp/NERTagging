# import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import (Sequence, Iterable, Dict, Tuple, Callable, Optional)
import logging
import datetime as dt
from models.embedding import WordEmbedding


TODAY = dt.datetime.today().strftime("%Y-%m-%d")
logging.basicConfig(filename="./log/exceptions.log")
LOGGER = logging.getLogger()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class TransformerTagger(nn.Module):
    def __init__(self, 
        pad_token_idx: int,
        d_model: int=512,
        nhead: int=8,
        vocab_size: int=30000,
        num_encoder_layers: int=10,
        num_decoder_layers: int=8,
        dim_feedforward: int=1024,
        dropout: float=.1,
        embedding_type: str="torch",
        num_dense_layers: int=5,
        activation: Callable=F.relu,
        batch_first: bool=True,
        layer_norm_eps: float=1e-4,
        # device: str=DEVICE,
        n_tags: int=10,
        ):
        super(TransformerTagger, self).__init__()
        assert d_model >= n_tags, "d_model must be higher than number of tags"
        self.d_model = d_model
        self.no_dense_layers = num_dense_layers
        self.pad_token_idx = pad_token_idx
        self.positional_encoder = PositionalEncoder(d_model, dropout)
        self.embedding = WordEmbedding(vocab_size=vocab_size,
            embedding_dim=d_model,
            embedding=embedding_type, 
            pad_token_idx=pad_token_idx)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            layer_norm_eps=layer_norm_eps,
            norm_first=True,
        )
        increment = math.floor(
            (d_model // n_tags) ** (1 / num_dense_layers)
            )
        in_features, out_features = d_model, d_model // increment
        for i in range(1, num_dense_layers):
            exec(f"""self.dense{i} = nn.Linear(in_features=in_features, 
                out_features=out_features)""")
            in_features, out_features = out_features, \
                out_features // increment
        exec(f"""self.dense{num_dense_layers} = nn.Linear(in_features=in_features, 
                out_features=n_tags)""")

        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, 
        source: torch.tensor, 
        target: Optional[torch.tensor]=None,
        src_key_padding_mask: Optional[torch.tensor]=None,
        tgt_key_padding_mask: Optional[torch.tensor]=None,
        ):
        source = self.embedding(source) * math.sqrt(self.d_model)
        source = self.positional_encoder(source)
        if target is not None:
            target = self.embedding(target) * math.sqrt(self.d_model)
            target = self.positional_encoder(target)
        else: target = source
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = src_key_padding_mask
        dense_in = self.transformer(source, target,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask)
        # dense_in = torch.relu(dense_in) # experiment with this
        # dense_in = torch.clip(dense_in, min=-0.99999, max=0.99999)
        for i in range(1, self.no_dense_layers + 1):
            out = eval(f"self.dense{i}(dense_in)")
            out = torch.tanh(out)
            # out = torch.clip(out, min=-0.99999, max=0.99999)
            dense_in = out

        out_prob = self.softmax(out)
        return out_prob

    def __call__(self, 
        source: torch.tensor, 
        target: Optional[torch.tensor]=None,
        src_key_padding_mask: Optional[torch.tensor]=None,
        tgt_key_padding_mask: Optional[torch.tensor]=None,):
        return self.forward(source, target, src_key_padding_mask, tgt_key_padding_mask)

# classes not included directly in torch

class PositionalEncoder(nn.Module):
    """Positional encoder"""
    def __init__(self, 
        d_model: int, 
        dropout: float=0.1, 
        max_len: int=1000):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('positional_encoder', pe)

    def forward(self, x):
        x = x + self.positional_encoder[:x.size(0), :]
        return self.dropout(x)



# custom functions and classes

def scaled_dot_product_attention(q, k, v, mask=None) -> Tuple[torch.tensor, torch.tensor]:
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

# def get_embedder(embedding: str="torch", **kwargs):
#     if embedding in ["torch", "pytorch"]:
#         embedder = nn.Embedding(**kwargs)
#     elif embedding in ["glove"]:
#         raise NotImplementedError
#     else:
#         raise NotImplementedError
#     return embedder

# class WordEmbedding(nn.Module):
#     def __init__(self, 
#         vocab_size: int,
#         embedding_dim: int,
#         pad_token_idx: int,
#         embedding: str="torch",
#         ):
#         """
#         :param vocab_size: vocabulary size
#         :param embedding_dim: dimensionality of the embeddings
#         :param embedding: str, takes "torch", "glove", "pytorch" 
#             (same as "torch"), and "bert"
#         """
#         super(WordEmbedding, self).__init__()
#         self.embedding = get_embedder(embedding, 
#             num_embeddings=vocab_size, 
#             embedding_dim=embedding_dim,
#             padding_idx=pad_token_idx,
#             max_norm=1e2,
#             norm_type=2.0)
#         self.embedding_dim = embedding_dim
#         self.pad_token_idx = pad_token_idx
        
#     def forward(self, x: torch.tensor):
#         x_ = self.embedding(x)
#         return x_