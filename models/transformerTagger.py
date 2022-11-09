import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from typing import (Sequence, Iterable, Dict, Tuple, Callable)

from dataset.dataset import NERDataset
from torch.utils.data import DataLoader

import logging

logging.basicConfig(filename="./log/exceptions.log")
LOGGER = logging.getLogger()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class TranformerTagger(nn.modules):
    def __init__(self, 
        d_model: int=512,
        n_head: int=8,
        vocab_size: int=23000,
        num_encoder_layers: int=8,
        num_decoder_layers: int=8,
        dim_feedforward: int=1024,
        dropout: float=.1,
        embedding_type: str="torch",
        no_dense_layers: int = 5,
        activation: Callable=F.relu,
        batch_first: bool=True,
        device: str=DEVICE,
        ):
        self.d_model = d_model
        self.positional_encoder = PositionalEncoder(d_model, dropout)
        self.embedding = get_embedder(embedding_type, 
            um_embeddings=vocab_size, 
            embedding_dim=d_model)


# classes not included directly in torch

class PositionalEncoder(nn.Module):
    """Positional encoder"""
    def __init__(self, 
        d_model: int, 
        dropout: float=0.1, 
        max_len: int=10000):
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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)




class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embed_dim must be multiples of num_heads!"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product_attention(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


# custom functions

def scaled_dot_product_attention(q, k, v, mask=None) -> Tuple[torch.tensor, torch.tensor]:
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

def get_embedder(embedding: str="torch", **kwargs):
    if embedding in ["torch", "pytorch"]:
        embedder = nn.Embedding(**kwargs)
    elif embedding in ["glove"]:
        raise NotImplementedError
    else:
        raise NotImplementedError
    return embedder