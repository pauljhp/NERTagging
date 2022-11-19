import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embedding import WordEmbedding
import math
import numpy as np
from typing import (Sequence, Iterable, Dict, Tuple, Callable, Optional)

import logging

logging.basicConfig(filename="./log/exceptions.log")
LOGGER = logging.getLogger()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class LSTMTagger(nn.Module):
    def __init__(self, 
        pad_token_idx: int,
        d_model: int=512,
        vocab_size: int=30000,
        num_encoder_layers: int=8,
        num_decoder_layers: int=8,
        dropout: float=.1,
        embedding_type: str="torch",
        num_dense_layers: int=5,
        input_size: int=64, 
        activation: Callable=F.relu,
        batch_first: bool=True,
        layer_norm_eps: float=1e-4,
        bias: bool=True,
        bidirectional: bool=True,
        proj_size: int=0,
        n_tags: int=10,
        device: torch.device=DEVICE,
        ):
        super(LSTMTagger, self).__init__()
        self.device = device
        self.lstmdecoder_exists = True if num_decoder_layers > 0 else False
        assert d_model >= n_tags, "d_model must be higher than number of tags"
        self.d_model = d_model
        self.num_dense_layers = num_dense_layers
        self.pad_token_idx = pad_token_idx
        self.positional_encoder = PositionalEncoder(input_size, dropout)
        num_directions = 2 if bidirectional else 1
        assert d_model > input_size and d_model % input_size == 0, "d_model must be multiples of input_size!"
        self.embedding = WordEmbedding(vocab_size=vocab_size,
            embedding_dim=input_size,
            embedding=embedding_type, 
            pad_token_idx=pad_token_idx)
        self.LSTMEncoder = nn.LSTM(
            input_size=input_size,
            hidden_size=d_model,
            num_layers=num_encoder_layers,
            proj_size=proj_size,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            bidirectional=bidirectional
        )
        if num_decoder_layers:
            self.LSTMDecoder = nn.LSTM(
                hidden_size=d_model,
                num_layers=num_decoder_layers,
                proj_size=proj_size,
                input_size=input_size,
                dropout=dropout,
                batch_first=batch_first,
                bias=bias,
                bidirectional=bidirectional
            )
        increment = math.floor(
            (d_model // n_tags) ** (1 / num_dense_layers)
            )
        in_features, out_features = d_model * num_directions, d_model * num_directions // increment
        for i in range(1, num_dense_layers):
            exec(f"""self.dense{i} = nn.Linear(in_features=in_features, 
                out_features=out_features)""")
            in_features, out_features = out_features, \
                out_features // increment
        exec(f"""self.dense{num_dense_layers} = nn.Linear(in_features=in_features, 
                out_features=n_tags)""")
        stride = int(d_model / input_size) * num_directions
        self.avgpool = nn.AvgPool1d(kernel_size=stride, stride=stride, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, 
        input: torch.tensor, 
        mask: Optional[torch.tensor]=None,
        ):
        input = self.embedding(input.to(self.device)) * math.sqrt(self.d_model)
        input = self.positional_encoder(input)
        if mask is not None:
            mask = mask.unsqueeze(-1).repeat(1, 1, input.shape[-1])
            mask = torch.where(mask, 
                torch.tensor(- 2. ** 32).to(self.device), 
                torch.tensor(0.).to(self.device))
            input += mask
        encoder_output, (encoder_h, _) = self.LSTMEncoder(input)
        # print(input.shape)
        print(encoder_output.shape, encoder_h.shape)
        if self.lstmdecoder_exists:
            decoder_input = self.avgpool(encoder_output)
            # print(decoder_input.shape)
            decoder_output, _ = self.LSTMDecoder(decoder_input)
            dense_in = decoder_output
        else:
            dense_in = encoder_output
        dense_in = torch.clip(dense_in, min=-0.99999, max=0.99999)
        dense_in = torch.tanh(dense_in) # experiment with this
        for i in range(1, self.num_dense_layers + 1):
            out = eval(f"self.dense{i}(dense_in)")
            out = torch.tanh(out)
            # out = torch.clip(out, min=-0.99999, max=0.99999)
            dense_in = out

        out_prob = self.softmax(out)
        return out_prob

    def __call__(self, 
        input: torch.tensor, 
        mask: Optional[torch.tensor]=None,):
        return self.forward(input, mask)

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