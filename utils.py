import torch
import numpy as np
from typing import Union, Sequence, Any, Optional
import itertools


def tagidx_to_prob(tagidx: Sequence[int], ntags: int, dtype: torch.long):
    return torch.tensor(
                [[1 if i == tag else 0 for i in range(ntags)] 
                    for tag in tagidx],
                dtype=dtype
            )

def pad_target_prob(
    targets: Sequence[torch.tensor], 
    pad_idx: int,
    max_len: int, 
    n_classes: int, 
    batch_size: int):
    """pad the padding classes into [0, 0, .., 1, 0..] where 1 represents the 
    pad class index"""
    pad_sequence = [0 for _ in range(n_classes)]
    padded, mask = [], []
    for target_prob in targets:
        pad = torch.tensor(
            pad_sequence * (max_len - target_prob.shape[0]), 
            dtype=torch.long)\
                .reshape(max_len - target_prob.shape[0], n_classes)
        # print(pad.shape, target_prob.shape)
        padded.append(torch.cat((target_prob, pad), dim=0))
        mask.append(torch.cat((torch.zeros(target_prob.shape).bool(),
            torch.ones(pad.shape).bool()), dim=0))
    target_prob_ = torch.cat(padded, dim=0).reshape(
        batch_size, max_len, n_classes,)
    target_mask = torch.cat(mask, dim=0).reshape(
        batch_size, max_len, n_classes,)
    return target_prob_, target_mask[slice(None), slice(None), 0]

def get_num_params(model: torch.nn.Module) -> int:
    """get the number of parameters in a model"""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def iter_by_chunk(iterable: Any, chunk_size: int):
    """iterate by chunk size"""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, chunk_size))
        if not chunk:
            break
        yield chunk