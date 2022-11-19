import torch
import torch.nn as nn

def get_embedder(embedding: str="torch", **kwargs):
    if embedding in ["torch", "pytorch"]:
        embedder = nn.Embedding(**kwargs)
    elif embedding in ["glove"]:
        raise NotImplementedError
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