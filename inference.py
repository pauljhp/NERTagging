import torch
import torch.nn as nn
from models.transformerTagger import TransformerTagger
# from models.transformerTagger import PositionalEncoder
from dataset.dataset import NERDataset
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import torch
import datetime as dt
# from torch.utils.tensorboard import SummaryWriter
from torchmetrics import F1Score, Accuracy, Precision, Recall
from typing import Sequence, Optional, Union, Tuple
import utils
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


train_data = NERDataset(tokenizer="spacy", cased=False, mode='train')
test_data = NERDataset(tokenizer="spacy", cased=False, mode='test')
val_data = NERDataset(tokenizer="spacy", cased=False, mode='valid')

TODAY = dt.datetime.today().strftime("%Y-%m-%d")

NO_DENSE_LAYERS = 5
D_MODEL = 512
LAYER_NORM_EPS = 0.01
NHEAD = 16
BATCH_SIZE = 256
BASE_LR = 1e-3
MAX_EPOCHS = 20
CHECKPOINT_DIR = "./checkpoints"
MODEL_NAME = "2022-11-15_runo11.pt"
MINIBATCH_SIZE = 16
OUTPUT_NAME = f"test_output_{TODAY}.txt"


parser = ArgumentParser()
parser.add_argument("-b", "--batch_size", type=int, default=BATCH_SIZE)
parser.add_argument("-c", "--checkpoint_dir", default=CHECKPOINT_DIR)
parser.add_argument("-mn", "--model_name", type=MODEL_NAME, type=str)
parser.add_argument("--model_type", dest="model_type", type=str)
parser.add_argument("-m", "--minibatch_size", type=int, default=MINIBATCH_SIZE)
parser.add_argument("--nhead", type=int, default=NHEAD)
parser.add_argument("--d_model", type=int, default=D_MODEL)
parser.add_argument("--no_dense_layers", type=int, default=NO_DENSE_LAYERS)
parser.add_argument("--layer_norm_eps", type=float, default=LAYER_NORM_EPS)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("--detect_anomaly", action="store_true")
parser.add_argument("--enable_autocast", action="store_true")
parser.add_argument("--output_name",default=OUTPUT_NAME, type=str)

torch.autograd.set_detect_anomaly(True)
torch.cuda.amp.autocast(enabled=True)

args = parser.parse_args()

f1 = F1Score(num_classes=train_data.ntargets, threshold=0.5)
accu = Accuracy(threshold=0.5, num_classes=train_data.ntargets)
precision = Precision(num_classes=train_data.ntargets)
recall = Recall(num_classes=train_data.ntargets, threshold=0.5)

feature_padding_value = train_data._tokenidx.get(train_data.pad_token)
tag_padding_value = train_data._targetidx.get(train_data.pad_token)

output_path = Path("./output").joinpath(args.output_name)

def collate_fn(data: Sequence[Tuple], 
    n_classes: int=train_data.ntargets,
    feature_padding_value=feature_padding_value,
    tag_padding_value=tag_padding_value,):
    """:return: features, target_prob, target, mask (save dims as target_prob)"""
    features, target_prob, targets, idx = zip(*data)
    features = pad_sequence(features, batch_first=True, padding_value=feature_padding_value)
    targets = pad_sequence(targets, batch_first=True, padding_value=tag_padding_value)
    max_len = targets.shape[1]
    batch_size=targets.shape[0]
    target_prob, target_mask = utils.pad_target_prob(target_prob, n_classes - 1, max_len, n_classes, batch_size)

    return (idx,
        features.long(), target_prob.to(torch.float64), 
        targets.long(), target_mask.bool()
        )

sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, 
    shuffle=False,
    sampler=sampler,  
    batch_size=args.batch_size, 
    collate_fn=collate_fn
        )

if args.model_type.lower() in ["transformer", "tranformertagger"]:
    model = TransformerTagger(d_model=args.d_model, 
        n_tags=train_data.ntargets, 
        vocab_size=train_data.vocab_size + 1,
        layer_norm_eps=args.layer_norm_eps,
        activation=torch.tanh,
        nhead=args.nhead, 
        batch_first=True, 
        no_dense_layers=args.no_dense_layers,
        pad_token_idx=train_data._tokenidx.get(train_data.pad_token))
elif args.model_type.lower() in ["lstm", "lstmtagger"]:
    raise NotImplementedError
else:
    raise ValueError(f"{args.model_type} is not a supported model type!")


model.load_state_dict(
    torch.load(Path(args.checkpoint_dir).joinpath(args.model_name).as_posix())
    )

model.eval()
counter, train_precision, train_recall, train_f1, train_accu = 0., 0., 0., 0., 0.,
for i, data in enumerate(test_dataloader):
    idx, src, tag_prob, tags, mask = data
    pred = model(src, src, mask)
    for j, (prd, truth, mk) in enumerate(zip(pred, tags, mask)):
        train_precision += precision(prd[~mk], truth.masked_select(~mk).long()).item()
        train_recall += recall(prd[~mk], truth.masked_select(~mk).long()).item()
        train_f1 += f1(prd[~mk], truth.masked_select(~mk).long()).item()
        train_accu += accu(prd[~mk], truth.masked_select(~mk).long()).item()
        counter += 1
    results = pd.DataFrame([
        src[~mask].numpy(), 
        pred[~mask].argmax(-1).numpy(),
        tags[~mask].numpy()],        
    columns=None,
    index=["tokens", "prediction", "truth"]).T
    results.loc[:, "tokens"] = results.tokens.apply(test_data.get_token_from_idx)
    results.loc[:, "prediction"] = results.prediction.map(test_data._target_lookup)
    text = "\n".join(results[["tokens", "prediction"]].apply(
        lambda x: f"{x[0]} {x[1]}", axis=1))
    with output_path.open("a") as f:
        f.write(text)