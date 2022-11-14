from argparse import ArgumentParser
import torch
from dataset.dataset import NERDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Sequence
import torch.optim as optim
import torch.nn as nn
import torch
from models.transformerTagger import TransformerTagger
from models.transformerTagger import PositionalEncoder
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import datetime as dt
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import F1Score, Accuracy, Precision, Recall
import utils
import time
from pathlib import Path


train_data = NERDataset(tokenizer="spacy", cased=False, mode='train')
test_data = NERDataset(tokenizer="spacy", cased=False, mode='test')
val_data = NERDataset(tokenizer="spacy", cased=False, mode='valid')

TODAY = dt.datetime.today().strftime("%Y-%m-%d")



BATCH_SIZE = 256
BASE_LR = 1e-3
MAX_EPOCHS = 20
# RUN_NO = 1
LOG_DIR = "./log/traininglog/"
MINIBATCH_SIZE = 16
SAVE_EVERY = 5


parser = ArgumentParser()
parser.add_argument("-b", "--batch_size", type=int, default=256)
parser.add_argument("-l", "--base_lr", type=float, default=1e-3)
parser.add_argument("-e", "--max_epochs", type=int, default=20)
parser.add_argument("-d", "--logdir", type=str, default=LOG_DIR)
parser.add_argument("-r", "--runno", type=int)
parser.add_argument("-m", "--minibatch_size", type=int, default=MINIBATCH_SIZE)
parser.add_argument("-s", "--save_every", default=SAVE_EVERY)
parser.add_argument("-c", "--checkpoint_dir", default=None)
parser.add_argument("--nhead", type=int, default=8)
parser.add_argument("--d_model", type=int, default=128)
parser.add_argument("--no_dense_layers", type=int, default=5)
parser.add_argument("--layer_norm_eps", type=float, default=1e-4)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("--detect_anomaly", action="store_true")
parser.add_argument("--enable_autocast", action="store_true")

args = parser.parse_args()
batch_size = args.batch_size
base_lr = args.base_lr
max_epochs = args.max_epochs
log_dir = args.logdir
runno = args.runno
minibatch_size = args.minibatch_size
save_every = args.save_every

SAVE_DIR = f"./checkpoints/{TODAY}_runo{runno}.pt"
save_dir = args.checkpoint_dir if args.checkpoint_dir else SAVE_DIR
log_dir = Path(log_dir).joinpath(
        f"{TODAY}/runno_{runno}_lr{base_lr:.6f}_{max_epochs}epochs_{args.d_model}dims_{args.no_dense_layers}denselayers_{args.nhead}heads")
torch.autograd.set_detect_anomaly(args.detect_anomaly)
torch.cuda.amp.autocast(enabled=args.enable_autocast)


if not log_dir.parent.exists():
    log_dir.parent.mkdir(parents=True, exist_ok=True)
WRITER = SummaryWriter(
    log_dir=log_dir.as_posix()
        )


f1 = F1Score(num_classes=10, threshold=0.5)
accu = Accuracy(threshold=0.5, num_classes=10)
precision = Precision(num_classes=10)
recall = Recall(num_classes=10, threshold=0.5)

feature_padding_value = train_data._tokenidx.get(train_data.pad_token)
tag_padding_value = train_data._targetidx.get(train_data.pad_token)

def collate_fn(data: Sequence[Tuple], 
    n_classes: int=train_data.ntargets,
    feature_padding_value=feature_padding_value,
    tag_padding_value=tag_padding_value,):
    """:return: features, target_prob, target, mask (save dims as target_prob)"""
    features, target_prob, targets = zip(*data)
    features = pad_sequence(features, batch_first=True, padding_value=feature_padding_value)
    targets = pad_sequence(targets, batch_first=True, padding_value=tag_padding_value)
    max_len = targets.shape[1]
    batch_size=targets.shape[0]
    target_prob, target_mask = utils.pad_target_prob(target_prob, n_classes - 1, max_len, n_classes, batch_size)

    return (
        features.long(), target_prob.to(torch.float64), 
        targets.long(), target_mask.bool()
        )

train_dataloader = DataLoader(train_data, 
    shuffle=True, 
    batch_size=BATCH_SIZE, 
    collate_fn=collate_fn
        )

val_dataloader = DataLoader(val_data, 
    shuffle=True, 
    batch_size=BATCH_SIZE, 
    collate_fn=collate_fn
        )

model = TransformerTagger(d_model=args.d_model, 
    n_tags=train_data.ntargets, 
    vocab_size=train_data.vocab_size + 1,
    layer_norm_eps=args.layer_norm_eps,
    nhead=args.nhead, 
    batch_first=True, 
    no_dense_layers=args.no_dense_layers,
    pad_token_idx=train_data._tokenidx.get(train_data.pad_token))

optimizer = optim.Adam(params=model.parameters(), 
    lr=BASE_LR,
    eps=1e-5,
    betas=(0.9, 0.99))

criterion = nn.CrossEntropyLoss()
start = time.time()

global_step = 0
for epoch in range(max_epochs):
    if args.verbose: print(f"running epoch {epoch}")
    running_loss = 0.
    model.train()
    for i, data in enumerate(train_dataloader):
        print(f"iter_no{i}")
        optimizer.zero_grad()
        src, tag_prob, tags, mask = data
        pred = model(src, src, mask)
        pred = torch.masked_fill(pred, 
            mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1]),
            torch.tensor(0., dtype=torch.float32)) # mask padded output
        tag_prob = torch.masked_fill(tag_prob, 
            mask.unsqueeze(-1).repeat(1, 1, tag_prob.shape[-1]), 
            torch.tensor(0., dtype=torch.float32)) # mask padded true values
        torch.nn.utils.clip_grad_norm_(model.parameters(), 
            max_norm=1e2, 
            norm_type=2.0, 
            error_if_nonfinite=False) # try clipping large weights
        loss = criterion(pred, tag_prob)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        WRITER.add_scalar("train/loss", loss.item(), 
                walltime=time.time()-start,
                global_step=global_step)
        global_step += i
        if i % minibatch_size == minibatch_size - 1:
            print(f"epoch{epoch}, pass{i}; loss={running_loss / i}")
            WRITER.add_scalar("train/loss", running_loss / i, 
                walltime=time.time()-start,
                global_step=global_step)
            running_loss = 0
    for tag, value in model.named_parameters():
        if value.grad is not None:
            WRITER.add_histogram(tag + "/grad", value.grad.cpu(), 
            walltime=time.time()-start,
            global_step=global_step)
    if args.verbose: print(f"finished epoch {epoch}\n------\n")
    if epoch % save_every == save_every - 1:
        torch.save(model.state_dict(), f=save_dir)
        model.eval()
        running_loss, running_precision, running_recall, running_f1, running_accu \
            = 0., 0., 0., 0., 0.
        counter = 0.
        for i, data in enumerate(val_dataloader):
            src, tag_prob, tags, mask = data
            pred = model(src, src, mask)
            loss = criterion(pred, tag_prob)
            running_loss += loss.item()
            for j, (pred, truth) in enumerate(zip(pred, tag_prob)):
                running_precision += precision(pred, truth.long()).item()
                running_recall += recall(pred, truth.long()).item()
                running_f1 += f1(pred, truth.long()).item()
                running_accu += accu(pred, truth.long()).item()
                counter += j
            if i % minibatch_size == minibatch_size - 1:
                if args.verbose:
                    print(f"epoch{epoch}, iter{i}; validation_loss={running_loss};\nvalidation_precision={running_precision / counter}")
                WRITER.add_scalar("val/loss", running_loss, 
                    walltime=time.time()-start,
                    global_step=global_step)
                WRITER.add_scaler("val/precision", running_precision / counter,
                    walltime=time.time()-start,
                    global_step=global_step)
                WRITER.add_scaler("val/recall", running_recall / counter,
                    walltime=time.time()-start,
                    global_step=global_step)
                WRITER.add_scaler("val/f1", running_f1 / counter,
                    walltime=time.time()-start,
                    global_step=global_step)
                WRITER.add_scaler("val/accuracy", running_accu / counter,
                    walltime=time.time()-start,
                    global_step=global_step)
                running_loss, running_precision, running_recall, running_f1, running_accu \
                = 0., 0., 0., 0., 0.
                counter = 0.
        