import copy

import torch
from torch.utils.data import DataLoader
import numpy as np

from models.transformer import Transformer
from dataset.copy import CopyDataset

TORCH_SEED = 0
NUMPY_SEED = 0
TRAIN_DATA_SEED = 1234
VAL_DATA_SEED = 4321

torch.manual_seed(TORCH_SEED)
np.random.seed(NUMPY_SEED)
wandb_project = "copy_transformer"
run_name = 'enc_dec_trafo_v01_seed0'

batch_size = 256
num_iters = 50000
lr = 4e-4

cuda = True
double = False

monitor_metric = "val/char_acc"
do_test_loop = False
test_val_freq = 1000
do_compile_eval = False
model_checkpoint_freq = 1000

train_dataset = CopyDataset(length=40, n=10000, seed=VAL_DATA_SEED)
val_dataset = CopyDataset(length=400, n=3000, seed=VAL_DATA_SEED)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = None

vocab_size = 10

model_kwargs = {"dim": 128, "n_layers": 3, "n_heads": 4, "ff_dim": 512, "vocab_size": vocab_size,
     "n_classes": 10, "dropout": 0.0, "pre_ln": False, "universal": False, "relative": False, 'emb_norm': False,
     'custom_ln': False, "causal": False, "append_cls": False, "padding_idx": None, 'reduce': 'none',
     'decoder': True}

model_class = Transformer
loss_fn = torch.nn.CrossEntropyLoss()
wandb_metrics = [('train/cross_entropy', 'min'), ('train/char_acc', 'max'), ('train/seq_acc', 'max'),
                 ('val/cross_entropy', 'min'), ('val/char_acc', 'max'), ('val/seq_acc', 'max')]

def get_seq_lens(x):
    return torch.full((len(x),), x.shape[1], device=x.device)

def compute_scores(model, batch, split='train'):
    device = next(model.parameters()).device
    x = batch
    x = x.to(device).to(torch.long)
    seq_lens = get_seq_lens(x)
    x_2 = x.clone()
    seq_lens_2 = get_seq_lens(x_2)

    if split == 'train':
        logits = model(x, seq_lens=seq_lens, x_2=x_2, seq_lens_2=seq_lens_2)

    if split == 'val':
        logits = model(x, seq_lens=seq_lens)

    loss = loss_fn(logits.transpose(1, 2), x_2)
    with torch.no_grad():
        pred_ids = logits.argmax(dim=2)
        matches = pred_ids == x_2
        char_acc = matches.to(torch.float).mean().item()
        seq_acc = torch.all(matches, dim=1).to(torch.float).mean().item()

    scores = {f"{split}/cross_entropy": loss.item(), f"{split}/char_acc": char_acc, f"{split}/seq_acc": seq_acc}
    return loss, scores, None
