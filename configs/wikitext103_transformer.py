import copy

import torch
from torch.utils.data import DataLoader
import numpy as np

from models import Transformer
from dataset.wikitext import WikiText103

wandb_project = "wikitext103_transformer"
batch_size = 64
num_iters = 50000
lr = 1e-4
test_val_freq = 1000
model_checkpoint_freq = 1000
do_test_loop = True
project_freq = 10
double = False
port_model = False

seq_length = 128
overlap = 16
train_dataset = WikiText103('train', seq_length=seq_length, overlap=16)
val_dataset = WikiText103('valid', seq_length=seq_length, overlap=16)
test_dataset = WikiText103('test', seq_length=seq_length, overlap=16)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_dataset.random_sampler(batch_size*num_iters))
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

vocab_size = train_dataset.tokenizer.get_vocab_size()
model_kwargs = {"dim": 384, "n_layers": 6, "n_heads": 8, "ff_dim": 1536, "vocab_size": vocab_size,
     "n_classes": 'auto', "dropout": 0.1, "pre_ln": False, "universal": False, "relative": False, 'emb_norm': False,
     'custom_ln': False, "causal": True, "append_cls": True}

model_class = Transformer

loss_fn = torch.nn.CrossEntropyLoss()

def compute_scores(model, batch, split='train'):
    x = batch
    device = next(model.parameters()).device
    x = x.to(device)

    logits = model(x)
    logits = logits[:, :-1]

    b, s = logits.shape[:2]
    loss = loss_fn(logits.reshape(b*s, -1), x.reshape(b*s))
    loss_float = loss.item()
    
    acc = (logits.argmax(dim=2) == x).to(torch.float).mean()
    return loss, {f'{split}/cross_entropy': loss_float, f'{split}/acc': acc.item(), f"{split}/perplexity": np.exp(loss_float)}, None