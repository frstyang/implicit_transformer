import copy

import torch
from torch.utils.data import DataLoader
import numpy as np

from models import Transformer
from dataset.wikitext import WikiText103

batch_size = 64
num_iters = 50000
lr = 1e-4
test_val_freq = 1000
model_checkpoint_freq = 1000
wandb_project = "wikitext103_transformer"
do_test_loop = True
project_freq = 10
double = False

train_dataset = WikiText103('train')
val_dataset = WikiText103('valid')
test_dataset = WikiText103('test')
collate_fn = train_dataset.make_collate_fn()

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=train_dataset.get_sampler(batch_size))
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=val_dataset.get_sampler(batch_size))
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=test_dataset.get_sampler(batch_size))

vocab_size = train_dataset.tokenizer.get_vocab_size()
model_kwargs = {"dim": 384, "n_layers": 8, "n_heads": 8, "ff_dim": 1536, "vocab_size": vocab_size,
     "n_classes": 'auto', "dropout": 0.1, "pre_ln": False, "universal": False, "relative": False, 'emb_norm': False,
     'custom_ln': False, "causal": True}

model_class = Transformer

loss_fn = torch.nn.CrossEntropyLoss()

def compute_scores(model, batch, split='train'):
    x, seq_lens = batch
    device = next(model.parameters()).device
    x = x.to(device)
    seq_lens = seq_lens.to(device)

    logits = model(x, seq_lens)
    logits = logits[:, :-1]

    pos = torch.arange(0, x.shape[1], device=x.device)
    where_ignore = (seq_lens[:, None] <= pos)
    x = x.masked_fill(where_ignore, -100)

    #print("logits.shape", logits.shape, "x.shape", x.shape, "x.dtype", x.dtype)

    b, s = logits.shape[:2]
    loss = loss_fn(logits.reshape(b*s, -1), x.reshape(b*s))
    loss_float = loss.item()

    
    acc = (logits.argmax(dim=2) == x)[~where_ignore].to(torch.float).mean()
    return loss, {f'{split}/cross_entropy': loss_float, f'{split}/acc': acc.item(), f"{split}/perplexity": np.exp(loss_float)}, None

port_model = False