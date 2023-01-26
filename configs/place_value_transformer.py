import copy

import torch
from torch.utils.data import DataLoader
import numpy as np

from models import Transformer
from models.implicit_transformer import ImplicitTransformer
from datasets import PlaceValueDataset

batch_size = 256
num_iters = 50000
lr = 1e-4
test_val_freq = 1000
model_checkpoint_freq = 1000
wandb_project = "place_value_transformer"
do_test_loop = True
project_freq = 30

train_dataset = PlaceValueDataset()
val_dataset = copy.copy(train_dataset)
val_dataset.split_index = 1
test_dataset = copy.copy(val_dataset)
test_dataset.split_index = 2
collate_fn = train_dataset.collate_fn

model_kwargs = {"relu_dim": 2048, "ln_dim": 1280, "dim": 512, "emb_dim": 256, "n_heads": 16,
    "vocab_size": len(train_dataset.vocabulary), "n_classes": 10, "n_layer_norms": 5}
# model_kwargs = {"dim": 256, "n_layers": 2, "n_heads": 8, "ff_dim": 1024, "vocab_size": len(train_dataset.vocabulary),
#      "n_classes": 10, "dropout": 0.1, "pre_ln": False, "universal": False, "relative": False, 'implicit': False, 
#      'emb_norm': False, 'custom_ln': False}

model_class = ImplicitTransformer
#model_class = Transformer

loss_fn = torch.nn.CrossEntropyLoss()

def compute_scores(model, batch, split='train'):
    x, y, seq_lens = batch
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    seq_lens = seq_lens.to(device)

    logits = model(x, seq_lens)
    loss = loss_fn(logits, y)
    acc = (logits.argmax(dim=1) == y).sum() / len(y)
    return loss, {f'{split}/cross_entropy': loss.item(), f'{split}/acc': acc.item()}

port_model = False
port_model_kwargs = {"dim": 256, "n_layers": 2, "n_heads": 8, "ff_dim": 1024, "vocab_size": 29,
    "n_classes": 10, "dropout": 0.0, "pre_ln": False, "universal": False, "relative": False, 'implicit': False,
    "custom_ln": True, "emb_norm": True}
port_model_class = Transformer
port_model_ckpt = './best_checkpoint_2L_cln_embn.pt'