import copy

import torch
from torch.utils.data import DataLoader
import numpy as np

from models import Transformer
from models.implicit_transformer import ImplicitTransformer
from dataset import PlaceValueDataset

wandb_project = "place_value_transformer"
batch_size = 256
num_iters = 50000
lr = 1e-4

cuda = True
double = False

test_val_freq = 1000
model_checkpoint_freq = 1000
do_test_loop = True

port_model = False
project = True
project_freq = 20

train_dataset = PlaceValueDataset()
val_dataset = copy.copy(train_dataset)
val_dataset.split_index = 1
test_dataset = copy.copy(val_dataset)
test_dataset.split_index = 2
collate_fn = train_dataset.collate_fn
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

Arguments for implicit transformer.
model_kwargs = {"relu_dim": 768, "ln_dim": 512, "dim": 512, "emb_dim": 256, "n_heads": 32,
    "vocab_size": len(train_dataset.vocabulary), "n_classes": 10, "n_layer_norms": 4, "n_relu_heads": 4, "jac_reg": True}

model_class = ImplicitTransformer

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
    return loss, {f'{split}/cross_entropy': loss.item(), f'{split}/acc': acc.item()}, None

def port_model_fn(model_to_port, model):
    model_to_port.load_state_dict(
        torch.load(config.port_model_ckpt)['model_state_dict']
    )
    model.port_transformer(port_model_kwargs['dim'], port_model_kwargs['ff_dim'], port_model_kwargs['n_layers'], model_to_port)
