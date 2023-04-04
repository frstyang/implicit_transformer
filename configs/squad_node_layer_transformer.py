import copy

import torch
from torch.utils.data import DataLoader
import numpy as np

from models import Transformer
from models.node_implicit_transformer import NodeLayerTransformer, t_to_nlt
from dataset.squad import load_preprocessed_squad_datasets, compute_metrics, collate_batch

TORCH_SEED = 0
NUMPY_SEED = 0

torch.manual_seed(TORCH_SEED)
np.random.seed(NUMPY_SEED)
wandb_project = "SQuAD_transformer"

batch_size = 32
num_iters = 50000
lr = 1e-4

cuda = True
double = False

monitor_metric = "val/f1"
do_test_loop = False
test_val_freq = 1000
do_compile_eval = True
model_checkpoint_freq = 1000

port_model = True
project = False
project_freq = 10

class ResuIter:
    def __init__(self, generator_fn):
        self.generator_fn = generator_fn
        
    def __iter__(self):
        self.generator = self.generator_fn()
        return self

    def __next__(self):
        return next(self.generator)

train_dataset, val_dataset, raw_val_dataset = load_preprocessed_squad_datasets()
train_loader = ResuIter(lambda: train_dataset.iter(batch_size))
val_loader = ResuIter(lambda: val_dataset.iter(batch_size))
test_loader = None

vocab_size = 28996

model_kwargs = {"dim": 384, "n_layers": 6, "n_heads": 8, "ff_dim": 1536, "vocab_size": vocab_size,
    "n_classes": 2, "padding_idx": 0, "append_cls": False, "autoreg": True, 'extra_connections': []}

model_class = NodeLayerTransformer

loss_fn = torch.nn.CrossEntropyLoss()

def compute_scores(model, batch, split='train'):
    batch = collate_batch(batch)
    device = next(model.parameters()).device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    x = batch['input_ids']
    src_mask = ~(batch['attention_mask'].to(bool))
    logits = model(x, src_mask=src_mask)
    start_logits = logits[:, :, 0]
    end_logits = logits[:, :, 1]

    if split == 'train':
        context_mask = batch["token_type_ids"].to(bool)
        context_mask[:, 0] = True
        start_logits = start_logits.masked_fill(~context_mask, float("-inf"))
        end_logits = end_logits.masked_fill(~context_mask, float("-inf"))

        start_loss = loss_fn(start_logits, batch['start_positions'])
        end_loss = loss_fn(end_logits, batch['end_positions'])
        loss = 0.5*(start_loss + end_loss)

        with torch.no_grad():
            start_acc = (start_logits.argmax(dim=1) == batch['start_positions']).to(torch.float).mean().item()
            end_acc = (end_logits.argmax(dim=1) == batch['end_positions']).to(torch.float).mean().item()

        scores = {f'{split}/cross_entropy': loss.item(), f'{split}/acc': 0.5*(start_acc+end_acc)}

        if hasattr(model, 'node_layers'):
            scores['train/forward_i'] = model.node_layers.forward_iter_info['i']
            scores['train/forward_err'] = model.node_layers.forward_iter_info['error']
            if hasattr(model.node_layers, 'backward_iter_info'):
                scores['train/bckward_i'] = model.node_layers.backward_iter_info['i']
                scores['train/bckward_err'] = model.node_layers.backward_iter_info['error']
        return (loss, scores, None)

    if split == "val":
        start_logits = start_logits.detach().cpu().numpy()
        end_logits = end_logits.detach().cpu().numpy()
        return None, {}, (start_logits, end_logits)

def compile(outputs):
    start_logits, end_logits = zip(*outputs)
    return np.concatenate(start_logits), np.concatenate(end_logits)

def evaluate(outputs, split='val'):
    start_logits, end_logits = outputs
    metrics = compute_metrics(start_logits, end_logits, val_dataset, raw_val_dataset)
    return {f"{split}/{k}": v for k,v in metrics.items()}

port_model_kwargs = {"dim": 384, "n_layers": 6, "n_heads": 8, "ff_dim": 1536, "vocab_size": vocab_size,
     "n_classes": 2, "dropout": 0.0, "pre_ln": False, "universal": False, "relative": False, 'emb_norm': False,
     'custom_ln': False, "causal": True, "append_cls": False, 'autoreg': True, "padding_idx": 0}

port_model_class = Transformer
port_model_fn = lambda t, nlt: t_to_nlt(t, nlt, 6)
