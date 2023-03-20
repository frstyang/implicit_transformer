import argparse
import importlib
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from einops import rearrange, repeat
from sklearn.decomposition import TruncatedSVD

CHECKPOINT_PATH = "./checkpoint.pt"
def wandb_log(scores):
    if len(scores) > 0:
        wandb.log(scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    train_loader = config.train_loader
    val_loader = config.val_loader
    test_loader = config.test_loader

    # when porting sparse implicit transformer the model to port should be loaded
    # first to use the random seed on the model to port's initialization

    if getattr(config, 'port_model', False):
        kwargs = config.port_model_kwargs
        model_to_port = config.port_model_class(**kwargs)
        model_to_port.cuda()

    print(f"Model class: {config.model_class.__name__}")
    model = config.model_class(**config.model_kwargs)
    if getattr(config, 'double', False):
        model.double()
    if getattr(config, 'cuda', True):
        model.cuda()

    if getattr(config, 'port_model', False):
        config.port_model_fn(model_to_port, model)

    if getattr(config, 'project', False):
        v = 0.9
        model.project(v=v, exact=True)

    optim = torch.optim.Adam(model.parameters(), config.lr)

    wandb.init(project=config.wandb_project, config=config.model_kwargs)
    wandb.config.update({'model': config.model_class.__name__,
    'batch_size': config.batch_size, 'num_iters': config.num_iters, 'lr': config.lr})
    wandb.define_metric("train/cross_entropy", summary='min')
    wandb.define_metric("train/acc", summary='max')

    np.random.seed(getattr(config, "NUMPY_SEED", 0))
    torch.manual_seed(getattr(config, "TORCH_SEED", 0))

    i = 0
    
    assert config.model_checkpoint_freq % config.test_val_freq == 0
    monitor_metric = getattr(config, 'monitor_metric', 'val/acc')
    best_val_metric = 0
    while i < config.num_iters:
        model.train()
        for batch in train_loader:
            loss, scores, outputs = config.compute_scores(model, batch, 'train')
            wandb.log(scores)
            loss.backward()
            optim.step()
            optim.zero_grad()

            if getattr(config, 'project') and (i % config.project_freq == 0):
                model.project(v=v, exact=True)

            i += 1
            if i % config.test_val_freq == 0:
                eval_splits = ['val'] + config.do_test_loop * ['test']
                eval_loaders = [val_loader] + config.do_test_loop * [test_loader]
                model.eval()
                with torch.no_grad():
                    for split, loader in zip(eval_splits, eval_loaders):
                        split_outputs = []
                        val_scores_for_each_batch = []
                        for batch in loader:
                            loss, scores, outputs = config.compute_scores(model, batch, split)
                            split_outputs.append(outputs)
                            val_scores_for_each_batch.append(scores)

                        scores_this_iteration = {k: np.mean([batch_scores[k] for batch_scores
                            in val_scores_for_each_batch]) for k in val_scores_for_each_batch[0].keys()}
                        if getattr(config, "do_compile_eval", False):
                            split_outputs = config.compile(split_outputs)
                            metrics = config.evaluate(split_outputs, split)
                            scores_this_iteration.update(metrics)
                        wandb_log(scores_this_iteration)
                        if split == 'val':
                            val_metric = scores_this_iteration[monitor_metric]

            if i % config.model_checkpoint_freq == 0:
                torch.save(
                    {'i': i, 'model_state_dict': model.state_dict(), 'optim_state_dict': optim.state_dict(), monitor_metric: val_metric},
                    CHECKPOINT_PATH,
                )
                if val_metric > best_val_metric:
                    print(f"{monitor_metric} of {val_metric} at iteration {i} was better than {best_val_metric}, checkpointing")
                    best_val_metric = val_metric
                    torch.save(
                        {'i': i, 'model_state_dict': model.state_dict(), 'optim_state_dict': optim.state_dict(), monitor_metric: val_metric},
                        './best_checkpoint.pt',
                    )

            if i == config.num_iters:
                break