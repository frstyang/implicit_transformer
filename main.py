import argparse
import importlib
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from einops import rearrange, repeat
from sklearn.decomposition import TruncatedSVD

def project(W, s_max, k):
    # W: a 2D torch.tensor
    W_numpy = W.cpu().numpy()
    tsvd = TruncatedSVD(k).fit(W_numpy)
    v = tsvd.components_
    s = tsvd.singular_values_
    greater_than_s_max = s > s_max
    if greater_than_s_max.sum() == 0:
        return W
    v_greater = v[greater_than_s_max]
    s_greater = s[greater_than_s_max]
    s_u_greater = W_numpy @ v_greater.T 
    s_max_minus_s_u_greater = s_u_greater * (s_max / s_greater - 1)
    return torch.tensor(W_numpy + s_max_minus_s_u_greater @ v_greater, device=W.device)

CHECKPOINT_PATH = "./checkpoint.pt"
def wandb_log(scores):
    if len(scores) > 0:
        wandb.log(scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--project", default=False)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    train_loader = config.train_loader
    val_loader = config.val_loader
    test_loader = config.test_loader

    # 3/8 reminder; when porting sparse implicit transformer the to-port model should be loaded
    # first to use the random seed on the to-port model initialization

    if config.port_model:
        kwargs = config.port_model_kwargs
        model_to_port = config.port_model_class(**kwargs)
        model_to_port.cuda()
        # model_to_port.load_state_dict(
        #     torch.load(config.port_model_ckpt)['model_state_dict']
        # )
        # model.port_transformer(kwargs['dim'], kwargs['ff_dim'], kwargs['n_layers'], model_to_port)
        # model.S_from_ported_transformer(kwargs['n_layers'], v=0.95)

    model = config.model_class(**config.model_kwargs)
    if config.double:
        model.double()
    model.cuda()

    if config.port_model:
        config.port_model_fn(model_to_port, model)

    if args.project:
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
    
    test_scores = []
    best_accuracy = 0
    while i < config.num_iters:
        model.train()
        for batch in train_loader:
            loss, scores, outputs = config.compute_scores(model, batch, 'train')
            wandb.log(scores)
            loss.backward()
            optim.step()
            optim.zero_grad()

            if args.project and (i % config.project_freq == 0):
                model.project(v=v, exact=True)
            # from models.projection import create_NA, create_NA_qkv
            # print(create_NA(model.A, model.relu_dim, model.ln_dim, model.dim, model.n_layer_norms, model.n_heads, model.n_relu_heads).sum(dim=1))
            # print(create_NA_qkv(model.A_qkv, model.n_heads, model.n_layer_norms).sum(dim=1))
            i += 1
            test_scores = []
            if i % config.test_val_freq == 0:
                eval_splits = ['val'] + config.do_test_loop * ['test']
                eval_loaders = [val_loader] + config.do_test_loop * [test_loader]
                model.eval()
                with torch.no_grad():
                    for split, loader in zip(eval_splits, eval_loaders):
                        split_outputs = []
                        for batch in loader:
                            loss, scores, outputs = config.compute_scores(model, batch, split)
                            split_outputs.append(outputs)
                            wandb_log(scores)
                        if getattr(config, "do_compile_eval", False):
                            split_outputs = config.compile(split_outputs)
                            metrics = config.evaluate(split_outputs, split)
                            wandb_log(metrics)
                        if split == 'test':
                            test_scores.append(scores)

            if i % config.model_checkpoint_freq == 0:
                torch.save(
                    {'i': i, 'model_state_dict': model.state_dict(), 'optim_state_dict': optim.state_dict()}, 
                    CHECKPOINT_PATH,
                )
                # if test_scores[-1]['test/acc'] > best_accuracy:
                #     print(f"Test accuracy of {test_scores[-1]['test/acc']} was better than {best_accuracy}, saving")
                #     best_accuracy = test_scores[-1]['test/acc']
                #     torch.save(
                #         {'i': i, 'model_state_dict': model.state_dict(), 'optim_state_dict': optim.state_dict()},
                #         './best_checkpoint.pt',
                #     )

            if i == config.num_iters:
                break