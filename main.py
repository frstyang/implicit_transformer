import argparse
import importlib
import sys

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--project", default=False)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)

    train_loader = DataLoader(config.train_dataset, batch_size=config.batch_size, collate_fn=config.collate_fn, shuffle=True)
    val_loader = DataLoader(config.val_dataset, batch_size=config.batch_size, collate_fn=config.collate_fn)
    test_loader = DataLoader(config.test_dataset, batch_size=config.batch_size, collate_fn=config.collate_fn)

    model = config.model_class(**config.model_kwargs)
    model.cuda()

    if config.port_model:
        kwargs = config.port_model_kwargs
        model_to_port = config.port_model_class(**kwargs)
        model_to_port.cuda()
        model_to_port.load_state_dict(
            torch.load(config.port_model_ckpt)['model_state_dict']
        )
        model.port_transformer(kwargs['dim'], kwargs['ff_dim'], kwargs['n_layers'], model_to_port)
        model.S_from_ported_transformer(kwargs['n_layers'], v=0.95)

    if args.project:
        model.project(v=0.95, exact=True)

    optim = torch.optim.Adam(model.parameters(), config.lr)

    wandb.init(project=config.wandb_project, config=config.model_kwargs)
    wandb.config.update({'model': config.model_class.__name__,
    'batch_size': config.batch_size, 'num_iters': config.num_iters, 'lr': config.lr})

    i = 0
    test_scores = []
    best_accuracy = 0
    while i < config.num_iters:
        model.train()
        for batch in train_loader:
            loss, scores = config.compute_scores(model, batch, 'train')
            loss.backward()
            optim.step()
            optim.zero_grad()
            if args.project and (i % config.project_freq == 0):
                model.project(v=0.95)

            wandb.log(scores)
            i += 1
            if i % config.test_val_freq == 0:
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        loss, scores = config.compute_scores(model, batch, 'val')
                        wandb.log(scores)
                    if config.do_test_loop:
                        for batch in test_loader:
                            loss, scores = config.compute_scores(model, batch, 'test')
                            wandb.log(scores)
                            test_scores.append(scores)
            if i % config.model_checkpoint_freq == 0:
                torch.save(
                    {'i': i, 'model_state_dict': model.state_dict(), 'optim_state_dict': optim.state_dict()}, 
                    CHECKPOINT_PATH,
                )
                if test_scores[-1]['test/acc'] > best_accuracy:
                    print(f"Test accuracy of {test_scores[-1]['test/acc']} was better than {best_accuracy}, saving")
                    best_accuracy = test_scores[-1]['test/acc']
                    torch.save(
                        {'i': i, 'model_state_dict': model.state_dict(), 'optim_state_dict': optim.state_dict()},
                        './best_checkpoint.pt',
                    )

            if i == config.num_iters:
                break