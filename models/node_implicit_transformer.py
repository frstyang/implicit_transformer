from abc import ABC, abstractmethod
from collections import OrderedDict
import copy

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn

from utils import dfs, get_back_edges, check_topo_order

class NodeLayer(ABC, nn.Module):
    def __init__(self, name, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.name = name
        self.input_names = []
        self.output_names = []
        self.input_maps = nn.ModuleDict()
        
    def connect(self, node):
        self.input_names.append(node.name)
        self.input_maps[node.name] = self.make_map(node)
        node.output_names.append(self.name)
        
    @abstractmethod
    def make_map(self, node):
        return
    
class InputNode(NodeLayer):
    def __init__(self, dim):
        super().__init__('input', dim)
        
    def make_map(self, node):
        return
    
    def forward(self, X, *args, **kwargs):
        return X['input']
        
from functools import reduce
def reduce_sum(tensor_list):
    return reduce(lambda a, b: a+b, tensor_list)

class FFLayer(NodeLayer):
    def __init__(self, name, out_dim, ff_dim):
        super().__init__(name, out_dim)
        self.ff_dim = ff_dim
        self.linear2 = nn.Linear(ff_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim, elementwise_affine=False)
        
    def make_map(self, node):
        return nn.Linear(node.out_dim, self.ff_dim)
    
    def forward(self, X, *args, **kwargs):
        mapped_inputs = [self.input_maps[name](X[name]) for name in self.input_names if name in X]
        out1 = torch.relu(reduce_sum(mapped_inputs))
        out2 = self.linear2(out1)
        return self.layer_norm(out2 + X[self.input_names[0]])
        
class AttentionLayer(NodeLayer):
    def __init__(self, name, out_dim, n_heads):
        super().__init__(name, out_dim)
        self.n_heads = n_heads
        self.W_o = nn.Linear(out_dim, out_dim, bias=False)
        self.layer_norm = nn.LayerNorm(out_dim, elementwise_affine=False)
        
    def make_map(self, node):
        return nn.Linear(node.out_dim, 3*self.out_dim, bias=False)
    
    def forward(self, X, src_mask=None, *args, **kwargs):
        mapped_inputs = [self.input_maps[name](X[name]) for name in self.input_names if name in X]
        x = reduce_sum(mapped_inputs)
        x = rearrange(x, 'b s (n_h d) -> (b n_h) s d', n_h=self.n_heads)
        q, k, v = x.chunk(3, dim=2)
        logits = torch.bmm(q, k.transpose(1, 2)) * (q.shape[-1]) ** (-0.5)
        # logits.shape: (batch_size*n_heads, max_seq_len, max_seq_len)
        if src_mask is not None:
            logits = rearrange(logits, '(b n_h) s1 s2 -> b n_h s1 s2', n_h=self.n_heads)
            logits.masked_fill_(src_mask[:, None, None, :], float("-inf"))
            logits = rearrange(logits, 'b n_h s1 s2 -> (b n_h) s1 s2', n_h=self.n_heads)
        self.causal_mask(logits)
        attn = torch.softmax(logits, dim=-1)
        out = torch.bmm(attn, v)
        
        out = rearrange(out, '(b n_h) s d -> b s (n_h d)', n_h=self.n_heads)
        out = self.W_o(out)
        return self.layer_norm(out + X[self.input_names[0]])

    def causal_mask(self, logits):
        i, j = logits.shape[-2:]
        causal_mask = torch.ones((i, j), dtype=torch.bool, device=logits.device).triu(j - i + 1)
        logits.masked_fill_(causal_mask, float("-inf"))
    
from models.transformer import PosEmb
class NodeLayerTransformer(nn.Module):
    def __init__(self, dim, n_layers, n_heads, ff_dim, vocab_size, n_classes,
                 padding_idx=None, append_cls=True, autoreg=False, extra_connections=[]):
        super().__init__()
        if padding_idx is None:
            self.embedding = nn.Embedding(vocab_size+1, dim, padding_idx=vocab_size)
        else:
            self.embedding = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.embedding.weight.data.normal_(0, dim**(-0.5))
        self.append_cls = append_cls
        if append_cls:
            self.cls_token = nn.Parameter(torch.normal(0, dim**(-0.5), (1, 1, dim)))
        self.pos_emb = PosEmb(dim)
        self.autoreg = (n_classes == 'auto') or autoreg
        if n_classes == 'auto':
            n_classes = vocab_size
        self.linear = nn.Linear(dim, n_classes)
        
        self.n_layers = n_layers
        self.node_layers = nn.ModuleDict({'input': InputNode(dim)})
        for i in range(n_layers):
            self.node_layers[f'attn_{i}'] = AttentionLayer(f'attn_{i}', dim, n_heads)
            self.node_layers[f'ff_{i}'] = FFLayer(f"ff_{i}", dim, ff_dim)
            if i == 0:
                self.node_layers[f'attn_{i}'].connect(self.node_layers['input'])
            else:
                self.node_layers[f'attn_{i}'].connect(self.node_layers[f'ff_{i-1}'])
            self.node_layers[f'ff_{i}'].connect(self.node_layers[f'attn_{i}'])

        for node1, node2 in extra_connections:
            self.node_layers[node2].connect(self.node_layers[node1])

        self.visited, self.post_nums, self.pre_nums = dfs(self.node_layers)
        self.back_edges = get_back_edges(self.node_layers, self.post_nums)
        if self.is_dag:
            assert check_topo_order(self.node_layers, self.post_nums)

    @property
    def is_dag(self):
        return len(self.back_edges) == 0
        
    def forward(self, x, seq_lens=None, src_mask=None):
        # x: (batch_size, max_seq_len)
        # seq_lens: (batch_size)
        assert (seq_lens is not None) ^ (src_mask is not None)
        x = self.embedding(x)
        x = x + self.pos_emb(x)
        if self.append_cls:
            x = torch.cat((self.cls_token.repeat(len(x), 1, 1), x), dim=1)
        if src_mask is None:
            pos = torch.arange(0, x.shape[1], device=x.device)
            src_mask = seq_lens[:, None] + self.append_cls <= pos
        X = ComputeGraph.apply(1e-5, 20, self.is_dag, self.node_layers,
                               {'src_mask': src_mask}, x, *self.node_layers.parameters())
        X = OrderedDict((name, v) for name, v in zip(self.node_layers.keys(), X))
        x = X[f'ff_{self.n_layers - 1}']
        if self.autoreg:
            logits = self.linear(x)
        else:
            logits = self.linear(x[:, 0, :])
        return logits
    
def graph_forward(node_layers, X, dynamic=True, *args, **kwargs):
    out = OrderedDict()
    for name, node in node_layers.items():
        out[name] = node(X, *args, **kwargs)
        if dynamic and name not in X:
            X[name] = out[name].detach().clone().requires_grad_()
        elif dynamic and name in X:
            X[name].data.copy_(out[name].data)
    return out

def err(node_outs1, node_outs2):
    diff = 0
    for key in node_outs1:
        diff = max(diff, (node_outs1[key] - node_outs2[key]).abs().max())
    return float(diff)
    
class ComputeGraph(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tol, max_iter, is_dag, node_layers, node_kwargs, x, *params):
        X = OrderedDict(input=x.detach().clone())
        names = list(node_layers.keys())
        node_layers_copy = copy.deepcopy(node_layers)
        for p in node_layers_copy.parameters(): p.requires_grad = False
        converged = is_dag # if not a DAG, check for convergence
        error = 0
        with torch.enable_grad():
            for i in range(max_iter):
                for name in list(X.keys())[1:]:
                    X[name] = X[name].detach().clone().requires_grad_()
                out = graph_forward(node_layers_copy, X, **node_kwargs)
                if is_dag:
                    break
                if i >= 1:
                    with torch.no_grad():
                        error = err(out, prev_out)
                    if error <= tol:
                        converged = True
                        break
                prev_out = out
        node_layers.forward_iter_info = {'i': i, 'error': error, 'converged': converged}
        ctx.save_for_backward(*out.values(), *X.values(), *node_layers.parameters())
        ctx.node_layers = node_layers
        ctx.node_layers_copy = node_layers_copy
        ctx.node_kwargs = node_kwargs
        ctx.max_iter = max_iter
        ctx.is_dag = is_dag
        ctx.tol = tol
        return tuple([x.detach().clone().requires_grad_() for x in out.values()])

    @staticmethod
    def backward(ctx, *node_grads):
        node_layers, node_kwargs = ctx.node_layers_copy, ctx.node_kwargs
        out_X_and_params = ctx.saved_tensors
        names = list(node_layers.keys())
        out = OrderedDict((name, tens) for name, tens in zip(names, out_X_and_params[:len(names)]))
        X = OrderedDict((name, tens) for name, tens in zip(names, out_X_and_params[len(names):2*len(names)]))
        params = out_X_and_params[2*len(names):]
        node_grads = OrderedDict((name, grad) for name, grad in zip(names, node_grads))

        new_node_grads = OrderedDict()
        error = 0
        converged = ctx.is_dag # if not a DAG, check for convergence
        for i in range(ctx.max_iter):
            for name in reversed(names[1:]):
                X[name].grad = None
                children = node_layers[name].output_names
                child_outs = []
                grads = []
                for child in children:
                    if out[child].requires_grad and child in new_node_grads:
                        child_outs.append(out[child])
                        grads.append(new_node_grads[child])
                if len(child_outs) > 0:
                    torch.autograd.backward(child_outs, grads, retain_graph=True)
                if X[name].grad is None:
                    new_node_grads[name] = node_grads[name]
                else:
                    new_node_grads[name] = node_grads[name] + X[name].grad
            if ctx.is_dag:
                break

            if i >= 1:
                error = err(new_node_grads, prev_new_node_grads)
                if error <= ctx.tol:
                    converged = True
                    break
            prev_new_node_grads = copy.copy(new_node_grads)

        ctx.node_layers.backward_iter_info = {'i': i, 'error': error, 'converged': converged}
        torch.autograd.backward(out['input'].requires_grad_(), torch.zeros_like(out['input']), retain_graph=False)
        
        for name in names[1:]: # still need to produce a gradient w.r.t. the input, x
            X[name] = X[name].detach()
        X['input'] = X['input'].requires_grad_()
        for p in node_layers.parameters(): p.requires_grad=True
                
        with torch.enable_grad():
            out = graph_forward(node_layers, X, False, **node_kwargs)
                
        torch.autograd.backward(list(out.values())[1:], reversed(new_node_grads.values()))
        return None, None, None, None, None, X['input'].grad, *map(lambda p: p.grad, node_layers.parameters())

def transfer(m1, m2):
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        p1.data.copy_(p2)
        
def t_to_nlt(t, nlt, n_layers):
    transfer(nlt.embedding, t.embedding)
    transfer(nlt.linear, t.linear)
    if hasattr(t, "cls_token"):
        nlt.cls_token.data.copy_(t.cls_token)
    for i in range(n_layers):
        if i == 0:
            attn_input = 'input'
        else:
            attn_input = f'ff_{i-1}'
        transfer(nlt.node_layers[f'attn_{i}'].W_o, t.transformer_blocks[i].mha.W_o)
        transfer(nlt.node_layers[f'attn_{i}'].input_maps[attn_input], t.transformer_blocks[i].mha.W_qkv)
        transfer(nlt.node_layers[f"ff_{i}"].linear2, t.transformer_blocks[i].ff_layer.ff[3])
        transfer(nlt.node_layers[f"ff_{i}"].input_maps[f"attn_{i}"], t.transformer_blocks[i].ff_layer.ff[0])