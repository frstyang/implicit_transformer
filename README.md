
# README
This repository implements a standard transformer, work-in-progress versions of an implicit transformer along with a training script that can be applied to various datasets. Instructions for downloading the necessary files for each dataset are given below.

To train on any dataset, run `python main.py --config configs/{config_file}`. Model specifications such as the width, depth, transformer class, and training hyperparameters can be set by directly modifying the specified config.

## Models
`Transformer`: this is a standard transformer, implemented in `models/transformer.py`.

`ImplicitTransformer`: an implicit transformer that divides its state vector into `n_relu_heads` ReLU blocks of size `relu_dim`, `n_layer_norms` LayerNorm blocks of size `ln_dim`, and `n_heads` Attention blocks of size `dim`. This model, its picard iteration-based forward and backward pass, and its projection algorithm for well-posedness are implemented in `models/implicit_transformer.py`, `models/implicit_transformer_function.py`, and `models/projection.py`.

`NodeLayerTransformer`: a transformer composed of `NodeLayer` modules. A `NodeLayer` represents a layer as a node in a directed graph. The architecture is completed by specifying the connections in the graph. Currently, `FFLayer` and `AttentionLayer` node layers are implemented. A standard transformer block computes the following:
```
# transformer block: attention block -> ff block
# attention block
attn_in = W_qkv(x)
out = multi_head_attention(attn_in)
x = layer_norm(x + out)

# ff block
ff_in = W_1@x + b_1
out = W_2@relu(ff_in) + b_2
x = layer_norm(x + out)
```
The above computation is represented inside a `NodeLayerTransformer` by instantiating an `AttentionLayer` and an `FFLayer` and connecting them with `attention_layer.connect(ff_layer)`. A standard transformer consists of a linear chain of these blocks. If a loop is added, the model becomes an implicit transformer. Extra connections can be added using the `extra_connections` argument. For example, `extra_connections = [(ff_3, attn_1)]` results in a loop from the 4th `FFLayer` connecting back to the 2nd `AttentionLayer`.
The model detects if the underlying graph is a DAG (i.e., strictly upper triangular `A` matrix), and if so, runs a single forward and backward computation through the network in topological order. This is equivalent to a standard forward/backward pass. If a cycle is detected, picard iteration is run instead, which repeats forward/backward computations until convergence or reaching a maximum number of iterations. The model is implemented in `models/node_implicit_transformer.py`.


## Results
**SQuAD**: Training a standard transformer (labeled `transformer_causal_seed0_0`), and training the same architecture and initialization ported to a node layer transformer (`nlt_causal_seed0_0`) obtain the same performance on SQuAD. This is a sanity check for the node layer transformer implementation in the non-implicit case.
[Performance curves](https://api.wandb.ai/links/forestyang/dxux823e)

Adding the single connection `(ff_3, attn_1)` to make the model implicit results in unstable training, and performance suffers as a result. As training progresses, the forward and backward pass start to always hit the iteration limit and do not converge. With that single connection, the model is already "too implicit" (hypothetically, the Perron-Frobenius norm is too large) to train without regularization or constraint. Perhaps connecting layers that are more adjacent, or "lightening" the connection somehow could help.
[Performance curves](https://api.wandb.ai/links/forestyang/pr1279v2)

To run the standard transformer in the first experiment, run `python main.py --config configs/squad_transformer.py`. To run the node layer transformer, run `python main.py--config configs/squad_node_layer_transformer.py`.

To train the `NodeLayerTransformer` in the second experiment, set `extra_connections` to `[(ff_3, attn_1)]`. Any desired extra connections may be passed as a list of tuples of the form `(u, v)` where `u` is the name of the node that is to be connected, i.e. feed into, the node named `v`. The node layer transformer with `n_layers=6` has nodes `attn_0, ff_0, ... attn_5, ff_5`.

**WikiText-103**: a standard transformer with the settings in `configs/wikitext103_transformer.py` fluctuates around 50 test perplexity after training for 25k iterations.
[Performance curve](https://api.wandb.ai/links/forestyang/a78hzo4i)

**Place Value Dataset**: a standard transformer with `configs/place_value_transformer.py` quickly reaches 0 training error and 100% training accuracy. The test accuracy however is limited to ~20%. Training with `relative=True` (relative positional embeddings) and `universal=True` ([universal transformers](https://arxiv.org/abs/1807.03819)) increases test accuracy to ~85%. This was observed in https://arxiv.org/abs/2108.12284. 
[Performance curves](https://api.wandb.ai/links/forestyang/jlme0ixh)

Training an `ImplicitTransformer` with `configs/place_value_implicit_transformer.py` causes the loss to shoot back up every time projection occurs, which is every 20 iterations. This suggests that the projection algorithm is imposing too stringent of a constraint on the model.
[Performance curves](https://api.wandb.ai/links/forestyang/zxn41eq6)

## Requirements
You will need the standard `numpy`, `torch` (preferably with cuda available), `einops`,`tqdm`, and `scikit-learn`packages.
This repo also makes use of several packages developed by Hugging Face for processing NLP datasets. They can be installed as follows:
```
pip install datasets==2.10.1
pip install evaluate==0.4.0
pip install tokenizers=0.12.1
pip install transformers=4.19.1
```
Finally, this repo uses wandb to conveniently log and plot runs in real time. Create an account at `wandb.ai`, install with `pip install wandb` and enter your API key when prompted to do so.

## Instructions for downloading datasets
### SQuAD
The Stanford Question-Answering Dataset (SQuAD) consists of question-passage pairs where the goal is to determine where the answer begins and ends in the passage. The dataset tokenization, preprocessing, and evaluation are taken from https://huggingface.co/course/chapter7/7?fw=tf. 
An example input text may look like
```
[CLS] What is the Grotto at Notre Dame? [SEP] it, is a copper statue of Christ with arms upraised with the legend " Venite Ad Me Omnes ". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette So [SEP]
```
where the question is separated from the passage by a `[SEP]` token. In this case, the 64th to 70th tokens of the input text contain the answer, `'a Marian place of prayer and reflection'`, so the model will try to assign a high probability to the 64th token of being the start token and a high probability of being the end token to the 70th token.

Download `train-v2.0.json` and `dev-v2.0.json` from https://rajpurkar.github.io/SQuAD-explorer/ by clicking on the buttons labeled "Training Set v2.0 (40 MB)" and "Dev Set v2.0 (4 MB)". Move these files to `cache/SQuAD`. In the end, the directory layout should look like:
```
cache/SQuAD/
    train-v2.0.json
    dev-v2.0.json
```
### WikiText-103
Text extracted from Wikipedia used for language modeling, where the task is to predict the next token. For example, the first 400 characters of the training set are:
```
= Valkyria Chronicles III = 

  Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series .
```
   The tokenizer, taken from https://huggingface.co/docs/tokenizers/quicktour, uses a Byte-Pair Encoding and tokenizes the previous text into the following 103 tokens separated by spaces:
```
   = V alk y ria Chronicles III = Sen j ō no V alk y ria 3 : Un recorded Chronicles ( Japanese : 戦 場 の ヴ ァ ル キ ュ リ ア 3 , lit . V alk y ria of the Battle field 3 ) , commonly referred to as V alk y ria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media . Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the V alk y ria series .
```
   Download the raw WikiText-103 data here: https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/. Scroll down to the link that says "Download WikiText-103 raw character level data" and click to download.
   Then, extract the contents of the downloaded zip file into the `cache/raw` directory in this repository. If this directory does not exist, create it. In the end, the directory layout should look like this:
```
cache/raw/
    wikitext-103-raw/
        wiki.test.raw
        wiki.train.raw
        wiki.valid.raw
```

### Place Value Dataset
This dataset contains examples of the form:
Input: What is the millions digit of 9967193?
Output: 9

The setup of this dataset is based on https://github.com/robertcsordas/transformer_generalization.
Download the mathematics_dataset-v1.0 folder from [here](https://drive.google.com/drive/folders/1PDcNYAkqEbQP-k5do1YohczgHtQGl8mM?usp=share_link) and move it to the `cache` directory in this repository. The directory layout should look like:
```
cache/dm_math/mathematics_dataset-v1.0/
    extrapolate/
    interpolate/
    train-easy/
    train-hard/
    train-medium/
```
