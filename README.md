# OASIS

**OASIS** is the framework proposed in our paper:

> **Towards Online Adaptive LLM-based Semantic Filtering**

OASIS performs semantic filtering by utilizing document similarity as a prior and adopting online active GAT training and sampling, together with progressive calibration sampling which adapts to queries with different difficulties. OASIS controls the labeling budgets with different ad hoc online queries while achives high query accuracy.

##  Environment Setup

We recommend using a dedicated Conda environment with Python 3.11.

```bash
conda create -n oasis python=3.11 -y
conda activate oasis
python -m pip install --upgrade pip setuptools wheel
```

Install the required dependencies:

```bash
pip install \
  "numpy>=1.26.0" \
  "scipy>=1.11.0" \
  "scikit-learn>=1.3.0" \
  "matplotlib>=3.8.0" \
  "torch>=2.1.0" \
  "torch-geometric>=2.5.0" \
  "vllm>=0.5.0" \
  "requests>=2.31.0" \
  "tqdm>=4.66.0"
```

> **Note:** `torch`, `torch-geometric`, and `vllm` depend on the CUDA and GPU configuration of the machine. If dependency resolution replaces an existing PyTorch installation, install these packages in a fresh Conda environment and ensure that the PyTorch/CUDA versions are compatible with your system.

## Configure the LLM Oracle

The default LLM oracle is **GPT-4o**, accessed through OpenRouter.

Set your OpenRouter API key before running OASIS:

```python
import os

os.environ["OPENROUTER_API_KEY"] = "YOUR_OPENROUTER_API_KEY"
```

##  Prepare the Input Data

OASIS requires a document collection and a semantic query predicate .

### Document input: `--doc-path`

`--doc-path` must point to data whose loaded content is a Python document list:

```python
list[str]
```

Each element is the text of one document. For example:

```python
[
    "The first document.",
    "The second document.",
    "The third document.",
]
```

### Query input: `--query-path`

`--query-path` must point to data whose loaded content is a single query string:

```python
str
```

For example:

```python
"Does the study indicate a positive effect of the treatment?"
```

In summary:

| Argument | Expected loaded value | Meaning |
|---|---|---|
| `--doc-path` | `list[str]` | A list containing the text of all documents |
| `--query-path` | `str` | The semantic query predicate |

##  Run OASIS

A minimal run using the default hyperparameters is:

```bash
python main.py   --doc-path /path/to/documents.pkl   --query-path /path/to/query.pkl   --target-accuracy 0.9
```

## Command-Line Arguments

### Sampling and online adaptation

| Argument | Type | Default | Description |
|---|---:|---:|---|
| `--target-accuracy` | `float` | `0.9` | Target accuracy. |
| `--budget` | `int` | `1000` | Total oracle labeling or oracle budget for proxy training. |
| `--mini-batches` | `int` | `10` | Number of mini-batches used during GAT training. |
| `--first-sample` | `int` | `400` | Number of samples selected in the initial sampling stage. |
| `--round-budget` | `int` | `100` | Maximum labeling budget for each interative round. |


### GAT Model and Active Learning

| Argument | Type | Default | Description |
|---|---:|---:|---|
| `--hidden-dim` | `int` | `128` | Hidden representation dimension. |
| `--num-layers` | `int` | `2` | Number of GAT layers. |
| `--training-steps` | `int` | `225` | Number of training steps. |
| `--lr` | `float` | `0.005` | Learning rate. |
| `--label-smoothing` | `float` | `0.1` | Label-smoothing coefficient. |
| `--supcon-weight` | `float` | `0.2` | Weight of the supervised contrastive loss. |
| `--smooth-weight` | `float` | `0.1` | Weight of the graph-smoothing regularizer. |
| `--lambda1` | `float` | `0.08` | Early-stopping threshold $\eta_1$. |
| `--lambda2` | `float` | `0.08` | Early-stopping threshold $\eta_2$. |
| `--lambda3` | `float` | `0.09` | Early-stopping threshold $\eta_3$. |
| `--temperature` | `float` | `0.3` | Temperature used by the contrastive objective. |
| `--unlabeled-temp` | `float` | `None` | Optional temperature specifically for unlabeled samples. |

### Graph construction

| Argument | Type | Default | Description |
|---|---:|---:|---|
| `--smooth-mode` | `str` | `unlabeled` | Selects which nodes are used by the graph-smoothing objective. |
| `--knn-k` | `int` | `200` | Number of nearest neighbors used during graph construction. |
| `--nsg-R` | `int` | `150` | Max neighborhood-size parameter used by NSG construction. |

