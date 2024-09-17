# NeuroBack: Improving CDCL SAT Solving using Graph Neural Networks (ICLR'24)

### Authors
Wenxi Wang, Yang Hu, Mohit Tiwari, Sarfraz Khurshid, Kenneth McMillan, Risto Miikkulainen

### Publication

If you use any part of our tool or data present in this repository or huggingface, please do cite our [ICLR'24 NeuroBack paper](https://iclr.cc/virtual/2024/poster/17641).

```
@inproceedings{
   wang2024neuroback,
   title={NeuroBack: Improving {CDCL} {SAT} Solving using Graph Neural Networks},
   author={Wenxi Wang and Yang Hu and Mohit Tiwari and Sarfraz Khurshid and Kenneth McMillan and Risto Miikkulainen},
   booktitle={The Twelfth International Conference on Learning Representations},
   year={2024},
}
```

### Overview

NeuroBack is an innovative SAT solver that integrates Graph Neural Networks (GNNs) to enhance Conflict-Driven Clause Learning (CDCL) SAT solving. The tool consists of two primary modules:

1. **GNN Module**: This module, built on [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), manages the pretraining, finetuning, and inference of the GNN model for backbone prediction.
2. **Solver Module**: Based on the [Kissat](https://github.com/arminbiere/kissat) solver, this module utilizes the GNN’s predictions to guide SAT solving.

### GNN Module

#### Python Environment

The GNN module requires Python 3, and it has been tested on Python 3.11 interpreter. To set up the python environment:

1. Install [Anaconda3](https://www.anaconda.com/), which includes most of the essential packages for machine learning and data science.
2. Follow the official documentation to install [PyTorch](https://pytorch.org/get-started/locally/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
3. Install additional dependencies using the following command:
   ```bash
   pip install xtract texttable
   ```

#### Datasets

##### Initial Small Dataset for the First Run

To facilitate initial testing and understanding of NeuroBack, a small dataset containing a few SAT formulas and their backbone variable phases is provided. The directory structure is as follows:

```
|-data
   |-cnf                # SAT formulas in CNF format
   |  |-pretrain        # For pretraining (backbone required)
   |  |-finetune        # For finetuning (backbone required)
   |  |-validation      # For validation (backbone required)
   |  |-test            # For testing (backbone NOT required)
   |
   |-backbone           # Backbone information
      |-pretrain        # For pretraining
      |-finetune        # For finetuning
      |-validation      # For validation
```

You can replace the provided datasets with your own SAT formulas and backbone variable information as needed.

##### **DataBack** Dataset

Our **DataBack** dataset, which includes 120,286 SAT formulas in CNF format along with their backbone variable phases, is available on [HuggingFace](https://huggingface.co/datasets/neuroback/DataBack). If you opt to use the **DataBack** dataset, please ensure it adheres to the same directory structure as the initial small dataset.

##### Customized Dataset

To use your own SAT formulas for pretraining or finetuning the GNN model, please utilize the [cadiback](https://github.com/arminbiere/cadiback) backbone extractor to compute the backbone variable phases. Please compress the backbone information in `xz` format and ensure your dataset is organized similarly to the initial small dataset.

#### Graph Representation Generation

Once your dataset is deployed, please generate graph representations for each SAT formula by running the following commands:

```bash
python3 graph.py pretrain
python3 graph.py finetune
python3 graph.py validation
python3 graph.py test
```

Note that it is normal to see "backbone file does not exist" messages when running the last command, as the `test` dataset does not include precomputed backbones.

Graph representations are saved in the `data/pt` folder. To accelerate the graph generation process, you can adjust the `n_cpu` variable in line 393 of `graph.py` to allocate more CPUs (by default, `n_cpu = 1`). However, please note that this may increase memory usage.

#### Pretraining and Finetuning

Once graph representations are ready, you can start the pretraining and finetuning process:

```bash
python3 learn.py pretrain   # Pretrain the GNN model
python3 learn.py finetune   # Finetune the GNN model
```

Pretrained model checkpoints are saved in `model/pretrain`, and finetuned model checkpoints in `model/finetune`. Each folder includes:

1. `[pretrain/finetune]-[i].ptg`: The model checkpoint after the i-th epoch of pretraining/finetuning.
2. `[pretrain/finetune]-best.ptg`: The model checkpoint with the best F1 score among those saved for each epoch. By default, pretrain-best.ptg will be loaded at the beginning of finetuning.

Logs regarding pretraining/finetuning are saved in `log/pretrain` or `log/finetune`, respectively. `gnn-load.log` (if it exists) records the performance metrics of the loaded model checkpoint on the validation set. `gnn-[i].log` contains the performance metrics of the model checkpoint for the i-th epoch. Metrics in the logs include confusion matrices, losses, precision, recall, and F1 scores.

You can customize hyperparameters such as learning rate and batch size by modifying lines 34-52 in `learn.py`. The default hyperparameter setting is the same as what has been introduced in our NeuroBack paper.

#### Backbone Prediction

To predict the backbone variables for SAT formulas in the `./data/cnf/test` folder using the finetuned GNN model (by default, the model we use is `./models/finetune/finetune-best.ptg`), you may choose to run the following command:

```bash
python3 predict_cuda.py
```

This command leverages GPU (cuda) for model inference and will skip formulas that trigger `cuda out of memory` errors. Alternatively, you can perform inference using CPUs:

```bash
python3 predict_cpu.py  # CPU-only inference
python3 predict_mix.py  # Mixed GPU and CPU inference (GPU first, then CPU if cuda out of memory)
```

Predictions are saved in the `./prediction/{cuda|cpu|mix}/cmb_predictions` folder. Each record contains a boolean variable ID and the estimated probability of being a positive or negative backbone (closer to 1 indicates a positive backbone; closer to 0 indicates a negative backbone).

Logs for predictions are saved in `./log/predict_cuda`, `./log/predict_cpu`, or `./log/predict_mix`. For each CNF file in the test dataset, a log file in csv format is generated, which records the CNF file name, the hardware used (i.e., cuda or cpu), and the time cost (in seconds) of model inference.

### Solver Module
(To be updated)

### Contact
For questions, please reach out to Wenxi Wang at [wenxiw@virginia.edu](mailto:wenxiw@virginia.edu) or Yang Hu at [huyang@utexas.edu](mailto:huyang@utexas.edu).