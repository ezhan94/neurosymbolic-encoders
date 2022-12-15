# Unsupervised Learning of Neurosymbolic Encoders

This repository contains the code for the TMLR paper titled [Unsupervised Learning of Neurosymbolic Encoders](https://openreview.net/forum?id=eWvBEMTlRq)

## Requirements

- Python 3.8.5+

- PyTorch 1.7.1+

- Pytorch Lightning 1.1.4+

- Comet.ml 3.2.12+

## Code Structure

- `configs/` contains experiment JSON files where all experiment hyperparameters are set. Some examples are included.

- `datasets/` contains the datasets and their respective DSLs.

- `lib/` contains all model files, as well as all distribution functions (Bernoulli, Gumbel-Softmax, Gaussian). We include code for learning programs for continuous latent variables, but it is not as thouroughly tested as we only experiment with discrete latent variables in our work. 

- `near/` contains the code used for program synthesis. See below. 

- `scripts/` currently contains one script for computing cluster metrics. Usage is `python scripts/compute_cluster_metrics.py --exp_folder <config_folder> --ckpt_name <model_name> --num_clusters <n_clusters> --comparison_file <file for labels>`

- `run_training.py` is the main file for starting experiments and log them on comet. Usage is `python run_training.py --config_dir <config_folder> -g <num_gpus>`

## Program Synthesis via NEAR

Our strategy for synthesizing programs is based on [NEAR](https://arxiv.org/abs/2007.12101). We integrate with their [code](https://github.com/trishullab/near) in our `update_neurosymbolic_encoder()` method in `run_training.py`.

We repurpose their algorithms for updating a program with an `update()` method in `near/algorithms/<algorithm>.py`. The only algorithms we've repurposed so far are `mc_sampling` and `iddfs_near`.

We also introduce and change DSL library functions in `near/dsl/`.

## Datasets

- Synthetic - the code is included in the repo and will generate new data and labels on the first run.

- CalMS21 (mouse) - the processed dataset can be downloaded at the following anonymized Google Drive [link](https://drive.google.com/drive/folders/10FUiGlaQjjjZELelgYXU0aT9ha2fN78s?usp=sharing)

- Basketball - the dataset can be downloaded for free [here](https://aws.amazon.com/marketplace/pp/prodview-7kigo63d3iln2?qid=1606330770194&sr=0-1&ref_=srh_res_product_title#offers)

## Adding a New Dataset

To add a new dataset to run an experiment requires the following steps:

1) Create a new `pl.LightningDataModule` in `datasets/`.

2) Create necessary library functions in `near/dsl/`.
 
3) Set the DSL in `datasets/<your dataset>/dsl.py` and reference it in the config JSON.

## Examples

`python run_training.py --config_dir synthetic/test -g 1` should run and terminate without errors.
