import os
import torch
import pytorch_lightning as pl
import argparse
import json
import numpy as np

import sys
sys.path.append(sys.path[0] + '/..')

from lib.models import get_model_class
from datasets import load_dataset

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import rand_score


# Usage
# python scripts/compute_cluster_metrics.py --exp_folder <config_folder> --ckpt_name <model_name> --num_clusters <n_clusters> --comparison_file <file for labels>
# Example
# python scripts/compute_cluster_metrics.py --exp_folder test --ckpt_name final_model --num_clusters 2 --comparison_file /some_path/labels.npz


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

def main(exp_dir, trial_id, args):
    # Get trial folder
    trial_dir = os.path.join(exp_dir, trial_id)

    # Load config
    with open(os.path.join(exp_dir, 'configs', '{}.json'.format(trial_id)), 'r') as f:
        config = json.load(f)
    data_config = config['data_config']
    model_config = config['model_config']

    # Initialize dataset
    datamodule = load_dataset(data_config)
    datamodule.setup()

    # Add state and action dims to model config
    model_config['state_dim'] = datamodule.state_dim
    model_config['action_dim'] = datamodule.action_dim

    # Get saved pytorch model
    print(trial_dir)
    model = torch.load(os.path.join(trial_dir, 'checkpoints', args.ckpt_name))

    gt_clusters = np.load(args.comparison_file, allow_pickle=True)['annotations']

    if model.has_clusters:
        n_clusters = model.n_clusters

        test_cluster_dist = torch.zeros(n_clusters)

        predicted_clusters = []

        for batch_idx, batch in enumerate(datamodule.test_dataloader()):
            clusters = model.get_clusters(batch)

            test_cluster_dist += torch.zeros(batch[0].size()[0], n_clusters).scatter_(-1, clusters, 1).sum(dim=0)

            predicted_clusters.append(clusters)

        test_cluster_dist /= torch.sum(test_cluster_dist)

        print("-- Cluster distribution on test set --")
        for c in range(n_clusters):
            print(f"cluster {c} | {100*test_cluster_dist[c].item():.2f}")

        pred_clusters = torch.cat(predicted_clusters)
    else:

        pred_encodings = []
        for batch_idx, batch in enumerate(datamodule.test_dataloader()):
            states, actions = batch
            encodings = model.encode_neural(states[:, :-1], actions, batch_first = True)

            pred_encodings.append(encodings.mean.detach().numpy())

        kmeans = KMeans(n_clusters=int(args.num_clusters), random_state=0).fit(np.concatenate(pred_encodings))

        pred_clusters = kmeans.labels_ 

    print("-- Purity metric on test set --")
    print(purity_score(gt_clusters, pred_clusters))

    print("-- NMI metric on test set --")
    print(normalized_mutual_info_score(gt_clusters.squeeze(), pred_clusters.squeeze()))

    print("-- RI metric on test set --")
    print(rand_score(gt_clusters.squeeze(), pred_clusters.squeeze()))


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_folder', type = str, required = True,
                     help = 'folder of experiments from which to load models')

    parser.add_argument('--ckpt_name', type = str, required = True,
                     help = 'name of trained checkpoint to load')

    parser.add_argument('--save_dir', type = str, required = False,
                     default = 'saved',
                     help = 'save directory for experiments')

    parser.add_argument('--comparison_file', type = str, required = True,
                     help = 'file containing annotations to compare against')

    parser.add_argument('--num_clusters', type = str, required = False,
                     help = 'number of clusters for k means, used only for neural encodings')

    args = parser.parse_args()

    # Get exp_directory
    exp_dir = os.path.join(os.getcwd(), args.save_dir, args.exp_folder)

    # Load master file
    print(exp_dir)
    assert os.path.isfile(os.path.join(exp_dir, 'master.json'))
    with open(os.path.join(exp_dir, 'master.json'), 'r') as f:
        master = json.load(f)

    for trial_id in master['summaries']:
        main(exp_dir, trial_id, args)
