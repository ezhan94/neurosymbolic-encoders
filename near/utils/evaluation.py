import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import hamming_loss, f1_score


def compute_f1_scores(predicted, truth, num_labels):
    assert isinstance(predicted, torch.Tensor)
    assert isinstance(truth, torch.Tensor)

    all_f1 = list(f1_score(truth, predicted, average=None))

    if num_labels > 1:
        weighted_avg_f1 = f1_score(truth, predicted, average='weighted')
        unweighted_avg_f1 = f1_score(truth, predicted, average='macro')
    else:
        avg_f1 = f1_score(truth, predicted, average='binary')
        unweighted_f1 = None

    return weighted_avg_f1, unweighted_avg_f1, all_f1


def label_correctness(predictions, truths, num_labels=1):
    if len(predictions.size()) == 1:
        predictions = torch.sigmoid(predictions) > 0.5
    else:
        assert len(predictions.size()) == 2
        predictions = torch.max(predictions, dim=-1)[1]

    info = {}

    # Log ground-truth label distribution
    info['label_dist'] = list(torch.bincount(truths.long().cpu()).numpy()/truths.size(0))
    while len(info['label_dist']) < num_labels:
        info['label_dist'].append(0.0)

    # Log prediction distribution
    info['pred_dist'] = list(torch.bincount(predictions.long().cpu()).numpy()/predictions.size(0))
    while len(info['pred_dist']) < num_labels:
        info['pred_dist'].append(0.0)

    # Log accuracy
    info['hamming_accuracy'] = 1 - hamming_loss(truths.squeeze().cpu(), predictions.squeeze().cpu())
    
    # Log f1-scores
    info['weighted_f1'], info['unweighted_f1'], info['all_f1s'] = compute_f1_scores(truths.squeeze().cpu(), predictions.squeeze().cpu(), num_labels)

    return 1-info['weighted_f1'], info

def mse_eval(predictions, truths, num_labels=1):
    return F.mse_loss(predictions, truths).item(), {}
