import random
import torch
from torch.utils.data import Dataset

TRAIN = 1
EVAL = 2


class TrajectoryDataset(Dataset):

    # Default parameters
    _state_dim = 0
    _action_dim = 0
    _seq_len = 0

    def __init__(self):
        assert hasattr(self, 'name')

        self.summary = {'name' : self.name}

        # Load data (and true labels, if any)
        self._load_data()

        # Assertions for train data
        assert hasattr(self, 'states') and isinstance(self.states, torch.Tensor)
        assert hasattr(self, 'actions') and isinstance(self.actions, torch.Tensor)
        assert self.states.size(0) == self.actions.size(0)
        assert self.states.size(1)-1 == self.actions.size(1) == self.seq_len

    def __len__(self):
        return self.states.size(0)

    def __getitem__(self, index):
        states = self.states[index,:,:self.state_dim]
        actions = self.actions[index,:,:self.action_dim]
        return states, actions

    @property
    def seq_len(self):
        assert self._seq_len > 0
        return self._seq_len

    @property
    def state_dim(self):
        assert self._state_dim > 0
        return self._state_dim

    @property
    def action_dim(self):
        assert self._action_dim > 0
        return self._action_dim
        
    def _load_data(self):
        raise NotImplementedError
