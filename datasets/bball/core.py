import os
import numpy as np
import torch
import pytorch_lightning as pl

from datasets import TrajectoryDataset
from torch.utils.data import random_split, DataLoader

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from skimage.transform import resize


# CONSTANTS
LENGTH = 94
WIDTH = 50
SCALE = 10

COORDS = {
    'ball' : [0,1],
    'offense' : [2,3,4,5,6,7,8,9,10,11],
    'defense' : [12,13,14,15,16,17,18,19,20,21]
}


class BBallDataset(TrajectoryDataset):

    name = 'bball_dataset'

    # Default config
    _seq_len = 50
    _state_dim = 22
    _action_dim = 22

    standardize_data = True
    single_agent = True
    player_types = {
        'ball' : False,
        'offense' : True,
        'defense' : False
    }
    subsample = 2

    def __init__(self, datapath):
        self.datapath = datapath
        super().__init__()

    def _load_data(self):

        data = np.load(self.datapath)['data']

        # Subsample timesteps
        data = data[:,::self.subsample]

        # Split up offense and defense
        offense = data[:,:,COORDS['offense']]
        defense = data[:,:,COORDS['defense']]
        data = np.concatenate([offense, defense], axis=0)

        # Save labels
        mode = self.datapath.split("/")[-1][:-4]
        labels = np.concatenate([np.ones(offense.shape[0]), np.zeros(defense.shape[0])])
        np.savez(f"datasets/bball/labels/offense_defense/{mode}.npz", annotations=labels, allow_pickle=True)

        # Standardize data
        if self.standardize_data:
            data = standardize(data)

        # Convert to states and actions
        states = data
        actions = states[:,1:] - states[:,:-1]

        # Update dimensions
        self._seq_len = actions.shape[1]
        self._state_dim = states.shape[-1]
        self._action_dim = actions.shape[-1]

        # Set data
        self.states = torch.Tensor(states)
        self.actions = torch.Tensor(actions)


class BBallDataModule(pl.LightningDataModule):

    data_dir = "datasets/bball/data"
    train_file = "train.npz"
    test_file = "test.npz"

    name = "bball"

    def __init__(self, data_config):
        super().__init__()

        self.batch_size = data_config['batch_size']
        self.val_split = data_config['val_split']

        self.summary = { 'name' : self.name }

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            bball_full = BBallDataset(os.path.join(self.data_dir, self.train_file))
            val_size = int(self.val_split*len(bball_full))
            train_size = len(bball_full)-val_size
            self.data_train, self.data_val = random_split(bball_full, [train_size, val_size])

            self._state_dim = bball_full.state_dim
            self._action_dim = bball_full.action_dim
            self._seq_len = bball_full.seq_len

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.data_test = BBallDataset(os.path.join(self.data_dir, self.test_file))

            self._state_dim = self.data_test.state_dim
            self._action_dim = self.data_test.action_dim
            self._seq_len = self.data_test.seq_len

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=4)

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

    def save(self, states, actions=[], save_path='', save_name='', burn_in=0, single_plot=False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if len(states.size()) == 2:
            states = states.unsqueeze(0)

        states = states.detach().numpy()
        states = unstandardize(states)

        if single_plot:
            states = np.swapaxes(states, 0, 1)
            states = np.reshape(states, (states.shape[0], 1, -1))
            states = np.swapaxes(states, 0, 1)

        n_players = int(states.shape[-1]/2)
        colormap = ['b'] * n_players

        for i in range(len(states)):
            seq = SCALE*states[i]

            fig, ax = _set_figax()

            for k in range(n_players):
                x = seq[:,(2*k)]
                y = seq[:,(2*k+1)]
                c = colormap[k]
                ax.plot(x, y, color=c, linewidth=5, alpha=0.5)
                ax.plot(x, y, color='k', linewidth=1, alpha=0.7) 

            # Starting positions
            x = seq[0,::2]
            y = seq[0,1::2]
            ax.plot(x, y, 'd', color='black', markersize=16)
        
            # Burn-ins (if any)
            if burn_in > 0:
                x = seq[:burn_in,0] if single_plot else seq[:burn_in,::2]
                y = seq[:burn_in,1] if single_plot else seq[:burn_in,1::2]

                ax.plot(x, y, color='black', linewidth=8, alpha=0.5)

            plt.tight_layout(pad=0)

            if len(save_name) == 0:
                plt.savefig(os.path.join(save_path, '{:03d}.png'.format(i)))
            else:
                plt.savefig(os.path.join(save_path, '{}.png'.format(save_name)))

            plt.close()


def standardize(data):
    """Scale by dimensions of court and mean-shift to center of left half-court."""
    state_dim = data.shape[2]
    shift = [int(WIDTH/2)] * state_dim
    scale = [LENGTH, WIDTH] * int(state_dim/2)
    return np.divide(data-shift, scale)

def unstandardize(data):
    """Undo stancaldardize."""
    state_dim = data.shape[2]
    shift = [int(WIDTH/2)] * state_dim
    scale = [LENGTH, WIDTH] * int(state_dim/2)
    return np.multiply(data, scale) + shift

def _set_figax():
    fig = plt.figure(figsize=(5,5))
    img = plt.imread("datasets/bball/data/court.png")
    img = resize(img,(SCALE*WIDTH,SCALE*LENGTH,3))

    ax = fig.add_subplot(111)
    ax.imshow(img)

    # Show just the left half-court
    ax.set_xlim([-50,550])
    ax.set_ylim([-50,550])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig, ax
