import os
import numpy as np
import torch
import pytorch_lightning as pl

from datasets import TrajectoryDataset
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class SyntheticDataset(TrajectoryDataset):

    name = 'synthetic_dataset'
    datapath = "datasets/synthetic/data"

    # Default config
    _seq_len = 25
    _state_dim = 2
    _action_dim = 2
    threshold_x = 10
    threshold_y = 10

    def __init__(self, mode, class1_dist=0.5, class2_dist=0.5):
        self.mode = mode
        self.class1_dist = class1_dist
        self.class2_dist = class2_dist
        super().__init__()

    def _load_data(self):
        if self.mode == 'train':
            n_seq = 10000
        elif self.mode == 'val':
            n_seq = 2000
        elif self.mode == 'test':
            n_seq = 2000

        class1_dist = int(100*self.class1_dist)
        class2_dist = int(100*self.class2_dist)
        self.datapath = os.path.join(self.datapath, 
            f"thx{self.threshold_x}_thy{self.threshold_y}_{100-class1_dist}-{class1_dist}_{100-class2_dist}-{class2_dist}")

        datafile = os.path.join(self.datapath, f"{self.mode}.npz")
        data = []

        if os.path.exists(datafile):
            data = np.load(datafile)['data']
            if data.shape[0] != n_seq:
                data = []
        elif not os.path.isdir(self.datapath):
            os.makedirs(self.datapath)

        if len(data) == 0:
            final_x = { True : [], False : [] }
            final_y = { True : [], False : [] }

            for i in range(n_seq):
                pos = np.random.normal(np.zeros(2))
                pos[0] += self.threshold_x
                pos[1] += self.threshold_y

                vel = np.random.normal(np.zeros(2))
                while np.linalg.norm(vel) < 0.05 or np.linalg.norm(vel) > 0.4:
                    vel = np.random.normal(np.zeros(2))

                dir_x = np.random.uniform() < self.class1_dist
                vel[0] += (2*int(dir_x) - 1)*0.4

                dir_y = np.random.uniform() < self.class2_dist
                vel[1] += (2*int(dir_y) - 1)*0.4

                pos_arr = []
                while len(pos_arr) == 0:
                    for j in range(self.seq_len):
                        if j == 0:
                            pos_arr.append(pos)
                        else:
                            noise = np.random.normal(np.zeros(2))/5.0
                            pos_arr.append(pos_arr[-1] + vel + noise)
                    # want synthetic data to be clean so threshold can be learned
                    if dir_x != (pos_arr[-1][0] > self.threshold_x) or dir_y != (pos_arr[-1][1] > self.threshold_y):
                        pos_arr = []

                final_x[dir_x].append(pos_arr[-1][0])
                final_y[dir_y].append(pos_arr[-1][1])
                data.append(np.stack(pos_arr))

            data = np.stack(data)

            np.savez(datafile, data=data)

        states = data
        actions = states[:,1:] - states[:,:-1]

        # Update dimensions
        self._seq_len = actions.shape[1]
        self._state_dim = states.shape[-1]
        self._action_dim = actions.shape[-1]

        # Set data
        self.states = torch.Tensor(states)
        self.actions = torch.Tensor(actions)


class SyntheticDataModule(pl.LightningDataModule):

    def __init__(self, data_config):
        super().__init__()
        self.batch_size = data_config['batch_size']

        self.class1_dist = data_config['class1_dist'] if 'class1_dist' in data_config else 0.5
        assert 0.0 <= self.class1_dist <= 1.0

        self.class2_dist = data_config['class2_dist'] if 'class2_dist' in data_config else 0.5
        assert 0.0 <= self.class2_dist <= 1.0

        self.summary = { 
            'name' : "synthetic_dataset",
            'class1_dist' : self.class1_dist,
            'class2_dist' : self.class2_dist
        }

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.data_train = SyntheticDataset(mode='train', class1_dist=self.class1_dist)
            self.data_val = SyntheticDataset(mode='val', class1_dist=self.class1_dist)

            self._state_dim = self.data_train.state_dim
            self._action_dim = self.data_train.action_dim
            self._seq_len = self.data_train.seq_len
            self.threshold_x = self.data_train.threshold_x
            self.threshold_y = self.data_train.threshold_y

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.data_test = SyntheticDataset(mode='test', class1_dist=self.class1_dist)

            self._state_dim = self.data_test.state_dim
            self._action_dim = self.data_test.action_dim
            self._seq_len = self.data_test.seq_len
            self.threshold_x = self.data_train.threshold_x
            self.threshold_y = self.data_train.threshold_y

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

    def save(self, states, save_path='', save_name='', burn_in=0):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        states = states.detach().numpy()

        fig = plt.figure(figsize=(6,6))

        plt.axvline(self.threshold_x, color='red')
        plt.axhline(self.threshold_y, color='red')

        for i in range(len(states)):
            seq = states[i]
            plt.plot(seq[:,0], seq[:,1], color='black', alpha=0.4)
            plt.plot(seq[0,0], seq[0,1], 'o', color='g', markersize=5, alpha=0.6)
            plt.plot(seq[-1,0], seq[-1,1], 'o', color='b', markersize=5)

        plt.xlim([-20+self.threshold_x, 20+self.threshold_x])
        plt.xlim([-20+self.threshold_y, 20+self.threshold_y])
        plt.xticks([], [])
        plt.yticks([], [])
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{save_name}.png"))
        plt.close()
