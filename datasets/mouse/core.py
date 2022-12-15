import torch
import pytorch_lightning as pl
import os
import numpy as np
from torch.utils.data import random_split, DataLoader
from datasets import TrajectoryDataset
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from sklearn.decomposition import TruncatedSVD



FRAME_WIDTH_TOP = 1024
FRAME_HEIGHT_TOP = 570

RESIDENT_COLOR = 'lawngreen'
INTRUDER_COLOR = 'skyblue'

PLOT_MOUSE_START_END = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4),
                        (3, 5), (4, 6), (5, 6), (1, 2)]


class MouseDataset(TrajectoryDataset):

    name = 'mouse_dataset'

    # Default config
    _seq_len = 21
    _state_dim = 28
    _action_dim = 28

    normalize_data = True
    compute_svd = False

    svd_computer_path = 'datasets/mouse/data/svd/svd_computer.pkl'
    mean_path = 'datasets/mouse/data/svd/mean.pkl'

    def __init__(self, root_dir, file_name, compute_svd = 5, save_svd = False, normalize = True):
        self.root_dir = root_dir
        self.file_name = file_name
        self.compute_svd = compute_svd
        self.save_svd = save_svd
        self.normalize_data = normalize

        super().__init__()

    def _load_data(self):
        self.states, self.actions = self._load_and_preprocess()

    def _load_and_preprocess(self):
        path = os.path.join(self.root_dir, self.file_name)
        file = np.load(path, allow_pickle = True)
        data = file['data']

        if self.normalize_data:
            data = normalize(data) 

        # Compute SVD on train data and apply to train and test data
        if self.compute_svd:
            seq_len = data.shape[1]

            data = data.reshape((-1, 2, 7, 2))
            # Input [seq_num x seq_len, 2, 7, 2]
            if self.save_svd or not os.path.isfile(self.svd_computer_path) or not os.path.isfile(self.mean_path):
                print("Computing SVD components ...")
                data_svd = transform_to_svd_components(
                    data, center_index=3, n_components=self.compute_svd,
                    save_svd_computer=self.svd_computer_path, save_mean=self.mean_path)
            else:
                with open(self.svd_computer_path, 'rb') as f:
                    self.svd_computer = pickle.load(f)
                with open(self.mean_path, 'rb') as f:        
                    self.mean = pickle.load(f)                
                data_svd = transform_to_svd_components(
                    data, center_index=3, n_components=self.compute_svd,
                    svd_computer=self.svd_computer, mean=self.mean)                   

            data = data_svd.reshape((-1, seq_len, data_svd.shape[-1] * 2))

        states = data
        actions = states[:, 1:] - states[:, :-1]

        # Update dimensions
        self._seq_len = actions.shape[1]
        self._state_dim = states.shape[-1]
        self._action_dim = actions.shape[-1]

        return torch.Tensor(states), torch.Tensor(actions)


class MouseDataModule(pl.LightningDataModule):

    # Default config
    # TODO this is repeated twice
    _seq_len = 21
    _state_dim = 28
    _action_dim = 28

    normalize_data = True

    compute_svd = False
    svd_computer_path = 'datasets/mouse/data/svd/svd_computer.pkl'
    mean_path = 'datasets/mouse/data/svd/mean.pkl' 

    name = "mouse"

    def __init__(self, data_config):
        super().__init__()

        self.data_dir = data_config['data_dir']
        self.train_name = data_config['train_name']
        self.val_name = data_config['val_name'] 
        self.test_name = data_config['test_name']    
        self.compute_svd = data_config['compute_svd']  
        self.normalize_data = data_config['normalize_data']
        self.batch_size = data_config['batch_size']

        self.summary = { 'name' : self.name }

        if self.compute_svd:
            self._state_dim = (4+self.compute_svd)*2
            self._action_dim = (4+self.compute_svd)*2            
        self.dims = (self._seq_len, self._state_dim) 

    def setup(self, stage=None):
        # called on every GPU

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.data_train = MouseDataset(self.data_dir, self.train_name, 
                compute_svd = self.compute_svd, save_svd = True, normalize = self.normalize_data)
            self.data_val = MouseDataset(self.data_dir, self.val_name, 
                compute_svd = self.compute_svd, normalize = self.normalize_data)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.data_test = MouseDataset(self.data_dir, self.test_name, 
                compute_svd = self.compute_svd, normalize = self.normalize_data)

        if self.compute_svd:
            with open(self.svd_computer_path, 'rb') as f:
                self.svd_computer = pickle.load(f)
            with open(self.mean_path, 'rb') as f:        
                self.mean = pickle.load(f)  

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle = False, num_workers = 4)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle = False, num_workers = 4)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle = False, num_workers = 4)

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

    def save(self, states,
             actions=[],
             plot_title = 'Mouse Plot',
             save_path='',
             save_name=''):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        states = states.detach().numpy()
        states = states.reshape((-1, self._seq_len-1, self._state_dim))

        # Restore absolute mouse positions if we computed svd
        if self.compute_svd:
            states = states.reshape((-1, states.shape[-1]))
            states = transform_svd_to_keypoints(
                states, self.svd_computer, self.mean)

            states = states.reshape((-1, self._seq_len-1, 28))

        if self.normalize_data:
            states = unnormalize(states)                

        for i in range(len(states)):
            current_save_path = os.path.join(save_path, "{:03}".format(i))
            if not os.path.exists(current_save_path):
                os.makedirs(current_save_path)

            seq = states[i].reshape((-1, 2, 7, 2))

            image_list = []
            for j in range(seq.shape[0]):
                fig, ax = _set_figax()

                plot_mouse(ax, seq[j, 0, :, :], color=RESIDENT_COLOR)
                plot_mouse(ax, seq[j, 1, :, :], color=INTRUDER_COLOR)

                ax.set_title(
                    plot_title + '\nseq {:03d}.png'.format(i) + ', frame {:03d}.png'.format(j))

                plt.tight_layout(pad=0)

                plt.savefig(os.path.join(
                    current_save_path, '{:03d}.png'.format(j)))

                image = np.fromstring(
                    fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                image = image.reshape(
                    fig.canvas.get_width_height()[::-1] + (3,))
                image_list.append(image)

                plt.close()

            # Plot animation.
            fig = plt.figure()
            im = plt.imshow(image_list[0])

            def animate(k):
                im.set_array(image_list[k])
                return im,
            ani = animation.FuncAnimation(fig, animate, frames=self._seq_len-1, blit=True)
            ani.save(os.path.join(save_path, '{:03d}_animation.gif'.format(i)),
                     writer='imagemagick', fps=10)
            plt.close()


def _set_figax():
    fig = plt.figure(figsize=(5, 2.7))

    img = np.zeros((FRAME_HEIGHT_TOP, FRAME_WIDTH_TOP, 3))

    ax = fig.add_subplot(111)
    ax.imshow(img)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig, ax


def plot_mouse(ax, pose, color):
    # Draw each keypoint
    for j in range(7):
        ax.plot(pose[j, 0], pose[j, 1], 'o', color=color, markersize=5)

    # Draw a line for each point pair to form the shape of the mouse

    for pair in PLOT_MOUSE_START_END:
        line_to_plot = pose[pair, :]
        ax.plot(line_to_plot[:, 0], line_to_plot[
                :, 1], color=color, linewidth=1)


def normalize(data):
    """Scale by dimensions of image and mean-shift to center of image."""
    state_dim = data.shape[2] // 2
    shift = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    scale = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    return np.divide(data - shift, scale)


def unnormalize(data):
    """Undo normalize."""
    state_dim = data.shape[2] // 2
    shift = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    scale = [int(FRAME_WIDTH_TOP / 2), int(FRAME_HEIGHT_TOP / 2)] * state_dim
    return np.multiply(data, scale) + shift

def transform_to_svd_components(data,
                                center_index,
                                n_components,
                                svd_computer = None,
                                mean = None,
                                save_svd_computer = None,
                                save_mean = None):
    assert not (svd_computer is None and save_svd_computer is None)
    assert not (mean is None and save_mean is None)    

    # data shape is num_seq x 2 x 7 x 2
    resident_keypoints = data[:, 0, :, :]
    intruder_keypoints = data[:, 1, :, :]
    data = np.concatenate([resident_keypoints, intruder_keypoints], axis=0)
    
    # Center the data using given center_index
    mouse_center = data[:, center_index, :]
    centered_data = data - mouse_center[:, np.newaxis, :]

    # Rotate such that keypoints 3 and 6 are parallel with the y axis
    mouse_rotation = np.arctan2(
        data[:, 3, 0] - data[:, 6, 0], data[:, 3, 1] - data[:, 6, 1])

    R = (np.array([[np.cos(mouse_rotation), -np.sin(mouse_rotation)],
                   [np.sin(mouse_rotation),  np.cos(mouse_rotation)]]).transpose((2, 0, 1)))

    # Encode mouse rotation as sine and cosine
    mouse_rotation = np.concatenate([np.sin(mouse_rotation)[:, np.newaxis], np.cos(
        mouse_rotation)[:, np.newaxis]], axis=-1)

    centered_data = np.matmul(R, centered_data.transpose(0, 2, 1))
    centered_data = centered_data.transpose((0, 2, 1))

    centered_data = centered_data.reshape((-1, 14))

    if mean is None:
        mean = np.mean(centered_data, axis=0)
    centered_data = centered_data - mean

    # Compute SVD components
    if svd_computer is None:
        svd_computer = TruncatedSVD(n_components=n_components)
        svd_data = svd_computer.fit_transform(centered_data)
        print("SVD explained_variance_ratio: " +
              str(svd_computer.explained_variance_ratio_.sum()))
    else:
        svd_data = svd_computer.transform(centered_data)
        explained_variances = np.var(svd_data, axis=0) / np.var(centered_data, axis=0).sum()
        print("SVD explained_variance_ratio: " +
              str(np.sum(explained_variances)))

    # Concatenate state as mouse center, mouse rotation and svd components
    data = np.concatenate([mouse_center, mouse_rotation, svd_data], axis=1)
    resident_keypoints = data[:data.shape[0] // 2, :]
    intruder_keypoints = data[data.shape[0] // 2:, :]
    data = np.stack([resident_keypoints, intruder_keypoints], axis=1)

    if save_svd_computer is not None:
        with open(save_svd_computer, 'wb') as f:
            pickle.dump(svd_computer, f)
    if save_mean is not None:            
        with open(save_mean, 'wb') as f:        
            pickle.dump(mean, f)
    
    return data


def unnormalize_keypoint_center_rotation(keypoints, center, rotation):

    keypoints = keypoints.reshape((-1, 7, 2))

    # Apply inverse rotation
    rotation = -1 * rotation
    R = np.array([[np.cos(rotation), -np.sin(rotation)],
                  [np.sin(rotation),  np.cos(rotation)]]).transpose((2, 0, 1))
    centered_data = np.matmul(R, keypoints.transpose(0, 2, 1))

    keypoints = centered_data + center[:, :, np.newaxis]
    keypoints = keypoints.transpose(0, 2, 1)

    return keypoints.reshape(-1, 14)


def transform_svd_to_keypoints(data, svd_computer, mean):
    num_components = data.shape[1] // 2
    resident_center = data[:, :2]
    resident_rotation = data[:, 2:4]
    resident_components = data[:, 4:num_components]
    intruder_center = data[:, num_components:num_components + 2]
    intruder_rotation = data[:, num_components + 2:num_components + 4]
    intruder_components = data[:, num_components + 4:]

    resident_keypoints = svd_computer.inverse_transform(resident_components)
    intruder_keypoints = svd_computer.inverse_transform(intruder_components)

    if mean is not None:
        resident_keypoints = resident_keypoints + mean
        intruder_keypoints = intruder_keypoints + mean

    # Compute rotation angle from sine and cosine representation
    resident_rotation = np.arctan2(
        resident_rotation[:, 0], resident_rotation[:, 1])
    intruder_rotation = np.arctan2(
        intruder_rotation[:, 0], intruder_rotation[:, 1])

    resident_keypoints = unnormalize_keypoint_center_rotation(
        resident_keypoints, resident_center, resident_rotation)
    intruder_keypoints = unnormalize_keypoint_center_rotation(
        intruder_keypoints, intruder_center, intruder_rotation)

    data = np.concatenate([resident_keypoints, intruder_keypoints], axis=-1)


    return data

