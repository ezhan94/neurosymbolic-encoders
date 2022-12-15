from .core import TrajectoryDataset
from .bball import BBallDataModule
from .mouse import MouseDataModule
from .synthetic import SyntheticDataModule


dataset_dict = {
    # 'bball' : BBallDataModule,
    'mouse' : MouseDataModule,
    'synthetic' : SyntheticDataModule
}


def load_dataset(data_config):
    dataset_name = data_config['name'].lower()

    if dataset_name in dataset_dict:
        return dataset_dict[dataset_name](data_config)
    else:
        raise NotImplementedError