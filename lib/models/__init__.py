from .tvae import TVAE
from .vq_tvae import VQ_TVAE
from .prog_tvae import Programmatic_TVAE
from .prog_tvae_info import Programmatic_TVAE_info

model_dict = {
    'tvae' : TVAE,
    'vq_tvae' : VQ_TVAE,
    'prog_tvae' : Programmatic_TVAE,
    'prog_tvae_info' : Programmatic_TVAE_info
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        raise NotImplementedError
        