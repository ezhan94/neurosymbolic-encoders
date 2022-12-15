import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.core import BaseSequentialModel
from lib.models import Programmatic_TVAE


class Programmatic_TVAE_info(BaseSequentialModel):

    name = 'prog_tvae_info'
    has_programmatic_encoder = True

    def __init__(self, model_config):
        super().__init__()

        self.adv_h_dim = model_config["adv_h_dim"] if "adv_h_dim" in model_config else 8
        self.adv_loss_coeff = model_config["adv_loss_coeff"] if "adv_loss_coeff" in model_config else 1.0
        self.adv_lr_scale = model_config["adv_lr_scale"] if "adv_lr_scale" in model_config else 1.0
        self.num_advs = model_config["num_advs"] if "num_advs" in model_config else 1

        self.prog_tvae = Programmatic_TVAE(model_config)
        self.init_model(init_prog_tvae=False)

    @property
    def num_progs_fixed(self):
        return self.prog_tvae.num_progs_fixed

    @property
    def num_progs_learn(self):
        return self.prog_tvae.num_progs_learn

    @property
    def num_progs(self):
        return self.prog_tvae.num_progs

    @property
    def enc_progs_fixed(self):
        return self.prog_tvae.enc_progs_fixed

    @property
    def enc_progs_learn(self):
        return self.prog_tvae.enc_progs_learn

    @property
    def state_dim(self):
        return self.prog_tvae.state_dim

    @property
    def action_dim(self):
        return self.prog_tvae.action_dim

    @property
    def z_dim(self):
        return self.prog_tvae.z_dim
    
    @property
    def z_neural_dim(self):
        return self.prog_tvae.z_neural_dim

    @property
    def z_prog_dim(self):
        return self.prog_tvae.z_prog_dim

    @property
    def z_prog_type(self):
        return self.prog_tvae.z_prog_type

    @property
    def n_clusters(self):
        return self.prog_tvae.n_clusters

    @property
    def has_clusters(self):
        return self.prog_tvae.has_clusters
    
    def init_model(self, init_prog_tvae=True):
        if init_prog_tvae:
            self.prog_tvae.init_model()

        # Have to init adversaries like this because z_neural_dim may have changed
        self.adversaries = nn.ModuleList([])
        for _ in range(self.num_advs):
            self.adversaries.append(MLP(self.z_neural_dim, self.z_prog_dim, self.adv_h_dim))

    def set_enc_prog(self, program, enc_prog_type=None, prog_ind=None):
        self.prog_tvae.set_enc_prog(program, enc_prog_type=enc_prog_type, prog_ind=prog_ind)

    def encode_neural(self, states, actions=None, batch_first=False):
        return self.prog_tvae.encode_neural(states, actions, batch_first)

    def get_clusters(self, batch, sample=False):
        return self.prog_tvae.get_clusters(batch, sample)

    def get_labels(self, batch, sample=False):
        return self.prog_tvae.get_labels(batch, sample)

    def configure_optimizers(self, lr=1e-3):
        opt_tvae = self.prog_tvae.configure_optimizers(lr=lr)
        opt_adv = torch.optim.Adam(self.adversaries.parameters(), lr=lr*self.adv_lr_scale, betas = (0.5, 0.999))

        return [opt_adv, opt_tvae], [] 

    def _shared_eval(self, batch, batch_idx, mode, optimizer_idx=None):
        prog_tvae_loss, prog_tvae_loss_dict, (z_prog_list, z_neural) = self.prog_tvae._shared_eval(batch, batch_idx, mode, return_all=True)

        # Adversarial loss
        adversarial_loss = 0.0
        for i in range(self.num_progs):
            adv_loss = self.adv_loss_coeff*F.binary_cross_entropy_with_logits(self.adversaries[i](z_neural), z_prog_list[i])
            self.log(f'{mode}_adv_{i}', adv_loss, prog_bar=True)
            adversarial_loss += adv_loss

        # Log other losses
        for i in range(self.num_progs_learn):
            self.log(f'{mode}_kl_prog_{i}', prog_tvae_loss_dict[f'kld_prog_{i}'])
        self.log(f'{mode}_kl_neur', prog_tvae_loss_dict['kld_neural'])
        self.log(f'{mode}_nll', prog_tvae_loss_dict['nll'], prog_bar=True)
        self.log(f'{mode}_loss', prog_tvae_loss-adversarial_loss)

        if mode == 'train':
            if optimizer_idx == 0:
                return adversarial_loss # train adversaries
            if optimizer_idx == 1:
                return prog_tvae_loss - adversarial_loss # train prog_tvae
        else:
            return prog_tvae_loss - adversarial_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        return self._shared_eval(batch, batch_idx, mode='train', optimizer_idx=optimizer_idx)

    def forward(self, states):
        return


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, h_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.h_dim = h_dim

        self.init_model()

    def init_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.output_dim))

    def forward(self, batch):
        return self.model(batch)
