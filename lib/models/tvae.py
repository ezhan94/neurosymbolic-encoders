import torch
import torch.nn as nn

from lib.models.core import BaseSequentialModel
from lib.distributions import Normal


class TVAE(BaseSequentialModel):

    name = 'tvae' # trajectory VAE
    has_programmatic_encoder = False
    has_clusters = False

    num_steps = 0
    use_beta = False

    def __init__(self, model_config):
        super().__init__()

        self.num_layers = model_config["num_layers"]
        self.rnn_dim = model_config["rnn_dim"]  
        self.z_dim = model_config["z_dim"]
        self.h_dim = model_config["h_dim"]
        self.state_dim = model_config["state_dim"]
        self.action_dim = model_config["action_dim"]

        self.cluster_loss_weight = model_config["cluster_loss"] if "cluster_loss" in model_config else 0.0
        self.decoder_type = model_config["decoder_type"] if "decoder_type" in model_config else "rnn"

        if "beta_params" in model_config.keys():
            self.beta_params = model_config["beta_params"]
            self.use_beta = True

        self.enc_birnn = nn.GRU(self.state_dim+self.action_dim, self.rnn_dim, num_layers=self.num_layers, bidirectional=True)

        self.enc_fc = nn.Sequential(
            nn.Linear(2*self.rnn_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(self.h_dim, self.z_dim)
        self.enc_logvar = nn.Linear(self.h_dim, self.z_dim)

        if self.decoder_type == 'rnn':
            self.dec_action_fc = nn.Sequential(
                nn.Linear(self.state_dim+self.z_dim+self.rnn_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU())
            self.dec_action_mean = nn.Linear(self.h_dim, self.action_dim)
            self.dec_action_logvar = nn.Linear(self.h_dim, self.action_dim)

            self.dec_rnn = nn.GRU(self.state_dim+self.action_dim, self.rnn_dim, num_layers=self.num_layers)

        elif self.decoder_type == 'linear':
            self.is_recurrent = False

            self.dec_action_mean = nn.Linear(self.state_dim+self.z_dim, self.action_dim)
            self.dec_action_logvar = nn.Linear(self.state_dim+self.z_dim, self.action_dim)

        else:
            raise NotImplementedError

    def init_model(self):
        self.num_steps = 0
        return

    def cluster_loss(self, H, kloss, lmbda, batch_size):
        # Implementation of cluster loss from VAME: https://github.com/LINCellularNeuroscience/VAME
        gram_matrix = (H.T @ H) / batch_size
        _ ,sv_2, _ = torch.svd(gram_matrix)
        sv = torch.sqrt(sv_2[:kloss])
        loss = torch.sum(sv)
        return lmbda*loss

    def _shared_eval(self, batch, batch_idx, mode):
        self.num_steps += 1

        states, actions = batch
        assert actions.size(1)+1 == states.size(1) # final state has no corresponding action
        states = states.transpose(0,1)
        actions = actions.transpose(0,1)

        posterior = self.encode(states[:-1], actions=actions)

        kld_true = type(posterior).kld(posterior, free_bits=0.0).mean().detach()
        self.log(f'{mode}_kld_true', kld_true)        

        kld = type(posterior).kld(posterior, free_bits=1/self.z_dim).mean()
        self.log(f'{mode}_kld', kld, prog_bar=True)       

        self.reset_policy(z=posterior.sample())

        nll = 0.0
        for t in range(actions.size(0)):
            action_likelihood = self(states[t])
            nll -= action_likelihood.log_prob(actions[t]).mean()

            if self.is_recurrent:
                self.update_hidden(states[t], actions[t])
        
        self.log(f'{mode}_nll', nll, prog_bar=True)        

        if self.use_beta:
            gamma, c_max, c_stop_iter, max_iter =  self.beta_params
            C = min(self.num_steps*c_max/c_stop_iter, c_max)
            loss = nll + gamma*(kld - C).abs()
        else:
            loss = kld + nll

        if self.cluster_loss_weight > 0:
            cluster_loss_val = self.cluster_loss(posterior.mean.T, 2, self.cluster_loss_weight, states.size(1))
            self.log(f'{mode}_cluster_loss', cluster_loss_val)            
            loss = loss + cluster_loss_val

        self.log(f'{mode}_loss', loss)        
        return loss

    def encode_neural(self, states, actions=None, batch_first=False):
        if batch_first:
            # transpose to 1st dim time, 2nd dim batch
            states = states.transpose(0,1) 
            actions = actions.transpose(0,1)

        return self.encode(states, actions)
        