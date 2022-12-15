import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.core import BaseSequentialModel
from lib.distributions import Bernoulli, Normal, GumbelSoftmax
from near.dsl import StartFunction


class Programmatic_TVAE(BaseSequentialModel):

    name = 'prog_tvae' # tvae with programmatic (neurosymbolic) encoder
    has_clusters = False
    use_cont_capacity = False
    use_disc_capacity = False

    def __init__(self, model_config):
        super().__init__()

        self.num_steps = 0

        self.h_dim = model_config["h_dim"]
        self.state_dim = model_config["state_dim"]
        self.action_dim = model_config["action_dim"]
        self.num_layers = model_config["num_layers"]
        self.rnn_dim = model_config["rnn_dim"]
        self.z_dim = model_config["z_dim"] # total z_dim (neural + programmatic)
        self.z_prog_dim = model_config["z_prog_dim"] # programmatic z dim
        self.z_prog_type = model_config["z_prog_type"]

        self.decoder_type = model_config["decoder_type"] if "decoder_type" in model_config else "rnn"
        self.prior_theta = model_config["prior_theta"] if "prior_theta" in model_config else 0.5 # prior on bernoulli latent variables

        if "cont_capacity" in model_config.keys():
            self.use_cont_capacity = True
            self.cont_capacity = model_config["cont_capacity"]
        if "disc_capacity" in model_config.keys():
            self.use_disc_capacity = True
            self.disc_capacity = model_config["disc_capacity"]

        if self.z_prog_dim > 0:
            self.enc_progs_fixed = nn.ModuleList([]) # encoder programs w/ fixed parameters
            self.enc_progs_learn = nn.ModuleList([]) # encoder programs for which we would like to learn parameters

        self.init_model()

    @property
    def num_progs_fixed(self):
        return len(self.enc_progs_fixed)

    @property
    def num_progs_learn(self):
        return len(self.enc_progs_learn)

    @property
    def num_progs(self):
        return self.num_progs_fixed+self.num_progs_learn

    @property
    def n_clusters(self):
        if self.z_prog_type == "discrete":
            if self.z_prog_dim == 1:
                return self.num_progs*2 # assume binary labels
            elif self.z_prog_dim > 1:
                return self.num_progs*self.z_prog_dim
        return 0 

    @property
    def z_neural_dim(self):
        z_neural_dim = self.z_dim - self.num_progs*self.z_prog_dim
        assert z_neural_dim >= 0, "total z_dim not enough dimensions given # of programs"
        return z_neural_dim
    
    def init_model(self):
        self.num_steps = 0

        if self.z_prog_dim > 0:
            self.has_programmatic_encoder = True

            if self.z_prog_type == "discrete":
                assert self.z_prog_dim >= 1 # at least 2 classes
                self.has_clusters = True

                if self.z_prog_dim == 1:
                    self.posterior_prog_dist = Bernoulli
                    self.posterior_prog_dist.prior_theta = self.prior_theta
                elif self.z_prog_dim > 1:
                    self.posterior_prog_dist = GumbelSoftmax

            elif self.z_prog_type == "continuous":
                assert self.z_prog_dim == 1
                self.has_clusters = False
                self.posterior_prog_dist = Normal

            else:
                raise NotImplementedError("please specify z_prog_type as discrete or continuous")

        # Programmatic Encoder
        if self.z_prog_dim > 0:
            if len(self.enc_progs_learn) > 0:
                for prog in self.enc_progs_learn:
                    prog.init_program()
            else:
                self.enc_progs_learn.append(StartFunction("list", "atom", self.state_dim+self.action_dim, self.z_prog_dim, self.rnn_dim))

            # Learn a state-independent log-variance, otherwise we'd need another program
            if self.z_prog_type == "continuous":
                self.encoder_prog_logvar = nn.Parameter(torch.randn(1), requires_grad=True)

        # Neural Encoder
        if self.z_neural_dim > 0:
            self.enc_birnn = nn.GRU(self.state_dim+self.action_dim, self.rnn_dim, 
                num_layers=self.num_layers, bidirectional=True)

            self.enc_fc = nn.Sequential(
                nn.Linear(2*self.rnn_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU())
            self.enc_mean = nn.Linear(self.h_dim, self.z_neural_dim)
            self.enc_logvar = nn.Linear(self.h_dim, self.z_neural_dim)

        # Decoder
        if self.decoder_type == 'rnn':
            self.dec_action_fc = nn.Sequential(
                nn.Linear(self.state_dim+self.z_dim+self.rnn_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU())
            self.dec_action_mean = nn.Linear(self.h_dim, self.action_dim)
            self.dec_action_logvar = nn.Linear(self.h_dim, self.action_dim)

            self.dec_rnn = nn.GRU(self.state_dim+self.action_dim, self.rnn_dim, num_layers=self.num_layers)
        
        elif self.decoder_type == 'mlp':
            self.is_recurrent = False
            
            self.dec_action_fc = nn.Sequential(
                nn.Linear(self.state_dim+self.z_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU())
            self.dec_action_mean = nn.Linear(self.h_dim, self.action_dim)
            self.dec_action_logvar = nn.Linear(self.h_dim, self.action_dim)

        elif self.decoder_type == 'linear':
            self.is_recurrent = False

            self.dec_action_mean = nn.Linear(self.state_dim+self.z_dim, self.action_dim)
            self.dec_action_logvar = nn.Linear(self.state_dim+self.z_dim, self.action_dim)
        
        else:
            raise NotImplementedError

    def configure_optimizers(self, lr=1e-3):
        # Turn off gradients for fixed programs
        for param in self.enc_progs_fixed.parameters():
            param.requires_grad = False

        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, betas = (0.5, 0.999))

    def set_enc_prog(self, program, enc_prog_type=None, prog_ind=None):
        if enc_prog_type == "fixed":
            self.enc_progs_fixed[prog_ind] = program
        elif enc_prog_type == "learn":
            self.enc_progs_learn[prog_ind] = program
        else:
            raise NotImplementedError

    def reset_policy(self, z=None, temperature=1.0, num_samples=0, device='cpu'):
        if z is None:
            assert num_samples > 0
            assert device is not None

            z_prog, z_neural = 0.0, 0.0

            if self.z_prog_dim > 0:
                z_prog_list = []
                for _ in range(self.num_progs):
                    z_prog_list.append(self.posterior_prog_dist.sample_prior(num_samples, self.z_prog_dim))
                z_prog = torch.cat(z_prog_list, dim=1)
            if self.z_neural_dim > 0:
                z_neural = Normal.sample_prior(num_samples, self.z_neural_dim)

            if self.z_neural_dim > 0 and self.z_prog_dim > 0:
                z = torch.cat([z_prog, z_neural], dim=1)
            else:
                z = z_prog + z_neural # one of them is 0.0
        
        self.z = z.to(device)
        self.temperature = temperature

        self.hidden = self.init_hidden_state(batch_size=z.size(0)).to(z.device)

    def encode_program_single(self, states, actions=None, batch_first=True, enc_prog_type=None, prog_ind=None):
        assert enc_prog_type in ["fixed", "learn"]
        assert isinstance(prog_ind, int)

        if not batch_first:
            states = states.transpose(0,1) # transpose to 1st dim time, 2nd dim batch

            if actions is not None:
                actions = actions.transpose(0,1)

        batch_lens = [states.size(1)] * states.size(0) # small hack to be compatible with NEAR code

        prog_input = states if actions is None else torch.cat([states, actions], dim=2)

        if enc_prog_type == "fixed":
            program_output = self.enc_progs_fixed[prog_ind].execute_on_batch(prog_input, batch_lens)
        elif enc_prog_type == "learn":
            program_output = self.enc_progs_learn[prog_ind].execute_on_batch(prog_input, batch_lens)
        else:
            raise NotImplementedError

        if self.z_prog_type == "discrete":
            if self.z_prog_dim == 1:
                enc_thetas = torch.sigmoid(program_output)
                return Bernoulli(enc_thetas)
            elif self.z_prog_dim > 1:
                enc_logits = F.log_softmax(program_output, dim=1)
                return GumbelSoftmax(enc_logits)

        elif self.z_prog_type == "continuous":
            enc_logvar = self.encoder_prog_logvar.expand(program_output.size())
            return Normal(program_output, enc_logvar)

    def encode_neural(self, states, actions=None, batch_first=False):
        if batch_first:
            # transpose to 1st dim time, 2nd dim batch
            states = states.transpose(0,1) 
            actions = actions.transpose(0,1)

        return self.encode(states, actions)

    def get_clusters(self, batch, sample=False):
        assert self.has_clusters
        assert self.n_clusters > 0

        labels_fixed, labels_learn = self.get_labels(batch, sample)
        labels = labels_fixed+labels_learn

        base = 2 if self.z_prog_dim == 1 else self.z_prog_dim
        clusters = 0.0

        for i in range(len(labels)):
            clusters += labels[i]*(base**i)

        return clusters.long()

    def get_labels(self, batch, sample=False):
        """Return labels of program outputs (both fixed and learn)."""

        labels_fixed = []
        for i in range(self.num_progs_fixed):
            labels_fixed.append(self.get_labels_single(batch, sample, enc_prog_type="fixed", prog_ind=i))

        labels_learn = []
        for i in range(self.num_progs_learn):
            labels_learn.append(self.get_labels_single(batch, sample, enc_prog_type="learn", prog_ind=i))

        return labels_fixed, labels_learn

    def get_labels_single(self, batch, sample=False, enc_prog_type=None, prog_ind=None):
        states, actions = batch
        assert actions.size(1)+1 == states.size(1) # final state has no corresponding action, also assert batch_first=True

        posterior_prog = self.encode_program_single(
            states[:,:-1], actions, enc_prog_type=enc_prog_type, prog_ind=prog_ind)

        if self.z_prog_type == "discrete":
            if sample:
                if self.z_prog_dim == 1:
                    return torch.distributions.Categorical(posterior_prog.theta).sample().unsqueeze(1)
                elif self.z_prog_dim > 1:
                    return torch.distributions.Categorical(posterior_prog.logits).sample()
            else:
                if self.z_prog_dim == 1:
                    return (posterior_prog.theta > 0.5).long()
                elif self.z_prog_dim > 1:
                    return torch.argmax(posterior_prog.logits, dim=1, keepdim=True)

        elif self.z_prog_type == "continuous":
            return posterior_prog.sample().detach() if sample else posterior_prog.mean.detach()

    def _shared_eval(self, batch, batch_idx, mode, return_all=False):
        if mode == 'train':
            self.num_steps += 1

        states, actions = batch
        batch_size = states.size(0)
        assert actions.size(1)+1 == states.size(1) # final state has no corresponding action
        
        # transpose to (time, batch, state/action_dim)
        states = states.transpose(0,1)
        actions = actions.transpose(0,1)

        loss_dict = {}
        z_prog_samples = [] # list of z_prog samples
        z_sample_prog, z_sample_neural = 0.0, 0.0

        # Encode programmatic latent and sample
        if self.z_prog_dim > 0:
            for i in range(len(self.enc_progs_fixed)):
                posterior_prog = self.encode_program_single(states[:-1], actions, batch_first=False, enc_prog_type="fixed", prog_ind=i)
                z_prog_samples.append(posterior_prog.sample()) # default for discrete is tau=0.5, hard=True

            for i in range(len(self.enc_progs_learn)):
                posterior_prog = self.encode_program_single(states[:-1], actions, batch_first=False, enc_prog_type="learn", prog_ind=i)
                z_prog_samples.append(posterior_prog.sample()) # default for discrete is tau=0.5, hard=True

                if not self.use_disc_capacity:
                    loss_dict[f'kld_prog_{i}'] = self.posterior_prog_dist.kld(posterior_prog).mean()
                    self.log(f'{mode}_kl_prog_{i}', loss_dict[f'kld_prog_{i}'])
                else:
                    # Linearly increase capacity of discrete channels
                    disc_min, disc_max, disc_num_iters, disc_gamma = self.disc_capacity
                    # Increase discrete capacity without exceeding disc_max or theoretical
                    # maximum (i.e. sum of log of dimension of each discrete variable)
                    disc_cap_current = (disc_max - disc_min) * self.num_steps / float(disc_num_iters) + disc_min
                    disc_cap_current = min(disc_cap_current, disc_max)
                    # Require float conversion here to not end up with numpy float
                    disc_theoretical_max = sum([float(np.log(disc_dim)) for disc_dim in [2]])#self.model.latent_spec['disc']])
                    disc_cap_current = min(disc_cap_current, disc_theoretical_max)
                    
                    # Calculate discrete capacity loss
                    disc_capacity_loss = disc_gamma * torch.abs(disc_cap_current - self.posterior_prog_dist.kld(posterior_prog).mean())

                    loss_dict[f'kld_prog_{i}'] = disc_capacity_loss
                    self.log(f'{mode}_kl_prog_{i}', self.posterior_prog_dist.kld(posterior_prog).mean())
                    self.log(f'{mode}_prog_capacity_{i}', disc_capacity_loss)

            z_sample_prog = torch.cat(z_prog_samples, dim=1)

        # Encode neural latent and sample
        if self.z_neural_dim > 0:
            posterior_neural = self.encode_neural(states[:-1], actions=actions, batch_first=False)

            kld_neural_true = Normal.kld(posterior_neural, free_bits=0.0).mean().detach()
            self.log(f'{mode}_kl_neur_true', kld_neural_true)      

            if not self.use_cont_capacity:
                loss_dict['kld_neural'] = Normal.kld(posterior_neural, free_bits=1/self.z_neural_dim).mean()
                self.log(f'{mode}_kl_neur', loss_dict['kld_neural'], prog_bar=True)
            else:
                # Linearly increase capacity of continuous channels
                cont_min, cont_max, cont_num_iters, cont_gamma = self.cont_capacity
                # Increase continuous capacity without exceeding cont_max
                cont_cap_current = (cont_max - cont_min) * self.num_steps / float(cont_num_iters) + cont_min
                cont_cap_current = min(cont_cap_current, cont_max)
                # Calculate continuous capacity loss
                cont_capacity_loss = cont_gamma * torch.abs(cont_cap_current - Normal.kld(posterior_neural, free_bits=1/self.z_neural_dim).mean())

                loss_dict['kld_neural'] = cont_capacity_loss
                self.log(f'{mode}_kl_neur', Normal.kld(posterior_neural, free_bits=1/self.z_neural_dim).mean(), prog_bar=True)
                self.log(f'{mode}_kl_neur_capacity', cont_capacity_loss, prog_bar=True)

            z_sample_neural = posterior_neural.sample()

        # Concat posterior(s) samples
        if self.z_neural_dim > 0 and self.z_prog_dim > 0:
            z_sample = torch.cat([z_sample_prog, z_sample_neural], dim=1)
        else:
            z_sample = z_sample_prog + z_sample_neural # one of them is 0.0

        self.reset_policy(z=z_sample, device=states.device)

        nll = 0.0
        for t in range(actions.size(0)):
            action_likelihood = self(states[t])
            nll -= action_likelihood.log_prob(actions[t]).mean()
            
            if self.is_recurrent:
                self.update_hidden(states[t], actions[t])
     
        loss_dict['nll'] = nll
        self.log(f'{mode}_nll', loss_dict['nll'], prog_bar=True)

        loss = sum(loss_dict.values())
        self.log(f'{mode}_loss', loss)

        if return_all:
            return loss, loss_dict, (z_prog_samples, z_sample_neural)
        else:
            return loss
