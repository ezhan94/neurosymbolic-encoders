import torch
import torch.nn as nn
import pytorch_lightning as pl

from lib.distributions import Normal


class BaseSequentialModel(pl.LightningModule):

    model_args = []
    is_recurrent = True

    def __init__(self):
        super().__init__()

        self.stage = 0 # some models have multi-stage training

    def _shared_eval(self, batch, batch_idx, mode):
        raise NotImplementedError        

    def training_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, mode='train')

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, mode='val')

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, mode='test')

    def forward(self, state):
        """Decodes the action for the next timestep."""

        dec_fc_input = torch.cat([state, self.z], dim=1)

        if self.is_recurrent:
            dec_fc_input = torch.cat([dec_fc_input, self.hidden[-1]], dim=1)

        dec_h = self.dec_action_fc(dec_fc_input) if hasattr(self, 'dec_action_fc') else dec_fc_input
        dec_mean = self.dec_action_mean(dec_h)

        if isinstance(self.dec_action_logvar, nn.Parameter):
            dec_logvar = self.dec_action_logvar
        else:
            dec_logvar = self.dec_action_logvar(dec_h)

        return Normal(dec_mean, dec_logvar)

    def encode(self, states, actions=None):
        enc_birnn_input = states
        if actions is not None:
            assert states.size(0) == actions.size(0)
            enc_birnn_input = torch.cat([states, actions], dim=-1)
        
        hiddens, _ = self.enc_birnn(enc_birnn_input)
        avg_hiddens = torch.mean(hiddens, dim=0)

        enc_fc_input = avg_hiddens

        enc_h = self.enc_fc(enc_fc_input) if hasattr(self, 'enc_fc') else enc_fc_input
        enc_mean = self.enc_mean(enc_h)
        enc_logvar = self.enc_logvar(enc_h)

        return Normal(enc_mean, enc_logvar)        

    def init_hidden_state(self, batch_size=1):
        return torch.zeros(self.num_layers, batch_size, self.rnn_dim)

    def update_hidden(self, state, action):
        state_action_pair = torch.cat([state, action], dim=1).unsqueeze(0)
        hiddens, self.hidden = self.dec_rnn(state_action_pair, self.hidden)
        return hiddens.to(state.device) # TODO weird bug, sometimes needs to move onto device

    def configure_optimizers(self, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer 

    def reset_policy(self, z=None, temperature=1.0, num_samples=0, device='cpu'):
        if z is None:
            assert num_samples > 0
            assert device is not None
            z = torch.randn(num_samples, self.z_dim).to(device)
        
        self.z = z
        self.temperature = temperature

        self.hidden = self.init_hidden_state(batch_size=z.size(0)).to(z.device)

    def act(self, state, sample=True):
        action_likelihood = self(state)
        action = action_likelihood.sample(temperature=self.temperature) if sample else action_likelihood.mean

        self.update_hidden(state, action)

        return action

    def generate_rollout(self, batch, temperature=1.0, burn_in=0, horizon=0, z=None):

        input_state, input_action = batch 

        self.reset_policy(z=z, temperature=temperature, num_samples=input_state.size(0), device=input_state.device)

        input_state = input_state.transpose(0,1)
        input_action = input_action.transpose(0,1)   

        states = [input_state[0].unsqueeze(0)]
        actions = []
        hiddens = []

        next_state = input_state[0]
        for t in range(horizon):
            curr_state = next_state
            if t < burn_in:
                action = input_action[t]
                curr_hiddens = self.update_hidden(curr_state, action)
                hiddens.append(curr_hiddens)
            else:
                action = self.act(curr_state)

            next_state = curr_state + action # transition model to next state
            states.append(next_state.unsqueeze(0))
            actions.append(action.unsqueeze(0))

        return torch.cat(states, dim=0).transpose(0,1), torch.cat(actions, dim=0).transpose(0,1) 


class VectorQuantizer(pl.LightningModule):
    """
    Based on Pytorch version here: https://github.com/MishaLaskin/vqvae
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, dim)
        quantization pipeline:
            1. get encoder input (B)
        """
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z.device)

        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        # TODO this loss is already averaged over batch
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach())**2)

        self.log('vq_embedding_loss', loss)        

        # preserve gradients
        z_q = z + (z_q - z).detach()
    
        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        self.log('vq_perplexity', perplexity)        

        return z_q, min_encodings, min_encoding_indices, loss, perplexity
