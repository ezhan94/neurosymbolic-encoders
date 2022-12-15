import torch
import torch.nn as nn

from lib.models.core import BaseSequentialModel, VectorQuantizer
from lib.distributions import Normal


class VQ_TVAE(BaseSequentialModel):

    name = 'vq_tvae' # vector quantized trajectory VAE
    has_programmatic_encoder = False
    has_clusters = True

    def __init__(self, model_config):
        super().__init__()

        self.num_layers = model_config["num_layers"]
        self.rnn_dim = model_config["rnn_dim"]  
        self.z_dim = model_config["z_dim"] 
        self.h_dim = model_config["h_dim"] 
        self.num_embeddings = model_config["num_embeddings"]     
        self.beta = model_config["beta"]     
        self.state_dim = model_config["state_dim"]
        self.action_dim = model_config["action_dim"]
        self.n_clusters = self.num_embeddings

        self.enc_birnn = nn.GRU(self.state_dim+self.action_dim, self.rnn_dim, 
            num_layers=self.num_layers, bidirectional=True)

        self.enc_fc = nn.Sequential(
            nn.Linear(2*self.rnn_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(self.h_dim, self.z_dim)

        self.vector_quantization = VectorQuantizer(
            self.num_embeddings, self.z_dim, self.beta)        

        self.dec_action_fc = nn.Sequential(
            nn.Linear(self.state_dim+self.z_dim+self.rnn_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())
        self.dec_action_mean = nn.Linear(self.h_dim, self.action_dim)
        self.dec_action_logvar = nn.Linear(self.h_dim, self.action_dim)

        self.dec_rnn = nn.GRU(self.state_dim+self.action_dim, 
            self.rnn_dim, num_layers=self.num_layers)

    def init_model(self):
        return

    def get_clusters(self, batch):
        states, actions = batch
        batch_size = states.size(0)

        assert actions.size(1)+1 == states.size(1) # final state has no corresponding action
        states = states.transpose(0,1)
        actions = actions.transpose(0,1)

        _, _, min_encoding_indices, _, _ = self.encode(states[:-1], actions=actions)

        return min_encoding_indices

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

        enc_q, min_encodings, min_encoding_indices, embedding_loss, perplexity = self.vector_quantization(enc_mean)

        return enc_q, min_encodings, min_encoding_indices, embedding_loss, perplexity

    def reset_policy(self, z=None, min_indices = None, temperature=1.0, num_samples=0, device='cpu'):
        if z is None:
            assert num_samples > 0
            assert device is not None

            if min_indices is None:
                min_indices = torch.randint(low = 0, high = self.num_embeddings, size = (num_samples,)).to(device)
            z = self.vector_quantization.embedding(min_indices)

        self.z = z
        self.temperature = temperature

        self.hidden = self.init_hidden_state(batch_size=z.size(0)).to(z.device)

    def _shared_eval(self, batch, batch_idx, mode):
        states, actions = batch

        assert actions.size(1)+1 == states.size(1) # final state has no corresponding action
        states = states.transpose(0,1)
        actions = actions.transpose(0,1)

        enc_q, min_encodings, min_encoding_indices, embedding_loss, perplexity = self.encode(states[:-1], actions=actions)

        self.log(f'{mode}_vq_embedding_loss', embedding_loss)      
        self.log(f'{mode}_vq_perplexity', perplexity)                

        self.reset_policy(z=enc_q)

        nll = 0.0
        for t in range(actions.size(0)):
            action_likelihood = self(states[t])
            nll -= action_likelihood.log_prob(actions[t]).mean()
            
            if self.is_recurrent:
                self.update_hidden(states[t], actions[t])
        
        self.log(f'{mode}_nll', nll, prog_bar=True)       

        loss = nll + embedding_loss
        self.log(f'{mode}_loss', loss)        
        return loss
