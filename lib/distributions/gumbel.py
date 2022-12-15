import math
import torch
import torch.nn.functional as F

from .core import Distribution


class GumbelSoftmax(Distribution):

    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def sample(self, tau=1.0, hard=False):
        # tau is temperature parameter
        return F.gumbel_softmax(self.logits, tau=tau, hard=hard)

    def log_prob(self, value, return_mean=False):
        import pdb; pdb.set_trace() # TODO

    @staticmethod
    def sample_prior(num_samples, sample_dim):
        # uniform prior
        unif = (1/sample_dim)*torch.ones(num_samples, sample_dim)
        inds = torch.multinomial(unif, 1)
        return torch.zeros(unif.size()).scatter_(-1, inds, 1)

    @staticmethod
    def kld(gumbel_1, gumbel_2=None, return_mean=False):
        '''
        Computes the kl-divergence between two gumble-softmax distributions.

        Args:
            gumbel_1 (GumbelSoftmax): first gumbel distribution
            gumbel_2 (GumbelSoftmax): second gumbel distribution (assume uniform if not provided)
        '''

        assert isinstance(gumbel_1, GumbelSoftmax)
        logits_1 = gumbel_1.logits

        if gumbel_2 is not None:
            assert isinstance(gumbel_2, GumbelSoftmax)
            logits_2 = gumbel_2.logits
        else:
            logits_2 = -math.log(logits_1.size(1)) # logits of uniform distribution

        kld_elements = torch.exp(logits_1) * (logits_1 - logits_2)
        kld = torch.sum(kld_elements, dim=-1)

        return kld
