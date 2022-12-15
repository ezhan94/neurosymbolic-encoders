import math
import torch
import torch.nn.functional as F

from .core import Distribution


class Bernoulli(Distribution):

    prior_theta = 0.5 # for class 1

    def __init__(self, theta):
        super().__init__()
        assert theta.size(1) == 1
        assert 0.0 <= torch.min(theta)
        assert torch.max(theta) <= 1.0
        self.theta = theta # for class 1

    def sample(self, tau=0.5, hard=True, eps=1e-8):
        # Sampling uses gumbel softmax under the hood
        logits = torch.log(torch.cat([1-self.theta, self.theta], dim=1) + eps)
        gumbel_samples = F.gumbel_softmax(logits, tau=tau, hard=hard)
        return gumbel_samples[:,1:2] # keep dims

    def log_prob(self, value, return_mean=False):
        import pdb; pdb.set_trace() # TODO

    def entropy(self, eps=1e-8):
        return - self.theta*torch.log(self.theta+eps) - (1.0-self.theta)*torch.log(1.0-self.theta+eps)

    @staticmethod
    def sample_prior(num_samples, sample_dim):
        return torch.bernoulli(self.prior_theta*torch.ones(num_samples, sample_dim))

    @staticmethod
    def kld(bernoulli_1, bernoulli_2=None, return_mean=False, eps=1e-8):
        '''
        Computes the kl-divergence between two bernoulli distributions.

        Args:
            bernoulli_1 (Bernoulli): first bernoulli distribution
            bernoulli_2 (Bernoulli): second bernoulli distribution (assume uniform if not provided)
        '''

        assert isinstance(bernoulli_1, Bernoulli)
        theta_1 = bernoulli_1.theta

        if bernoulli_2 is not None:
            assert isinstance(bernoulli_2, Bernoulli)
            theta_2 = bernoulli_2.theta
        else:
            theta_2 = Bernoulli.prior_theta

        kld_elements = theta_1*torch.log(theta_1/theta_2 + eps) + (1-theta_1)*torch.log((1-theta_1)/(1-theta_2) + eps)
        kld = torch.sum(kld_elements, dim=-1)

        return kld
