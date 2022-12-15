import torch
from .library_functions import LibraryFunction
from .neural_functions import init_neural_function


class BBallSpeed(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, player_reduce=None):
        super().__init__("atom", "atom", 20, 1, num_units, name="PlayerSpeed")

        if player_reduce is None:
            player_reduce = init_neural_function("atom", "atom", 5, 1, num_units)
        self.submodules["player_reduce"] = player_reduce

    def init_params(self):
        self.params['threshold'] = torch.nn.Parameter(torch.rand(1))
        self.params['scale'] = torch.nn.Parameter(torch.rand(1))

    def execute_on_batch(self, batch, batch_lens=None):
        n_players = 5
        assert batch.size(1) % n_players == 0

        actions = batch[:,2*n_players:].view(batch.size(0),-1,2) # actions are 2nd half of features
        speed = torch.linalg.norm(actions, dim=2)

        t = self.params['threshold'].to(batch.device)
        s = self.params['scale'].to(batch.device)
        out = self.submodules["player_reduce"].execute_on_batch(speed)

        return s*out - t

class BBallXPos(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, player_reduce=None):
        super().__init__("atom", "atom", 20, 1, num_units, name="PlayerXPos")

        if player_reduce is None:
            player_reduce = init_neural_function("atom", "atom", 5, 1, num_units)
        self.submodules["player_reduce"] = player_reduce

    def init_params(self):
        self.params['threshold'] = torch.nn.Parameter(torch.rand(1))
        self.params['scale'] = torch.nn.Parameter(torch.rand(1))

    def execute_on_batch(self, batch, batch_lens=None):
        n_players = 5
        assert batch.size(1) % n_players == 0

        pos = batch[:,:2*n_players].view(batch.size(0),-1,2) # positions are 1st half of features
        x_pos = pos[:,:,0]

        t = self.params['threshold'].to(batch.device)
        s = self.params['scale'].to(batch.device)
        out = self.submodules["player_reduce"].execute_on_batch(x_pos)

        return s*out - t

class BBallYPos(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, player_reduce=None):
        super().__init__("atom", "atom", 20, 1, num_units, name="PlayerYPos")

        if player_reduce is None:
            player_reduce = init_neural_function("atom", "atom", 5, 1, num_units)
        self.submodules["player_reduce"] = player_reduce

    def init_params(self):
        self.params['threshold'] = torch.nn.Parameter(torch.rand(1))
        self.params['scale'] = torch.nn.Parameter(torch.rand(1))

    def execute_on_batch(self, batch, batch_lens=None):
        n_players = 5
        assert batch.size(1) % n_players == 0

        pos = batch[:,:2*n_players].view(batch.size(0),-1,2) # positions are 1st half of features
        y_pos = pos[:,:,1]

        t = self.params['threshold'].to(batch.device)
        s = self.params['scale'].to(batch.device)
        out = self.submodules["player_reduce"].execute_on_batch(y_pos)

        return s*out - t

class BBallDist2Basket(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, player_reduce=None):
        super().__init__("atom", "atom", 20, 1, num_units, name="PlayerDist2Basket")

        if player_reduce is None:
            player_reduce = init_neural_function("atom", "atom", 5, 1, num_units)
        self.submodules["player_reduce"] = player_reduce

    def init_params(self):
        self.params['threshold'] = torch.nn.Parameter(torch.rand(1))
        self.params['scale'] = torch.nn.Parameter(torch.rand(1))

    def execute_on_batch(self, batch, batch_lens=None):
        n_players = 5
        assert batch.size(1) % n_players == 0

        pos = batch[:,:2*n_players].view(batch.size(0),-1,2) # positions are 1st half of features

        x = pos[:,:,0] + 0.213
        y = pos[:,:,1]
        dist2basket = torch.sqrt(x**2 + y**2)

        t = self.params['threshold'].to(batch.device)
        s = self.params['scale'].to(batch.device)
        out = self.submodules["player_reduce"].execute_on_batch(dist2basket)

        return s*out - t

class BBallPlayerAvg(LibraryFunction):

    def __init__(self, input_size, output_size, num_units):
        super().__init__("atom", "atom", 5, 1, num_units, name="PlayerAvg")

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        return torch.mean(batch, dim=1, keepdim=True)

class BBallPlayerMax(LibraryFunction):

    def __init__(self, input_size, output_size, num_units):
        super().__init__("atom", "atom", 5, 1, num_units, name="PlayerMax")

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        return torch.logsumexp(100*batch, dim=1, keepdim=True)/100

class BBallPlayerMin(LibraryFunction):

    def __init__(self, input_size, output_size, num_units):
        super().__init__("atom", "atom", 5, 1, num_units, name="PlayerMin")

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        return -torch.logsumexp(-100*batch, dim=1, keepdim=True)/100
